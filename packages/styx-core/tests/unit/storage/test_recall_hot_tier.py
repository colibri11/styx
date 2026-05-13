"""Тесты интеграции hot_tier в recall_full — host-level (без real БД).

Полный pipeline через psycopg+pgvector — в test_recall.py / e2e волны 11.
Здесь — supplement, dedup, put-on-success, snapshot fence, disabled fallback.
"""

from __future__ import annotations

import datetime as _dt
import math
import uuid

import pytest

from styx.embedding import FakeEmbeddingClient
from styx.engine import hot_tier
from styx.storage.queries import MemoryHit
from styx.storage.recall import recall_full
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG
from styx.turn_state import RecallSnapshot


@pytest.fixture(autouse=True)
def _reset_hot():
    hot_tier.reset_all()
    yield
    hot_tier.reset_all()


# -- fakes ----------------------------------------------------------------


class _FakeConn:
    """Cursor падает — read_baseline_for_scoring fail-open вернёт None."""

    def cursor(self, *args, **kwargs):
        raise RuntimeError("baseline read disabled in fake")


class _FakeQueries:
    def __init__(self, *, db_hits: list[MemoryHit], agent_id: str = "test-agent"):
        self._db_hits = db_hits
        self.agent_id = agent_id
        self.conn = _FakeConn()
        self.events_recorded: list[uuid.UUID] = []
        self.last_accessed: list[uuid.UUID] = []

    def search_similar(self, **kwargs) -> list[MemoryHit]:  # noqa: ANN001
        return list(self._db_hits)

    def record_recall_event(self, *, memory_id, query_hash, match_score, session_id) -> int:
        self.events_recorded.append(memory_id)
        return len(self.events_recorded)

    def update_last_accessed_at(self, ids):
        self.last_accessed = list(ids)


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _hit(
    *,
    embedding: list[float],
    score: float = 0.5,
    content: str = "memory",
    kind_src: str = "subjective",
    agent_id: str = "test-agent",
    created_at: _dt.datetime | None = None,
    mid: uuid.UUID | None = None,
) -> MemoryHit:
    return MemoryHit(
        id=mid or uuid.uuid4(),
        agent_id=agent_id,
        kind="subjective_dialogue",
        kind_src=kind_src,
        role="user",
        content=content,
        metadata={},
        created_at=created_at or _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc),
        score=score,
        match_score=score,
        embedding=embedding,
    )


# -- supplement -----------------------------------------------------------


def test_supplement_adds_hot_when_db_empty() -> None:
    """БД пуста, hot имеет item с близким embedding'ом → возвращается из hot."""
    hot_tier.configure("test-agent")
    embed = FakeEmbeddingClient()
    qvec = embed.embed("focus topic")
    hot_tier.put_many("test-agent", [_hit(embedding=qvec, content="from hot", score=0.8)])

    queries = _FakeQueries(db_hits=[])
    result = recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="focus topic",
    )
    assert len(result.memories) == 1
    assert result.memories[0].content == "from hot"


def test_supplement_below_min_score_filtered() -> None:
    """Hot item с cosine ниже min_score → filter отсечёт после объединения."""
    hot_tier.configure("test-agent")
    embed = FakeEmbeddingClient()
    # Hot item с ортогональным embed'ом — cosine ≈ 0.
    hot_tier.put_many("test-agent", [_hit(embedding=_unit([1.0, 0.0, 0.0] + [0.0] * 765), content="far")])

    queries = _FakeQueries(db_hits=[])
    result = recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="something completely different",
    )
    # FakeEmbedding для query — не совпадает с _unit'ом → cosine низкий → filter отсёк.
    assert result.memories == []


def test_dedup_db_wins_on_id_collision() -> None:
    """Один и тот же id в БД и hot → БД-версия (с composite score)."""
    hot_tier.configure("test-agent")
    embed = FakeEmbeddingClient()
    qvec = embed.embed("query")
    shared_id = uuid.uuid4()

    db_hit = _hit(
        embedding=qvec, mid=shared_id, content="db version", score=0.95
    )
    hot_hit = _hit(
        embedding=qvec, mid=shared_id, content="hot version", score=0.5
    )
    hot_tier.put_many("test-agent", [hot_hit])

    queries = _FakeQueries(db_hits=[db_hit])
    result = recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="query",
    )
    assert len(result.memories) == 1
    assert result.memories[0].content == "db version"
    assert math.isclose(result.memories[0].score, 0.95, abs_tol=1e-6)


def test_put_after_successful_recall() -> None:
    """После recall_full hits в hot — следующий scan их находит."""
    hot_tier.configure("test-agent")
    embed = FakeEmbeddingClient()
    qvec = embed.embed("query A")
    db_hit = _hit(embedding=qvec, content="A", score=0.9)

    queries = _FakeQueries(db_hits=[db_hit])
    result = recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="query A",
    )
    assert len(result.memories) == 1
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert db_hit.id in s.entries


def test_no_put_on_empty_recall() -> None:
    """Recall с пустым результатом — ничего не кладёт в hot."""
    hot_tier.configure("test-agent")
    embed = FakeEmbeddingClient()
    queries = _FakeQueries(db_hits=[])
    recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="nothing",
    )
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert s.entries == {}


# -- snapshot fence -------------------------------------------------------


def test_snapshot_fence_skips_hot_after_cycle_start() -> None:
    """Hot item с created_at > cycle_start и kind_src=objective → отсекается."""
    hot_tier.configure("test-agent")
    embed = FakeEmbeddingClient()
    qvec = embed.embed("topic")
    cycle = _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    after_cycle = cycle + _dt.timedelta(minutes=5)
    hot_tier.put_many("test-agent", [_hit(
        embedding=qvec, kind_src="objective", created_at=after_cycle,
        content="future objective",
    )])

    queries = _FakeQueries(db_hits=[])
    snap = RecallSnapshot(cycle_start=cycle, agent_id="test-agent")
    result = recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="topic",
        snapshot=snap,
    )
    assert result.memories == []


# -- disabled fallback ----------------------------------------------------


def test_disabled_hot_no_supplement_no_put() -> None:
    """Hot disabled → supplement пуст, put no-op (state остаётся None)."""
    # Не зовём configure() — hot_tier остаётся disabled.
    embed = FakeEmbeddingClient()
    qvec = embed.embed("query")
    db_hit = _hit(embedding=qvec, content="db only", score=0.9)

    queries = _FakeQueries(db_hits=[db_hit])
    result = recall_full(
        queries=queries,  # type: ignore[arg-type]
        embed_client=embed,
        query="query",
    )
    assert len(result.memories) == 1
    assert hot_tier.get_state("test-agent") is None
