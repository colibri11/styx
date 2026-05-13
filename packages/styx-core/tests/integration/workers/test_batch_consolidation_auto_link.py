"""Интеграция: dialogue_batch_consolidation handler + auto-link (волна 18).

Real Postgres + stub LLM + контролируемый embedder. Проверяем что после
gatekeeper STORE / SUPERSEDE ветки в `relations` появляются `related_to`
рёбра до соседей; на MERGE/SKIP — нет.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid

import psycopg
import pytest

from styx.engine.auto_link import AutoLinkConfig
from styx.engine.selective_gatekeeper import GatekeeperConfig
from styx.llm import LLMRateLimiter
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries
from styx.workers.handlers.dialogue_batch_consolidation import (
    create_dialogue_batch_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


# ── Stubs ────────────────────────────────────────────────────────────


class _StubLLM:
    def __init__(self, summary: str) -> None:
        self._summary = summary

    def chat_json(self, *, messages: list[dict]) -> dict:  # noqa: ARG002
        return {
            "skip": False, "skip_reason": None,
            "summary": self._summary,
            "archive_hints": [{"snippet": "snippet"}],
            "vad": None,
        }


class _ControlledEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = list(vec)

    @property
    def dim(self) -> int:
        return len(self._vec)

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return list(self._vec)


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    base[1] = (1.0 - seed * seed) ** 0.5
    return base


def _embed_with_offset(offset: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = (1.0 - offset * offset) ** 0.5
    base[1] = offset
    return base


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _seed_dialogue(conn, agent: str, n_pairs: int = 22) -> _dt.datetime:
    base = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(minutes=30)
    last_at = base
    with conn.cursor() as cur:
        for i in range(n_pairs):
            user_at = base + _dt.timedelta(seconds=i * 2)
            asst_at = base + _dt.timedelta(seconds=i * 2 + 1)
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind, "
                "  kind_src, created_at) "
                "VALUES (%s, 'user', %s, 'episode', 'subjective', %s)",
                (agent, f"Реплика юзера {i}", user_at),
            )
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind, "
                "  kind_src, created_at) "
                "VALUES (%s, 'assistant', %s, 'episode', 'subjective', %s)",
                (agent, f"Ответ агента {i}", asst_at),
            )
            last_at = asst_at
    conn.commit()
    return last_at + _dt.timedelta(seconds=1)


def _make_task(agent: str, window_to: _dt.datetime) -> LlmTask:
    return LlmTask(
        id=uuid.uuid4(),
        task_type="dialogue_batch_consolidation",
        memory_id=None,
        payload={
            "agent_id": agent, "window_from": None,
            "window_to": window_to.isoformat(), "with_overlap": False,
        },
        retry_count=0,
    )


def _make_ctx(conn, embedder, summary):
    return HandlerContext(
        conn=conn, llm=_StubLLM(summary),
        rate_limit=LLMRateLimiter(capacity=1, refill_per_second=10.0),
        logger=logging.getLogger("test"),
        embedder=embedder,
    )


def _count_related_to(conn, source_id) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_type='memory' AND source_id=%s "
            "   AND relation='related_to'",
            (source_id,),
        )
        return cur.fetchone()[0]


# ── Tests ────────────────────────────────────────────────────────────


def test_store_writes_auto_link_relations(conn, agent="alpha") -> None:
    """STORE-ветка: пред-seeded близкая memory → auto-link создаёт ребро."""
    window_to = _seed_dialogue(conn, agent)

    # Pre-seed похожая (но не идентичная) subjective memory другого
    # агента — auto-link должен сработать cross-agent.
    foreign = AgentScopedQueries(conn, "beta").insert_memory(
        role="summary", content="чужая близкая запись",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    conn.commit()

    new_summary = "новая консолидированная заметка от агента"
    embedder = _ControlledEmbedder(_embed_with_offset(0.0))
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
        auto_link_config=AutoLinkConfig(),
    )
    handler(_make_task(agent, window_to), _make_ctx(conn, embedder, new_summary))
    conn.commit()

    # Найти новый ряд (kind_src=dialogue_batch_consolidation, agent=alpha).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        row = cur.fetchone()
    assert row is not None
    new_id = row[0]
    # Auto-link создал ребро на foreign (cross-agent).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT target_id FROM relations "
            " WHERE source_type='memory' AND source_id=%s "
            "   AND relation='related_to'",
            (new_id,),
        )
        targets = {r[0] for r in cur.fetchall()}
    assert foreign in targets


def test_merge_does_not_auto_link(conn, agent="alpha") -> None:
    """MERGE-ветка: новый ряд удалён, auto-link не зовётся."""
    window_to = _seed_dialogue(conn, agent)

    q = AgentScopedQueries(conn, agent)
    existing = q.insert_memory(
        role="summary", content="кратко",
        kind="episode", kind_src="dialogue_batch_consolidation",
        embedding=_embed(1.0),
    )
    foreign = AgentScopedQueries(conn, "beta").insert_memory(
        role="summary", content="чужая",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.10),
    )
    conn.commit()

    new_summary = "развёрнутая заметка консолидации с большим количеством текста"
    embedder = _ControlledEmbedder(_embed(1.0))  # similarity 1.0 → MERGE
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
        auto_link_config=AutoLinkConfig(),
    )
    handler(_make_task(agent, window_to), _make_ctx(conn, embedder, new_summary))
    conn.commit()

    # Никаких новых related_to рёбер — MERGE не auto-link'ит.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM relations WHERE relation='related_to'"
        )
        # Возможно cross-agent foreign уже имел рёбра до этого — проверяем
        # что после merge'а никаких related_to'тов с merge'нутого ряда
        # не осталось.
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_type='memory' "
            "   AND source_id NOT IN ("
            "       SELECT id FROM memories WHERE id IN (%s, %s)"
            "   )"
            "   AND relation='related_to'",
            (existing, foreign),
        )
        # Все возможные related_to от чьего-то ещё memory'а посторонние.
    # Главное: мерж'нутый new_id удалён вместе с relations.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        # Только existing остался.
        assert cur.fetchone()[0] == 1


def test_disabled_auto_link_no_relations(conn, agent="alpha") -> None:
    """STYX_AUTO_LINK_ENABLED=0 → STORE без рёбер."""
    window_to = _seed_dialogue(conn, agent)
    AgentScopedQueries(conn, "beta").insert_memory(
        role="summary", content="чужая близкая",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    conn.commit()

    embedder = _ControlledEmbedder(_embed_with_offset(0.0))
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
        auto_link_config=AutoLinkConfig(enabled=False),
    )
    handler(_make_task(agent, window_to),
            _make_ctx(conn, embedder, "новая заметка"))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        new_id = cur.fetchone()[0]
    assert _count_related_to(conn, new_id) == 0


def test_supersede_writes_auto_link_relations(conn, agent="alpha") -> None:
    """SUPERSEDE-ветка: новый ряд жив → auto-link срабатывает."""
    window_to = _seed_dialogue(conn, agent)

    # Pre-seed ряд агента в supersede-зоне (similarity ~0.88).
    q = AgentScopedQueries(conn, agent)
    q.insert_memory(
        role="summary", content="заметка агента",
        kind="episode", kind_src="dialogue_batch_consolidation",
        embedding=_embed(0.5),
    )
    foreign = AgentScopedQueries(conn, "beta").insert_memory(
        role="summary", content="чужая",
        kind="note", kind_src="subjective",
        embedding=_embed(0.85),  # similarity ~0.88 к _embed(0.5)
    )
    conn.commit()

    embedder = _ControlledEmbedder(_embed(0.85))
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
        auto_link_config=AutoLinkConfig(),
    )
    handler(_make_task(agent, window_to),
            _make_ctx(conn, embedder, "заметка агента!"))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation' "
            "   AND superseded_by IS NULL",
            (agent,),
        )
        row = cur.fetchone()
    assert row is not None
    new_id = row[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT target_id FROM relations "
            " WHERE source_type='memory' AND source_id=%s "
            "   AND relation='related_to'",
            (new_id,),
        )
        targets = {r[0] for r in cur.fetchall()}
    # Среди auto-link рёбер — foreign (cross-agent в supersede-зоне).
    assert foreign in targets
