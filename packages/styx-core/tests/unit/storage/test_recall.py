"""Тесты recall_full pipeline."""

from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

from styx.embedding import EmbeddingError, FakeEmbeddingClient
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries
from styx.storage.recall import format_recall_text, recall_full
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _seed(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    content: str,
    embed_client: FakeEmbeddingClient,
    role: str = "user",
    kind: str = "episode",
) -> uuid.UUID:
    sid = uuid.uuid4()
    vec = embed_client.embed(content)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, agent_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (sid, agent_id),
        )
        cur.execute(
            "INSERT INTO memories "
            "(agent_id, session_id, role, content, kind, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (
                agent_id,
                sid,
                role,
                content,
                kind,
                "[" + ",".join(repr(x) for x in vec) + "]",
            ),
        )
        return cur.fetchone()[0]


def test_recall_full_returns_relevant_memories(conn: psycopg.Connection) -> None:
    """Same FakeEmbeddingClient → одинаковый seed → query 'apples' максимально
    близко к memory 'apples and pears'."""
    embed = FakeEmbeddingClient()
    agent = "alpha"
    target = _seed(conn, agent_id=agent, content="apples and pears", embed_client=embed)
    _seed(conn, agent_id=agent, content="completely different topic", embed_client=embed)
    _seed(conn, agent_id=agent, content="another unrelated thing", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id=agent)
    result = recall_full(
        queries=q, embed_client=embed, query="apples and pears",
    )
    conn.commit()

    assert len(result.memories) >= 1
    # Топ-1 — точное совпадение query == content (FakeEmbedding детерминирован).
    assert result.memories[0].id == target
    assert result.memories[0].score > 0.32  # min_score дефолт (волна 8: 0.6 → 0.32)


def test_recall_full_filters_by_min_score(conn: psycopg.Connection) -> None:
    """Если все score ниже min_score — пустой результат."""
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="beta", content="x", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="beta")
    # Высокий min_score → ничего не пройдёт (ортогональный query).
    from dataclasses import replace
    cfg = replace(DEFAULT_RECALL_CONFIG.full, min_score=0.99)

    result = recall_full(
        queries=q, embed_client=embed, query="totally different query",
        full_config=cfg,
    )
    assert result.memories == []
    assert result.queried_count >= 0


def test_recall_full_records_recall_events(conn: psycopg.Connection) -> None:
    embed = FakeEmbeddingClient()
    target = _seed(conn, agent_id="gamma", content="record me", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="gamma")
    recall_full(queries=q, embed_client=embed, query="record me")
    conn.commit()

    qhash = hashlib.sha256(b"record me").digest()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT match_score, query_hash FROM recall_events "
            "WHERE memory_id = %s",
            (target,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    score, stored_hash = rows[0]
    assert bytes(stored_hash) == qhash
    assert score > 0


def test_recall_full_skips_recall_events_when_disabled(
    conn: psycopg.Connection,
) -> None:
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="delta", content="x", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="delta")
    recall_full(
        queries=q, embed_client=embed, query="x", record_events=False,
    )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM recall_events")
        assert cur.fetchone()[0] == 0


def test_recall_full_handles_embed_error_gracefully() -> None:
    """EmbeddingError при embed query → пустой результат, без падения."""

    class _BadEmbed:
        @property
        def dim(self) -> int:
            return 768

        def embed(self, text: str) -> list[float]:
            raise EmbeddingError("ollama unavailable")

    # Без conn здесь не запустить search_similar — но мы вылетаем раньше
    # через embed_client.embed(query).
    class _StubQueries:
        agent_id = "x"

        def search_similar(self, **kwargs):
            raise AssertionError("не должно быть вызвано")

        def record_recall_event(self, **kwargs):
            raise AssertionError("не должно быть вызвано")

    result = recall_full(
        queries=_StubQueries(),  # type: ignore[arg-type]
        embed_client=_BadEmbed(),
        query="anything",
    )
    assert result.memories == []
    assert result.queried_count == 0


def test_recall_full_internal_dedup_runs(conn: psycopg.Connection) -> None:
    """Два почти-идентичных текста → одна победа."""
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="epsilon", content="apples are great", embed_client=embed)
    # Идентичный embedding (FakeEmbedding детерминирован) → cluster collapse.
    a2 = _seed(conn, agent_id="epsilon", content="apples are great", embed_client=embed)
    _seed(conn, agent_id="epsilon", content="absolutely different topic", embed_client=embed)
    conn.commit()
    # Заметка: оба ряда «apples are great» имеют одинаковые embedding и
    # почти-одинаковые scores. internal_dedup склеит их в один.
    _ = a2

    q = AgentScopedQueries(conn, agent_id="epsilon")
    result = recall_full(
        queries=q, embed_client=embed, query="apples are great",
    )
    conn.commit()

    # Дубликат должен быть отсечён.
    assert result.internal_duplicates_removed >= 1
    contents = [m.content for m in result.memories]
    assert contents.count("apples are great") == 1


def test_format_recall_text_empty() -> None:
    from styx.storage.recall import RecallResult
    r = RecallResult(memories=[], queried_count=0, internal_duplicates_removed=0)
    assert format_recall_text(r) == "<no memories matched>"


def test_format_recall_text_renders_memories(conn: psycopg.Connection) -> None:
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="zeta", content="hello world", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="zeta")
    result = recall_full(queries=q, embed_client=embed, query="hello world")
    conn.commit()

    text = format_recall_text(result)
    assert "hello world" in text
    assert "score=" in text
    assert "role=user" in text
