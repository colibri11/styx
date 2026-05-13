"""Тесты AgentScopedQueries для search_archive (волна 20).

Покрытия:
- ``search_chunks_for_archive`` — hybrid score, agent isolation через
  JOIN на documents, фильтры (date_from/date_to/snapshot), exclusion
  по NULL embedding.
- ``search_dialogue_for_archive`` — hybrid поверх memories
  WHERE role IN ('user','assistant'), фильтры, agent isolation.

Требует ``STYX_TEST_DATABASE_URL``.
"""

from __future__ import annotations

import datetime as _dt

import psycopg
import pytest

from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _embed(seed: float, dim: int = 768) -> list[float]:
    """Embedding с одним ненулевым компонентом — даёт control над cosine."""
    base = [0.0] * dim
    base[0] = seed
    return base


# ── chunks ────────────────────────────────────────────────────────


def test_search_chunks_returns_topK_by_score(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=1000)
    q.insert_chunks_batch(doc_id, [
        (0, "первый chunk про яблоки", _embed(1.0), 0, 30),
        (1, "второй chunk про груши", _embed(0.5), 30, 60),
        (2, "третий chunk про арбузы", _embed(0.1), 60, 90),
    ])
    conn.commit()

    hits = q.search_chunks_for_archive(
        query_vector=_embed(1.0),
        query_text="яблоки",
        limit=2,
    )
    assert len(hits) == 2
    # Первый chunk (seed=1.0) match'ится embedding'у запроса (seed=1.0)
    # лучше всего → score выше.
    assert hits[0].position == 0
    assert hits[0].content == "первый chunk про яблоки"
    assert hits[0].score >= hits[1].score


def test_search_chunks_excludes_null_embedding(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [
        (0, "chunk c embedding", _embed(1.0), 0, 20),
        (1, "chunk БЕЗ embedding", None, 20, 40),
    ])
    conn.commit()
    hits = q.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="chunk", limit=10,
    )
    assert len(hits) == 1
    assert hits[0].position == 0


def test_search_chunks_agent_isolation(conn: psycopg.Connection) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    doc_a = qa.insert_document(source="memory_store", char_count=100)
    doc_b = qb.insert_document(source="memory_store", char_count=100)
    qa.insert_chunks_batch(doc_a, [(0, "alpha content", _embed(1.0), 0, 13)])
    qb.insert_chunks_batch(doc_b, [(0, "beta content", _embed(1.0), 0, 12)])
    conn.commit()

    hits_a = qa.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="content", limit=10,
    )
    hits_b = qb.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="content", limit=10,
    )
    assert len(hits_a) == 1
    assert hits_a[0].content == "alpha content"
    assert len(hits_b) == 1
    assert hits_b[0].content == "beta content"


def test_search_chunks_snapshot_filter(conn: psycopg.Connection) -> None:
    """chunks.created_at > cycle_start исключаются."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [(0, "test content", _embed(1.0), 0, 12)])
    conn.commit()

    # Snapshot в прошлом → chunk вне scope.
    cs = _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)
    hits = q.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="test",
        limit=10, snapshot_cycle_start=cs,
    )
    assert hits == []

    # Snapshot в будущем → chunk видимы.
    cs_future = _dt.datetime(2099, 1, 1, tzinfo=_dt.timezone.utc)
    hits2 = q.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="test",
        limit=10, snapshot_cycle_start=cs_future,
    )
    assert len(hits2) == 1


def test_search_chunks_date_filters(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [(0, "test content", _embed(1.0), 0, 12)])
    conn.commit()

    # date_from в будущем → пусто.
    hits = q.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="test",
        limit=10, date_from=_dt.datetime(2099, 1, 1, tzinfo=_dt.timezone.utc),
    )
    assert hits == []

    # date_to в прошлом → пусто.
    hits = q.search_chunks_for_archive(
        query_vector=_embed(1.0), query_text="test",
        limit=10, date_to=_dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc),
    )
    assert hits == []


# ── dialogue ──────────────────────────────────────────────────────


def test_search_dialogue_role_filter(conn: psycopg.Connection) -> None:
    """memories с role NOT IN ('user','assistant') не попадают."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_message(role="user", content="user реплика", embedding=_embed(1.0))
    q.insert_message(role="assistant", content="assistant ответ", embedding=_embed(0.9))
    q.insert_message(role="system", content="system note", embedding=_embed(0.95))
    conn.commit()

    hits = q.search_dialogue_for_archive(
        query_vector=_embed(1.0), query_text="реплика", limit=10,
    )
    roles = {h.role for h in hits}
    assert roles == {"user", "assistant"}
    assert "system" not in roles


def test_search_dialogue_agent_isolation(conn: psycopg.Connection) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    qa.insert_message(role="user", content="alpha user msg", embedding=_embed(1.0))
    qb.insert_message(role="user", content="beta user msg", embedding=_embed(1.0))
    conn.commit()

    hits_a = qa.search_dialogue_for_archive(
        query_vector=_embed(1.0), query_text="msg", limit=10,
    )
    hits_b = qb.search_dialogue_for_archive(
        query_vector=_embed(1.0), query_text="msg", limit=10,
    )
    assert len(hits_a) == 1
    assert hits_a[0].content == "alpha user msg"
    assert len(hits_b) == 1
    assert hits_b[0].content == "beta user msg"


def test_search_dialogue_excludes_superseded(conn: psycopg.Connection) -> None:
    """superseded_by IS NOT NULL — не возвращается."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    alive_id = q.insert_message(role="user", content="живая", embedding=_embed(1.0))
    dead_id = q.insert_message(role="user", content="мёртвая", embedding=_embed(1.0))
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by = %s WHERE id = %s",
            (alive_id, dead_id),
        )
    conn.commit()

    hits = q.search_dialogue_for_archive(
        query_vector=_embed(1.0), query_text="живая", limit=10,
    )
    ids = {h.memory_id for h in hits}
    assert alive_id in ids
    assert dead_id not in ids


def test_search_dialogue_excludes_null_embedding(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_message(role="user", content="с embed", embedding=_embed(1.0))
    q.insert_message(role="user", content="без embed", embedding=None)
    conn.commit()
    hits = q.search_dialogue_for_archive(
        query_vector=_embed(1.0), query_text="embed", limit=10,
    )
    contents = {h.content for h in hits}
    assert "с embed" in contents
    assert "без embed" not in contents
