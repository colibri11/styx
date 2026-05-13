"""Integration: engine.search_archive поверх real Postgres (волна 20).

Использует synthetic embeddings (контролируем cosine — детерминированно)
вместо реального Ollama. Проверяет end-to-end: queries hybrid SQL +
stitch + orchestrator forms.

Требует ``STYX_TEST_DATABASE_URL``.
"""

from __future__ import annotations

import datetime as _dt
import uuid

import psycopg
import pytest

from styx.engine import search_archive as _engine
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    return base


class _FakeEmbedder:
    """Возвращает заранее заданный embedding (для детерминированности)."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, text: str) -> list[float]:
        return self._vector


def test_search_documents_stitches_real_chunks(conn: psycopg.Connection) -> None:
    """Insert document + 3 contiguous chunks с overlap'ом → engine
    search_documents возвращает 1 stitched region."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [
        (0, "hello world ", _embed(1.0), 0, 12),
        (1, "world foo ",   _embed(0.9), 6, 16),
        (2, "foo bar",      _embed(0.8), 12, 19),
    ])
    conn.commit()

    embedder = _FakeEmbedder(_embed(1.0))
    resp = _engine.search_documents(
        queries=q, embedder=embedder, query="hello", limit=10,
    )
    assert resp.total_matched == 1
    region = resp.results[0]
    assert region.scope == "document"
    assert region.document_id == str(doc_id)
    assert region.chunk_positions == (0, 1, 2)
    # Overlap dedup: 'hello world ' + 'foo ' (chars 12-16 → overlap 6) +
    # 'bar' — порядок ASC.
    assert "hello" in region.text
    assert "bar" in region.text


def test_search_all_interleaves_docs_and_dialogue(conn: psycopg.Connection) -> None:
    """search_all возвращает alternating (doc, dialogue, doc, dialogue)."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [
        (0, "doc chunk first", _embed(1.0), 0, 15),
    ])
    q.insert_message(role="user", content="dialogue user msg", embedding=_embed(1.0))
    q.insert_message(role="assistant", content="dialogue assistant msg", embedding=_embed(0.95))
    conn.commit()

    embedder = _FakeEmbedder(_embed(1.0))
    resp = _engine.search_all(
        queries=q, embedder=embedder, query="msg chunk", limit=10,
    )
    scopes = [r.scope for r in resp.results]
    assert "document" in scopes
    assert "dialogue" in scopes
    # Alternate: первый — document (search_documents в search_all первым).
    assert scopes[0] == "document"
    assert scopes[1] == "dialogue"


def test_search_chunks_returns_individual_no_stitch(conn: psycopg.Connection) -> None:
    """scope='chunks' — каждый chunk как отдельный hit, БЕЗ группировки."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [
        (0, "chunk zero", _embed(1.0), 0, 10),
        (1, "chunk one",  _embed(0.9), 10, 19),
    ])
    conn.commit()

    embedder = _FakeEmbedder(_embed(1.0))
    resp = _engine.search_chunks(
        queries=q, embedder=embedder, query="chunk", limit=10,
    )
    assert resp.total_matched == 2
    # Оба — отдельные, scope='chunk'.
    assert all(r.scope == "chunk" for r in resp.results)
    positions = sorted(r.chunk_position for r in resp.results)
    assert positions == [0, 1]


def test_search_dialogue_finds_role_filtered(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_message(role="user", content="user реплика", embedding=_embed(1.0))
    q.insert_message(role="assistant", content="assistant ответ", embedding=_embed(0.9))
    q.insert_message(role="system", content="system note", embedding=_embed(0.95))
    conn.commit()

    embedder = _FakeEmbedder(_embed(1.0))
    resp = _engine.search_dialogue(
        queries=q, embedder=embedder, query="реплика ответ", limit=10,
    )
    roles = {r.role for r in resp.results}
    assert roles == {"user", "assistant"}
    assert "system" not in roles


def test_agent_isolation_full_stack(conn: psycopg.Connection) -> None:
    """Agent A не видит chunks или dialogue agent B."""
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")

    doc_a = qa.insert_document(source="memory_store", char_count=20)
    qa.insert_chunks_batch(doc_a, [(0, "alpha private chunk", _embed(1.0), 0, 19)])
    doc_b = qb.insert_document(source="memory_store", char_count=20)
    qb.insert_chunks_batch(doc_b, [(0, "beta private chunk", _embed(1.0), 0, 18)])

    qa.insert_message(role="user", content="alpha user msg", embedding=_embed(1.0))
    qb.insert_message(role="user", content="beta user msg", embedding=_embed(1.0))
    conn.commit()

    embedder = _FakeEmbedder(_embed(1.0))

    resp_a = _engine.search_all(
        queries=qa, embedder=embedder, query="chunk msg", limit=10,
    )
    resp_b = _engine.search_all(
        queries=qb, embedder=embedder, query="chunk msg", limit=10,
    )

    texts_a = {r.text for r in resp_a.results}
    texts_b = {r.text for r in resp_b.results}
    assert "alpha private chunk" in texts_a
    assert "alpha user msg" in texts_a
    assert "beta private chunk" not in texts_a
    assert "beta user msg" not in texts_a

    assert "beta private chunk" in texts_b
    assert "beta user msg" in texts_b
    assert "alpha private chunk" not in texts_b
    assert "alpha user msg" not in texts_b


def test_search_with_snapshot_filters_recent(conn: psycopg.Connection) -> None:
    """snapshot_cycle_start = past → ничего не возвращает."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=20)
    q.insert_chunks_batch(doc_id, [(0, "fresh chunk", _embed(1.0), 0, 11)])
    q.insert_message(role="user", content="fresh dialogue", embedding=_embed(1.0))
    conn.commit()

    embedder = _FakeEmbedder(_embed(1.0))
    cs_past = _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)
    resp = _engine.search_all(
        queries=q, embedder=embedder, query="fresh", limit=10,
        snapshot_cycle_start=cs_past,
    )
    assert resp.results == []
