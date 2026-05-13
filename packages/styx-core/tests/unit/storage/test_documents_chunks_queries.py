"""Тесты AgentScopedQueries методов для documents + chunks (волна 19).

Покрытия:
- ``insert_document`` — agent-scoped INSERT, RETURNING id, метаданные.
- ``insert_chunks_batch`` — batch INSERT, sequential position'ы,
  embedding'и сохраняются.
- ``insert_memory`` с ``archive_ref`` — поле jsonb сохраняется и
  читается обратно.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import uuid

import psycopg
import psycopg.errors
import pytest

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


def test_insert_document_returns_id(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(
        source="memory_store",
        char_count=4500,
        summary="первый chunk truncate",
        metadata={"k": "v"},
    )
    conn.commit()
    assert isinstance(doc_id, uuid.UUID)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT agent_id, source, char_count, summary, metadata "
            "FROM documents WHERE id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    assert row is not None
    agent_id, source, char_count, summary, metadata = row
    assert agent_id == "alpha"
    assert source == "memory_store"
    assert char_count == 4500
    assert summary == "первый chunk truncate"
    assert metadata == {"k": "v"}


def test_insert_document_no_metadata_default_empty(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(
        source="memory_store", char_count=3000,
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT metadata FROM documents WHERE id = %s", (doc_id,)
        )
        assert cur.fetchone()[0] == {}


def test_insert_chunks_batch_writes_all_positions(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(
        source="memory_store", char_count=3000,
    )
    chunks = [
        (0, "первый chunk", _embed(0.1), 0, 50),
        (1, "второй chunk", _embed(0.2), 30, 80),
        (2, "третий chunk", _embed(0.3), 60, 110),
    ]
    q.insert_chunks_batch(doc_id, chunks)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT position, content, char_start, char_end, "
            "       (embedding IS NOT NULL) AS has_embed "
            "FROM chunks WHERE document_id = %s ORDER BY position",
            (doc_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 3
    for i, (position, content, char_start, char_end, has_embed) in enumerate(rows):
        assert position == i
        assert content == chunks[i][1]
        assert char_start == chunks[i][3]
        assert char_end == chunks[i][4]
        assert has_embed is True


def test_insert_chunks_batch_with_null_embedding(conn: psycopg.Connection) -> None:
    """embedding=None → INSERT с NULL (chunk без вектора, recall не
    найдёт но row не теряется)."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [(0, "no embed chunk", None, 0, 14)])
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT (embedding IS NULL) AS no_embed FROM chunks "
            "WHERE document_id = %s",
            (doc_id,),
        )
        assert cur.fetchone()[0] is True


def test_insert_chunks_batch_empty_noop(conn: psycopg.Connection) -> None:
    """Empty list — никаких INSERT'ов, никакой ошибки."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=100)
    q.insert_chunks_batch(doc_id, [])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s", (doc_id,)
        )
        assert cur.fetchone()[0] == 0


def test_insert_memory_with_archive_ref(conn: psycopg.Connection) -> None:
    """Расширение insert_memory: archive_ref jsonb сохраняется."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(source="memory_store", char_count=4000)

    archive_ref = {
        "kind": "document",
        "id": str(doc_id),
        "locator": f"styx://store/{doc_id}",
        "snippet": "первые 1000 chars первого chunk'а",
    }
    tail_id = q.insert_memory(
        role="system",
        content="truncate summary…",
        kind="note",
        kind_src="subjective_tail",
        embedding=_embed(0.1),
        archive_ref=archive_ref,
    )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT archive_ref, kind_src FROM memories WHERE id = %s",
            (tail_id,),
        )
        row = cur.fetchone()
    assert row is not None
    saved_ref, kind_src = row
    assert saved_ref == archive_ref
    assert kind_src == "subjective_tail"


def test_insert_memory_without_archive_ref_keeps_null(conn: psycopg.Connection) -> None:
    """Backward compat: insert_memory без archive_ref оставляет колонку
    NULL (не пишет {} dict)."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = q.insert_memory(
        role="summary",
        content="без archive",
        kind="note",
        kind_src="subjective",
        embedding=_embed(0.1),
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT archive_ref FROM memories WHERE id = %s", (mid,)
        )
        assert cur.fetchone()[0] is None


def test_documents_agent_isolation(conn: psycopg.Connection) -> None:
    """Документы alpha не видны при alpha-scoped query через
    information_schema-pull (sanity)."""
    q_alpha = AgentScopedQueries(conn, agent_id="alpha")
    q_beta = AgentScopedQueries(conn, agent_id="beta")
    doc_alpha = q_alpha.insert_document(source="memory_store", char_count=100)
    doc_beta = q_beta.insert_document(source="memory_store", char_count=100)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT agent_id, count(*) FROM documents GROUP BY agent_id")
        rows = dict(cur.fetchall())
    assert rows["alpha"] == 1
    assert rows["beta"] == 1
    assert doc_alpha != doc_beta
