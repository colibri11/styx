"""Tests расширенных query-методов для documents (волна 28).

Покрытия:
- ``insert_document`` с новыми параметрами (file_path, mime_type, etc).
- ``find_document_by_content_hash`` — hit/miss, agent-scoped.

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


def test_insert_document_with_file_metadata(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(
        source="ingest_document",
        char_count=4500,
        content_hash="deadbeef" * 8,
        file_path="/srv/docs/x.pdf",
        original_name="x.pdf",
        mime_type="application/pdf",
        source_ref="upload-42",
        size_bytes=123456,
        visibility="private",
        metadata={"page_count": 7},
    )
    conn.commit()
    assert isinstance(doc_id, uuid.UUID)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_path, original_name, mime_type, source_ref, "
            "       size_bytes, visibility, content_hash, metadata "
            "FROM documents WHERE id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    assert row is not None
    fp, name, mime, sref, size, vis, hsh, meta = row
    assert fp == "/srv/docs/x.pdf"
    assert name == "x.pdf"
    assert mime == "application/pdf"
    assert sref == "upload-42"
    assert size == 123456
    assert vis == "private"
    assert hsh == "deadbeef" * 8
    assert meta.get("page_count") == 7


def test_insert_document_legacy_nulls(conn: psycopg.Connection) -> None:
    """Без новых параметров — все колонки NULL (волна 19 backwards
    compat для store-routed рядов)."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    doc_id = q.insert_document(
        source="memory_store",
        char_count=3000,
        summary="tail summary",
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_path, mime_type, size_bytes, visibility "
            "FROM documents WHERE id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    assert row == (None, None, None, None)


def test_find_document_by_content_hash_hit(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    h = "feedface" * 8
    doc_id = q.insert_document(
        source="ingest_document",
        char_count=1000,
        content_hash=h,
        file_path="/x.pdf",
        original_name="x.pdf",
        mime_type="application/pdf",
        size_bytes=1000,
    )
    conn.commit()
    found = q.find_document_by_content_hash(h)
    assert found == doc_id


def test_find_document_by_content_hash_miss(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    assert q.find_document_by_content_hash("abcd" * 16) is None


def test_find_document_by_content_hash_agent_scoped(
    conn: psycopg.Connection,
) -> None:
    """Hash другого agent_id — не виден."""
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    h = "cafe" * 16
    qa.insert_document(
        source="ingest_document",
        char_count=500,
        content_hash=h,
        file_path="/a.pdf",
        original_name="a.pdf",
        mime_type="application/pdf",
        size_bytes=500,
    )
    conn.commit()
    assert qa.find_document_by_content_hash(h) is not None
    assert qb.find_document_by_content_hash(h) is None


def test_insert_document_unique_violation_on_dup_hash(
    conn: psycopg.Connection,
) -> None:
    """Тот же agent_id + тот же hash → UniqueViolation."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    h = "aabbccdd" * 8
    q.insert_document(
        source="ingest_document",
        char_count=100,
        content_hash=h,
        file_path="/x.pdf",
        original_name="x.pdf",
        mime_type="application/pdf",
        size_bytes=100,
    )
    conn.commit()
    with pytest.raises(psycopg.errors.UniqueViolation):
        q.insert_document(
            source="ingest_document",
            char_count=200,
            content_hash=h,
            file_path="/y.pdf",
            original_name="y.pdf",
            mime_type="application/pdf",
            size_bytes=200,
        )
        conn.commit()
    conn.rollback()
