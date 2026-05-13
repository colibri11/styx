"""Tests миграции 0005_documents_chunks.sql (волна 19).

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import uuid

import psycopg
import psycopg.errors
import pytest

from styx.storage import migrate


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
            (table,),
        )
        return cur.fetchone() is not None


def _index_exists(conn: psycopg.Connection, index: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s", (index,)
        )
        return cur.fetchone() is not None


def _constraint_exists(conn: psycopg.Connection, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_constraint WHERE conname = %s", (name,)
        )
        return cur.fetchone() is not None


def test_migration_creates_documents_and_chunks(clean_db: str) -> None:
    """0005 на свежей БД создаёт обе таблицы + индексы."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        assert _table_exists(conn, "documents")
        assert _table_exists(conn, "chunks")
        # Indices.
        assert _index_exists(conn, "idx_documents_agent_created")
        assert _index_exists(conn, "idx_documents_source")
        assert _index_exists(conn, "uq_chunks_document_position")
        assert _index_exists(conn, "chunks_embedding_hnsw_idx")
        # CHECK constraints.
        assert _constraint_exists(conn, "chunks_position_nonneg")
        assert _constraint_exists(conn, "chunks_char_range_ordered")


def test_migration_idempotent(clean_db: str) -> None:
    """Повторный run() — no-op (идемпотентность migrate)."""
    migrate.run(clean_db)
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        assert _table_exists(conn, "documents")
        assert _table_exists(conn, "chunks")


def test_chunks_cascade_delete_on_documents(clean_db: str) -> None:
    """ON DELETE CASCADE для chunks при удалении document'а."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, char_count) "
                "VALUES ('alpha', 'memory_store', 1000) RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunks "
                "  (document_id, position, content, char_start, char_end) "
                "VALUES (%s, 0, %s, 0, 50)",
                (doc_id, "первый chunk"),
            )
            cur.execute(
                "INSERT INTO chunks "
                "  (document_id, position, content, char_start, char_end) "
                "VALUES (%s, 1, %s, 30, 80)",
                (doc_id, "второй chunk"),
            )
            conn.commit()

            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            conn.commit()

            cur.execute(
                "SELECT count(*) FROM chunks WHERE document_id = %s",
                (doc_id,),
            )
            assert cur.fetchone()[0] == 0


def test_chunks_position_uniqueness(clean_db: str) -> None:
    """UNIQUE (document_id, position) — двойной INSERT той же позиции
    падает."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, char_count) "
                "VALUES ('alpha', 'memory_store', 1000) RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunks "
                "  (document_id, position, content, char_start, char_end) "
                "VALUES (%s, 0, 'first', 0, 50)",
                (doc_id,),
            )
        conn.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks "
                    "  (document_id, position, content, char_start, char_end) "
                    "VALUES (%s, 0, 'duplicate position', 100, 150)",
                    (doc_id,),
                )
            conn.commit()
        conn.rollback()


def test_chunks_position_nonnegative_check(clean_db: str) -> None:
    """CHECK position ≥ 0."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, char_count) "
                "VALUES ('alpha', 'memory_store', 1000) RETURNING id"
            )
            doc_id = cur.fetchone()[0]
        conn.commit()

        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks "
                    "  (document_id, position, content, char_start, char_end) "
                    "VALUES (%s, -1, 'neg position', 0, 50)",
                    (doc_id,),
                )
            conn.commit()
        conn.rollback()


def test_chunks_char_range_ordered_check(clean_db: str) -> None:
    """CHECK char_start ≤ char_end."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, char_count) "
                "VALUES ('alpha', 'memory_store', 1000) RETURNING id"
            )
            doc_id = cur.fetchone()[0]
        conn.commit()

        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks "
                    "  (document_id, position, content, char_start, char_end) "
                    "VALUES (%s, 0, 'reversed', 100, 50)",
                    (doc_id,),
                )
            conn.commit()
        conn.rollback()
