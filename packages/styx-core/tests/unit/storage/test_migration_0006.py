"""Tests миграции 0006_chunks_fts.sql (волна 20).

Покрытия:
- ALTER ADD content_tsv generated применился, колонка существует.
- GIN idx_chunks_fts создан.
- Идемпотентность (повторный run — no-op).
- to_tsvector('simple', content) генерирует tsvector автоматически
  при INSERT.
- ts_rank против plainto_tsquery возвращает > 0 для matched.

Требует ``STYX_TEST_DATABASE_URL``.
"""

from __future__ import annotations

import psycopg
import pytest

from styx.storage import migrate


def _column_exists(conn: psycopg.Connection, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s",
            (table, column),
        )
        return cur.fetchone() is not None


def _index_exists(conn: psycopg.Connection, index: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s", (index,)
        )
        return cur.fetchone() is not None


def test_migration_creates_content_tsv_and_index(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        assert _column_exists(conn, "chunks", "content_tsv")
        assert _index_exists(conn, "idx_chunks_fts")


def test_migration_idempotent(clean_db: str) -> None:
    migrate.run(clean_db)
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        assert _column_exists(conn, "chunks", "content_tsv")
        assert _index_exists(conn, "idx_chunks_fts")


def test_content_tsv_auto_generated_on_insert(clean_db: str) -> None:
    """Generated STORED column — tsvector присутствует сразу после INSERT."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (agent_id, source, char_count) "
                "VALUES ('alpha', 'memory_store', 100) RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunks "
                "  (document_id, position, content, char_start, char_end) "
                "VALUES (%s, 0, 'hello world test', 0, 16) RETURNING id",
                (doc_id,),
            )
            chunk_id = cur.fetchone()[0]
            conn.commit()

            cur.execute(
                "SELECT content_tsv IS NOT NULL FROM chunks WHERE id = %s",
                (chunk_id,),
            )
            assert cur.fetchone()[0] is True


def test_ts_rank_returns_positive_for_match(clean_db: str) -> None:
    """ts_rank(content_tsv, plainto_tsquery('simple', 'hello')) > 0
    для chunk'а где есть слово 'hello'."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (agent_id, source, char_count) "
                "VALUES ('alpha', 'memory_store', 100) RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO chunks "
                "  (document_id, position, content, char_start, char_end) "
                "VALUES (%s, 0, 'hello world test', 0, 16)",
                (doc_id,),
            )
            conn.commit()

            cur.execute(
                "SELECT ts_rank(content_tsv, plainto_tsquery('simple', 'hello'), 32) "
                "FROM chunks WHERE document_id = %s",
                (doc_id,),
            )
            rank = cur.fetchone()[0]
            assert rank > 0

            cur.execute(
                "SELECT ts_rank(content_tsv, plainto_tsquery('simple', 'banana'), 32) "
                "FROM chunks WHERE document_id = %s",
                (doc_id,),
            )
            rank_no_match = cur.fetchone()[0]
            assert rank_no_match == 0
