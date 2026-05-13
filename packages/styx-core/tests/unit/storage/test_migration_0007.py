"""Tests миграции 0007_documents_pipeline.sql (волна 28).

Покрытия:
- ALTER TABLE — 6 nullable колонок добавляются.
- Backfill из metadata JSONB корректно поднимает promoted ключи.
- Cleanup metadata убирает promoted ключи.
- Повторная apply — идемпотентна.
- Partial UNIQUE на (agent_id, content_hash) WHERE NOT NULL работает.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import json
import uuid

import psycopg
import psycopg.errors
import pytest

from styx.storage import migrate
from styx.storage.migrate import Migration, apply, discover_migrations


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


def test_migration_adds_six_columns(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        for col in (
            "file_path",
            "original_name",
            "mime_type",
            "source_ref",
            "size_bytes",
            "visibility",
        ):
            assert _column_exists(conn, "documents", col), col


def test_migration_partial_unique_index(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        assert _index_exists(conn, "uq_documents_agent_content_hash")


def test_migration_idempotent(clean_db: str) -> None:
    """Повторный run() — no-op."""
    migrate.run(clean_db)
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        for col in ("file_path", "mime_type", "size_bytes", "visibility"):
            assert _column_exists(conn, "documents", col)


def _split_migrations() -> tuple[list[Migration], list[Migration]]:
    """Раздели миграции на ≤ 0006 и [0007]. Для тестов backfill'а:
    применяем pre-0007 миграции, INSERT-имитируем post-migration-kit-27
    ряд с metadata, потом применяем 0007 → backfill срабатывает."""
    all_migs = discover_migrations()
    target = "0007_documents_pipeline.sql"
    before = [m for m in all_migs if m.name < target]
    at_target = [m for m in all_migs if m.name == target]
    assert at_target, f"миграция {target} не найдена"
    return before, at_target


def test_backfill_from_metadata_jsonb(clean_db: str) -> None:
    """Backfill из metadata JSONB поднимает promoted ключи; cleanup
    убирает их из JSONB. Симуляция post-migration-kit-27 состояния:
    ряд в documents существовал ДО миграции 0007 с file_* в metadata."""
    before, at_target = _split_migrations()
    # Step 1 — применить миграции до 0007.
    with psycopg.connect(clean_db) as conn:
        apply(conn, before)
        # Step 2 — INSERT ряд с file_* в metadata (имитация
        # migration kit 27 / 06_documents.sql).
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, char_count, metadata) "
                "VALUES ('alpha', 'memorybox_legacy', 1234, %s) "
                "RETURNING id",
                (json.dumps({
                    "file_path": "/home/alyona/doc.pdf",
                    "original_name": "doc.pdf",
                    "mime_type": "application/pdf",
                    "source_ref": "legacy-42",
                    "size_bytes": "9876",
                    "visibility": "private",
                    "kept_key": "value",
                }),),
            )
            row = cur.fetchone()
            assert row is not None
            doc_id = row[0]
        conn.commit()
        # Step 3 — apply 0007, она должна backfill'нуть.
        apply(conn, at_target)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_path, original_name, mime_type, source_ref, "
                "       size_bytes, visibility, metadata "
                "FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
        assert row is not None
        fp, name, mime, sref, size, vis, meta = row
        assert fp == "/home/alyona/doc.pdf"
        assert name == "doc.pdf"
        assert mime == "application/pdf"
        assert sref == "legacy-42"
        assert size == 9876
        assert vis == "private"
        # Promoted ключи убраны из metadata.
        assert "file_path" not in meta
        assert "mime_type" not in meta
        assert "visibility" not in meta
        # Не-promoted ключ сохранён.
        assert meta.get("kept_key") == "value"


def test_partial_unique_enforces_dedup(clean_db: str) -> None:
    """UNIQUE на (agent_id, content_hash) WHERE content_hash NOT NULL —
    дубликат отвергается psycopg.errors.UniqueViolation."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, content_hash, char_count) "
                "VALUES ('alpha', 'ingest_document', %s, 100) "
                "RETURNING id",
                ("deadbeef" * 8,),
            )
            row = cur.fetchone()
            assert row is not None
        conn.commit()

        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO documents "
                    "  (agent_id, source, content_hash, char_count) "
                    "VALUES ('alpha', 'ingest_document', %s, 100)",
                    ("deadbeef" * 8,),
                )
        conn.rollback()


def test_partial_unique_allows_null_content_hash(clean_db: str) -> None:
    """Partial UNIQUE НЕ применяется к NULL — несколько NULL'ов
    сосуществуют (волна 19 store-routed ряды)."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            for _ in range(3):
                cur.execute(
                    "INSERT INTO documents "
                    "  (agent_id, source, char_count) "
                    "VALUES ('alpha', 'memory_store', 100)"
                )
            cur.execute(
                "SELECT count(*) FROM documents WHERE agent_id='alpha'"
            )
            assert cur.fetchone()[0] == 3
        conn.commit()


def test_partial_unique_per_agent_scope(clean_db: str) -> None:
    """Тот же content_hash разных agent_id — не конфликтуют."""
    migrate.run(clean_db)
    h = "feedface" * 8
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, content_hash, char_count) "
                "VALUES ('alpha', 'ingest_document', %s, 100)",
                (h,),
            )
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, content_hash, char_count) "
                "VALUES ('beta', 'ingest_document', %s, 100)",
                (h,),
            )
        conn.commit()


def test_insert_document_extended_fields(clean_db: str) -> None:
    """После 0007 INSERT в новые колонки сохраняется и читается."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, char_count, "
                "   file_path, original_name, mime_type, source_ref, "
                "   size_bytes, visibility) "
                "VALUES ('alpha', 'ingest_document', 5000, "
                "        %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    "/srv/docs/sample.pdf",
                    "sample.pdf",
                    "application/pdf",
                    "upload-7",
                    12345,
                    "shared",
                ),
            )
            row = cur.fetchone()
            assert row is not None
            doc_id = row[0]
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_path, original_name, mime_type, source_ref, "
                "       size_bytes, visibility FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
        assert row == (
            "/srv/docs/sample.pdf",
            "sample.pdf",
            "application/pdf",
            "upload-7",
            12345,
            "shared",
        )
