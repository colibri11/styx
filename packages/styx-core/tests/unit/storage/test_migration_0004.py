"""Tests миграции 0004_relations_unique.sql (волна 18).

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import uuid

import psycopg
import psycopg.errors
import pytest

from styx.storage import migrate


def test_migration_applies_to_clean_db(clean_db: str) -> None:
    """0004 применяется на свежей БД без ошибок."""
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_constraint "
                " WHERE conname = 'relations_unique' "
                "   AND conrelid = 'relations'::regclass"
            )
            assert cur.fetchone() is not None


def test_migration_dedupes_existing_duplicates(clean_db: str) -> None:
    """Если в БД уже есть дубли (source, target, relation) — миграция
    их чистит (оставляет самую раннюю по created_at)."""
    # Применяем все миграции до 0003 включительно (миграция 0002
    # создаёт таблицу relations, 0003 — working_set).
    with psycopg.connect(clean_db) as conn:
        # Apply 0001..0003 вручную через запуск migrate.run без
        # последней (мы хотим вставить дубль ДО 0004 миграции).
        # Но migrate.run применяет всё сразу. Поэтому сначала run,
        # потом дропаем constraint, вставляем дубль, и run снова —
        # он подхватит cleanup'ом.
        pass
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            # Снимаем UNIQUE constraint, чтобы вставить дубли.
            cur.execute("ALTER TABLE relations DROP CONSTRAINT IF EXISTS relations_unique")
            src = uuid.uuid4()
            tgt = uuid.uuid4()
            for _ in range(3):
                cur.execute(
                    "INSERT INTO relations "
                    "  (source_type, source_id, target_type, target_id, relation) "
                    "VALUES ('memory', %s, 'memory', %s, 'related_to')",
                    (src, tgt),
                )
            cur.execute(
                "DELETE FROM _styx_migrations WHERE name = '0004_relations_unique.sql'"
            )
        conn.commit()

    # Re-apply 0004 → cleanup + UNIQUE.
    migrate.run(clean_db)

    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM relations "
                " WHERE source_type='memory' AND source_id=%s "
                "   AND target_type='memory' AND target_id=%s "
                "   AND relation='related_to'",
                (src, tgt),
            )
            assert cur.fetchone()[0] == 1


def test_unique_constraint_blocks_explicit_duplicates(clean_db: str) -> None:
    """После миграции 0004 INSERT с тем же (src, tgt, rel) без ON CONFLICT
    падает на UNIQUE constraint."""
    migrate.run(clean_db)
    src = uuid.uuid4()
    tgt = uuid.uuid4()
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relations "
                "  (source_type, source_id, target_type, target_id, relation) "
                "VALUES ('memory', %s, 'memory', %s, 'related_to')",
                (src, tgt),
            )
        conn.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO relations "
                    "  (source_type, source_id, target_type, target_id, relation) "
                    "VALUES ('memory', %s, 'memory', %s, 'related_to')",
                    (src, tgt),
                )
            conn.commit()
        conn.rollback()
