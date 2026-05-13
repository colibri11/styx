"""Интеграционные тесты для миграции 0002_memorybox_port.

Покрытия:
- Чистая БД: все колонки/таблицы/индексы появляются.
- БД с данными от 0001 (sessions/memories/recall_events) — миграция
  чистая, существующие ряды получают per-kind backfill.
- Идемпотентность: повторный прогон — no-op.
- RLS не включён (decisions § 17.1).
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


def _embedding_dim(conn: psycopg.Connection) -> int:
    """Возвращает dim из atttypmod для memories.embedding."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'"
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def test_migration_runs_clean(migrated_db: str) -> None:
    """0002 успешно применяется на чистой БД."""
    with psycopg.connect(migrated_db) as conn:
        # Все добавленные колонки на memories.
        for col in [
            "visibility",
            "kind",
            "kind_src",
            "archive_ref",
            "superseded_by",
            "relevance",
            "access_count",
            "last_accessed_at",
            "lifecycle",
            "usefulness",
            "importance_provisional",
            "importance_final",
            "unique_query_count",
            "recall_score_sum",
            "estimated_tokens",
            "emotional_context_valence",
            "emotional_context_arousal",
            "emotional_context_dominance",
            "content_hash",
            "content_tsv",
        ]:
            assert _column_exists(conn, "memories", col), f"memories.{col} missing"

        # recall_events расширения.
        for col in ["query_hash", "match_score", "used_in_output", "classifier_run_at"]:
            assert _column_exists(conn, "recall_events", col), f"recall_events.{col} missing"

        # match_score переименован, score удалён.
        assert not _column_exists(conn, "recall_events", "score")


def test_new_tables_exist(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        for table in [
            "relations",
            "llm_tasks",
            "consolidation_state",
            "sweep_runs",
            "emotional_state",
            "emotional_baseline",
            "memory_reinterpretations",
            "reinterpret_applications",
            "memory_consolidation_applications",
        ]:
            assert _table_exists(conn, table), f"table {table} missing"


def test_indexes_exist(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        for ix in [
            "idx_memories_visibility",
            "idx_memories_kind",
            "idx_memories_superseded",
            "idx_memories_estimated_tokens",
            "idx_memories_fts",
            "idx_memories_agent_kind_src",
            "idx_memories_archive_ref_kind",
            "memories_agent_content_hash_uniq",
            "memories_embedding_hnsw_idx",
            "idx_recall_events_unique",
            "idx_recall_events_unconfirmed",
            "idx_recall_events_classifier_pending",
            "idx_relations_source",
            "idx_relations_target",
            "idx_llm_tasks_queue",
            "idx_llm_tasks_memory",
            "idx_sweep_runs_started",
            "idx_emotional_state_agent_at",
            "idx_memory_reinterpretations_memory",
            "idx_reinterpret_applications_pending",
            "idx_memory_consolidation_applications_pending",
        ]:
            assert _index_exists(conn, ix), f"index {ix} missing"


def test_check_constraints_exist(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        for name in [
            "memories_visibility_check",
            "memories_kind_check",
            "memories_kind_src_check",
            "memories_lifecycle_check",
            "memories_content_length_check",
            "llm_tasks_status_check",
            "sweep_runs_status_check",
        ]:
            assert _constraint_exists(conn, name), f"constraint {name} missing"


def test_embedding_dim_is_768(migrated_db: str) -> None:
    """vector(768) должен сидеть в схеме после миграции 0002."""
    with psycopg.connect(migrated_db) as conn:
        assert _embedding_dim(conn) == 768


def test_no_rls_enabled_anywhere(migrated_db: str) -> None:
    """decisions § 17.1 — RLS не тащим. Ни одна таблица не должна
    иметь rowsecurity=true."""
    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND rowsecurity = true"
            )
            rls_tables = [r[0] for r in cur.fetchall()]
    assert rls_tables == [], f"RLS включён на: {rls_tables}"


def test_idempotent(migrated_db: str) -> None:
    """Повторный прогон migrate.run — no-op."""
    applied = migrate.run(migrated_db)
    assert applied == []


def test_per_kind_backfill_applies_to_existing_rows(clean_db: str) -> None:
    """Если в БД уже есть memories с разным kind, importance_provisional
    бэкфилится по таблице (decision=0.85, episode=0.40 и т.д.)."""
    # Сначала применить только 0001 (через manual SQL — обходим discover).
    from importlib import resources

    pkg = resources.files("styx.storage.schema")
    sql_0001 = pkg.joinpath("0001_init.sql").read_text(encoding="utf-8")

    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_0001)
        conn.commit()

        # Положим один ряд в memories — без kind (его ещё нет в схеме).
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content) VALUES (%s, %s, %s)",
                ("test_agent", "user", "hello"),
            )
        conn.commit()

    # Теперь применяем все миграции (включая 0002).
    migrate.run(clean_db)

    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT kind, importance_provisional FROM memories "
                "WHERE agent_id = 'test_agent'"
            )
            rows = cur.fetchall()

    assert len(rows) == 1
    kind, prov = rows[0]
    # Default kind после ALTER ADD = 'episode', backfill → 0.40.
    assert kind == "episode"
    assert abs(prov - 0.40) < 1e-6


def test_content_length_check_rejects_long_inserts(migrated_db: str) -> None:
    """memories_content_length_check ≤ 2400 символов — INSERT длиннее
    должен падать."""
    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO memories (agent_id, role, content) "
                    "VALUES (%s, %s, %s)",
                    ("test_agent", "user", "x" * 2401),
                )
        conn.rollback()


def test_visibility_check_rejects_unknown(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO memories (agent_id, role, content, visibility) "
                    "VALUES (%s, %s, %s, %s)",
                    ("test_agent", "user", "ok", "public"),
                )
        conn.rollback()


def test_trigger_enqueues_importance_scoring(migrated_db: str) -> None:
    """memory_importance_scoring AFTER INSERT — вставляет llm_tasks task."""
    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content) "
                "VALUES (%s, %s, %s) RETURNING id",
                ("trigger_agent", "user", "trigger me"),
            )
            row = cur.fetchone()
        conn.commit()

        memory_id = row[0]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT task_type, status, payload "
                "FROM llm_tasks WHERE memory_id = %s",
                (memory_id,),
            )
            tasks = cur.fetchall()

    assert len(tasks) == 1
    task_type, status, payload = tasks[0]
    assert task_type == "importance_scoring_from_content"
    assert status == "pending"
    assert payload["kind"] == "episode"
    assert payload["length"] == len("trigger me")
