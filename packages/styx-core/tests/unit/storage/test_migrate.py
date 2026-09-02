"""Интеграционные тесты идемпотентного мигратора."""

from __future__ import annotations

import psycopg

from styx.storage import migrate


EXPECTED_TABLES = {
    "sessions", "memories", "recall_events", "_styx_migrations",
    "cognitive_acts", "cognitive_snapshots", "cognitive_actions",
    "cognitive_consequences", "cognitive_presentations", "memory_lineage", "line_state",
    "will_projections",
    "cognitive_act_reductions",
}
EXPECTED_INDEXES = {
    "sessions_agent_started_idx",
    "memories_agent_seq_idx",
    "memories_session_seq_idx",
    "memories_embedding_hnsw_idx",
    "recall_events_memory_idx",
    "recall_events_session_idx",
    "memories_subjective_line_idx",
    "cognitive_acts_parent_key_idx",
    "cognitive_snapshots_agent_created_idx",
    "cognitive_snapshots_agent_host_uq",
    "cognitive_consequences_inbox_idx",
    "cognitive_presentations_active_idx",
    "memory_lineage_source_idx",
    "memory_lineage_target_idx",
    "memories_validated_act_residue_uq",
    "memories_line_provenance_idx",
    "cognitive_act_reductions_pending_idx",
    "llm_tasks_act_residue_active_uq",
    "llm_tasks_act_residue_attempt_uq",
}


def _public_tables(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        return {row[0] for row in cur.fetchall()}


def _public_indexes(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
        )
        return {row[0] for row in cur.fetchall()}


def test_migration_applies_to_empty_db(clean_db: str) -> None:
    applied = migrate.run(clean_db)
    assert applied == [
        "0001_init.sql", "0002_memorybox_port.sql",
        "0003_working_set.sql", "0004_relations_unique.sql",
        "0005_documents_chunks.sql", "0006_chunks_fts.sql",
        "0007_documents_pipeline.sql", "0008_emotional_evidence.sql",
        "0009_cognitive_continuity.sql",
        "0010_act_residue_carrier.sql",
    ]

    with psycopg.connect(clean_db) as conn:
        tables = _public_tables(conn)
        assert EXPECTED_TABLES.issubset(tables), f"missing: {EXPECTED_TABLES - tables}"

        indexes = _public_indexes(conn)
        assert EXPECTED_INDEXES.issubset(indexes), f"missing: {EXPECTED_INDEXES - indexes}"


def test_migration_is_idempotent(clean_db: str) -> None:
    first = migrate.run(clean_db)
    second = migrate.run(clean_db)
    assert first == [
        "0001_init.sql", "0002_memorybox_port.sql",
        "0003_working_set.sql", "0004_relations_unique.sql",
        "0005_documents_chunks.sql", "0006_chunks_fts.sql",
        "0007_documents_pipeline.sql", "0008_emotional_evidence.sql",
        "0009_cognitive_continuity.sql",
        "0010_act_residue_carrier.sql",
    ]
    assert second == []


def test_schema_supports_basic_io(clean_db: str) -> None:
    migrate.run(clean_db)

    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES "
                "('00000000-0000-0000-0000-000000000001', 'test-agent')"
            )
            cur.execute(
                "INSERT INTO memories (agent_id, session_id, role, content) "
                "VALUES ('test-agent', "
                "'00000000-0000-0000-0000-000000000001', 'user', 'hello') "
                "RETURNING id"
            )
            memory_id = cur.fetchone()[0]
            # После 0002 поле score переименовано в match_score (real),
            # focus остаётся Styx-specific. query_hash NULL допустим
            # благодаря partial UNIQUE.
            cur.execute(
                    "INSERT INTO recall_events "
                    "(memory_id, session_id, agent_id, focus, match_score) "
                    "VALUES (%s, '00000000-0000-0000-0000-000000000001', "
                    "'test-agent', "
                "'greeting', 0.91)",
                (memory_id,),
            )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories WHERE agent_id = 'test-agent'"
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT count(*) FROM recall_events WHERE memory_id = %s",
                (memory_id,),
            )
            assert cur.fetchone()[0] == 1


def test_recall_events_has_agent_owned_session_fk(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'recall_events'"
            )
            cols = {row[0] for row in cur.fetchall()}
    assert "agent_id" in cols
    assert "memory_id" in cols


def test_role_check_constraint_rejects_invalid(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES "
                "('00000000-0000-0000-0000-000000000002', 'a')"
            )
        conn.commit()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memories (agent_id, session_id, role, content) "
                    "VALUES ('a', '00000000-0000-0000-0000-000000000002', "
                    "'invalid_role', 'x')"
                )
            conn.commit()
        except psycopg.errors.CheckViolation:
            conn.rollback()
        else:
            conn.rollback()
            raise AssertionError("CHECK на role не сработал")
