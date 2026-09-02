"""Интеграционные тесты идемпотентного мигратора."""

from __future__ import annotations

import uuid

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
    "cognitive_observations_source_key_uq",
    "cognitive_observations_source_sequence_uq",
    "cognitive_observations_pending_order_idx",
    "cognitive_observations_pending_correlation_idx",
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
        "0011_durable_observations.sql",
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
        "0011_durable_observations.sql",
    ]
    assert second == []


def test_0011_upgrades_legacy_inbox_without_inventing_provenance(
    clean_db: str,
) -> None:
    migrations = migrate.discover_migrations()
    legacy = [item for item in migrations if item.name <= "0010_act_residue_carrier.sql"]
    current = [item for item in migrations if item.name == "0011_durable_observations.sql"]
    with psycopg.connect(clean_db) as conn:
        migrate.apply(conn, legacy)
        act_id = uuid.uuid4()
        consequence_id = uuid.uuid4()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cognitive_acts "
                "(id,agent_id,host_key,status,channel_input,channel_output,completed_at) "
                "VALUES (%s,'agent-a','legacy-act','completed','{}','{}',now())",
                (act_id,),
            )
            cur.execute(
                "INSERT INTO cognitive_consequences "
                "(id,agent_id,act_id,ordinal,kind,content,metadata,status) "
                "VALUES (%s,'agent-a',%s,0,'legacy-result','legacy evidence','{}','pending')",
                (consequence_id, act_id),
            )
            cur.execute(
                "INSERT INTO cognitive_snapshots "
                "(token,agent_id,line_version,lease_expires_at,presentation_completed_at) "
                "VALUES ('legacy-snapshot','agent-a',0,now()+interval '1 hour',now())"
            )
            cur.execute(
                "INSERT INTO cognitive_presentations "
                "(snapshot_token,consequence_id,agent_id,lease_expires_at) "
                "VALUES ('legacy-snapshot',%s,'agent-a',now()+interval '1 hour')",
                (consequence_id,),
            )
        conn.commit()

        assert migrate.apply(conn, current) == ["0011_durable_observations.sql"]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_id,correlation_status FROM cognitive_consequences "
                "WHERE id=%s",
                (consequence_id,),
            )
            assert cur.fetchone() == (None, "legacy")
            cur.execute(
                "SELECT presented_payload,payload_hash,presentation_version "
                "FROM cognitive_presentations WHERE consequence_id=%s",
                (consequence_id,),
            )
            assert cur.fetchone() == (None, None, None)


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
