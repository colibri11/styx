"""Regression tests for causal emotional evidence migration 0008."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from styx.storage import migrate
from styx.storage.migrate import apply, discover_migrations


TARGET = "0008_emotional_evidence.sql"


def _columns(conn: psycopg.Connection, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s",
            (table,),
        )
        return {row[0] for row in cur.fetchall()}


def test_migration_adds_evidence_and_provenance_schema(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        assert {
            "idempotency_key",
            "source_kind",
            "source_ref",
            "confidence",
            "cause_status",
        }.issubset(_columns(conn, "emotional_events"))
        assert {
            "parent_state_id",
            "event_id",
            "delta_valence",
            "transition_confidence",
            "causal_context",
            "computation_version",
        }.issubset(_columns(conn, "emotional_state"))
        assert {
            "cause_event_id",
            "status",
            "lease_expires_at",
            "support_valence",
            "status_source_event_id",
        }.issubset(_columns(conn, "emotional_cause_status"))
        assert {
            "source_window_from",
            "sample_size",
            "source_state_id",
        }.issubset(_columns(conn, "emotional_baseline"))
        assert {
            "emotional_context_state_id",
            "emotional_context_at",
            "emotional_context_confidence",
            "emotional_context_causes",
        }.issubset(_columns(conn, "memories"))


def test_upgrade_preserves_legacy_rows_as_unknown(clean_db: str) -> None:
    migrations = discover_migrations()
    before = [migration for migration in migrations if migration.name < TARGET]
    target = [migration for migration in migrations if migration.name == TARGET]
    assert target

    with psycopg.connect(clean_db) as conn:
        apply(conn, before)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_state "
                "(agent_id, valence, arousal, dominance, source) "
                "VALUES ('legacy-agent', 0.2, -0.1, 0.3, 'hot_sentiment') "
                "RETURNING id"
            )
            state_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO memories "
                "(agent_id, role, content, emotional_context_valence, "
                " emotional_context_arousal, emotional_context_dominance) "
                "VALUES ('legacy-agent', 'summary', 'legacy', 0.2, -0.1, 0.3) "
                "RETURNING id"
            )
            memory_id = cur.fetchone()[0]
        conn.commit()

        apply(conn, target)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT parent_state_id, event_id, confidence, causal_context "
                "FROM emotional_state WHERE id=%s",
                (state_id,),
            )
            assert cur.fetchone() == (None, None, None, [])
            cur.execute(
                "SELECT emotional_context_state_id, emotional_context_confidence, "
                "       emotional_context_causes "
                "FROM memories WHERE id=%s",
                (memory_id,),
            )
            assert cur.fetchone() == (None, None, None)


def test_event_idempotency_is_per_agent(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_events "
                "(agent_id, occurred_at, source_kind, idempotency_key) "
                "VALUES ('alpha', now(), 'turn', 'same-key')"
            )
            with pytest.raises(psycopg.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO emotional_events "
                    "(agent_id, occurred_at, source_kind, idempotency_key) "
                    "VALUES ('alpha', now(), 'turn', 'same-key')"
                )
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_events "
                "(agent_id, occurred_at, source_kind, idempotency_key) VALUES "
                "('alpha', now(), 'turn', 'same-key'), "
                "('beta', now(), 'turn', 'same-key')"
            )
        conn.commit()


def test_terminal_diary_part_idempotency_is_per_agent(clean_db: str) -> None:
    migrate.run(clean_db)
    metadata = {
        "styx_sync_turn_key": "context:key",
        "styx_sync_turn_role": "user",
        "styx_sync_turn_part": 0,
    }
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, metadata) "
                "VALUES ('alpha', 'user', 'first', %s)",
                (Jsonb(metadata),),
            )
            with pytest.raises(psycopg.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO memories (agent_id, role, content, metadata) "
                    "VALUES ('alpha', 'user', 'retry', %s)",
                    (Jsonb(metadata),),
                )
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, metadata) "
                "VALUES ('alpha', 'user', 'first', %s), "
                "       ('beta', 'user', 'first', %s)",
                (Jsonb(metadata), Jsonb(metadata)),
            )


def test_active_cause_lease_must_be_strictly_future(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_events "
                "(agent_id, occurred_at, source_kind) "
                "VALUES ('alpha', now(), 'turn') RETURNING id"
            )
            event_id = cur.fetchone()[0]
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO emotional_cause_status "
                    "(agent_id, cause_event_id, at, status, lease_expires_at) "
                    "VALUES ('alpha', %s, now(), 'active', now())",
                    (event_id,),
                )


@pytest.mark.parametrize(
    ("column", "value"),
    (("confidence", 1.1), ("intensity", -0.1)),
)
def test_event_unit_interval_constraints(
    clean_db: str, column: str, value: float
) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur, pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                f"INSERT INTO emotional_events "
                f"(agent_id, occurred_at, source_kind, {column}) "
                f"VALUES ('alpha', now(), 'turn', %s)",
                (value,),
            )


def test_cross_agent_event_and_memory_links_are_rejected(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_events "
                "(agent_id, occurred_at, source_kind) "
                "VALUES ('alpha', now(), 'turn') RETURNING id"
            )
            event_id = cur.fetchone()[0]
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO emotional_state "
                    "(agent_id, valence, arousal, dominance, event_id) "
                    "VALUES ('beta', 0, 0, 0, %s)",
                    (event_id,),
                )
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_events "
                "(agent_id, occurred_at, source_kind) "
                "VALUES ('alpha', now(), 'turn') RETURNING id"
            )
            event_id = cur.fetchone()[0]
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO emotional_cause_status "
                    "(agent_id, cause_event_id, status, lease_expires_at) "
                    "VALUES ('beta', %s, 'active', now() + interval '15 minutes')",
                    (event_id,),
                )
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_state "
                "(agent_id, valence, arousal, dominance) "
                "VALUES ('alpha', 0, 0, 0) RETURNING id"
            )
            state_id = cur.fetchone()[0]
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO memories "
                    "(agent_id, role, content, emotional_context_state_id) "
                    "VALUES ('beta', 'summary', %s, %s)",
                    (str(uuid.uuid4()), state_id),
                )
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_state "
                "(agent_id, valence, arousal, dominance) "
                "VALUES ('alpha', 0, 0, 0) RETURNING id"
            )
            state_id = cur.fetchone()[0]
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO emotional_baseline "
                    "(agent_id, valence, arousal, dominance, source_state_id) "
                    "VALUES ('beta', 0, 0, 0, %s)",
                    (state_id,),
                )
