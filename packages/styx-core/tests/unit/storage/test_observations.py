"""Wave 39 durable observation inbox invariants on real PostgreSQL."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.storage.cognition import commit_cognitive_act, record_snapshot
from styx.storage.observations import (
    ObservationBackpressure,
    ObservationConflict,
    ingest_observation,
    present_pending_observations,
)


def _observation(sequence: int = 1, **overrides):
    value = {
        "source_id": "workspace-monitor",
        "source_stream": "workspace/main",
        "source_sequence": sequence,
        "observation_key": f"event-{sequence}",
        "difference_kind": "state_change",
        "content": f"Workspace state changed at sequence {sequence}.",
        "salience": 0.8,
        "confidence": 0.95,
        "reducer_name": "workspace-diff",
        "reducer_version": "1",
        "metadata": {"scope": "tests"},
    }
    value.update(overrides)
    return value


def test_ingest_is_exactly_idempotent_and_source_sequence_is_unique(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        first = ingest_observation(conn, "agent-a", **_observation())
        conn.commit()
        duplicate = ingest_observation(conn, "agent-a", **_observation())
        conn.commit()
        assert duplicate.duplicate is True
        assert duplicate.observation_id == first.observation_id
        assert duplicate.payload_hash == first.payload_hash

        with pytest.raises(ObservationConflict, match="different payload"):
            ingest_observation(
                conn,
                "agent-a",
                **_observation(content="Different state under the same key."),
            )
        conn.rollback()
        with pytest.raises(ObservationConflict, match="source_sequence"):
            ingest_observation(
                conn,
                "agent-a",
                **_observation(observation_key="different-key"),
            )
        conn.rollback()

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT count(*) OVER()::int AS count,act_id,ordinal,source_id,"
                " payload_hash,correlation_status FROM cognitive_consequences "
                "WHERE agent_id='agent-a'"
            )
            row = cur.fetchone()
        assert row["count"] == 1
        assert row["act_id"] is None and row["ordinal"] is None
        assert row["source_id"] == "workspace-monitor"
        assert row["payload_hash"] == first.payload_hash
        assert row["correlation_status"] == "uncorrelated"


def test_late_and_backpressure_are_explicit_without_deletion(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        ingest_observation(conn, "agent-a", **_observation(sequence=2))
        conn.commit()
        late = ingest_observation(conn, "agent-a", **_observation(sequence=1))
        conn.commit()
        assert late.late is True
        with pytest.raises(ObservationBackpressure) as caught:
            ingest_observation(
                conn,
                "agent-a",
                **_observation(sequence=3),
                pending_cap=2,
            )
        assert caught.value.pending_count == 2
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM cognitive_consequences "
                "WHERE agent_id='agent-a' AND status='pending'"
            )
            assert cur.fetchone()[0] == 2


def test_presentation_orders_streams_by_first_ingest_and_events_by_sequence(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        ingest_observation(
            conn,
            "agent-a",
            **_observation(
                sequence=2,
                source_id="source-a",
                source_stream="stream-a",
                observation_key="a-2",
            ),
        )
        ingest_observation(
            conn,
            "agent-a",
            **_observation(
                sequence=0,
                source_id="source-b",
                source_stream="stream-b",
                observation_key="b-0",
            ),
        )
        late = ingest_observation(
            conn,
            "agent-a",
            **_observation(
                sequence=1,
                source_id="source-a",
                source_stream="stream-a",
                observation_key="a-1",
            ),
        )
        assert late.late is True

        record_snapshot(conn, "agent-a", "snapshot-order", 0)
        shown = present_pending_observations(
            conn, "agent-a", "snapshot-order", limit=4
        )

    assert [
        (item["source_id"], item["source_stream"], item["source_sequence"])
        for item in shown
    ] == [
        ("source-a", "stream-a", 1),
        ("source-a", "stream-a", 2),
        ("source-b", "stream-b", 0),
    ]


def test_observation_before_action_resolves_or_conflicts_on_commit(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        pending = ingest_observation(
            conn,
            "agent-a",
            **_observation(
                action_ref={"host_key": "turn-1", "action_ordinal": 0}
            ),
        )
        conflict = ingest_observation(
            conn,
            "agent-a",
            **_observation(
                sequence=2,
                action_ref={"host_key": "turn-1", "action_ordinal": 9},
            ),
        )
        assert pending.correlation_status == "pending"
        assert conflict.correlation_status == "pending"
        act = commit_cognitive_act(
            conn,
            "agent-a",
            host_key="turn-1",
            parent_host_key=None,
            session_id=None,
            snapshot_token=None,
            status="completed",
            input_line_version=0,
            channel_input={},
            channel_output={},
            actions=[{
                "kind": "call",
                "tool_event_id": "tool-1",
                "name": "check",
                "content": "",
                "metadata": {},
            }],
            consequences=[],
            metadata={},
        )
        conn.commit()
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id,correlation_status,action_act_id "
                "FROM cognitive_consequences WHERE agent_id='agent-a' "
                "ORDER BY source_sequence"
            )
            rows = list(cur.fetchall())
        assert rows[0]["id"] == pending.observation_id
        assert rows[0]["correlation_status"] == "resolved"
        assert rows[0]["action_act_id"] == act.act_id
        assert rows[1]["correlation_status"] == "conflict"
        assert rows[1]["action_act_id"] is None

        with pytest.raises(ObservationConflict, match="resolve exactly"):
            ingest_observation(
                conn,
                "agent-a",
                **_observation(
                    sequence=3,
                    action_ref={"host_key": "turn-1", "action_event_id": "wrong"},
                ),
            )


def test_cross_agent_action_reference_is_rejected(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        commit_cognitive_act(
            conn,
            "agent-b",
            host_key="foreign-turn",
            parent_host_key=None,
            session_id=None,
            snapshot_token=None,
            status="completed",
            input_line_version=0,
            channel_input={},
            channel_output={},
            actions=[{
                "kind": "call", "tool_event_id": "x", "name": "x",
                "content": "", "metadata": {},
            }],
            consequences=[],
            metadata={},
        )
        conn.commit()
        with pytest.raises(ObservationConflict, match="another agent"):
            ingest_observation(
                conn,
                "agent-a",
                **_observation(
                    action_ref={
                        "agent_id": "agent-b",
                        "host_key": "foreign-turn",
                        "action_ordinal": 0,
                    }
                ),
            )
        pending = ingest_observation(
            conn,
            "agent-a",
            **_observation(
                sequence=2,
                action_ref={"host_key": "foreign-turn", "action_ordinal": 0},
            ),
        )
        assert pending.correlation_status == "pending"


def test_presentation_is_frozen_single_lease_and_late_commit_cannot_steal(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        first = ingest_observation(conn, "agent-a", **_observation())
        record_snapshot(conn, "agent-a", "snapshot-old", 0, lease_seconds=1.0)
        shown_old = present_pending_observations(
            conn, "agent-a", "snapshot-old", limit=1
        )
        assert shown_old[0]["observation_id"] == str(first.observation_id)
        assert shown_old == present_pending_observations(
            conn, "agent-a", "snapshot-old", limit=1
        )

        record_snapshot(conn, "agent-a", "snapshot-overlap", 0)
        assert present_pending_observations(
            conn, "agent-a", "snapshot-overlap", limit=1
        ) == []
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT pg_sleep(1.05)")
        record_snapshot(conn, "agent-a", "snapshot-new", 0)
        shown_new = present_pending_observations(
            conn, "agent-a", "snapshot-new", limit=1
        )
        assert shown_new == shown_old

        old_act = commit_cognitive_act(
            conn,
            "agent-a",
            host_key="late-old",
            parent_host_key=None,
            session_id=None,
            snapshot_token="snapshot-old",
            status="completed",
            input_line_version=0,
            channel_input={},
            channel_output={},
            actions=[],
            consequences=[],
            metadata={},
        )
        assert old_act.acknowledged_count == 0
        new_act = commit_cognitive_act(
            conn,
            "agent-a",
            host_key="current-new",
            parent_host_key=None,
            session_id=None,
            snapshot_token="snapshot-new",
            status="failed",
            input_line_version=0,
            channel_input={},
            channel_output={},
            actions=[],
            consequences=[],
            metadata={},
        )
        assert new_act.acknowledged_count == 1
        conn.commit()
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status,acknowledged_by_act_id FROM cognitive_consequences "
                "WHERE id=%s",
                (first.observation_id,),
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT presented_payload,payload_hash FROM cognitive_presentations "
                "WHERE consequence_id=%s ORDER BY presented_at",
                (first.observation_id,),
            )
            presentations = list(cur.fetchall())
        assert row == {
            "status": "acknowledged",
            "acknowledged_by_act_id": new_act.act_id,
        }
        assert len(presentations) == 2
        assert presentations[0]["presented_payload"] == shown_old[0]
        assert presentations[1]["presented_payload"] == shown_new[0]
        assert presentations[0]["payload_hash"] == presentations[1]["payload_hash"]


def test_database_rejects_canonical_source_payload_mutation(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        result = ingest_observation(conn, "agent-a", **_observation())
        conn.commit()
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE cognitive_consequences SET content='mutated' WHERE id=%s",
                    (result.observation_id,),
                )


def test_storage_rejects_untyped_time_and_unknown_action_fields(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        with pytest.raises(ValueError, match="timezone-aware datetime"):
            ingest_observation(
                conn,
                "agent-a",
                **_observation(source_observed_at="2026-09-02T00:00:00Z"),
            )
        with pytest.raises(ValueError, match="unknown fields"):
            ingest_observation(
                conn,
                "agent-a",
                **_observation(
                    action_ref={
                        "host_key": "turn-1",
                        "action_ordinal": 0,
                        "causal_claim": True,
                    }
                ),
            )
