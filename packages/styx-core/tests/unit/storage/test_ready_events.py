from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.storage.cognition import CognitiveCommitConflict, commit_cognitive_act
from styx.storage.observations import ingest_observation, present_pending_observations
from styx.storage.ready_events import (
    ReadyEventConflict,
    claim_ready_events,
    create_observation_ready_event,
    resolve_ready_events,
)
from styx.engine.execution_provenance import execution_provenance_hash, normalize_execution_provenance


def _ingest(conn: psycopg.Connection, agent: str = "agent-a"):
    return ingest_observation(
        conn, agent, source_id="sensor", source_stream="events", source_sequence=1,
        observation_key="event-1", difference_kind="external_signal", content="changed",
        salience=0.8, confidence=0.9, reducer_name="fixture", reducer_version="1",
    )


def test_observation_event_is_idempotent_and_content_free(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        observation = _ingest(conn)
        first = create_observation_ready_event(
            conn, "agent-a", source_generation=observation.ingest_seq,
            observation_high_water=observation.ingest_seq,
            pending_count=observation.pending_count,
        )
        replay = create_observation_ready_event(
            conn, "agent-a", source_generation=observation.ingest_seq,
            observation_high_water=observation.ingest_seq,
            pending_count=observation.pending_count,
        )
        conn.commit()
        assert first["ready_generation"] == 1
        assert replay["duplicate"] is True
        with conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(e)-'id'-'created_at'-'available_after' FROM cognitive_ready_events e")
            payload = cur.fetchone()[0]
        assert "changed" not in str(payload)


def test_claim_defer_and_expired_redelivery(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        observation = _ingest(conn)
        create_observation_ready_event(
            conn, "agent-a", source_generation=observation.ingest_seq,
            observation_high_water=observation.ingest_seq, pending_count=1,
        )
        claim = claim_ready_events(conn, "agent-a", consumer_id="host-a")
        assert claim.claim_token and len(claim.events) == 1
        blocked = claim_ready_events(conn, "agent-a", consumer_id="host-b")
        assert blocked.events == ()
        resolved = resolve_ready_events(
            conn, "agent-a", consumer_id="host-a", claim_token=str(claim.claim_token),
            outcome="deferred", discard_cooldown_s=0,
        )
        assert resolved["outcome"] == "deferred"
        again = claim_ready_events(conn, "agent-a", consumer_id="host-b")
        assert again.events[0]["ready_generation"] == 1
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE cognitive_ready_events SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE agent_id='agent-a' AND status='claimed'"
            )
        crashed = claim_ready_events(conn, "agent-a", consumer_id="host-c")
        assert crashed.events[0]["redelivery_count"] == 1


def test_operator_signal_is_content_free_and_idempotent(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        first = create_observation_ready_event(
            conn, "agent-a", source_generation=9, observation_high_water=None,
            pending_count=0, reason="operator_signal",
        )
        replay = create_observation_ready_event(
            conn, "agent-a", source_generation=9, observation_high_water=None,
            pending_count=0, reason="operator_signal",
        )
        assert first["ready_generation"] == replay["ready_generation"]
        assert replay["duplicate"] is True


def test_presented_requires_exact_snapshot_observation_coverage(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        observation = _ingest(conn)
        create_observation_ready_event(
            conn, "agent-a", source_generation=observation.ingest_seq,
            observation_high_water=observation.ingest_seq, pending_count=1,
        )
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cognitive_snapshots(token,agent_id,line_version,lease_expires_at) "
                "VALUES ('snap','agent-a',0,clock_timestamp()+interval '1 hour')"
            )
        claim = claim_ready_events(conn, "agent-a", consumer_id="host-a")
        with pytest.raises(ReadyEventConflict, match="did not present"):
            resolve_ready_events(
                conn, "agent-a", consumer_id="host-a", claim_token=str(claim.claim_token),
                outcome="presented", snapshot_token="snap",
            )
        present_pending_observations(conn, "agent-a", "snap")
        result = resolve_ready_events(
            conn, "agent-a", consumer_id="host-a", claim_token=str(claim.claim_token),
            outcome="presented", snapshot_token="snap",
        )
        assert result["resolved_count"] == 1


def test_actual_provenance_is_immutable_under_host_key_retry(migrated_db: str) -> None:
    base = normalize_execution_provenance(None, legacy_model="a", legacy_platform="ollama")
    changed = normalize_execution_provenance(None, legacy_model="b", legacy_platform="sglang")
    kwargs = dict(
        host_key="turn", parent_host_key=None, session_id=None, snapshot_token=None,
        status="completed", input_line_version=0, channel_input={}, channel_output={},
        actions=[], consequences=[], metadata={},
    )
    with psycopg.connect(migrated_db) as conn:
        commit_cognitive_act(
            conn, "agent-a", **kwargs, execution_provenance=base,
            execution_provenance_hash=execution_provenance_hash(base),
        )
        conn.commit()
        with pytest.raises(CognitiveCommitConflict):
            commit_cognitive_act(
                conn, "agent-a", **kwargs, execution_provenance=changed,
                execution_provenance_hash=execution_provenance_hash(changed),
            )
