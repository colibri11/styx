"""Durable, content-free host readiness ledger (wave 41)."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row


ReadyOutcome = Literal["presented", "deferred", "discarded"]


class ReadyEventConflict(ValueError):
    """A claim token, consumer, or resolution coordinate did not match."""


class ReadyEventBackpressure(RuntimeError):
    """The event ledger or consumer outstanding bound was reached."""


@dataclass(frozen=True)
class ReadyClaim:
    claim_token: uuid.UUID | None
    lease_expires_at: dt.datetime | None
    events: tuple[dict[str, Any], ...]


def _consumer(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError("consumer_id must contain 1..128 characters")
    return value


def create_observation_ready_event(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    source_generation: int,
    observation_high_water: int | None,
    pending_count: int,
    event_cap: int = 1024,
    global_event_cap: int = 100_000,
    reason: str = "observation_available",
    available_after_s: float = 0.0,
) -> dict[str, Any]:
    """Create exactly one generation for a newly persisted external difference."""
    if reason not in {"observation_available", "observation_redeliverable", "operator_signal"}:
        raise ValueError("unsupported ready event reason")
    if source_generation < 0 or (
        observation_high_water is not None and observation_high_water < 0
    ) or pending_count < 0:
        raise ValueError("ready event coordinates must be non-negative")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended('styx:ready_events:global', 0))"
        )
        cur.execute(
            "INSERT INTO ready_event_state(agent_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (agent_id,),
        )
        cur.execute(
            "SELECT next_generation,last_source_generation FROM ready_event_state "
            "WHERE agent_id=%s FOR UPDATE",
            (agent_id,),
        )
        state = cur.fetchone()
        assert state is not None
        cur.execute(
            "SELECT id,ready_generation,status FROM cognitive_ready_events "
            "WHERE agent_id=%s AND reason=%s AND source_generation=%s",
            (agent_id, reason, source_generation),
        )
        duplicate = cur.fetchone()
        if duplicate is not None:
            return {**duplicate, "duplicate": True}
        cur.execute(
            "SELECT count(*)::int AS count FROM cognitive_ready_events "
            "WHERE agent_id=%s AND status IN ('pending','claimed')",
            (agent_id,),
        )
        active = int(cur.fetchone()["count"])
        if active >= max(1, min(int(event_cap), 100_000)):
            raise ReadyEventBackpressure("ready event pending cap reached")
        cur.execute(
            "SELECT count(*)::int AS count FROM cognitive_ready_events "
            "WHERE status IN ('pending','claimed')"
        )
        if int(cur.fetchone()["count"]) >= max(1, min(int(global_event_cap), 1_000_000)):
            raise ReadyEventBackpressure("global ready event pending cap reached")
        generation = int(state["next_generation"])
        event_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO cognitive_ready_events "
            "(id,agent_id,ready_generation,reason,source_generation,"
            " observation_high_water,pending_count,available_after) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,clock_timestamp()+(%s * interval '1 second'))",
            (
                event_id, agent_id, generation, reason, source_generation,
                observation_high_water, pending_count, max(0.0, min(3600.0, available_after_s)),
            ),
        )
        cur.execute(
            "UPDATE ready_event_state SET next_generation=%s,last_source_generation="
            "GREATEST(last_source_generation,%s),updated_at=clock_timestamp() WHERE agent_id=%s",
            (generation + 1, source_generation, agent_id),
        )
        return {
            "id": event_id,
            "ready_generation": generation,
            "status": "pending",
            "duplicate": False,
        }


def claim_ready_events(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    consumer_id: str,
    after_generation: int = 0,
    limit: int = 1,
    lease_seconds: float = 30.0,
    outstanding_cap: int = 8,
) -> ReadyClaim:
    """Atomically claim a bounded generation range; delivery is at-least-once."""
    consumer_id = _consumer(consumer_id)
    bounded_limit = max(1, min(int(limit), 32))
    bounded_lease = max(1.0, min(float(lease_seconds), 3600.0))
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "UPDATE cognitive_ready_events SET status='pending',claim_token=NULL,"
            "claimed_by=NULL,lease_expires_at=NULL,redelivery_count=redelivery_count+1 "
            "WHERE agent_id=%s AND status='claimed' AND lease_expires_at<=clock_timestamp()",
            (agent_id,),
        )
        cur.execute(
            "SELECT count(*)::int AS count FROM cognitive_ready_events "
            "WHERE agent_id=%s AND status='claimed' AND claimed_by=%s "
            "AND lease_expires_at>clock_timestamp()",
            (agent_id, consumer_id),
        )
        outstanding = int(cur.fetchone()["count"])
        allowance = max(0, min(bounded_limit, max(1, int(outstanding_cap)) - outstanding))
        if allowance <= 0:
            raise ReadyEventBackpressure("consumer outstanding claim cap reached")
        cur.execute(
            "SELECT id,ready_generation,reason,source_generation,observation_high_water,"
            "pending_count,created_at,redelivery_count FROM cognitive_ready_events "
            "WHERE agent_id=%s AND status='pending' AND ready_generation>%s "
            "AND available_after<=clock_timestamp() ORDER BY ready_generation "
            "LIMIT %s FOR UPDATE SKIP LOCKED",
            (agent_id, max(0, int(after_generation)), allowance),
        )
        rows = list(cur.fetchall())
        if not rows:
            return ReadyClaim(None, None, ())
        token = uuid.uuid4()
        ids = [row["id"] for row in rows]
        cur.execute(
            "UPDATE cognitive_ready_events SET status='claimed',claim_token=%s,claimed_by=%s,"
            "lease_expires_at=clock_timestamp()+(%s * interval '1 second'),"
            "delivery_count=delivery_count+1 WHERE agent_id=%s AND id=ANY(%s) "
            "RETURNING lease_expires_at",
            (token, consumer_id, bounded_lease, agent_id, ids),
        )
        lease = cur.fetchone()["lease_expires_at"]
        events = tuple(
            {
                "event_id": str(row["id"]),
                "ready_generation": int(row["ready_generation"]),
                "reason": row["reason"],
                "source_generation": int(row["source_generation"]),
                "observation_high_water": row["observation_high_water"],
                "pending_count": int(row["pending_count"]),
                "created_at": row["created_at"],
                "redelivery_count": int(row["redelivery_count"]),
            }
            for row in rows
        )
        return ReadyClaim(token, lease, events)


def resolve_ready_events(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    consumer_id: str,
    claim_token: str,
    outcome: ReadyOutcome,
    snapshot_token: str | None = None,
    policy_reason: str | None = None,
    discard_cooldown_s: float = 30.0,
    event_cap: int = 1024,
    global_event_cap: int = 100_000,
) -> dict[str, Any]:
    """Resolve one claim group without deleting its source observations."""
    consumer_id = _consumer(consumer_id)
    try:
        token = uuid.UUID(str(claim_token))
    except ValueError as exc:
        raise ReadyEventConflict("claim_token is invalid") from exc
    if outcome not in {"presented", "deferred", "discarded"}:
        raise ValueError("unsupported ready event outcome")
    if outcome == "presented" and not snapshot_token:
        raise ReadyEventConflict("presented outcome requires snapshot_token")
    if outcome == "discarded" and (
        not isinstance(policy_reason, str) or not 1 <= len(policy_reason) <= 128
    ):
        raise ReadyEventConflict("discarded outcome requires bounded policy_reason")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM cognitive_ready_events WHERE agent_id=%s AND claim_token=%s "
            "AND status='claimed' FOR UPDATE",
            (agent_id, token),
        )
        rows = list(cur.fetchall())
        if not rows:
            raise ReadyEventConflict("claim is unknown, expired, or already resolved")
        if any(row["claimed_by"] != consumer_id for row in rows):
            raise ReadyEventConflict("claim belongs to another consumer")
        if any(row["lease_expires_at"] <= dt.datetime.now(dt.timezone.utc) for row in rows):
            raise ReadyEventConflict("claim lease expired")
        if outcome == "presented":
            cur.execute(
                "SELECT token FROM cognitive_snapshots WHERE token=%s AND agent_id=%s",
                (snapshot_token, agent_id),
            )
            if cur.fetchone() is None:
                raise ReadyEventConflict("snapshot is unknown for this agent")
            for row in rows:
                if row["observation_high_water"] is None:
                    continue
                cur.execute(
                    "SELECT 1 FROM cognitive_presentations p JOIN cognitive_consequences c "
                    "ON c.id=p.consequence_id AND c.agent_id=p.agent_id "
                    "WHERE p.agent_id=%s AND p.snapshot_token=%s "
                    "AND c.source_id IS NOT NULL AND c.ingest_seq=%s LIMIT 1",
                    (agent_id, snapshot_token, row["observation_high_water"]),
                )
                if cur.fetchone() is None:
                    raise ReadyEventConflict(
                        "snapshot did not present the event observation coordinate"
                    )
        ids = [row["id"] for row in rows]
        if outcome == "deferred":
            cur.execute(
                "UPDATE cognitive_ready_events SET status='pending',claim_token=NULL,"
                "claimed_by=NULL,lease_expires_at=NULL,resolve_outcome=NULL,"
                "resolved_snapshot_token=NULL,policy_reason=NULL,"
                "available_after=clock_timestamp()+(%s * interval '1 second') "
                "WHERE agent_id=%s AND id=ANY(%s)",
                (max(0.0, min(3600.0, discard_cooldown_s)), agent_id, ids),
            )
            return {"resolved_count": len(rows), "outcome": outcome, "redelivered": False}
        cur.execute(
            "UPDATE cognitive_ready_events SET status='resolved',claim_token=NULL,claimed_by=NULL,"
            "lease_expires_at=NULL,resolve_outcome=%s,resolved_snapshot_token=%s,policy_reason=%s,"
            "resolved_at=clock_timestamp() WHERE agent_id=%s AND id=ANY(%s)",
            (outcome, snapshot_token, policy_reason, agent_id, ids),
        )
        redelivered = False
        if outcome == "discarded":
            for row in rows:
                high_water = row["observation_high_water"]
                if high_water is None:
                    continue
                cur.execute(
                    "SELECT count(*)::int AS count FROM cognitive_consequences WHERE agent_id=%s "
                    "AND source_id IS NOT NULL AND status<>'acknowledged'",
                    (agent_id,),
                )
                pending = int(cur.fetchone()["count"])
                if pending:
                    replay = create_observation_ready_event(
                        conn, agent_id,
                        source_generation=int(row["source_generation"]),
                        observation_high_water=int(high_water), pending_count=pending,
                        event_cap=event_cap, reason="observation_redeliverable",
                        global_event_cap=global_event_cap,
                        available_after_s=discard_cooldown_s,
                    )
                    redelivered = redelivered or not replay["duplicate"]
        return {"resolved_count": len(rows), "outcome": outcome, "redelivered": redelivered}


def ready_event_stats(conn: psycopg.Connection, agent_id: str) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT count(*) FILTER (WHERE status='pending')::int AS pending,"
            "count(*) FILTER (WHERE status='claimed')::int AS claimed,"
            "count(*) FILTER (WHERE status='resolved')::int AS resolved,"
            "coalesce(sum(redelivery_count),0)::int AS redeliveries,"
            "coalesce(max(ready_generation),0)::bigint AS latest_generation,"
            "extract(epoch FROM clock_timestamp()-min(created_at) FILTER "
            "(WHERE status='pending'))::double precision AS oldest_pending_age_s,"
            "extract(epoch FROM clock_timestamp()-min(created_at) FILTER "
            "(WHERE status='claimed'))::double precision AS oldest_claimed_age_s,"
            "coalesce(avg(extract(epoch FROM resolved_at-created_at)) FILTER "
            "(WHERE resolve_outcome='presented'),0)::double precision AS avg_wake_latency_s,"
            "count(*) FILTER (WHERE resolve_outcome='presented')::int AS presented,"
            "count(*) FILTER (WHERE resolve_outcome='discarded')::int AS discarded "
            "FROM cognitive_ready_events WHERE agent_id=%s",
            (agent_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else {
            "pending": 0, "claimed": 0, "resolved": 0,
            "redeliveries": 0, "latest_generation": 0,
        }


def execution_provenance_stats(
    conn: psycopg.Connection, agent_id: str
) -> dict[str, Any]:
    """Aggregate families only; never expose model ids or endpoints."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT coalesce(execution_provenance->>'provider_family','unknown') AS provider,"
            "coalesce(execution_provenance->>'runtime_family','unknown') AS runtime,"
            "count(*)::int AS count FROM cognitive_acts WHERE agent_id=%s "
            "GROUP BY provider,runtime ORDER BY provider,runtime",
            (agent_id,),
        )
        rows = list(cur.fetchall())
    total = sum(int(row["count"]) for row in rows)
    unknown = sum(
        int(row["count"]) for row in rows
        if row["provider"] == "unknown" or row["runtime"] == "unknown"
    )
    return {
        "families": [dict(row) for row in rows],
        "total": total,
        "unknown": unknown,
        "unknown_rate": (unknown / total if total else 0.0),
    }
