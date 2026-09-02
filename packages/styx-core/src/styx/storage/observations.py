"""Durable preliminary-reduced external observations (wave 39).

The physical inbox remains ``cognitive_consequences`` for additive migration
compatibility.  Canonical rows are distinguished by non-NULL ``source_id``;
they are immutable external evidence until a later cognitive act presents and
consumes their frozen projection.  This module never writes memories.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from styx.storage.cognition import (
    redact_journal_json,
    redact_journal_metadata,
    redact_journal_text,
    validate_journal_json,
)


PRESENTATION_VERSION = "observation_presentation_v1"
DIFFERENCE_KINDS = frozenset({
    "state_change",
    "delivery_receipt",
    "action_result",
    "action_error",
    "external_signal",
})
MAX_PRESENTED_OBSERVATIONS = 4
MAX_PRESENTED_CONTENT = 512


class ObservationConflict(ValueError):
    """A stable source coordinate was reused with different semantics."""


class ObservationBackpressure(RuntimeError):
    """The per-agent durable pending bound was reached."""

    def __init__(self, *, pending_count: int, retry_after_s: int = 5) -> None:
        self.pending_count = pending_count
        self.retry_after_s = retry_after_s
        super().__init__("observation inbox pending cap reached")


@dataclass(frozen=True)
class ObservationIngestResult:
    observation_id: uuid.UUID
    duplicate: bool
    payload_hash: str
    correlation_status: str
    action_act_id: uuid.UUID | None
    late: bool
    pending_count: int
    created_at: dt.datetime
    ingest_seq: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.isoformat() if isinstance(item, dt.datetime) else str(item),
    )


def observation_payload_hash(value: Mapping[str, Any]) -> str:
    """Hash the validated caller payload before persistence redaction."""
    try:
        encoded = _canonical_json(dict(value)).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("observation payload must be canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_identifier(value: str, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{field} must contain 1..{maximum} characters")
    return value


def _validate_score(value: float, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be a finite number in 0..1")
    return float(value)


def _lock_inbox(conn: psycopg.Connection, agent_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"styx:observation_inbox:{agent_id}",),
        )


def _resolve_declared_action(
    cur: psycopg.Cursor[Any],
    *,
    agent_id: str,
    host_key: str,
    action_ordinal: int | None,
    action_event_id: str | None,
) -> tuple[str, uuid.UUID | None]:
    cur.execute(
        "SELECT id FROM cognitive_acts WHERE agent_id=%s AND host_key=%s",
        (agent_id, host_key),
    )
    act = cur.fetchone()
    if act is None:
        # Host keys are agent-scoped.  A same-named act belonging to another
        # agent is irrelevant; the declared agent coordinate was already
        # validated at the API/storage boundary.
        return "pending", None

    clauses = ["agent_id=%s", "act_id=%s"]
    params: list[Any] = [agent_id, act["id"]]
    if action_ordinal is not None:
        clauses.append("ordinal=%s")
        params.append(action_ordinal)
    if action_event_id is not None:
        clauses.append("event_id=%s")
        params.append(action_event_id)
    cur.execute(
        "SELECT id FROM cognitive_actions WHERE " + " AND ".join(clauses) + " LIMIT 2",
        tuple(params),
    )
    matches = list(cur.fetchall())
    if len(matches) != 1:
        raise ObservationConflict("action coordinates do not resolve exactly")
    return "resolved", act["id"]


def ingest_observation(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    source_id: str,
    source_stream: str,
    source_sequence: int,
    observation_key: str,
    difference_kind: str,
    content: str,
    salience: float,
    confidence: float,
    reducer_name: str,
    reducer_version: str,
    action_ref: Mapping[str, Any] | None = None,
    source_observed_at: dt.datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    pending_cap: int = 1024,
) -> ObservationIngestResult:
    """Append or replay one canonical observation inside the caller tx."""
    agent_id = _validate_identifier(agent_id, "agent_id", 256)
    source_id = _validate_identifier(source_id, "source_id", 256)
    source_stream = _validate_identifier(source_stream, "source_stream", 256)
    observation_key = _validate_identifier(observation_key, "observation_key", 256)
    reducer_name = _validate_identifier(reducer_name, "reducer_name", 128)
    reducer_version = _validate_identifier(reducer_version, "reducer_version", 64)
    if difference_kind not in DIFFERENCE_KINDS:
        raise ValueError("difference_kind is not in the controlled vocabulary")
    if isinstance(source_sequence, bool) or not isinstance(source_sequence, int) or source_sequence < 0:
        raise ValueError("source_sequence must be a non-negative integer")
    if not isinstance(content, str) or not 1 <= len(content) <= 2000:
        raise ValueError("content must contain 1..2000 characters")
    salience = _validate_score(salience, "salience")
    confidence = _validate_score(confidence, "confidence")
    raw_metadata = dict(metadata or {})
    validate_journal_json(raw_metadata, max_string=1000)
    if source_observed_at is not None and (
        not isinstance(source_observed_at, dt.datetime)
        or source_observed_at.tzinfo is None
    ):
        raise ValueError("source_observed_at must be a timezone-aware datetime")

    action_payload: dict[str, Any] | None = None
    action_host_key: str | None = None
    action_ordinal: int | None = None
    action_event_id: str | None = None
    if action_ref is not None:
        action_payload = dict(action_ref)
        unknown_action_fields = set(action_payload) - {
            "agent_id", "host_key", "action_ordinal", "action_event_id",
        }
        if unknown_action_fields:
            raise ValueError("action_ref contains unknown fields")
        declared_agent = action_payload.get("agent_id")
        if declared_agent is not None and declared_agent != agent_id:
            raise ObservationConflict("action reference belongs to another agent")
        action_host_key = _validate_identifier(
            action_payload.get("host_key"), "action_ref.host_key", 512
        )
        action_ordinal = action_payload.get("action_ordinal")
        action_event_id = action_payload.get("action_event_id")
        if action_ordinal is None and action_event_id is None:
            raise ValueError("action_ref requires action_ordinal or action_event_id")
        if action_ordinal is not None and (
            isinstance(action_ordinal, bool)
            or not isinstance(action_ordinal, int)
            or action_ordinal < 0
        ):
            raise ValueError("action_ref.action_ordinal must be non-negative")
        if action_event_id is not None:
            action_event_id = _validate_identifier(
                action_event_id, "action_ref.action_event_id", 256
            )

    hash_payload = {
        "version": "observation_ingest_v1",
        "agent_id": agent_id,
        "source_id": source_id,
        "source_stream": source_stream,
        "source_sequence": source_sequence,
        "observation_key": observation_key,
        "difference_kind": difference_kind,
        "content": content,
        "salience": salience,
        "confidence": confidence,
        "reducer_name": reducer_name,
        "reducer_version": reducer_version,
        "action_ref": action_payload,
        "source_observed_at": source_observed_at,
        "metadata": raw_metadata,
    }
    payload_hash = observation_payload_hash(hash_payload)
    bounded_cap = min(100_000, max(1, int(pending_cap)))
    _lock_inbox(conn, agent_id)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,payload_hash,correlation_status,action_act_id,late,created_at,ingest_seq "
            "FROM cognitive_consequences "
            "WHERE agent_id=%s AND source_id=%s AND observation_key=%s FOR UPDATE",
            (agent_id, source_id, observation_key),
        )
        existing = cur.fetchone()
        if existing is not None:
            if existing["payload_hash"] != payload_hash:
                raise ObservationConflict(
                    "observation_key was already used for a different payload"
                )
            cur.execute(
                "SELECT count(*)::int AS count FROM cognitive_consequences "
                "WHERE agent_id=%s AND source_id IS NOT NULL "
                "AND status<>'acknowledged'",
                (agent_id,),
            )
            pending_count = int(cur.fetchone()["count"])
            return ObservationIngestResult(
                observation_id=existing["id"],
                duplicate=True,
                payload_hash=payload_hash,
                correlation_status=existing["correlation_status"],
                action_act_id=existing["action_act_id"],
                late=bool(existing["late"]),
                pending_count=pending_count,
                created_at=existing["created_at"],
                ingest_seq=int(existing["ingest_seq"]),
            )

        cur.execute(
            "SELECT observation_key FROM cognitive_consequences "
            "WHERE agent_id=%s AND source_id=%s AND source_stream=%s "
            "AND source_sequence=%s FOR UPDATE",
            (agent_id, source_id, source_stream, source_sequence),
        )
        sequence_owner = cur.fetchone()
        if sequence_owner is not None:
            raise ObservationConflict(
                "source_sequence was already used by another observation"
            )

        cur.execute(
            "SELECT count(*)::int AS count FROM cognitive_consequences "
            "WHERE agent_id=%s AND source_id IS NOT NULL "
            "AND status<>'acknowledged'",
            (agent_id,),
        )
        pending_before = int(cur.fetchone()["count"])
        if pending_before >= bounded_cap:
            raise ObservationBackpressure(pending_count=pending_before)

        cur.execute(
            "SELECT max(source_sequence) AS maximum FROM cognitive_consequences "
            "WHERE agent_id=%s AND source_id=%s AND source_stream=%s",
            (agent_id, source_id, source_stream),
        )
        maximum = cur.fetchone()["maximum"]
        late = maximum is not None and source_sequence < int(maximum)

        if action_host_key is None:
            correlation_status, action_act_id = "uncorrelated", None
        else:
            correlation_status, action_act_id = _resolve_declared_action(
                cur,
                agent_id=agent_id,
                host_key=action_host_key,
                action_ordinal=action_ordinal,
                action_event_id=action_event_id,
            )

        observation_id = uuid.uuid4()
        safe_content = redact_journal_text(content, limit=2000)
        safe_metadata = redact_journal_metadata(raw_metadata)
        cur.execute(
            "INSERT INTO cognitive_consequences ("
            "id,agent_id,act_id,ordinal,kind,content,metadata,status,"
            "source_id,source_stream,source_sequence,observation_key,"
            "difference_kind,reducer_name,reducer_version,payload_hash,"
            "salience,confidence,source_observed_at,declared_action_host_key,"
            "declared_action_ordinal,declared_action_event_id,action_act_id,"
            "correlation_status,late) "
            "VALUES (%s,%s,NULL,NULL,%s,%s,%s,'pending',%s,%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING created_at,ingest_seq",
            (
                observation_id,
                agent_id,
                difference_kind,
                safe_content,
                Jsonb(safe_metadata),
                source_id,
                source_stream,
                source_sequence,
                observation_key,
                difference_kind,
                reducer_name,
                reducer_version,
                payload_hash,
                salience,
                confidence,
                source_observed_at,
                action_host_key,
                action_ordinal,
                action_event_id,
                action_act_id,
                correlation_status,
                late,
            ),
        )
        inserted = cur.fetchone()
        created_at = inserted["created_at"]
        ingest_seq = int(inserted["ingest_seq"])
    return ObservationIngestResult(
        observation_id=observation_id,
        duplicate=False,
        payload_hash=payload_hash,
        correlation_status=correlation_status,
        action_act_id=action_act_id,
        late=late,
        pending_count=pending_before + 1,
        created_at=created_at,
        ingest_seq=ingest_seq,
    )


def resolve_pending_observations_for_act(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    host_key: str,
    act_id: uuid.UUID,
) -> tuple[int, int]:
    """Resolve early observations after this act's actions are durable."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,declared_action_ordinal,declared_action_event_id "
            "FROM cognitive_consequences WHERE agent_id=%s "
            "AND source_id IS NOT NULL AND correlation_status='pending' "
            "AND declared_action_host_key=%s ORDER BY ingest_seq FOR UPDATE",
            (agent_id, host_key),
        )
        rows = list(cur.fetchall())
        resolved = 0
        conflicts = 0
        for row in rows:
            clauses = ["agent_id=%s", "act_id=%s"]
            params: list[Any] = [agent_id, act_id]
            if row["declared_action_ordinal"] is not None:
                clauses.append("ordinal=%s")
                params.append(row["declared_action_ordinal"])
            if row["declared_action_event_id"] is not None:
                clauses.append("event_id=%s")
                params.append(row["declared_action_event_id"])
            cur.execute(
                "SELECT id FROM cognitive_actions WHERE " + " AND ".join(clauses) + " LIMIT 2",
                tuple(params),
            )
            matches = list(cur.fetchall())
            if len(matches) == 1:
                cur.execute(
                    "UPDATE cognitive_consequences SET correlation_status='resolved',"
                    " action_act_id=%s WHERE id=%s AND agent_id=%s",
                    (act_id, row["id"], agent_id),
                )
                resolved += 1
            else:
                cur.execute(
                    "UPDATE cognitive_consequences SET correlation_status='conflict',"
                    " action_act_id=NULL WHERE id=%s AND agent_id=%s",
                    (row["id"], agent_id),
                )
                conflicts += 1
    return resolved, conflicts


def _presentation_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    canonical = row.get("source_id") is not None
    return {
        "observation_id": str(row["id"]),
        "observation_status": "canonical" if canonical else "legacy",
        "source_id": str(row["source_id"])[:256] if canonical else None,
        "source_stream": str(row["source_stream"])[:256] if canonical else None,
        "source_sequence": int(row["source_sequence"]) if canonical else None,
        "observation_key": str(row["observation_key"])[:256] if canonical else None,
        "difference_kind": (
            str(row["difference_kind"])[:64]
            if canonical else str(row["kind"])[:64]
        ),
        "content": redact_journal_text(row["content"], limit=MAX_PRESENTED_CONTENT),
        "salience": round(float(row["salience"]), 6) if canonical else None,
        "confidence": round(float(row["confidence"]), 6) if canonical else None,
        "reducer_name": str(row["reducer_name"])[:128] if canonical else None,
        "reducer_version": str(row["reducer_version"])[:64] if canonical else None,
        "correlation_status": str(row["correlation_status"]),
        "action_ordinal": (
            int(row["declared_action_ordinal"])
            if row.get("declared_action_ordinal") is not None else None
        ),
        "action_event_id": (
            str(row["declared_action_event_id"])[:256]
            if row.get("declared_action_event_id") is not None else None
        ),
        "source_observed_at": (
            row["source_observed_at"].isoformat()
            if row.get("source_observed_at") is not None else None
        ),
        "ingested_at": row["created_at"].isoformat(),
        "late": bool(row.get("late", False)),
    }


def present_pending_observations(
    conn: psycopg.Connection,
    agent_id: str,
    snapshot_token: str,
    *,
    limit: int = MAX_PRESENTED_OBSERVATIONS,
) -> list[dict[str, Any]]:
    """Lease observations and return only immutable frozen presentations."""
    bounded_limit = min(max(1, int(limit)), MAX_PRESENTED_OBSERVATIONS)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT lease_expires_at,presentation_completed_at,"
            " lease_expires_at>clock_timestamp() AS active "
            "FROM cognitive_snapshots WHERE token=%s AND agent_id=%s FOR UPDATE",
            (snapshot_token, agent_id),
        )
        snapshot = cur.fetchone()
        if snapshot is None:
            raise ValueError("snapshot_token is unknown for this agent")

        if snapshot["active"] and snapshot["presentation_completed_at"] is None:
            cur.execute(
                "WITH ranked AS ("
                " SELECT c.id,c.source_sequence,c.ingest_seq,"
                " min(c.ingest_seq) OVER ("
                "   PARTITION BY c.agent_id,c.source_id,c.source_stream"
                " ) AS stream_first "
                " FROM cognitive_consequences c "
                " WHERE c.agent_id=%s AND c.status<>'acknowledged' "
                " AND NOT EXISTS ("
                "   SELECT 1 FROM cognitive_presentations active "
                "   WHERE active.agent_id=c.agent_id "
                "   AND active.consequence_id=c.id "
                "   AND active.lease_expires_at>clock_timestamp()"
                " )) "
                "SELECT c.* FROM cognitive_consequences c "
                "JOIN ranked r ON r.id=c.id "
                "ORDER BY r.stream_first NULLS LAST,"
                " r.source_sequence NULLS LAST,r.ingest_seq NULLS LAST,"
                " c.created_at,c.ordinal NULLS LAST,c.id "
                "LIMIT %s FOR UPDATE OF c SKIP LOCKED",
                (agent_id, bounded_limit),
            )
            selected = list(cur.fetchall())
            for row in selected:
                payload = _presentation_payload(row)
                payload_hash = hashlib.sha256(
                    _canonical_json(payload).encode("utf-8")
                ).hexdigest()
                cur.execute(
                    "INSERT INTO cognitive_presentations ("
                    "snapshot_token,consequence_id,agent_id,lease_expires_at,"
                    "presented_payload,payload_hash,presentation_version) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (
                        snapshot_token,
                        row["id"],
                        agent_id,
                        snapshot["lease_expires_at"],
                        Jsonb(payload),
                        payload_hash,
                        PRESENTATION_VERSION,
                    ),
                )
            cur.execute(
                "UPDATE cognitive_snapshots SET presentation_completed_at=clock_timestamp() "
                "WHERE token=%s AND agent_id=%s "
                "AND presentation_completed_at IS NULL",
                (snapshot_token, agent_id),
            )

        cur.execute(
            "SELECT p.presented_payload,"
            "p.payload_hash AS presentation_payload_hash,"
            "p.presentation_version,c.* "
            "FROM cognitive_presentations p "
            "JOIN cognitive_consequences c "
            "ON (c.id,c.agent_id)=(p.consequence_id,p.agent_id) "
            "WHERE p.snapshot_token=%s AND p.agent_id=%s "
            "AND p.lease_expires_at>clock_timestamp() "
            "AND c.status<>'acknowledged' "
            "ORDER BY p.presented_at,p.consequence_id LIMIT %s",
            (snapshot_token, agent_id, bounded_limit),
        )
        rows = list(cur.fetchall())

    result: list[dict[str, Any]] = []
    for row in rows:
        payload = row["presented_payload"]
        if not isinstance(payload, dict):
            # Mixed-version presentation created before migration 0011.
            payload = _presentation_payload(row)
        elif row["presentation_version"] != PRESENTATION_VERSION or (
            hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            != row["presentation_payload_hash"]
        ):
            raise ObservationConflict("frozen observation presentation hash mismatch")
        result.append(dict(payload))
    return result


def legacy_consequence_mirror(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One-cycle deprecated response mirror; never used in the prompt."""
    return [
        {
            "consequence_id": item["observation_id"],
            "source_act_id": "",
            "ordinal": (
                item["source_sequence"]
                if isinstance(item.get("source_sequence"), int) else 0
            ),
            "kind": item["difference_kind"],
            "content": item["content"],
            "metadata": {},
            "created_at": item["ingested_at"],
        }
        for item in observations
    ]


def observation_queue_stats(
    conn: psycopg.Connection, agent_id: str
) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT count(*)::int AS pending_count,"
            " min(created_at) FILTER (WHERE status<>'acknowledged') AS oldest,"
            " count(*) FILTER (WHERE correlation_status='pending')::int "
            " AS pending_correlation_count,"
            " count(*) FILTER (WHERE correlation_status='conflict')::int "
            " AS conflict_count,"
            " count(*) FILTER (WHERE late)::int AS late_count "
            "FROM cognitive_consequences WHERE agent_id=%s "
            "AND source_id IS NOT NULL AND status<>'acknowledged'",
            (agent_id,),
        )
        row = cur.fetchone()
    oldest = row["oldest"]
    age_s = None
    if oldest is not None:
        now = dt.datetime.now(dt.timezone.utc)
        age_s = max(0.0, (now - oldest).total_seconds())
    return {
        "pending_count": int(row["pending_count"]),
        "oldest_pending_age_s": round(age_s, 3) if age_s is not None else None,
        "pending_correlation_count": int(row["pending_correlation_count"]),
        "conflict_count": int(row["conflict_count"]),
        "late_count": int(row["late_count"]),
    }
