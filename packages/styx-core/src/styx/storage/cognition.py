"""Durable cognitive-act journal and technical line projection (wave 37).

This module models continuity as explicit database provenance.  The will
projection is a deterministic technical reduction over every live eligible
trace; it is deliberately not a claim of selfhood, personality, or sentience.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


WILL_COMPUTATION_VERSION = "causal_carrier_v1"
MAX_SUPPORTS = 8
MAX_PENDING = 16
MAX_PRESENTED_PENDING = 4
MAX_PRESENTED_PENDING_CONTENT = 512
MAX_TRACES = 8
MAX_SYSTEM_PROMPT_ADDITION = 16_000
MAX_JSON_DEPTH = 6
MAX_JSON_NODES = 256
MAX_JSON_BYTES = 32_768
MAX_PARENT_DEPTH = 1024
COMMIT_REQUEST_HASH_METADATA_KEY = "_styx_commit_request_sha256"

_MODEL_PRIVATE_COORDINATE_KEYS = frozenset(
    {
        "hostkey",
        "parenthostkey",
        "predecessorhostkey",
        "cognitivehostkey",
        "callerhostkey",
    }
)
_MODEL_REDUCTION_STATUSES = frozenset(
    {
        "absent",
        "applied",
        "no_residue",
        "pending",
        "predecessor_pending",
        "retryable",
        "running",
        "terminal_failure",
        "unscheduled",
        "untracked",
    }
)
_SHA256_COORDINATE = re.compile(r"^[0-9a-f]{64}$")


class SnapshotReplayConflict(ValueError):
    """A keyed snapshot can only replay its original live, unused envelope."""


class CognitiveCommitConflict(ValueError):
    """A committed host key cannot be reused for different caller semantics."""

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b(\s*[:=]\s*)[^\s,;]+"),
)
_SECRET_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|token|password|passwd|secret|credential|cookie)(?:$|[_-])"
)


@dataclass
class _JsonBudget:
    nodes_left: int = MAX_JSON_NODES
    bytes_left: int = MAX_JSON_BYTES

    def consume(self, value: Any) -> bool:
        self.nodes_left -= 1
        if self.nodes_left < 0:
            return False
        if isinstance(value, (dict, list, tuple)):
            size = 1
        elif isinstance(value, str):
            size = len(value.encode("utf-8"))
        elif isinstance(value, (bytes, bytearray)):
            size = len(value)
        else:
            size = len(str(value).encode("utf-8"))
        self.bytes_left -= size
        return self.bytes_left >= 0


def redact_journal_text(value: Any, *, limit: int = 8000) -> str:
    """Bound and redact common credential shapes without losing event shape."""
    text = str(value or "")[:limit]
    text = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", text)
    text = _SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]", text)
    return text


def redact_journal_json(
    value: Any, *, _depth: int = 0, _budget: _JsonBudget | None = None
) -> Any:
    """Recursively redact under one aggregate node/UTF-8-byte budget."""
    budget = _budget or _JsonBudget()
    if _depth > MAX_JSON_DEPTH or not budget.consume(value):
        return "[TRUNCATED]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:16]:
            key = str(raw_key)[:64]
            if not budget.consume(key):
                result["truncated"] = "[TRUNCATED]"
                break
            result[key] = (
                "[REDACTED]"
                if _SECRET_KEY.search(key)
                else redact_journal_json(item, _depth=_depth + 1, _budget=budget)
            )
            if budget.nodes_left <= 0 or budget.bytes_left <= 0:
                break
        return result
    if isinstance(value, (list, tuple)):
        return [
            redact_journal_json(item, _depth=_depth + 1, _budget=budget)
            for item in value[:16]
            if budget.nodes_left > 0 and budget.bytes_left > 0
        ]
    if isinstance(value, str):
        return redact_journal_text(value, limit=1000)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_journal_text(value, limit=1000)


def redact_journal_metadata(value: Any) -> dict[str, Any]:
    redacted = redact_journal_json(value)
    return redacted if isinstance(redacted, dict) else {}


def validate_journal_json(value: Any, *, max_string: int = 16_000) -> None:
    """Validate the same aggregate shape that the redactor can safely walk."""
    budget = _JsonBudget()

    def _visit(item: Any, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"metadata nesting exceeds {MAX_JSON_DEPTH} levels")
        if not budget.consume(item):
            if budget.nodes_left < 0:
                raise ValueError(f"metadata exceeds {MAX_JSON_NODES} aggregate nodes")
            raise ValueError(f"metadata exceeds {MAX_JSON_BYTES} aggregate UTF-8 bytes")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, str):
            if len(item) > max_string:
                raise ValueError(f"metadata string exceeds {max_string} characters")
            return
        if isinstance(item, (int, float)):
            if isinstance(item, float) and (item != item or abs(item) == float("inf")):
                raise ValueError("metadata number must be finite")
            return
        if isinstance(item, list):
            if len(item) > 16:
                raise ValueError("metadata array exceeds 16 items")
            for child in item:
                _visit(child, depth + 1)
            return
        if isinstance(item, dict):
            if len(item) > 16:
                raise ValueError("metadata object exceeds 16 keys")
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 64:
                    raise ValueError("metadata keys must contain 1..64 characters")
                if not budget.consume(key):
                    raise ValueError(
                        f"metadata exceeds {MAX_JSON_BYTES} aggregate UTF-8 bytes"
                    )
                _visit(child, depth + 1)
            return
        raise ValueError("metadata values must be JSON-compatible")

    _visit(value, 0)


def _vector_literal(vector: Sequence[float] | None) -> str | None:
    if vector is None:
        return None
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _parse_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().strip("[]")
        return [] if not raw else [float(item) for item in raw.split(",")]
    return [float(item) for item in value]


def lock_agent_line(conn: psycopg.Connection, agent_id: str) -> None:
    """Serialize one agent's preturn/commit transaction across processes."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"styx:cognitive_line:{agent_id}",),
        )


def _representative_indices(total: int, limit: int) -> list[int]:
    """Evenly cover the full line instead of returning only a recent tail."""
    if total <= limit:
        return list(range(total))
    if limit <= 1:
        return [total - 1]
    return sorted({round(index * (total - 1) / (limit - 1)) for index in range(limit)})


def ensure_will_projection(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    support_limit: int = MAX_SUPPORTS,
) -> dict[str, Any]:
    """Return/rebuild the query-independent carrier of the complete live line.

    Embeddings remain diagnostic coordinates.  The active model receives the
    bounded extractive carrier built from the complete eligible set, while
    legacy/unfenced rows remain explicitly provisional.  Pending act
    reductions make an otherwise usable carrier visibly stale.
    """
    from styx.engine.causal_carrier import build_causal_carrier

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO line_state(agent_id, version, dirty) VALUES (%s, 0, true) "
            "ON CONFLICT (agent_id) DO NOTHING",
            (agent_id,),
        )
        cur.execute(
            "SELECT version,dirty,causal_root_hash,causal_root_version,"
            " causal_frontier FROM line_state WHERE agent_id = %s FOR UPDATE",
            (agent_id,),
        )
        state = cur.fetchone()
        assert state is not None
        version = int(state["version"])
        cur.execute(
            "SELECT "
            " count(*) FILTER (WHERE status IN ('pending','running','retryable'))::int "
            "   AS pending_count, "
            " count(*) FILTER (WHERE status='terminal_failure')::int AS failure_count "
            "FROM cognitive_act_reductions WHERE agent_id=%s",
            (agent_id,),
        )
        reduction_state = cur.fetchone() or {"pending_count": 0, "failure_count": 0}
        pending_count = int(reduction_state["pending_count"] or 0)
        failure_count = int(reduction_state["failure_count"] or 0)
        if not state["dirty"]:
            cur.execute(
                "SELECT formed,source_count,source_hash,supports,"
                " computation_version,embedding,projection_status,"
                " projection_available,covered_line_version,coverage_count,"
                " coverage_hash,carrier_text,carrier_payload,carrier_version "
                "FROM will_projections WHERE agent_id=%s AND line_version=%s",
                (agent_id, version),
            )
            cached = cur.fetchone()
            if cached is not None:
                payload = dict(cached["carrier_payload"] or {})
                base_status = str(
                    payload.get("base_projection_status")
                    or cached["projection_status"]
                )
                effective_status = (
                    "degraded" if failure_count
                    else "stale" if pending_count and cached["projection_available"]
                    else base_status
                )
                formed = bool(
                    effective_status == "ready"
                    and cached["projection_available"]
                    and int(cached["covered_line_version"]) == version
                    and int(cached["coverage_count"]) == int(cached["source_count"])
                    and cached["coverage_hash"] == cached["source_hash"]
                )
                cur.execute(
                    "UPDATE will_projections SET projection_status=%s,formed=%s,"
                    " pending_reduction_count=%s WHERE agent_id=%s AND line_version=%s",
                    (effective_status, formed, pending_count, agent_id, version),
                )
                return {
                    "formed": formed,
                    "technical_projection": True,
                    "line_version": version,
                    "source_count": int(cached["source_count"]),
                    "source_hash": cached["source_hash"],
                    "supports": cached["supports"],
                    "computation_version": cached["computation_version"],
                    "embedding": _parse_vector(cached["embedding"]),
                    "projection_status": effective_status,
                    "projection_available": bool(cached["projection_available"]),
                    "covered_line_version": int(cached["covered_line_version"]),
                    "coverage_count": int(cached["coverage_count"]),
                    "coverage_hash": cached["coverage_hash"],
                    "carrier_text": cached["carrier_text"] or "",
                    "carrier_version": cached["carrier_version"],
                    "pending_reduction_count": pending_count,
                    "reduction_failure_count": failure_count,
                    "technical_strength": float(payload.get("technical_strength", 0.0)),
                    "coherence": payload.get("coherence"),
                    "diagnostics": payload.get("diagnostics", {}),
                    "causal_root_hash": str(state["causal_root_hash"]),
                    "causal_root_version": int(state["causal_root_version"]),
                    "causal_frontier": list(state["causal_frontier"] or []),
                    "root_coverage_hash": payload.get("root_coverage_hash", ""),
                    "root_count": int(payload.get("root_count", 0)),
                    "covered_node_count": int(
                        payload.get("covered_node_count", 0)
                    ),
                }

        cur.execute(
            "SELECT id,seq,role,kind,content,embedding,created_at,updated_at,"
            " line_provenance,cognitive_act_id,residue_ordinal,residue_causal_role,"
            " residue_predecessors,residue_line_root_hash,residue_affect "
            "FROM memories WHERE agent_id=%s "
            "  AND memory_domain='subjective_trace' AND line_eligible=true "
            "  AND superseded_by IS NULL ORDER BY seq ASC",
            (agent_id,),
        )
        rows = list(cur.fetchall())

        carrier_rows = [
            {
                **dict(row),
                "embedding": _parse_vector(row["embedding"]),
            }
            for row in rows
        ]
        carrier = build_causal_carrier(
            carrier_rows,
            max_supports=max(1, support_limit),
        )

        vectors = [_parse_vector(row["embedding"]) for row in rows]
        vectors = [vector for vector in vectors if vector]
        projection_vector: list[float] | None = None
        if vectors:
            dimension = len(vectors[0])
            compatible = [vector for vector in vectors if len(vector) == dimension]
            projection_vector = [
                sum(vector[index] for vector in compatible) / len(compatible)
                for index in range(dimension)
            ]

        supports = list(carrier["supports"])
        source_hash = carrier["coverage_hash"]
        base_status = carrier["projection_status"]
        effective_status = (
            "degraded" if failure_count
            else "stale" if pending_count and carrier["projection_available"]
            else base_status
        )
        projection_available = bool(carrier["projection_available"])
        carrier_text = carrier["carrier_text"] or None
        carrier_version = (
            str(carrier["diagnostics"].get("algorithm_version"))
            if projection_available else None
        )
        formed = bool(
            effective_status == "ready"
            and projection_available
            and carrier["coverage_count"] == len(rows)
        )
        carrier_payload = {
            "base_projection_status": base_status,
            "technical_strength": carrier["technical_strength"],
            "coherence": carrier["coherence"],
            "root_coverage_hash": carrier["root_coverage_hash"],
            "root_count": carrier["root_count"],
            "covered_node_count": carrier["covered_node_count"],
            "diagnostics": carrier["diagnostics"],
        }
        cur.execute(
            "INSERT INTO will_projections "
            "(agent_id,line_version,formed,source_count,source_hash,embedding,supports,"
            " computation_version,projection_status,projection_available,"
            " covered_line_version,coverage_count,coverage_hash,carrier_text,"
            " carrier_payload,carrier_version,pending_reduction_count,carrier_built_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
            " clock_timestamp()) "
            "ON CONFLICT (agent_id,line_version) DO UPDATE SET "
            "formed=excluded.formed, source_count=excluded.source_count, "
            "source_hash=excluded.source_hash, embedding=excluded.embedding, "
            "supports=excluded.supports, computation_version=excluded.computation_version,"
            "projection_status=excluded.projection_status,"
            "projection_available=excluded.projection_available,"
            "covered_line_version=excluded.covered_line_version,"
            "coverage_count=excluded.coverage_count,coverage_hash=excluded.coverage_hash,"
            "carrier_text=excluded.carrier_text,carrier_payload=excluded.carrier_payload,"
            "carrier_version=excluded.carrier_version,"
            "pending_reduction_count=excluded.pending_reduction_count,"
            "carrier_built_at=excluded.carrier_built_at",
            (
                agent_id,
                version,
                formed,
                len(rows),
                source_hash,
                _vector_literal(projection_vector),
                Jsonb(supports),
                WILL_COMPUTATION_VERSION,
                effective_status,
                projection_available,
                version,
                carrier["coverage_count"],
                carrier["coverage_hash"],
                carrier_text,
                Jsonb(carrier_payload),
                carrier_version,
                pending_count,
            ),
        )
        cur.execute(
            "UPDATE line_state SET dirty=false, updated_at=clock_timestamp() "
            "WHERE agent_id=%s AND version=%s",
            (agent_id, version),
        )
    return {
        "formed": formed,
        "technical_projection": True,
        "line_version": version,
        "source_count": len(rows),
        "source_hash": source_hash,
        "supports": supports,
        "computation_version": WILL_COMPUTATION_VERSION,
        "embedding": projection_vector,
        "projection_status": effective_status,
        "projection_available": projection_available,
        "covered_line_version": version,
        "coverage_count": carrier["coverage_count"],
        "coverage_hash": carrier["coverage_hash"],
        "carrier_text": carrier["carrier_text"],
        "carrier_version": carrier_version,
        "pending_reduction_count": pending_count,
        "reduction_failure_count": failure_count,
        "technical_strength": carrier["technical_strength"],
        "coherence": carrier["coherence"],
        "diagnostics": carrier["diagnostics"],
        "causal_root_hash": str(state["causal_root_hash"]),
        "causal_root_version": int(state["causal_root_version"]),
        "causal_frontier": list(state["causal_frontier"] or []),
        "root_coverage_hash": carrier["root_coverage_hash"],
        "root_count": carrier["root_count"],
        "covered_node_count": carrier["covered_node_count"],
    }


def strict_reconstruction(
    conn: psycopg.Connection,
    agent_id: str,
    query_vector: Sequence[float] | None,
    *,
    limit: int = MAX_TRACES,
) -> list[dict[str, Any]]:
    """Query only validated live line traces; no query vector means no guesses.

    Legacy rows remain available through explicit archive/dialogue tools, but
    they cannot silently enter the active cognitive envelope as subjective
    evidence merely because an older schema labelled them ``subjective``.
    """
    if not query_vector:
        return []
    qvec = _vector_literal(query_vector)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, role, kind, content, created_at, "
            "       1 - (embedding <=> %s::vector) AS score "
            "FROM memories WHERE agent_id=%s "
            "  AND memory_domain='subjective_trace' AND line_eligible=true "
            "  AND line_provenance = 'validated_act_residue' "
            "  AND superseded_by IS NULL AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %s::vector, seq DESC LIMIT %s",
            (qvec, agent_id, qvec, min(max(1, limit), MAX_TRACES)),
        )
        return [
            {
                "memory_id": str(row["id"]),
                "role": row["role"],
                "kind": row["kind"],
                "content": row["content"],
                "created_at": row["created_at"].isoformat(),
                "score": float(row["score"]),
            }
            for row in cur.fetchall()
        ]


def present_pending_consequences(
    conn: psycopg.Connection,
    agent_id: str,
    snapshot_token: str,
    *,
    limit: int = MAX_PENDING,
) -> list[dict[str, Any]]:
    """Lease pending rows to a physical snapshot without rebinding history.

    The presentation rows are immutable.  An uncommitted expired lease merely
    stops fencing the consequence, allowing a later snapshot to create a new
    association.  Retrying the same still-live snapshot reads its association.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT lease_expires_at,presentation_completed_at, "
            "       lease_expires_at > clock_timestamp() AS active "
            "FROM cognitive_snapshots "
            "WHERE token=%s AND agent_id=%s FOR UPDATE",
            (snapshot_token, agent_id),
        )
        snapshot = cur.fetchone()
        if snapshot is None:
            raise ValueError("snapshot_token is unknown for this agent")
        # Presentation is the acknowledgement boundary, not merely a DB
        # selection.  Lease only the subset that is guaranteed to fit the
        # active prompt; omitted rows remain pending for a later snapshot.
        bounded_limit = min(max(1, limit), MAX_PENDING, MAX_PRESENTED_PENDING)
        if snapshot["active"] and snapshot["presentation_completed_at"] is None:
            cur.execute(
                "SELECT c.id FROM cognitive_consequences c "
                "WHERE c.agent_id=%s AND c.status <> 'acknowledged' "
                "  AND NOT EXISTS ("
                "    SELECT 1 FROM cognitive_presentations active "
                "    WHERE active.agent_id=c.agent_id "
                "      AND active.consequence_id=c.id "
                "      AND active.lease_expires_at > clock_timestamp()"
                "  ) "
                "ORDER BY c.created_at,c.ordinal LIMIT %s FOR UPDATE OF c",
                (agent_id, bounded_limit),
            )
            available_ids = [row["id"] for row in cur.fetchall()]
            for consequence_id in available_ids:
                cur.execute(
                    "INSERT INTO cognitive_presentations "
                    "(snapshot_token,consequence_id,agent_id,lease_expires_at) "
                    "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (
                        snapshot_token,
                        consequence_id,
                        agent_id,
                        snapshot["lease_expires_at"],
                    ),
                )
            cur.execute(
                "UPDATE cognitive_snapshots "
                "SET presentation_completed_at=clock_timestamp() "
                "WHERE token=%s AND agent_id=%s "
                "  AND presentation_completed_at IS NULL",
                (snapshot_token, agent_id),
            )
        cur.execute(
            "SELECT c.id,c.act_id,c.ordinal,c.kind,c.content,c.metadata,c.created_at "
            "FROM cognitive_presentations p "
            "JOIN cognitive_consequences c "
            "  ON (c.id,c.agent_id)=(p.consequence_id,p.agent_id) "
            "WHERE p.snapshot_token=%s AND p.agent_id=%s "
            "  AND p.lease_expires_at > clock_timestamp() "
            "  AND c.status <> 'acknowledged' "
            "ORDER BY c.created_at,c.ordinal LIMIT %s",
            (snapshot_token, agent_id, bounded_limit),
        )
        rows = list(cur.fetchall())
    return [
        {
            "consequence_id": str(row["id"]),
            "source_act_id": str(row["act_id"]),
            "ordinal": int(row["ordinal"]),
            "kind": row["kind"],
            "content": redact_journal_text(
                row["content"], limit=MAX_PRESENTED_PENDING_CONTENT
            ),
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


def record_snapshot(
    conn: psycopg.Connection,
    agent_id: str,
    token: str,
    line_version: int,
    *,
    session_id: uuid.UUID | None = None,
    host_key: str | None = None,
    request_hash: str | None = None,
    lease_seconds: float = 60.0,
) -> str:
    """Acquire a new physical snapshot.

    Keyed retries are resolved by :func:`load_snapshot_replay` before this
    insert. Keeping replay separate prevents a caller from accidentally
    recomputing an envelope around an old token.
    """
    bounded_lease = min(3600.0, max(1.0, float(lease_seconds)))
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO cognitive_snapshots"
            "(token,agent_id,session_id,host_key,request_hash,line_version,lease_expires_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,clock_timestamp() + %s * interval '1 second')",
            (
                token, agent_id, session_id, host_key, request_hash,
                line_version, bounded_lease,
            ),
        )
    return token


def load_snapshot_replay(
    conn: psycopg.Connection,
    agent_id: str,
    host_key: str | None,
    request_hash: str,
    *,
    session_id: uuid.UUID | None = None,
) -> dict[str, Any] | None:
    """Return the exact live envelope for a physical-attempt retry.

    Hermes supplies a host key and receives strict conflict detection. The
    current OpenClaw assemble ABI has no durable turn key, so it replays only
    a byte-identical, live, unused envelope in the same session.
    """
    if not host_key and session_id is None:
        return None
    with conn.cursor(row_factory=dict_row) as cur:
        if host_key:
            cur.execute(
                "SELECT request_hash,response_payload,used_by_act_id,"
                "       lease_expires_at > clock_timestamp() AS active "
                "FROM cognitive_snapshots "
                "WHERE agent_id=%s AND host_key=%s FOR UPDATE",
                (agent_id, host_key),
            )
        else:
            cur.execute(
                "SELECT request_hash,response_payload,used_by_act_id,"
                "       lease_expires_at > clock_timestamp() AS active "
                "FROM cognitive_snapshots "
                "WHERE agent_id=%s AND session_id=%s AND host_key IS NULL "
                "  AND request_hash=%s AND used_by_act_id IS NULL "
                "  AND lease_expires_at > clock_timestamp() "
                "ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                (agent_id, session_id, request_hash),
            )
        existing = cur.fetchone()
    if existing is None:
        return None
    if host_key and existing["request_hash"] != request_hash:
        raise SnapshotReplayConflict(
            "host_key was already used for a different preturn request"
        )
    if existing["used_by_act_id"] is not None:
        raise SnapshotReplayConflict("host_key snapshot has already been committed")
    if not existing["active"]:
        raise SnapshotReplayConflict("host_key snapshot lease has expired")
    payload = existing["response_payload"]
    if not isinstance(payload, dict):
        raise SnapshotReplayConflict("host_key snapshot envelope is incomplete")
    return dict(payload)


def complete_snapshot_response(
    conn: psycopg.Connection,
    agent_id: str,
    snapshot_token: str,
    response: dict[str, Any],
) -> None:
    """Freeze the public envelope beside the token before committing it."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cognitive_snapshots SET response_payload=%s "
            "WHERE token=%s AND agent_id=%s AND response_payload IS NULL",
            (Jsonb(response), snapshot_token, agent_id),
        )
        if cur.rowcount != 1:
            raise RuntimeError("cognitive snapshot envelope was already completed")


@dataclass(frozen=True)
class CommitResult:
    act_id: uuid.UUID
    duplicate: bool
    acknowledged_count: int
    consequence_ids: tuple[uuid.UUID, ...]


def _assert_acyclic_parentage(
    cur: psycopg.Cursor[Any], agent_id: str, host_key: str, parent_host_key: str | None
) -> None:
    """Reject self-parent and cycles in the declared host-key graph.

    The declared graph is authoritative even before all parents arrive, so
    this covers both direct resolution and late child resolution.
    """
    if parent_host_key is None:
        return
    if parent_host_key == host_key:
        raise ValueError("cognitive act cannot be its own parent")
    cur.execute(
        "WITH RECURSIVE ancestry(host_key,declared_parent_key,depth,path,cycle) AS ("
        " SELECT a.host_key,a.declared_parent_key,1,ARRAY[a.host_key],false "
        " FROM cognitive_acts a WHERE a.agent_id=%s AND a.host_key=%s "
        " UNION ALL "
        " SELECT p.host_key,p.declared_parent_key,x.depth+1,"
        "        x.path || p.host_key,p.host_key=ANY(x.path) "
        " FROM ancestry x JOIN cognitive_acts p "
        "   ON p.agent_id=%s AND p.host_key=x.declared_parent_key "
        " WHERE x.declared_parent_key IS NOT NULL "
        "   AND x.depth < %s AND NOT x.cycle"
        ") "
        "SELECT COALESCE(bool_or(cycle),false) AS existing_cycle, "
        "       COALESCE(bool_or(host_key=%s OR declared_parent_key=%s),false) "
        "           AS reaches_new_act, "
        "       COALESCE(bool_or(depth=%s AND declared_parent_key IS NOT NULL),false) "
        "           AS depth_guard "
        "FROM ancestry",
        (
            agent_id,
            parent_host_key,
            agent_id,
            MAX_PARENT_DEPTH,
            host_key,
            host_key,
            MAX_PARENT_DEPTH,
        ),
    )
    audit = cur.fetchone()
    if audit and (audit["existing_cycle"] or audit["reaches_new_act"]):
        raise ValueError("cognitive act parentage would create a causal cycle")
    if audit and audit["depth_guard"]:
        raise ValueError("cognitive act parentage exceeds bounded ancestry depth")


def _commit_request_fingerprint(
    *,
    agent_id: str,
    host_key: str,
    parent_host_key: str | None,
    session_id: uuid.UUID | None,
    snapshot_token: str | None,
    snapshot_policy: str,
    parent_policy: str,
    status: str,
    channel_input: dict[str, Any],
    channel_output: dict[str, Any],
    actions: Sequence[dict[str, Any]],
    consequences: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Hash only bounded caller coordinates, before dynamic resolution.

    ``latest_session`` parent/snapshot selection and the current line version
    deliberately do not participate: an identical delivery must remain an
    idempotent duplicate even if external state advanced after the first
    commit.  The returned metadata has internal keys removed so a host cannot
    forge either the fingerprint or resolution audit fields.
    """
    caller_metadata = dict(metadata)
    caller_metadata.pop(COMMIT_REQUEST_HASH_METADATA_KEY, None)
    caller_metadata.pop("continuity_resolution", None)
    safe_metadata = redact_journal_metadata(caller_metadata)

    def _safe_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items:
            safe = redact_journal_metadata(item)
            # Journal content has a wider intentional bound than generic
            # nested metadata, so preserve the exact persisted text here.
            safe["content"] = redact_journal_text(item.get("content", ""))
            safe["metadata"] = redact_journal_metadata(item.get("metadata", {}))
            normalized.append(safe)
        return normalized

    payload = {
        "version": "cognition_commit_request_v1",
        "agent_id": agent_id,
        "host_key": host_key,
        "session_id": str(session_id) if session_id is not None else None,
        "declared_parent_key": parent_host_key,
        "snapshot_token": snapshot_token,
        "snapshot_policy": snapshot_policy,
        "parent_policy": parent_policy,
        "status": status,
        "channel_input": redact_journal_json(channel_input),
        "channel_output": redact_journal_json(channel_output),
        "actions": _safe_items(actions),
        "consequences": _safe_items(consequences),
        "metadata": safe_metadata,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest, safe_metadata


def commit_cognitive_act(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    host_key: str,
    parent_host_key: str | None,
    session_id: uuid.UUID | None,
    snapshot_token: str | None,
    status: str,
    input_line_version: int,
    channel_input: dict[str, Any],
    channel_output: dict[str, Any],
    actions: Sequence[dict[str, Any]],
    consequences: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    snapshot_policy: str = "explicit",
    parent_policy: str = "explicit",
) -> CommitResult:
    """Append one idempotent act and its ordered journals in the caller tx."""
    if snapshot_policy not in {"explicit", "latest_session"}:
        raise ValueError("unsupported snapshot_policy")
    if parent_policy not in {"explicit", "latest_session"}:
        raise ValueError("unsupported parent_policy")
    request_hash, safe_metadata = _commit_request_fingerprint(
        agent_id=agent_id,
        host_key=host_key,
        parent_host_key=parent_host_key,
        session_id=session_id,
        snapshot_token=snapshot_token,
        snapshot_policy=snapshot_policy,
        parent_policy=parent_policy,
        status=status,
        channel_input=channel_input,
        channel_output=channel_output,
        actions=actions,
        consequences=consequences,
        metadata=metadata,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,metadata FROM cognitive_acts WHERE agent_id=%s AND host_key=%s",
            (agent_id, host_key),
        )
        existing = cur.fetchone()
        if existing is not None:
            existing_metadata = existing["metadata"]
            stored_hash = (
                existing_metadata.get(COMMIT_REQUEST_HASH_METADATA_KEY)
                if isinstance(existing_metadata, dict)
                else None
            )
            if stored_hash is not None and stored_hash != request_hash:
                raise CognitiveCommitConflict(
                    "host_key was already committed with a different request"
                )
            cur.execute(
                "SELECT count(*)::int AS acknowledged_count FROM cognitive_consequences "
                "WHERE agent_id=%s AND acknowledged_by_act_id=%s",
                (agent_id, existing["id"]),
            )
            acknowledged = int(cur.fetchone()["acknowledged_count"])
            cur.execute(
                "SELECT id FROM cognitive_consequences "
                "WHERE agent_id=%s AND act_id=%s ORDER BY ordinal",
                (agent_id, existing["id"]),
            )
            return CommitResult(
                existing["id"], True, acknowledged,
                tuple(row["id"] for row in cur.fetchall()),
            )

        effective_parent_host_key = parent_host_key
        if (
            parent_policy == "latest_session"
            and effective_parent_host_key is None
            and session_id is not None
        ):
            cur.execute(
                "SELECT host_key FROM cognitive_acts "
                "WHERE agent_id=%s AND session_id=%s "
                "ORDER BY completed_at DESC NULLS LAST,created_at DESC,id DESC LIMIT 1",
                (agent_id, session_id),
            )
            latest_parent = cur.fetchone()
            if latest_parent is not None:
                effective_parent_host_key = latest_parent["host_key"]

        effective_snapshot_token = snapshot_token
        if (
            snapshot_policy == "latest_session"
            and effective_snapshot_token is None
            and session_id is not None
        ):
            cur.execute(
                "SELECT token FROM cognitive_snapshots "
                "WHERE agent_id=%s AND session_id=%s AND host_key IS NULL "
                "  AND used_by_act_id IS NULL AND response_payload IS NOT NULL "
                "  AND lease_expires_at > clock_timestamp() "
                "ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                (agent_id, session_id),
            )
            latest_snapshot = cur.fetchone()
            if latest_snapshot is not None:
                effective_snapshot_token = latest_snapshot["token"]

        metadata = {
            **safe_metadata,
            "continuity_resolution": {
                "snapshot_policy": snapshot_policy,
                "snapshot_claimed": effective_snapshot_token is not None,
                "parent_policy": parent_policy,
                "parent_resolved": effective_parent_host_key is not None,
            },
            COMMIT_REQUEST_HASH_METADATA_KEY: request_hash,
        }

        _assert_acyclic_parentage(
            cur, agent_id, host_key, effective_parent_host_key
        )

        if effective_snapshot_token:
            cur.execute(
                "SELECT line_version,host_key,used_by_act_id FROM cognitive_snapshots "
                "WHERE token=%s AND agent_id=%s FOR UPDATE",
                (effective_snapshot_token, agent_id),
            )
            snapshot = cur.fetchone()
            if snapshot is None:
                raise ValueError("snapshot_token is unknown for this agent")
            if snapshot["host_key"] is not None and snapshot["host_key"] != host_key:
                raise ValueError("snapshot_token is fenced to a different host_key")
            if snapshot["used_by_act_id"] is not None:
                raise ValueError("snapshot_token has already been committed")
            input_line_version = int(snapshot["line_version"])

        parent_id = None
        if effective_parent_host_key:
            cur.execute(
                "SELECT id FROM cognitive_acts WHERE agent_id=%s AND host_key=%s",
                (agent_id, effective_parent_host_key),
            )
            parent = cur.fetchone()
            parent_id = parent["id"] if parent is not None else None

        act_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO cognitive_acts "
            "(id,agent_id,host_key,session_id,declared_parent_key,parent_act_id,"
            " input_line_version,input_snapshot_token,status,channel_input,"
            " channel_output,metadata,completed_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,clock_timestamp())",
            (
                act_id, agent_id, host_key, session_id,
                effective_parent_host_key, parent_id,
                input_line_version, effective_snapshot_token, status,
                Jsonb(redact_journal_json(channel_input)),
                Jsonb(redact_journal_json(channel_output)),
                Jsonb(redact_journal_json(metadata)),
            ),
        )
        if effective_snapshot_token:
            cur.execute(
                "UPDATE cognitive_snapshots SET used_by_act_id=%s, used_at=clock_timestamp() "
                "WHERE token=%s AND agent_id=%s",
                (act_id, effective_snapshot_token, agent_id),
            )
        # Resolve children that declared this act before it arrived.
        cur.execute(
            "UPDATE cognitive_acts SET parent_act_id=%s "
            "WHERE agent_id=%s AND declared_parent_key=%s AND parent_act_id IS NULL",
            (act_id, agent_id, host_key),
        )

        for ordinal, action in enumerate(actions):
            cur.execute(
                "INSERT INTO cognitive_actions "
                "(agent_id,act_id,ordinal,kind,event_id,name,content,metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    agent_id, act_id, ordinal, action["kind"],
                    action.get("tool_event_id", ""), action.get("name", ""),
                    redact_journal_text(action.get("content", "")),
                    Jsonb(redact_journal_metadata(action.get("metadata", {}))),
                ),
            )

        consequence_ids: list[uuid.UUID] = []
        for ordinal, consequence in enumerate(consequences):
            consequence_id = uuid.uuid4()
            cur.execute(
                "INSERT INTO cognitive_consequences "
                "(id,agent_id,act_id,ordinal,kind,content,metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (
                    consequence_id, agent_id, act_id, ordinal,
                    consequence["kind"], redact_journal_text(consequence["content"]),
                    Jsonb(redact_journal_metadata(consequence.get("metadata", {}))),
                ),
            )
            consequence_ids.append(consequence_id)

        acknowledged = 0
        if effective_snapshot_token:
            cur.execute(
                "UPDATE cognitive_consequences SET status='acknowledged', "
                "acknowledged_by_act_id=%s, acknowledged_at=clock_timestamp() "
                "WHERE agent_id=%s AND status <> 'acknowledged' "
                "  AND EXISTS ("
                "    SELECT 1 FROM cognitive_presentations p "
                "    WHERE p.agent_id=%s AND p.consequence_id=cognitive_consequences.id "
                "      AND p.snapshot_token=%s "
                "      AND p.lease_expires_at > clock_timestamp()"
                "  )",
                (act_id, agent_id, agent_id, effective_snapshot_token),
            )
            acknowledged = cur.rowcount or 0
    return CommitResult(act_id, False, acknowledged, tuple(consequence_ids))


def append_lineage(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    transform: str,
    source_memory_id: uuid.UUID | None = None,
    target_memory_id: uuid.UUID | None = None,
    cognitive_act_id: uuid.UUID | None = None,
    ordinal: int = 0,
    retained_weight: float | None = None,
    source_coordinates: dict[str, Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memory_lineage "
            "(agent_id,source_memory_id,target_memory_id,cognitive_act_id,"
            " transform,ordinal,retained_weight,source_coordinates) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                agent_id, source_memory_id, target_memory_id, cognitive_act_id,
                transform, ordinal, retained_weight, Jsonb(source_coordinates or {}),
            ),
        )


def current_line_version(conn: psycopg.Connection, agent_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM line_state WHERE agent_id=%s", (agent_id,))
        row = cur.fetchone()
        return 0 if row is None else int(row[0])


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _without_private_model_coordinates(value: Any) -> Any:
    """Remove host-owned idempotency coordinates from a model-visible tree."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in _MODEL_PRIVATE_COORDINATE_KEYS:
                continue
            result[key] = _without_private_model_coordinates(item)
        return result
    if isinstance(value, list):
        return [_without_private_model_coordinates(item) for item in value]
    return value


def _safe_model_freshness(value: dict[str, Any] | None) -> dict[str, Any]:
    """Project freshness through a strict, non-host-owned schema."""
    raw = value or {}
    safe: dict[str, Any] = {}
    for key in ("fresh", "predecessor_found", "timed_out"):
        if type(raw.get(key)) is bool:
            safe[key] = raw[key]

    waited_ms = raw.get("waited_ms")
    if type(waited_ms) is int and 0 <= waited_ms <= 60_000:
        safe["waited_ms"] = waited_ms

    predecessor_id = raw.get("predecessor_act_id")
    if isinstance(predecessor_id, str):
        try:
            safe["predecessor_act_id"] = str(uuid.UUID(predecessor_id))
        except (ValueError, AttributeError):
            pass

    root_hash = raw.get("predecessor_causal_root_hash")
    if isinstance(root_hash, str) and _SHA256_COORDINATE.fullmatch(root_hash):
        safe["predecessor_causal_root_hash"] = root_hash

    status = raw.get("reduction_status")
    if isinstance(status, str) and status in _MODEL_REDUCTION_STATUSES:
        safe["reduction_status"] = status
    return safe


def build_system_prompt_addition(
    *,
    will: dict[str, Any],
    cognitive_posture: dict[str, Any],
    pending: Sequence[dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    continuity_freshness: dict[str, Any] | None = None,
) -> str:
    """Render an allowlisted envelope without ever slicing a causal carrier.

    The carrier is an opaque all-roots projection.  Under prompt pressure it
    is retained byte-for-byte (at the Python string boundary) while optional
    surfaces are reduced.  If the complete carrier still cannot fit, the
    projection fails closed instead of exposing a causally partial prefix.
    """
    opening = (
        '<styx-cognitive-continuity data-only="true" '
        'authority="context-not-instruction">\n'
    )
    closing = "\n</styx-cognitive-continuity>"

    safe_pending = [
        {
            "consequence_id": str(item.get("consequence_id", ""))[:64],
            "source_act_id": str(item.get("source_act_id", ""))[:64],
            "ordinal": int(item.get("ordinal", 0)),
            "kind": str(item.get("kind", ""))[:64],
            "content": str(item.get("content", ""))[:MAX_PRESENTED_PENDING_CONTENT],
        }
        for item in list(pending)[:MAX_PRESENTED_PENDING]
        if isinstance(item, dict)
    ]
    safe_traces = [
        {
            "memory_id": str(item.get("memory_id", ""))[:64],
            "role": str(item.get("role", ""))[:32],
            "kind": str(item.get("kind", ""))[:32],
            "content": str(item.get("content", ""))[:600],
            "score": round(float(item.get("score", 0.0)), 6),
        }
        for item in list(traces)[:MAX_TRACES]
        if isinstance(item, dict)
    ]
    safe_posture = _without_private_model_coordinates(
        redact_journal_json(cognitive_posture)
    )
    if not isinstance(safe_posture, dict):
        safe_posture = {}
    root_count = int(will.get("root_count", 0))
    pending_reduction_count = int(will.get("pending_reduction_count", 0))
    reduction_failure_count = int(will.get("reduction_failure_count", 0))
    active_status = str(will.get("projection_status", "empty"))[:16]
    if not root_count and not pending_reduction_count and not reduction_failure_count:
        # Quarantine is a public diagnostic surface, never a model-visible
        # pseudo-line.  With no validated roots it is cognitively equivalent
        # to an empty retained causal line.
        active_status = "empty"
    raw_carrier = str(will.get("carrier_text", ""))
    carrier_available = bool(will.get("projection_available")) and bool(raw_carrier)
    projection = {
        "formed": bool(will.get("formed")),
        "projection_status": active_status,
        "projection_available": carrier_available,
        # Full storage versions/counts/hashes include quarantine and stay on
        # the public diagnostic API.  Only reducer-owned root coordinates are
        # visible to the model.
        "causal_root_hash": (
            str(will.get("causal_root_hash"))
            if _SHA256_COORDINATE.fullmatch(str(will.get("causal_root_hash", "")))
            else ""
        ),
        "causal_root_version": int(will.get("causal_root_version", 0)),
        "root_count": root_count,
        "covered_node_count": int(will.get("covered_node_count", 0)),
        "pending_reduction_count": pending_reduction_count,
        "reduction_failure_count": reduction_failure_count,
        # This is the query-independent lossy carrier computed from the whole
        # eligible line. It is quoted data, not an imperative or identity claim.
        "carrier_text": raw_carrier if carrier_available else "",
    }
    safe_freshness = _safe_model_freshness(continuity_freshness)
    payload = {
        "technical_projection": projection,
        "continuity_freshness": safe_freshness,
        "cognitive_posture": safe_posture,
        "pending_consequences": safe_pending,
        "reconstructed_subjective_traces": safe_traces,
    }

    def _render(value: dict[str, Any]) -> str:
        return opening + json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + closing

    prompt = _render(payload)
    if len(prompt) <= MAX_SYSTEM_PROMPT_ADDITION:
        return prompt

    # Preserve the complete carrier and every actually leased consequence.
    # Query-dependent recall is the first optional surface removed.
    payload["reconstructed_subjective_traces"] = []
    prompt = _render(payload)
    if len(prompt) <= MAX_SYSTEM_PROMPT_ADDITION:
        return prompt

    payload["cognitive_posture"] = {}
    payload["pending_consequences"] = [
        {**item, "content": item["content"][:256]} for item in safe_pending
    ]
    payload["details_omitted"] = True
    prompt = _render(payload)
    if len(prompt) <= MAX_SYSTEM_PROMPT_ADDITION:
        return prompt

    compact = {
        "technical_projection": {
            "formed": projection["formed"],
            "projection_status": projection["projection_status"],
            "projection_available": projection["projection_available"],
            "root_count": projection["root_count"],
            "carrier_text": projection["carrier_text"],
        },
        "continuity_freshness": safe_freshness,
        "cognitive_posture": {},
        "pending_consequences": [
            {**item, "content": item["content"][:128]} for item in safe_pending
        ],
        "reconstructed_subjective_traces": [],
        "details_omitted": True,
    }
    prompt = _render(compact)
    if len(prompt) <= MAX_SYSTEM_PROMPT_ADDITION:
        return prompt

    # The only remaining way to respect the hard envelope bound is to withhold
    # the whole carrier.  Never substitute a prefix: it would silently erase
    # some roots while claiming the projection remains available.
    compact["technical_projection"] = {
        "formed": False,
        "projection_status": "degraded",
        "projection_available": False,
        "root_count": projection["root_count"],
        "carrier_text": "",
        "carrier_unavailable_reason": "complete_carrier_exceeds_prompt_budget",
    }
    prompt = _render(compact)
    assert len(prompt) <= MAX_SYSTEM_PROMPT_ADDITION
    return prompt
