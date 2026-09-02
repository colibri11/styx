"""Durable, agent-scoped reduction of completed cognitive acts (wave 38).

The module is deliberately storage-only.  It does not decide what an act
"means" and does not claim that a technical residue or carrier is a self,
personality, will, or consciousness.  It provides a bounded evidence envelope,
an idempotent outbox, and the atomic validation boundary through which a
reducer result may become an eligible subjective trace.

No raw dialogue is persisted in ``llm_tasks.payload``.  Queue tasks carry only
agent/act/version coordinates and a digest of the already-redacted journal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from styx.storage.cognition import (
    lock_agent_line,
    redact_journal_json,
    redact_journal_metadata,
    redact_journal_text,
)


ACT_RESIDUE_TASK_TYPE = "act_residue_reduction"
DEFAULT_REDUCER_VERSION = "act_residue_v1"
MAX_ACTIONS = 64
MAX_OBSERVATIONS = 16
MAX_RESIDUES = 4
MAX_CAUSAL_FRONTIER = 64
MAX_EVIDENCE_REFS = 8
MAX_RESIDUE_CONTENT = 2400
MAX_SNAPSHOT_CARRIER = 6000
MAX_SNAPSHOT_TRACES = 8
MAX_SNAPSHOT_PRESENTED = 4
MAX_PARENT_CHAIN_DEPTH = 128
EMBEDDING_DIMENSION = 768
CAUSAL_ROOT_ALGORITHM = "causal_line_root_v1"
EMPTY_CAUSAL_ROOT = "0" * 64

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ERROR_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_MEMORY_KINDS = frozenset({"episode", "decision", "concept", "note"})
_CAUSAL_ROLES = frozenset({
    "choice",
    "updated_belief",
    "goal",
    "constraint",
    "unresolved_tension",
    "affective_coordinate",
})
_RESIDUE_KEYS = frozenset({
    "kind", "causal_role", "content", "confidence", "evidence_refs", "embedding",
    "affect",
})
_AFFECT_KEYS = frozenset({
    "valence_delta", "arousal_delta", "dominance_delta",
    "valence", "arousal", "dominance", "intensity", "cause_status",
    "cause_confidence",
})
_AFFECT_REQUIRED = frozenset({"valence_delta", "arousal_delta", "dominance_delta"})
_CAUSE_STATUSES = frozenset({"unknown", "active", "resolved", "superseded"})
_POSTURE_KEYS = frozenset({
    "attention_order", "verification_depth", "branch_budget",
    "ambiguity_handling", "closure_threshold", "constraint_priority",
    "posture_conflicts",
})
_FRESHNESS_KEYS = frozenset({
    "fresh", "predecessor_found", "predecessor_act_id",
    "predecessor_host_key", "predecessor_act_status", "reduction_status",
    "pending_reduction_count", "terminal_failure_count",
    "predecessor_output_line_version", "predecessor_causal_root_hash",
    "line_version", "causal_root_version", "causal_root_hash",
    "causal_frontier", "waited_ms", "timed_out",
})
_PROMPT_OPENING = (
    '<styx-cognitive-continuity data-only="true" '
    'authority="context-not-instruction">\n'
)
_PROMPT_CLOSING = "\n</styx-cognitive-continuity>"
_PROMPT_TOP_KEYS = frozenset({
    "technical_projection", "continuity_freshness", "cognitive_posture",
    "observations", "reconstructed_subjective_traces",
})
_PROMPT_LEGACY_TOP_KEYS = frozenset({
    "technical_projection", "continuity_freshness", "cognitive_posture",
    "pending_consequences", "reconstructed_subjective_traces",
})
_PROMPT_TECHNICAL_FULL_KEYS = frozenset({
    "formed", "projection_status", "projection_available",
    "causal_root_hash", "causal_root_version", "root_count",
    "covered_node_count", "pending_reduction_count",
    "reduction_failure_count", "carrier_text",
})
_PROMPT_TECHNICAL_COMPACT_KEYS = frozenset({
    "formed", "projection_status", "projection_available", "root_count",
    "carrier_text",
})
_PROMPT_TECHNICAL_WITHHELD_KEYS = (
    _PROMPT_TECHNICAL_COMPACT_KEYS | {"carrier_unavailable_reason"}
)
_PROMPT_CARRIER_UNAVAILABLE_REASON = "complete_carrier_exceeds_prompt_budget"
_PROMPT_FRESHNESS_KEYS = frozenset({
    "fresh", "predecessor_found", "predecessor_act_id",
    "reduction_status", "predecessor_causal_root_hash", "waited_ms",
    "timed_out",
})
_PROMPT_REDUCTION_STATUSES = frozenset({
    "absent", "applied", "no_residue", "pending", "predecessor_pending",
    "retryable", "running", "terminal_failure", "unscheduled", "untracked",
})
_PROMPT_OBSERVATION_KEYS = frozenset({
    "observation_id", "observation_status", "source_id", "source_stream",
    "source_sequence", "observation_key", "difference_kind", "content",
    "salience", "confidence", "reducer_name", "reducer_version",
    "correlation_status", "action_ordinal", "action_event_id",
    "source_observed_at", "ingested_at", "late",
})
_PROMPT_LEGACY_CONSEQUENCE_KEYS = frozenset({
    "consequence_id", "source_act_id", "ordinal", "kind", "content",
})
_PROMPT_TRACE_KEYS = frozenset({
    "memory_id", "role", "kind", "content", "score",
})


class ActReductionError(ValueError):
    """Base class for invalid or conflicting reduction operations."""


class ActReductionConflict(ActReductionError):
    """The requested transition conflicts with an existing durable outcome."""


class ActReductionValidationError(ActReductionError):
    """Reducer output or evidence coordinates are outside the allowlist."""


class ActReductionDependencyPending(ActReductionError):
    """A declared causal parent has no usable terminal reduction yet.

    Worker handlers must classify this exception as transient/retryable and
    catch it before the broader :class:`ActReductionError`.  Its fields are
    controlled storage coordinates only; no host content is exposed.
    """

    def __init__(
        self,
        *,
        parent_act_id: uuid.UUID | None,
        reduction_status: str,
    ) -> None:
        self.parent_act_id = parent_act_id
        self.reduction_status = reduction_status
        super().__init__(
            "declared parent reduction is not applied or no_residue: "
            f"status={reduction_status}"
        )


@dataclass(frozen=True)
class ReductionSchedule:
    reduction_id: uuid.UUID
    task_id: uuid.UUID | None
    status: str
    input_hash: str
    outcome_version: int
    duplicate: bool


@dataclass(frozen=True)
class ReductionApply:
    reduction_id: uuid.UUID
    status: str
    memory_ids: tuple[uuid.UUID, ...]
    result_hash: str
    duplicate: bool
    line_version: int
    causal_root_hash: str
    predecessor_frontier: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class CausalLineState:
    """Technical coordinate of the currently retained residue DAG."""

    line_version: int
    causal_root_hash: str
    causal_root_version: int
    causal_root_act_id: uuid.UUID | None
    frontier: tuple[uuid.UUID, ...]


def _as_uuid(value: uuid.UUID | str, field: str) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ActReductionValidationError(f"{field} must be a UUID") from exc


def _validate_agent_id(agent_id: str) -> str:
    if not isinstance(agent_id, str) or not agent_id or len(agent_id) > 256:
        raise ActReductionValidationError("agent_id must contain 1..256 characters")
    return agent_id


def _validate_reducer_version(value: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ActReductionValidationError(
            "reducer_version must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}"
        )
    return value


def _validate_hash(value: str, field: str = "input_hash") -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ActReductionValidationError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _safe_json_object(value: Any) -> dict[str, Any]:
    redacted = redact_journal_json(value)
    return redacted if isinstance(redacted, dict) else {}


def reduction_input_hash(value: Mapping[str, Any]) -> str:
    """Return the canonical digest used by both scheduler and reducer handler."""
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActReductionValidationError("reduction input must be canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def causal_line_root_hash(
    *,
    previous_root_hash: str,
    previous_root_version: int,
    output_line_version: int,
    act_id: uuid.UUID | str,
    reducer_version: str,
    input_hash: str,
    predecessor_frontier: Sequence[uuid.UUID | str],
    residues: Sequence[Mapping[str, Any]],
) -> str:
    """Build the deterministic root for one atomically incorporated batch.

    The digest commits to the previous retained root, the exact predecessor
    frontier and every ordered residue.  It proves storage coverage of these
    coordinates; it does not prove preservation of every semantic nuance.
    """
    previous_root_hash = _validate_hash(previous_root_hash, "previous_root_hash")
    input_hash = _validate_hash(input_hash)
    act_uuid = _as_uuid(act_id, "act_id")
    reducer_version = _validate_reducer_version(reducer_version)
    if previous_root_version < 0 or output_line_version <= previous_root_version:
        raise ActReductionValidationError("causal root versions must advance")
    frontier = [str(_as_uuid(item, "predecessor_frontier")) for item in predecessor_frontier]
    if len(frontier) > MAX_CAUSAL_FRONTIER or len(set(frontier)) != len(frontier):
        raise ActReductionValidationError("predecessor frontier must contain 0..4 unique ids")
    nodes: list[dict[str, Any]] = []
    for ordinal, residue in enumerate(residues):
        memory_id = str(_as_uuid(residue.get("memory_id"), "memory_id"))
        content = residue.get("content")
        if not isinstance(content, str):
            raise ActReductionValidationError("root residue content must be a string")
        nodes.append({
            "memory_id": memory_id,
            "ordinal": ordinal,
            "kind": residue.get("kind"),
            "causal_role": residue.get("causal_role"),
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "confidence": residue.get("confidence"),
            "evidence_hash": reduction_input_hash({
                "evidence_refs": residue.get("evidence_refs", [])
            }),
            "affect_hash": (
                reduction_input_hash({"affect": residue.get("affect")})
                if residue.get("affect") is not None else None
            ),
        })
    if not nodes or len(nodes) > MAX_RESIDUES:
        raise ActReductionValidationError("causal root requires 1..4 residue nodes")
    document = {
        "algorithm": CAUSAL_ROOT_ALGORITHM,
        "previous_root_hash": previous_root_hash,
        "previous_root_version": previous_root_version,
        "output_line_version": output_line_version,
        "act_id": str(act_uuid),
        "reducer_version": reducer_version,
        "input_hash": input_hash,
        "predecessor_frontier": frontier,
        "residues": nodes,
    }
    return reduction_input_hash(document)


def load_causal_line_state(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    for_update: bool = False,
) -> CausalLineState:
    """Read the current technical root/frontier, returning an empty coordinate.

    The function never creates a row.  This is important for ``no_residue``:
    observing an empty line must not manufacture a line-state transition.
    """
    agent_id = _validate_agent_id(agent_id)
    suffix = " FOR UPDATE" if for_update else ""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT version,causal_root_hash,causal_frontier,causal_root_version,"
            " causal_root_act_id FROM line_state WHERE agent_id=%s" + suffix,
            (agent_id,),
        )
        row = cur.fetchone()
    if row is None:
        return CausalLineState(0, EMPTY_CAUSAL_ROOT, 0, None, ())
    raw_frontier = row["causal_frontier"] or []
    if not isinstance(raw_frontier, list):
        raise ActReductionConflict("line_state causal_frontier is malformed")
    frontier = tuple(_as_uuid(item, "causal_frontier") for item in raw_frontier)
    if len(frontier) > MAX_RESIDUES or len(set(frontier)) != len(frontier):
        raise ActReductionConflict("line_state causal_frontier is not a bounded set")
    root_hash = _validate_hash(str(row["causal_root_hash"]), "causal_root_hash")
    state = CausalLineState(
        line_version=int(row["version"]),
        causal_root_hash=root_hash,
        causal_root_version=int(row["causal_root_version"]),
        causal_root_act_id=row["causal_root_act_id"],
        frontier=frontier,
    )
    if state.causal_root_version > state.line_version:
        raise ActReductionConflict("causal root version is ahead of line version")
    if state.frontier:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM memories WHERE agent_id=%s AND id=ANY(%s) "
                "AND memory_domain='subjective_trace' AND line_eligible=true "
                "AND line_provenance IN "
                "('validated_act_residue','validated_transform') "
                "AND line_status='active'",
                (agent_id, list(state.frontier)),
            )
            live = {item[0] for item in cur.fetchall()}
        if live != set(state.frontier):
            raise ActReductionConflict("causal frontier contains a non-live or foreign residue")
    return state


def read_predecessor_freshness(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    parent_host_key: str | None = None,
    session_id: uuid.UUID | str | None = None,
    reducer_version: str = DEFAULT_REDUCER_VERSION,
) -> dict[str, Any]:
    """Read the predecessor reduction/root fence without waiting or writing.

    Resolution uses an explicit parent host key when supplied, otherwise the
    latest terminal act in the supplied session.  The provider can bounded-poll
    this helper; storage deliberately performs no sleep and owns no deadline.
    """
    agent_id = _validate_agent_id(agent_id)
    reducer_version = _validate_reducer_version(reducer_version)
    parent_key = parent_host_key.strip() if isinstance(parent_host_key, str) else None
    if parent_key is not None and not 1 <= len(parent_key) <= 512:
        raise ActReductionValidationError("parent_host_key must contain 1..512 characters")
    session_uuid = _as_uuid(session_id, "session_id") if session_id is not None else None
    if parent_key is None and session_uuid is None:
        raise ActReductionValidationError(
            "parent_host_key or session_id is required for predecessor resolution"
        )

    with conn.cursor(row_factory=dict_row) as cur:
        if parent_key is not None:
            cur.execute(
                "SELECT id,host_key,status,session_id FROM cognitive_acts "
                "WHERE agent_id=%s AND host_key=%s",
                (agent_id, parent_key),
            )
        else:
            cur.execute(
                "SELECT id,host_key,status,session_id FROM cognitive_acts "
                "WHERE agent_id=%s AND session_id=%s "
                "AND status IN ('completed','failed') "
                "ORDER BY completed_at DESC NULLS LAST,created_at DESC,id DESC LIMIT 1",
                (agent_id, session_uuid),
            )
        predecessor = cur.fetchone()

        cur.execute(
            "SELECT status,count(*)::int AS count FROM cognitive_act_reductions "
            "WHERE agent_id=%s GROUP BY status",
            (agent_id,),
        )
        reduction_counts = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}

    line = load_causal_line_state(conn, agent_id)
    missing_explicit_parent = predecessor is None and parent_key is not None
    base = {
        "fresh": predecessor is None and not missing_explicit_parent,
        "predecessor_found": predecessor is not None,
        "predecessor_act_id": str(predecessor["id"]) if predecessor else None,
        "predecessor_host_key": str(predecessor["host_key"]) if predecessor else None,
        "predecessor_act_status": str(predecessor["status"]) if predecessor else None,
        "reduction_status": (
            "predecessor_pending" if missing_explicit_parent
            else "absent" if predecessor is None else "unscheduled"
        ),
        "reduction_task_counts": {
            "pending": 0, "running": 0, "done": 0, "failed": 0,
        },
        "pending_reduction_count": sum(
            reduction_counts.get(status, 0)
            for status in ("pending", "running", "retryable")
        ),
        "terminal_failure_count": reduction_counts.get("terminal_failure", 0),
        "predecessor_output_line_version": None,
        "predecessor_causal_root_hash": None,
        "line_version": line.line_version,
        "causal_root_version": line.causal_root_version,
        "causal_root_hash": line.causal_root_hash,
        "causal_frontier": [str(item) for item in line.frontier],
    }
    if predecessor is None:
        return base

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT status,output_line_version,causal_root_hash "
            "FROM cognitive_act_reductions WHERE agent_id=%s AND act_id=%s "
            "AND reducer_version=%s",
            (agent_id, predecessor["id"], reducer_version),
        )
        reduction = cur.fetchone()
        cur.execute(
            "SELECT status,count(*)::int AS count FROM llm_tasks "
            "WHERE task_type=%s AND payload->>'agent_id'=%s "
            "AND payload->>'act_id'=%s AND payload->>'reducer_version'=%s "
            "GROUP BY status",
            (
                ACT_RESIDUE_TASK_TYPE, agent_id, str(predecessor["id"]),
                reducer_version,
            ),
        )
        task_counts = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
    if reduction is not None:
        status = str(reduction["status"])
        base["reduction_status"] = status
        base["fresh"] = status in {"applied", "no_residue"}
        base["predecessor_output_line_version"] = reduction["output_line_version"]
        base["predecessor_causal_root_hash"] = reduction["causal_root_hash"]
    base["reduction_task_counts"] = {
        key: task_counts.get(key, 0) for key in ("pending", "running", "done", "failed")
    }
    return base


def _snapshot_hash(value: Any) -> str | None:
    candidate = str(value or "")
    return candidate if _HASH_RE.fullmatch(candidate) is not None else None


def _snapshot_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _snapshot_uuid_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:limit]:
        try:
            parsed = str(_as_uuid(item, "snapshot coordinate"))
        except ActReductionValidationError:
            continue
        if parsed not in result:
            result.append(parsed)
    return result


def _model_visible_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and project the exact continuity envelope shown to the model.

    The parallel public response fields are diagnostic and may contain traces
    or posture which the prompt renderer dropped to satisfy its hard limit.
    Consequently they are never used as reducer evidence.  The rendered field
    is accepted only in Styx's fixed wrapper and strict JSON schema; prose
    outside that envelope, unknown fields, and malformed coordinates fail
    closed instead of becoming trusted host input.
    """
    rendered = payload.get("system_prompt_addition")
    if (
        not isinstance(rendered, str)
        or len(rendered) > 16_000
        or not rendered.startswith(_PROMPT_OPENING)
        or not rendered.endswith(_PROMPT_CLOSING)
    ):
        raise ActReductionConflict(
            "frozen system_prompt_addition has no valid Styx wrapper"
        )
    encoded = rendered[len(_PROMPT_OPENING):-len(_PROMPT_CLOSING)]
    try:
        raw = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ActReductionConflict(
            "frozen system_prompt_addition is not valid JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise ActReductionConflict("frozen continuity envelope must be an object")
    details_omitted = "details_omitted" in raw
    raw_keys = set(raw)
    current_top = (
        _PROMPT_TOP_KEYS | {"details_omitted"}
        if details_omitted else _PROMPT_TOP_KEYS
    )
    legacy_top = (
        _PROMPT_LEGACY_TOP_KEYS | {"details_omitted"}
        if details_omitted else _PROMPT_LEGACY_TOP_KEYS
    )
    if raw_keys == current_top:
        prompt_shape = "observations"
    elif raw_keys == legacy_top:
        # One-cycle mixed-version compatibility: acts can finish after a
        # daemon upgrade with a pre-upgrade, already-frozen preturn envelope.
        prompt_shape = "legacy_consequences"
    else:
        raise ActReductionConflict("frozen continuity fields do not match allowlist")
    if details_omitted and raw["details_omitted"] is not True:
        raise ActReductionConflict("details_omitted must be true when present")

    technical = raw.get("technical_projection")
    freshness = raw.get("continuity_freshness")
    posture = raw.get("cognitive_posture")
    observations = raw.get(
        "observations" if prompt_shape == "observations" else "pending_consequences"
    )
    traces = raw.get("reconstructed_subjective_traces")
    if not isinstance(technical, dict):
        raise ActReductionConflict("technical_projection fields do not match allowlist")
    technical_keys = set(technical)
    if technical_keys == _PROMPT_TECHNICAL_FULL_KEYS:
        technical_shape = "full"
    elif (
        details_omitted
        and technical_keys == _PROMPT_TECHNICAL_COMPACT_KEYS
    ):
        technical_shape = "compact"
    elif (
        details_omitted
        and technical_keys == _PROMPT_TECHNICAL_WITHHELD_KEYS
    ):
        technical_shape = "withheld"
    else:
        raise ActReductionConflict(
            "technical_projection fields do not match a rendered shape"
        )
    if not isinstance(freshness, dict) or set(freshness) - _PROMPT_FRESHNESS_KEYS:
        raise ActReductionConflict("continuity_freshness fields do not match allowlist")
    if not isinstance(posture, dict) or set(posture) - _POSTURE_KEYS:
        raise ActReductionConflict("cognitive_posture fields do not match allowlist")
    if not isinstance(observations, list) or len(observations) > MAX_SNAPSHOT_PRESENTED:
        raise ActReductionConflict("observations is not bounded")
    if not isinstance(traces, list) or len(traces) > MAX_SNAPSHOT_TRACES:
        raise ActReductionConflict("reconstructed traces are not bounded")
    if type(technical.get("formed")) is not bool or type(
        technical.get("projection_available")
    ) is not bool:
        raise ActReductionConflict("technical projection booleans are invalid")

    def _required_nonnegative_int(value: Any, field: str) -> int:
        parsed = _snapshot_int(value)
        if parsed is None:
            raise ActReductionConflict(f"{field} must be a non-negative integer")
        return parsed

    projection_status = technical.get("projection_status")
    carrier_text = technical.get("carrier_text")
    if projection_status not in {
        "empty", "provisional", "ready", "stale", "degraded",
    }:
        raise ActReductionConflict("projection_status is invalid")
    if not isinstance(carrier_text, str) or len(carrier_text) > MAX_SNAPSHOT_CARRIER:
        raise ActReductionConflict("carrier_text is invalid")
    projection_available = technical["projection_available"]
    if projection_available != bool(carrier_text):
        raise ActReductionConflict(
            "projection_available must match complete carrier visibility"
        )

    root_count = _required_nonnegative_int(
        technical.get("root_count"), "root_count"
    )
    if technical_shape == "full":
        root_hash_raw = technical.get("causal_root_hash")
        root_hash = _snapshot_hash(root_hash_raw)
        if root_hash is None and root_hash_raw not in {"", None}:
            raise ActReductionConflict("causal_root_hash is invalid")
        causal_root_version = _required_nonnegative_int(
            technical.get("causal_root_version"), "causal_root_version"
        )
        covered_node_count = _required_nonnegative_int(
            technical.get("covered_node_count"), "covered_node_count"
        )
        pending_reduction_count = _required_nonnegative_int(
            technical.get("pending_reduction_count"),
            "pending_reduction_count",
        )
        reduction_failure_count = _required_nonnegative_int(
            technical.get("reduction_failure_count"),
            "reduction_failure_count",
        )
    else:
        # The compact renderer intentionally withholds these coordinates.  Do
        # not invent zeroes: null is the exact evidence placeholder for a
        # field which was not model-visible.
        root_hash = None
        causal_root_version = None
        covered_node_count = None
        pending_reduction_count = None
        reduction_failure_count = None

    if technical_shape == "withheld" and (
        technical.get("formed") is not False
        or projection_status != "degraded"
        or projection_available is not False
        or carrier_text != ""
        or technical.get("carrier_unavailable_reason")
        != _PROMPT_CARRIER_UNAVAILABLE_REASON
    ):
        raise ActReductionConflict("withheld carrier projection is invalid")

    for key in ("fresh", "predecessor_found", "timed_out"):
        if key in freshness and type(freshness[key]) is not bool:
            raise ActReductionConflict(f"continuity_freshness.{key} is invalid")
    if "waited_ms" in freshness and (
        type(freshness["waited_ms"]) is not int
        or not 0 <= freshness["waited_ms"] <= 60_000
    ):
        raise ActReductionConflict("continuity_freshness.waited_ms is invalid")
    if "predecessor_act_id" in freshness and not _snapshot_uuid_list(
        [freshness["predecessor_act_id"]], limit=1
    ):
        raise ActReductionConflict(
            "continuity_freshness.predecessor_act_id is invalid"
        )
    if "predecessor_causal_root_hash" in freshness and _snapshot_hash(
        freshness["predecessor_causal_root_hash"]
    ) is None:
        raise ActReductionConflict(
            "continuity_freshness.predecessor_causal_root_hash is invalid"
        )
    if (
        "reduction_status" in freshness
        and freshness["reduction_status"] not in _PROMPT_REDUCTION_STATUSES
    ):
        raise ActReductionConflict(
            "continuity_freshness.reduction_status is invalid"
        )

    if details_omitted and (posture or traces):
        raise ActReductionConflict(
            "details_omitted requires empty posture and reconstructed traces"
        )

    safe_freshness = redact_journal_json(freshness)
    safe_posture = redact_journal_json(posture)
    if not isinstance(safe_freshness, dict) or not isinstance(safe_posture, dict):
        raise ActReductionConflict("continuity JSON projection is invalid")

    presented_ids: list[str] = []
    for item in observations:
        if prompt_shape == "legacy_consequences":
            if (
                not isinstance(item, dict)
                or set(item) != _PROMPT_LEGACY_CONSEQUENCE_KEYS
            ):
                raise ActReductionConflict(
                    "legacy consequence fields do not match allowlist"
                )
            ids = _snapshot_uuid_list([item.get("consequence_id")], limit=1)
            source_ids = _snapshot_uuid_list([item.get("source_act_id")], limit=1)
            ordinal = item.get("ordinal")
            kind = item.get("kind")
            content = item.get("content")
            if (
                not ids
                or not source_ids
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or not isinstance(kind, str)
                or not 1 <= len(kind) <= 64
                or not isinstance(content, str)
                or len(content) > 512
            ):
                raise ActReductionConflict("legacy consequence coordinate is invalid")
            if ids[0] in presented_ids:
                raise ActReductionConflict("legacy consequence ids must be unique")
            presented_ids.append(ids[0])
            continue
        if not isinstance(item, dict) or set(item) != _PROMPT_OBSERVATION_KEYS:
            raise ActReductionConflict("observation fields do not match allowlist")
        ids = _snapshot_uuid_list([item.get("observation_id")], limit=1)
        status = item.get("observation_status")
        difference_kind = item.get("difference_kind")
        content = item.get("content")
        ingested_at = item.get("ingested_at")
        if (
            not ids
            or status not in {"canonical", "legacy"}
            or not isinstance(difference_kind, str)
            or not 1 <= len(difference_kind) <= 64
            or not isinstance(content, str)
            or len(content) > 512
            or not isinstance(ingested_at, str)
            or not 1 <= len(ingested_at) <= 64
            or type(item.get("late")) is not bool
        ):
            raise ActReductionConflict("observation coordinate is invalid")
        if status == "canonical":
            for key, maximum in (
                ("source_id", 256),
                ("source_stream", 256),
                ("observation_key", 256),
                ("reducer_name", 128),
                ("reducer_version", 64),
            ):
                value = item.get(key)
                if not isinstance(value, str) or not 1 <= len(value) <= maximum:
                    raise ActReductionConflict(f"observation.{key} is invalid")
            sequence = item.get("source_sequence")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                raise ActReductionConflict("observation.source_sequence is invalid")
            for key in ("salience", "confidence"):
                value = item.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ActReductionConflict(f"observation.{key} is invalid")
            if item.get("correlation_status") not in {
                "uncorrelated", "pending", "resolved", "conflict",
            }:
                raise ActReductionConflict(
                    "observation.correlation_status is invalid"
                )
        else:
            nullable = {
                "source_id", "source_stream", "source_sequence",
                "observation_key", "salience", "confidence", "reducer_name",
                "reducer_version", "action_ordinal", "action_event_id",
                "source_observed_at",
            }
            if any(item.get(key) is not None for key in nullable) or (
                item.get("correlation_status") != "legacy"
            ):
                raise ActReductionConflict("legacy observation projection is invalid")
        action_ordinal = item.get("action_ordinal")
        if action_ordinal is not None and (
            isinstance(action_ordinal, bool)
            or not isinstance(action_ordinal, int)
            or action_ordinal < 0
        ):
            raise ActReductionConflict("observation.action_ordinal is invalid")
        for key in ("action_event_id", "source_observed_at"):
            value = item.get(key)
            if value is not None and (
                not isinstance(value, str)
                or not 1 <= len(value) <= (
                    256 if key == "action_event_id" else 64
                )
            ):
                raise ActReductionConflict(f"observation.{key} is invalid")
        if ids[0] in presented_ids:
            raise ActReductionConflict("observation ids must be unique")
        presented_ids.append(ids[0])

    visible_traces: list[dict[str, Any]] = []
    for item in traces:
        if not isinstance(item, dict) or set(item) != _PROMPT_TRACE_KEYS:
            raise ActReductionConflict("trace fields do not match allowlist")
        memory_ids = _snapshot_uuid_list([item.get("memory_id")], limit=1)
        score = item.get("score")
        if (
            not memory_ids
            or not isinstance(item.get("role"), str)
            or len(item["role"]) > 32
            or not isinstance(item.get("kind"), str)
            or len(item["kind"]) > 32
            or not isinstance(item.get("content"), str)
            or len(item["content"]) > 600
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ActReductionConflict("trace coordinate is invalid")
        if any(trace["memory_id"] == memory_ids[0] for trace in visible_traces):
            raise ActReductionConflict("trace memory ids must be unique")
        visible_traces.append({
            "memory_id": memory_ids[0],
            "role": redact_journal_text(item["role"], limit=32),
            "kind": redact_journal_text(item["kind"], limit=32),
            "content": redact_journal_text(item["content"], limit=600),
            "score": round(float(score), 6),
        })

    return {
        # Keep the existing worker schema, but populate it exclusively from
        # fields present in the parsed model-visible envelope.  Non-visible
        # diagnostic coordinates are explicit null/empty placeholders.
        "carrier": {
            "text": redact_journal_text(carrier_text, limit=MAX_SNAPSHOT_CARRIER),
            "version": "",
            "projection_status": projection_status,
            "projection_available": projection_available,
            "line_version": None,
            "covered_line_version": None,
            "causal_root_hash": root_hash,
            "causal_root_version": causal_root_version,
            "causal_frontier": [],
            "root_coverage_hash": None,
            "root_count": root_count,
            "covered_node_count": covered_node_count,
            "pending_reduction_count": pending_reduction_count,
            "reduction_failure_count": reduction_failure_count,
        },
        "cognitive_posture": safe_posture,
        "continuity_freshness": safe_freshness,
        "presented_observation_ids": presented_ids,
        "trace_coordinates": visible_traces,
    }


def _project_input_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the exact frozen preturn response into bounded reducer input.

    Raw messages are deliberately absent.  The evidence is derived only from
    the strictly wrapped JSON which actually entered the model prompt; public
    response fields cannot re-introduce traces or posture dropped by renderer.
    """
    # The snapshot token identifies the frozen ledger row but was never shown
    # to the model.  It remains in the top-level storage/hash envelope and must
    # not be synthesized into the model-visible evidence projection.
    return _model_visible_snapshot(payload)


def load_act_reduction_input(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
) -> dict[str, Any] | None:
    """Load the bounded/redacted evidence envelope for exactly one agent's act.

    ``presented_observations`` come from the immutable payload of the exact
    presentation consumed by this act.  Source-row changes and same-act tool
    events therefore cannot be replayed as future-world observations.
    """
    agent_id = _validate_agent_id(agent_id)
    act_uuid = _as_uuid(act_id, "act_id")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,host_key,status,session_id,declared_parent_key,parent_act_id,"
            "       input_line_version,input_snapshot_token,channel_input,channel_output "
            "FROM cognitive_acts WHERE agent_id=%s AND id=%s",
            (agent_id, act_uuid),
        )
        act = cur.fetchone()
        if act is None:
            return None

        cur.execute(
            "SELECT ordinal,kind,event_id,name,content,metadata,count(*) OVER() AS total_count "
            "FROM cognitive_actions WHERE agent_id=%s AND act_id=%s "
            "ORDER BY ordinal LIMIT %s",
            (agent_id, act_uuid, MAX_ACTIONS + 1),
        )
        raw_actions = list(cur.fetchall())

        cur.execute(
            "SELECT c.id AS observation_id,c.act_id AS source_act_id,c.ordinal,"
            " c.kind,c.content,c.metadata,c.created_at,p.snapshot_token,"
            " p.presented_payload,p.payload_hash,p.presentation_version,"
            " count(*) OVER() AS total_count "
            "FROM cognitive_presentations p "
            "JOIN cognitive_consequences c "
            " ON (c.id,c.agent_id)=(p.consequence_id,p.agent_id) "
            "WHERE p.snapshot_token=%s AND p.agent_id=%s "
            "AND c.acknowledged_by_act_id=%s "
            "ORDER BY p.presented_at,p.consequence_id LIMIT %s",
            (
                act["input_snapshot_token"], agent_id, act_uuid,
                MAX_OBSERVATIONS + 1,
            ),
        )
        raw_observations = list(cur.fetchall())

        snapshot_projection = None
        if act["input_snapshot_token"] is not None:
            cur.execute(
                "SELECT response_payload FROM cognitive_snapshots "
                "WHERE token=%s AND agent_id=%s AND used_by_act_id=%s",
                (act["input_snapshot_token"], agent_id, act_uuid),
            )
            snapshot = cur.fetchone()
            if snapshot is None or not isinstance(snapshot["response_payload"], dict):
                raise ActReductionConflict(
                    "claimed cognitive snapshot has no frozen response envelope"
                )
            snapshot_projection = _project_input_snapshot(
                snapshot["response_payload"]
            )

    actions = [
        {
            "ordinal": int(row["ordinal"]),
            "kind": str(row["kind"])[:32],
            "event_id": str(row["event_id"] or "")[:256],
            "name": str(row["name"] or "")[:256],
            "content": redact_journal_text(row["content"], limit=8000),
            "metadata": redact_journal_metadata(row["metadata"] or {}),
        }
        for row in raw_actions[:MAX_ACTIONS]
    ]
    observations: list[dict[str, Any]] = []
    for row in raw_observations[:MAX_OBSERVATIONS]:
        payload = row["presented_payload"]
        if isinstance(payload, dict):
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if (
                row["presentation_version"] != "observation_presentation_v1"
                or hashlib.sha256(encoded).hexdigest() != row["payload_hash"]
            ):
                raise ActReductionConflict(
                    "consumed observation presentation hash mismatch"
                )
            observations.append(dict(payload))
            continue
        # Mixed-version row: preserve old evidence without inventing reducer
        # provenance.  New snapshots always freeze the canonical shape.
        observations.append({
            "observation_id": str(row["observation_id"]),
            "observation_status": "legacy",
            "source_id": None,
            "source_stream": None,
            "source_sequence": None,
            "observation_key": None,
            "difference_kind": str(row["kind"])[:64],
            "content": redact_journal_text(row["content"], limit=512),
            "salience": None,
            "confidence": None,
            "reducer_name": None,
            "reducer_version": None,
            "correlation_status": "legacy",
            "action_ordinal": None,
            "action_event_id": None,
            "source_observed_at": None,
            "ingested_at": row["created_at"].isoformat(),
            "late": False,
        })
    return {
        "agent_id": agent_id,
        "act_id": str(act["id"]),
        "host_key": str(act["host_key"])[:512],
        "status": str(act["status"]),
        "session_id": str(act["session_id"]) if act["session_id"] else None,
        "declared_parent_key": (
            str(act["declared_parent_key"])[:512]
            if act["declared_parent_key"] is not None else None
        ),
        # ``parent_act_id`` is a derived late-resolution cache.  The immutable
        # declared key is hashed; apply resolves the current parent under the
        # line lock so a parent arriving after schedule cannot invalidate the
        # reducer input hash.
        "input_line_version": int(act["input_line_version"]),
        "input_snapshot_token": (
            str(act["input_snapshot_token"])[:128]
            if act["input_snapshot_token"] is not None else None
        ),
        "channel_input": _safe_json_object(act["channel_input"] or {}),
        "channel_output": _safe_json_object(act["channel_output"] or {}),
        "input_snapshot": snapshot_projection,
        "action_count": (
            int(raw_actions[0]["total_count"]) if raw_actions else 0
        ),
        "actions_truncated": len(raw_actions) > MAX_ACTIONS,
        "actions": actions,
        "presented_observation_count": (
            int(raw_observations[0]["total_count"]) if raw_observations else 0
        ),
        "presented_observations_truncated": len(raw_observations) > MAX_OBSERVATIONS,
        "presented_observations": observations,
    }


def _row_to_schedule(row: Mapping[str, Any], *, duplicate: bool) -> ReductionSchedule:
    return ReductionSchedule(
        reduction_id=row["id"],
        task_id=row.get("task_id"),
        status=str(row["status"]),
        input_hash=str(row["input_hash"]),
        outcome_version=int(row["outcome_version"]),
        duplicate=duplicate,
    )


def _insert_task(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    act_id: uuid.UUID,
    reducer_version: str,
    input_hash: str,
    attempt_no: int,
) -> uuid.UUID:
    payload = {
        "agent_id": agent_id,
        "act_id": str(act_id),
        "reducer_version": reducer_version,
        "input_hash": input_hash,
        "attempt_no": attempt_no,
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO llm_tasks(task_type,payload,status,task_version) "
            "VALUES (%s,%s,'pending',1) RETURNING id",
            (ACT_RESIDUE_TASK_TYPE, Jsonb(payload)),
        )
        row = cur.fetchone()
    assert row is not None
    return row[0]


def schedule_act_reduction(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
    *,
    reducer_version: str = DEFAULT_REDUCER_VERSION,
    retry: bool = False,
) -> ReductionSchedule:
    """Create one durable outcome and coordinate-only queue task.

    The caller owns the transaction.  A failed act never creates an LLM task;
    when it declares a missing/non-terminal parent its durable ledger remains
    retryable until the parent's causal outcome can be inherited.  Calling
    this function again is idempotent.  A completed row explicitly marked
    ``retryable`` is re-enqueued only when ``retry=True``.
    """
    agent_id = _validate_agent_id(agent_id)
    act_uuid = _as_uuid(act_id, "act_id")
    reducer_version = _validate_reducer_version(reducer_version)
    evidence = load_act_reduction_input(conn, agent_id, act_uuid)
    if evidence is None:
        raise ActReductionValidationError("act_id is unknown for this agent")
    digest = reduction_input_hash(evidence)
    if evidence["status"] not in {"completed", "failed"}:
        raise ActReductionValidationError("act is not terminal")

    lock_agent_line(conn, agent_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,task_id,status,input_hash,outcome_version,attempt_count "
            "FROM cognitive_act_reductions "
            "WHERE agent_id=%s AND act_id=%s AND reducer_version=%s FOR UPDATE",
            (agent_id, act_uuid, reducer_version),
        )
        existing = cur.fetchone()

    if evidence["status"] == "failed":
        if existing is None:
            reduction_id = uuid.uuid4()
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "INSERT INTO cognitive_act_reductions "
                    "(id,agent_id,act_id,reducer_version,input_hash,status,"
                    " last_error_code) VALUES (%s,%s,%s,%s,%s,'retryable',"
                    " 'dependency_pending') "
                    "RETURNING id,task_id,status,input_hash,outcome_version,"
                    " attempt_count,residue_count,result_hash,output_line_version,"
                    " causal_root_hash,predecessor_frontier",
                    (reduction_id, agent_id, act_uuid, reducer_version, digest),
                )
                existing = cur.fetchone()
            created = True
        else:
            created = False
            if existing["input_hash"] != digest:
                if existing["status"] in {
                    "applied", "no_residue", "terminal_failure", "running"
                }:
                    raise ActReductionConflict(
                        "act input changed after reduction was fixed"
                    )
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "UPDATE cognitive_act_reductions SET input_hash=%s,"
                        " outcome_version=outcome_version+1,updated_at=clock_timestamp() "
                        "WHERE id=%s RETURNING id,task_id,status,input_hash,"
                        "outcome_version,attempt_count,residue_count,result_hash,"
                        "output_line_version,causal_root_hash,predecessor_frontier",
                        (digest, existing["id"]),
                    )
                    existing = cur.fetchone()
                assert existing is not None
                created = True
        if existing["task_id"] is not None:
            raise ActReductionConflict("failed act reduction must not have an LLM task")
        if existing["status"] in {"no_residue", "terminal_failure"}:
            return _row_to_schedule(existing, duplicate=not created)
        return _resolve_failed_reduction(
            conn,
            agent_id=agent_id,
            act_id=act_uuid,
            reducer_version=reducer_version,
            input_hash=digest,
            reduction_id=existing["id"],
            duplicate=not created,
        )

    if existing is not None:
        if existing["input_hash"] != digest:
            if existing["status"] in {"applied", "no_residue", "terminal_failure", "running"}:
                raise ActReductionConflict("act input changed after reduction was fixed")
            with conn.cursor() as cur:
                if existing["task_id"] is not None:
                    cur.execute(
                        "UPDATE llm_tasks SET status='failed',error='input_superseded',"
                        " completed_at=clock_timestamp() "
                        "WHERE id=%s AND status='pending'",
                        (existing["task_id"],),
                    )
            attempt_no = max(
                int(existing["attempt_count"]) + 1,
                int(existing["outcome_version"]) + 1,
            )
            task_id = _insert_task(
                conn, agent_id=agent_id, act_id=act_uuid,
                reducer_version=reducer_version, input_hash=digest,
                attempt_no=attempt_no,
            )
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "UPDATE cognitive_act_reductions SET input_hash=%s,task_id=%s,"
                    " status='pending',outcome_version=outcome_version+1,"
                    " last_error_code=NULL,started_at=NULL,completed_at=NULL,"
                    " updated_at=clock_timestamp() WHERE id=%s "
                    "RETURNING id,task_id,status,input_hash,outcome_version",
                    (digest, task_id, existing["id"]),
                )
                refreshed = cur.fetchone()
            assert refreshed is not None
            return _row_to_schedule(refreshed, duplicate=False)

        if existing["status"] == "retryable" and retry:
            if existing["task_id"] is not None:
                with conn.cursor() as cur:
                    cur.execute("SELECT status FROM llm_tasks WHERE id=%s", (existing["task_id"],))
                    task_row = cur.fetchone()
                if task_row is not None and task_row[0] in {"pending", "running"}:
                    return _row_to_schedule(existing, duplicate=True)
            attempt_no = max(
                int(existing["attempt_count"]) + 1,
                int(existing["outcome_version"]) + 1,
            )
            task_id = _insert_task(
                conn, agent_id=agent_id, act_id=act_uuid,
                reducer_version=reducer_version, input_hash=digest,
                attempt_no=attempt_no,
            )
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "UPDATE cognitive_act_reductions SET task_id=%s,status='pending',"
                    " last_error_code=NULL,started_at=NULL,updated_at=clock_timestamp() "
                    "WHERE id=%s RETURNING id,task_id,status,input_hash,outcome_version",
                    (task_id, existing["id"]),
                )
                retried = cur.fetchone()
            assert retried is not None
            return _row_to_schedule(retried, duplicate=False)
        return _row_to_schedule(existing, duplicate=True)

    reduction_id = uuid.uuid4()
    task_id = _insert_task(
        conn, agent_id=agent_id, act_id=act_uuid,
        reducer_version=reducer_version, input_hash=digest, attempt_no=1,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO cognitive_act_reductions "
            "(id,agent_id,act_id,reducer_version,input_hash,status,task_id) "
            "VALUES (%s,%s,%s,%s,%s,'pending',%s) "
            "RETURNING id,task_id,status,input_hash,outcome_version",
            (reduction_id, agent_id, act_uuid, reducer_version, digest, task_id),
        )
        row = cur.fetchone()
    assert row is not None
    return _row_to_schedule(row, duplicate=False)


def _load_locked_reduction(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID,
    reducer_version: str,
) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,task_id,status,input_hash,outcome_version,attempt_count,"
            "       residue_count,result_hash,output_line_version,causal_root_hash,"
            "       predecessor_frontier "
            "FROM cognitive_act_reductions "
            "WHERE agent_id=%s AND act_id=%s AND reducer_version=%s FOR UPDATE",
            (agent_id, act_id, reducer_version),
        )
        row = cur.fetchone()
    if row is None:
        raise ActReductionValidationError("act reduction was not scheduled")
    return row


def _stored_frontier(row: Mapping[str, Any]) -> tuple[uuid.UUID, ...]:
    value = row.get("predecessor_frontier") or []
    if not isinstance(value, list):
        raise ActReductionConflict("stored predecessor frontier is malformed")
    frontier = tuple(_as_uuid(item, "predecessor_frontier") for item in value)
    if len(frontier) > MAX_RESIDUES or len(set(frontier)) != len(frontier):
        raise ActReductionConflict("stored predecessor frontier is not bounded and unique")
    return frontier


@dataclass(frozen=True)
class _TerminalCausalOutcome:
    output_line_version: int
    causal_root_hash: str
    frontier: tuple[uuid.UUID, ...]


class _ParentOutcomePending(Exception):
    pass


class _ParentOutcomeCycle(Exception):
    pass


class _ParentOutcomeFailed(Exception):
    pass


def _terminal_reduction_outcome(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    act_id: uuid.UUID,
    reducer_version: str,
    row: Mapping[str, Any],
) -> _TerminalCausalOutcome:
    status = str(row["status"])
    if status not in {"applied", "no_residue"}:
        raise ActReductionConflict("parent reduction is not a causal outcome")
    line_version = row.get("output_line_version")
    root_hash = row.get("causal_root_hash")
    if isinstance(line_version, bool) or not isinstance(line_version, int):
        raise ActReductionConflict("parent output_line_version is invalid")
    root_hash = _validate_hash(str(root_hash), "parent causal_root_hash")
    if status == "no_residue":
        frontier = _stored_frontier(row)
    else:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM memories WHERE agent_id=%s AND cognitive_act_id=%s "
                "AND line_provenance='validated_act_residue' "
                "AND residue_reducer_version=%s ORDER BY residue_ordinal",
                (agent_id, act_id, reducer_version),
            )
            frontier = tuple(item[0] for item in cur.fetchall())
        if len(frontier) != int(row.get("residue_count") or 0):
            raise ActReductionConflict(
                "parent terminal residue frontier is incomplete"
            )
        if not 1 <= len(frontier) <= MAX_RESIDUES:
            raise ActReductionConflict("parent applied frontier is invalid")
    return _TerminalCausalOutcome(line_version, root_hash, frontier)


def _resolve_failed_parent_outcome(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    act_id: uuid.UUID,
    reducer_version: str,
    visited: frozenset[uuid.UUID],
    depth: int,
) -> _TerminalCausalOutcome:
    if depth > MAX_PARENT_CHAIN_DEPTH or act_id in visited:
        raise _ParentOutcomeCycle
    path = visited | {act_id}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT status,declared_parent_key,parent_act_id FROM cognitive_acts "
            "WHERE agent_id=%s AND id=%s FOR UPDATE",
            (agent_id, act_id),
        )
        act = cur.fetchone()
        if act is None:
            raise _ParentOutcomePending
        declared_key = act["declared_parent_key"]
        if declared_key is None:
            line = load_causal_line_state(conn, agent_id, for_update=True)
            return _TerminalCausalOutcome(
                line.line_version, line.causal_root_hash, line.frontier
            )
        parent_id = act["parent_act_id"]
        if parent_id is None:
            cur.execute(
                "SELECT id,status FROM cognitive_acts "
                "WHERE agent_id=%s AND host_key=%s",
                (agent_id, declared_key),
            )
            parent = cur.fetchone()
            if parent is None:
                raise _ParentOutcomePending
            parent_id = parent["id"]
            cur.execute(
                "UPDATE cognitive_acts SET parent_act_id=%s "
                "WHERE agent_id=%s AND id=%s AND parent_act_id IS NULL",
                (parent_id, agent_id, act_id),
            )
        if parent_id in path:
            raise _ParentOutcomeCycle
        cur.execute(
            "SELECT id,task_id,status,input_hash,outcome_version,attempt_count,"
            "residue_count,result_hash,output_line_version,causal_root_hash,"
            "predecessor_frontier FROM cognitive_act_reductions "
            "WHERE agent_id=%s AND act_id=%s AND reducer_version=%s FOR UPDATE",
            (agent_id, parent_id, reducer_version),
        )
        reduction = cur.fetchone()
        cur.execute(
            "SELECT status FROM cognitive_acts WHERE agent_id=%s AND id=%s",
            (agent_id, parent_id),
        )
        parent_act = cur.fetchone()
    if parent_act is None:
        raise _ParentOutcomePending
    if reduction is not None and reduction["status"] in {"applied", "no_residue"}:
        return _terminal_reduction_outcome(
            conn,
            agent_id=agent_id,
            act_id=parent_id,
            reducer_version=reducer_version,
            row=reduction,
        )
    if reduction is not None and reduction["status"] == "terminal_failure":
        raise _ParentOutcomeFailed
    if str(parent_act["status"]) != "failed":
        raise _ParentOutcomePending
    if reduction is None:
        # Every terminal cognition commit creates its ledger atomically.  Do
        # not synthesize a hash from partially visible state if that invariant
        # was violated; a later repair/schedule can establish it safely.
        raise _ParentOutcomePending
    if reduction["task_id"] is not None:
        raise _ParentOutcomePending

    inherited = _resolve_failed_parent_outcome(
        conn,
        agent_id=agent_id,
        act_id=parent_id,
        reducer_version=reducer_version,
        visited=path,
        depth=depth + 1,
    )
    result_hash = reduction_input_hash({"outcome": "no_residue", "residues": []})
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "UPDATE cognitive_act_reductions SET status='no_residue',"
            "residue_count=0,output_line_version=%s,causal_root_hash=%s,"
            "predecessor_frontier=%s,result_hash=%s,last_error_code=NULL,"
            "completed_at=clock_timestamp(),updated_at=clock_timestamp() "
            "WHERE id=%s AND status NOT IN ('applied','no_residue','terminal_failure') "
            "RETURNING id,task_id,status,input_hash,outcome_version,attempt_count,"
            "residue_count,result_hash,output_line_version,causal_root_hash,"
            "predecessor_frontier",
            (
                inherited.output_line_version,
                inherited.causal_root_hash,
                Jsonb([str(item) for item in inherited.frontier]),
                result_hash,
                reduction["id"],
            ),
        )
        resolved = cur.fetchone()
    if resolved is None:
        raise ActReductionConflict("failed parent outcome raced")
    return inherited


def _resolve_failed_reduction(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    act_id: uuid.UUID,
    reducer_version: str,
    input_hash: str,
    reduction_id: uuid.UUID,
    duplicate: bool,
) -> ReductionSchedule:
    try:
        inherited = _resolve_failed_parent_outcome(
            conn,
            agent_id=agent_id,
            act_id=act_id,
            reducer_version=reducer_version,
            visited=frozenset(),
            depth=0,
        )
    except _ParentOutcomePending:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "UPDATE cognitive_act_reductions SET status='retryable',"
                "last_error_code='dependency_pending',started_at=NULL,"
                "completed_at=NULL,updated_at=clock_timestamp() WHERE id=%s "
                "RETURNING id,task_id,status,input_hash,outcome_version",
                (reduction_id,),
            )
            row = cur.fetchone()
        assert row is not None
        return _row_to_schedule(row, duplicate=duplicate)
    except (_ParentOutcomeCycle, _ParentOutcomeFailed) as exc:
        error_code = (
            "parent_cycle"
            if isinstance(exc, _ParentOutcomeCycle)
            else "parent_terminal_failure"
        )
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "UPDATE cognitive_act_reductions SET status='terminal_failure',"
                "last_error_code=%s,residue_count=0,output_line_version=NULL,"
                "causal_root_hash=NULL,predecessor_frontier='[]'::jsonb,"
                "result_hash=NULL,started_at=NULL,completed_at=clock_timestamp(),"
                "updated_at=clock_timestamp() WHERE id=%s "
                "RETURNING id,task_id,status,input_hash,outcome_version",
                (error_code, reduction_id),
            )
            row = cur.fetchone()
        assert row is not None
        return _row_to_schedule(row, duplicate=duplicate)

    result_hash = reduction_input_hash({"outcome": "no_residue", "residues": []})
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "UPDATE cognitive_act_reductions SET status='no_residue',"
            "residue_count=0,output_line_version=%s,causal_root_hash=%s,"
            "predecessor_frontier=%s,result_hash=%s,last_error_code=NULL,"
            "started_at=NULL,completed_at=clock_timestamp(),"
            "updated_at=clock_timestamp() WHERE id=%s "
            "RETURNING id,task_id,status,input_hash,outcome_version",
            (
                inherited.output_line_version,
                inherited.causal_root_hash,
                Jsonb([str(item) for item in inherited.frontier]),
                result_hash,
                reduction_id,
            ),
        )
        row = cur.fetchone()
    assert row is not None
    return _row_to_schedule(row, duplicate=duplicate)


def mark_act_reduction_running(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
    *,
    reducer_version: str,
    task_id: uuid.UUID | str,
    input_hash: str,
) -> None:
    agent_id = _validate_agent_id(agent_id)
    act_uuid = _as_uuid(act_id, "act_id")
    task_uuid = _as_uuid(task_id, "task_id")
    reducer_version = _validate_reducer_version(reducer_version)
    input_hash = _validate_hash(input_hash)
    row = _load_locked_reduction(conn, agent_id, act_uuid, reducer_version)
    if row["task_id"] != task_uuid or row["input_hash"] != input_hash:
        raise ActReductionConflict("task coordinates do not match reduction ledger")
    if row["status"] == "running":
        return
    if row["status"] != "pending":
        raise ActReductionConflict(f"cannot start reduction in {row['status']} status")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cognitive_act_reductions SET status='running',"
            " attempt_count=attempt_count+1,started_at=clock_timestamp(),"
            " updated_at=clock_timestamp() WHERE id=%s",
            (row["id"],),
        )


def _normalise_evidence_ref(
    raw: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ActReductionValidationError("evidence reference must be an object")
    source = raw.get("source")
    if source in {"channel_input", "channel_output"}:
        if set(raw) != {"source", "key"}:
            raise ActReductionValidationError("channel evidence accepts only source/key")
        key = raw.get("key")
        channel = evidence[source]
        if not isinstance(key, str) or not key or len(key) > 64 or key not in channel:
            raise ActReductionValidationError("channel evidence key is unknown for this act")
        return {"source": source, "key": key}
    if source == "action":
        if set(raw) != {"source", "ordinal"} or isinstance(raw.get("ordinal"), bool):
            raise ActReductionValidationError("action evidence requires only an ordinal")
        try:
            ordinal = int(raw["ordinal"])
        except (TypeError, ValueError) as exc:
            raise ActReductionValidationError("action evidence ordinal must be an integer") from exc
        known = {item["ordinal"] for item in evidence["actions"]}
        if ordinal not in known:
            raise ActReductionValidationError("action evidence ordinal is unknown for this act")
        return {"source": "action", "ordinal": ordinal}
    if source == "observation":
        if set(raw) != {"source", "observation_id"}:
            raise ActReductionValidationError(
                "observation evidence requires only an observation_id"
            )
        observation_id = str(_as_uuid(raw.get("observation_id"), "observation_id"))
        known = {item["observation_id"] for item in evidence["presented_observations"]}
        if observation_id not in known:
            raise ActReductionValidationError(
                "observation evidence is not acknowledged by this act"
            )
        return {"source": "observation", "observation_id": observation_id}
    raise ActReductionValidationError("evidence source is not allowlisted")


def _normalise_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != EMBEDDING_DIMENSION:
        raise ActReductionValidationError(
            f"embedding must contain exactly {EMBEDDING_DIMENSION} values"
        )
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ActReductionValidationError("embedding values must be finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ActReductionValidationError("embedding values must be finite numbers") from exc
        if not math.isfinite(number):
            raise ActReductionValidationError("embedding values must be finite numbers")
        vector.append(number)
    return vector


def _bounded_float(value: Any, field: str, *, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ActReductionValidationError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ActReductionValidationError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ActReductionValidationError(f"{field} must be within {low}..{high}")
    return number


def _normalise_affect(value: Any, causal_role: str) -> dict[str, Any] | None:
    if causal_role != "affective_coordinate":
        if value is not None:
            raise ActReductionValidationError(
                "affect is allowed only for affective_coordinate residues"
            )
        return None
    if not isinstance(value, Mapping):
        raise ActReductionValidationError(
            "affective_coordinate residue requires an affect object"
        )
    if set(value) - _AFFECT_KEYS or not _AFFECT_REQUIRED.issubset(value):
        raise ActReductionValidationError("affect fields do not match the allowlist")
    absolute = {"valence", "arousal", "dominance"}
    present_absolute = absolute.intersection(value)
    if present_absolute and present_absolute != absolute:
        raise ActReductionValidationError("absolute valence/arousal/dominance are all-or-none")
    result: dict[str, Any] = {
        key: _bounded_float(value[key], f"affect.{key}", low=-1.0, high=1.0)
        for key in ("valence_delta", "arousal_delta", "dominance_delta")
    }
    for key in ("valence", "arousal", "dominance"):
        if key in value:
            result[key] = _bounded_float(
                value[key], f"affect.{key}", low=-1.0, high=1.0
            )
    for key in ("intensity", "cause_confidence"):
        if key in value:
            result[key] = _bounded_float(
                value[key], f"affect.{key}", low=0.0, high=1.0
            )
    if "cause_status" in value:
        if value["cause_status"] not in _CAUSE_STATUSES:
            raise ActReductionValidationError("affect.cause_status is not allowlisted")
        result["cause_status"] = value["cause_status"]
    return result


def _normalise_residues(
    residues: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(residues, (str, bytes)) or not isinstance(residues, Sequence):
        raise ActReductionValidationError("residues must be an array")
    if len(residues) > MAX_RESIDUES:
        raise ActReductionValidationError(f"at most {MAX_RESIDUES} residues are allowed")
    normalised: list[dict[str, Any]] = []
    for raw in residues:
        if not isinstance(raw, Mapping):
            raise ActReductionValidationError("each residue must be an object")
        unknown = set(raw) - _RESIDUE_KEYS
        required = _RESIDUE_KEYS - {"embedding", "affect"}
        if unknown or not required.issubset(raw):
            raise ActReductionValidationError("residue fields do not match the allowlist")
        kind = raw["kind"]
        causal_role = raw["causal_role"]
        if kind not in _MEMORY_KINDS:
            raise ActReductionValidationError("residue kind is not allowlisted")
        if causal_role not in _CAUSAL_ROLES:
            raise ActReductionValidationError("residue causal_role is not allowlisted")
        if not isinstance(raw["content"], str):
            raise ActReductionValidationError("residue content must be a string")
        content = raw["content"].strip()
        if not content or len(content) > MAX_RESIDUE_CONTENT:
            raise ActReductionValidationError(
                f"residue content must contain 1..{MAX_RESIDUE_CONTENT} characters"
            )
        if isinstance(raw["confidence"], bool):
            raise ActReductionValidationError("residue confidence must be numeric")
        try:
            confidence = float(raw["confidence"])
        except (TypeError, ValueError) as exc:
            raise ActReductionValidationError("residue confidence must be numeric") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ActReductionValidationError("residue confidence must be within 0..1")
        refs = raw["evidence_refs"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_EVIDENCE_REFS:
            raise ActReductionValidationError(
                f"residue evidence_refs must contain 1..{MAX_EVIDENCE_REFS} items"
            )
        safe_refs = [_normalise_evidence_ref(item, evidence) for item in refs]
        # Exact duplicates add no provenance and make result hashing ambiguous.
        if len({json.dumps(item, sort_keys=True) for item in safe_refs}) != len(safe_refs):
            raise ActReductionValidationError("residue evidence_refs must be unique")
        normalised.append({
            "kind": kind,
            "causal_role": causal_role,
            "content": redact_journal_text(content, limit=MAX_RESIDUE_CONTENT),
            "confidence": confidence,
            "evidence_refs": safe_refs,
            "embedding": _normalise_embedding(raw.get("embedding")),
            "affect": _normalise_affect(raw.get("affect"), causal_role),
        })
    if sum(
        item["causal_role"] == "affective_coordinate" for item in normalised
    ) > 1:
        raise ActReductionValidationError(
            "an act may contain at most one affective_coordinate residue"
        )
    return normalised


def _vector_literal(vector: Sequence[float] | None) -> str | None:
    if vector is None:
        return None
    return "[" + ",".join(str(value) for value in vector) + "]"


def _apply_affective_residue(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    act_id: uuid.UUID,
    memory_id: uuid.UUID,
    ordinal: int,
    reducer_version: str,
    line_root_hash: str,
    residue: Mapping[str, Any],
) -> None:
    """Materialize an affective coordinate from the same validated residue.

    This is intentionally part of the reduction transaction.  The legacy
    synchronous turn observer remains an API compatibility surface, but the
    canonical cognition path must not run a second independent interpreter of
    the same act.
    """
    affect = residue.get("affect")
    if residue.get("causal_role") != "affective_coordinate" or not isinstance(
        affect, Mapping
    ):
        return

    from styx.emotional.state import (
        EmotionalVector,
        append_emotional_event,
        append_emotional_transition,
        read_last_state_record,
    )
    from styx.emotional.transition import redact_cause_summary

    delta = EmotionalVector(
        float(affect["valence_delta"]),
        float(affect["arousal_delta"]),
        float(affect["dominance_delta"]),
    )
    absolute_keys = {"valence", "arousal", "dominance"}
    signal = (
        EmotionalVector(
            float(affect["valence"]),
            float(affect["arousal"]),
            float(affect["dominance"]),
        )
        if absolute_keys.issubset(affect)
        else delta
    )
    intensity = float(affect.get("intensity", residue["confidence"]))
    cause_confidence = float(
        affect.get("cause_confidence", residue["confidence"])
    )
    event = append_emotional_event(
        conn,
        agent_id,
        source_kind="cognitive_act_residue",
        source_ref=str(act_id),
        # Reduction idempotency is owned by the locked terminal ledger row.
        # A public event key could be pre-claimed by an unrelated caller and
        # make this internal coordinate reuse foreign immutable evidence.
        idempotency_key=None,
        signal=signal,
        intensity=intensity,
        confidence=cause_confidence,
        cause_summary=redact_cause_summary(str(residue["content"]))[:1000],
        cause_status=str(affect.get("cause_status", "unknown")),
        metadata={
            "cognitive_act_id": str(act_id),
            "residue_memory_id": str(memory_id),
            "residue_ordinal": ordinal,
            "reducer_version": reducer_version,
            "line_root_hash": line_root_hash,
            "evidence_refs": list(residue["evidence_refs"]),
            "coordinate_only": True,
        },
        sync_cause_lifecycle=False,
    )
    previous = read_last_state_record(conn, agent_id)
    cause_status = str(affect.get("cause_status", "unknown"))
    contribution = {
        "evidence_id": event.event.id,
        "source_ref": str(act_id),
        "status": cause_status,
        "intensity": intensity,
        "confidence": cause_confidence,
        "observed_at": event.event.observed_at.isoformat(),
        "weighted_delta": [
            delta.valence,
            delta.arousal,
            delta.dominance,
        ],
        "cognitive_act_id": str(act_id),
        "residue_memory_id": str(memory_id),
        "line_root_hash": line_root_hash,
    }
    if cause_status != "unknown":
        contribution["cause_active"] = cause_status == "active"
    causal_context = [
        *(dict(item) for item in previous.causal_context),
        contribution,
    ] if previous is not None else [contribution]
    append_emotional_transition(
        conn,
        agent_id,
        delta,
        source="act_residue",
        event_id=event.event.id,
        intensity=intensity,
        confidence=cause_confidence,
        causal_context=causal_context,
        computation_version="act_residue_v1",
        metadata={
            "cognitive_act_id": str(act_id),
            "residue_memory_id": str(memory_id),
            "line_root_hash": line_root_hash,
        },
    )


def _declared_parent_frontier(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    act_id: uuid.UUID,
    reducer_version: str,
) -> tuple[uuid.UUID, ...] | None:
    """Resolve an immutable declared parent to its terminal causal output.

    ``None`` means the act declares no causal parent.  An empty tuple is a
    valid terminal frontier (for example, a root ``no_residue`` act).  The
    mutable ``parent_act_id`` cache may be filled after scheduling without
    changing the already-frozen reduction input.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT declared_parent_key,parent_act_id FROM cognitive_acts "
            "WHERE agent_id=%s AND id=%s FOR UPDATE",
            (agent_id, act_id),
        )
        act = cur.fetchone()
        if act is None:
            raise ActReductionValidationError("act_id is unknown for this agent")
        declared_parent_key = act["declared_parent_key"]
        if declared_parent_key is None:
            return None
        parent_act_id = act["parent_act_id"]
        if parent_act_id is None:
            cur.execute(
                "SELECT id FROM cognitive_acts WHERE agent_id=%s AND host_key=%s",
                (agent_id, declared_parent_key),
            )
            parent = cur.fetchone()
            if parent is None:
                raise ActReductionDependencyPending(
                    parent_act_id=None,
                    reduction_status="parent_unresolved",
                )
            parent_act_id = parent["id"]
            cur.execute(
                "UPDATE cognitive_acts SET parent_act_id=%s "
                "WHERE agent_id=%s AND id=%s AND parent_act_id IS NULL",
                (parent_act_id, agent_id, act_id),
            )

        cur.execute(
            "SELECT status,residue_count,predecessor_frontier "
            "FROM cognitive_act_reductions "
            "WHERE agent_id=%s AND act_id=%s AND reducer_version=%s",
            (agent_id, parent_act_id, reducer_version),
        )
        parent_reduction = cur.fetchone()
        if parent_reduction is None:
            raise ActReductionDependencyPending(
                parent_act_id=parent_act_id,
                reduction_status="unscheduled",
            )
        parent_status = str(parent_reduction["status"])
        if parent_status not in {"applied", "no_residue"}:
            raise ActReductionDependencyPending(
                parent_act_id=parent_act_id,
                reduction_status=parent_status,
            )
        if parent_status == "no_residue":
            return _stored_frontier(parent_reduction)

        cur.execute(
            "SELECT id FROM memories WHERE agent_id=%s AND cognitive_act_id=%s "
            "AND line_provenance='validated_act_residue' "
            "AND residue_reducer_version=%s ORDER BY residue_ordinal",
            (agent_id, parent_act_id, reducer_version),
        )
        memory_ids = tuple(item["id"] for item in cur.fetchall())
    if len(memory_ids) != int(parent_reduction["residue_count"]):
        raise ActReductionConflict(
            "declared parent terminal residue frontier is incomplete"
        )
    if not 1 <= len(memory_ids) <= MAX_RESIDUES:
        raise ActReductionConflict(
            "declared parent applied outcome has an invalid frontier"
        )
    return memory_ids


def apply_act_reduction(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
    *,
    reducer_version: str,
    input_hash: str,
    residues: Sequence[Mapping[str, Any]],
    task_id: uuid.UUID | str | None = None,
) -> ReductionApply:
    """Validate and atomically incorporate zero to four act residues.

    The caller owns the transaction.  Any validation/constraint error rolls
    back the ledger transition, memories, lineage, and line-version trigger as
    one unit.
    """
    agent_id = _validate_agent_id(agent_id)
    act_uuid = _as_uuid(act_id, "act_id")
    reducer_version = _validate_reducer_version(reducer_version)
    input_hash = _validate_hash(input_hash)
    task_uuid = _as_uuid(task_id, "task_id") if task_id is not None else None
    evidence = load_act_reduction_input(conn, agent_id, act_uuid)
    if evidence is None or evidence["status"] != "completed":
        raise ActReductionValidationError("only a completed act can yield residues")
    if reduction_input_hash(evidence) != input_hash:
        raise ActReductionConflict("act input no longer matches scheduled input_hash")
    normalised = _normalise_residues(residues, evidence)
    result_document = {
        "outcome": "applied" if normalised else "no_residue",
        "residues": normalised,
    }
    result_hash = reduction_input_hash(result_document)

    lock_agent_line(conn, agent_id)
    row = _load_locked_reduction(conn, agent_id, act_uuid, reducer_version)
    if row["input_hash"] != input_hash:
        raise ActReductionConflict("input_hash does not match reduction ledger")
    if task_uuid is not None and row["task_id"] != task_uuid:
        raise ActReductionConflict("task_id does not match reduction ledger")
    if row["status"] in {"applied", "no_residue"}:
        if row["result_hash"] != result_hash:
            raise ActReductionConflict("act already has a different terminal outcome")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM memories WHERE agent_id=%s AND cognitive_act_id=%s "
                "AND line_provenance='validated_act_residue' "
                "AND residue_reducer_version=%s ORDER BY residue_ordinal",
                (agent_id, act_uuid, reducer_version),
            )
            ids = tuple(item[0] for item in cur.fetchall())
        stored_root = _validate_hash(
            str(row["causal_root_hash"]), "stored causal_root_hash"
        )
        return ReductionApply(
            row["id"], row["status"], ids, result_hash, True,
            int(row["output_line_version"]), stored_root, _stored_frontier(row),
        )
    if row["status"] == "terminal_failure":
        raise ActReductionConflict("act reduction is terminal_failure")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories WHERE agent_id=%s AND cognitive_act_id=%s "
            "AND line_provenance='validated_act_residue' "
            "AND residue_reducer_version=%s",
            (agent_id, act_uuid, reducer_version),
        )
        if int(cur.fetchone()[0]) != 0:
            raise ActReductionConflict("partial residue rows exist without a terminal outcome")

    declared_parent_frontier = _declared_parent_frontier(
        conn,
        agent_id=agent_id,
        act_id=act_uuid,
        reducer_version=reducer_version,
    )

    if normalised:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO line_state(agent_id,version,dirty) VALUES (%s,0,true) "
                "ON CONFLICT (agent_id) DO NOTHING",
                (agent_id,),
            )
    causal_state = load_causal_line_state(conn, agent_id, for_update=True)
    if bool(causal_state.frontier) != (causal_state.causal_root_hash != EMPTY_CAUSAL_ROOT):
        raise ActReductionConflict("causal root and frontier disagree")
    if causal_state.causal_root_hash == EMPTY_CAUSAL_ROOT:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories WHERE agent_id=%s "
                "AND memory_domain='subjective_trace' AND line_eligible=true "
                "AND line_provenance IN "
                "('validated_act_residue','validated_transform') "
                "AND line_status='active'",
                (agent_id,),
            )
            if int(cur.fetchone()[0]) != 0:
                raise ActReductionConflict(
                    "validated residues exist without a causal line root"
                )

    predecessor_frontier = (
        declared_parent_frontier
        if declared_parent_frontier is not None
        else causal_state.frontier
    )
    memory_ids = [uuid.uuid4() for _ in normalised]
    expected_line_version = causal_state.line_version + len(memory_ids)
    rooted_residues = [
        {**residue, "memory_id": memory_ids[ordinal]}
        for ordinal, residue in enumerate(normalised)
    ]
    predecessor_json = [str(item) for item in predecessor_frontier]

    # Wave 40 promotes the edge ledger to the source of truth.  Node/root
    # hashes contain semantic material only; row ids, clocks and embeddings do
    # not alter the causal coordinate.
    from styx.engine.causal_graph import (
        GraphEdge,
        GraphNode,
        causal_edge_hash,
        causal_node_hash,
        validate_graph,
    )
    from styx.storage.causal_graph import load_causal_graph

    current_nodes, current_edges = load_causal_graph(conn, agent_id)
    validate_graph(current_nodes, current_edges)
    current_by_id = {node.node_id: node for node in current_nodes}
    predecessor_hashes: list[str] = []
    for predecessor in predecessor_frontier:
        node = current_by_id.get(str(predecessor))
        if node is None or node.line_status != "active":
            raise ActReductionConflict(
                "predecessor frontier is absent from the active causal graph"
            )
        predecessor_hashes.append(node.node_hash)
    residue_hashes = [
        causal_node_hash(
            node_kind="act_residue",
            content=residue["content"],
            causal_role=residue["causal_role"],
            predecessor_hashes=predecessor_hashes,
        )
        for residue in normalised
    ]
    planned_nodes = list(current_nodes) + [
        GraphNode(str(memory_ids[index]), residue_hashes[index], "active")
        for index in range(len(memory_ids))
    ]
    planned_edges = list(current_edges)
    for ordinal, memory_id in enumerate(memory_ids):
        for predecessor in predecessor_frontier:
            source = current_by_id[str(predecessor)]
            planned_edges.append(GraphEdge(
                f"planned:{act_uuid}:{ordinal}:{predecessor}",
                str(predecessor), str(memory_id), "incorporated",
                causal_edge_hash(
                    source_hash=source.node_hash,
                    target_hash=residue_hashes[ordinal],
                    relation="incorporated",
                ),
            ))
    planned_validation = validate_graph(planned_nodes, planned_edges)
    next_root = (
        planned_validation.graph_root_hash
        if rooted_residues else causal_state.causal_root_hash
    )

    if memory_ids:
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('styx.causal_operation','1',true)")

    for ordinal, residue in enumerate(normalised):
        memory_id = memory_ids[ordinal]
        metadata = {
            "act_residue": {
                "causal_role": residue["causal_role"],
                "confidence": residue["confidence"],
                "evidence_refs": residue["evidence_refs"],
                "reducer_version": reducer_version,
                "input_hash": input_hash,
                "predecessor_frontier": predecessor_json,
                "previous_root_hash": causal_state.causal_root_hash,
                "line_root_hash": next_root,
                **({"affect": residue["affect"]} if residue["affect"] else {}),
            }
        }
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories "
                "(id,agent_id,role,kind,kind_src,content,embedding,metadata,"
                " memory_domain,line_eligible,cognitive_act_id,line_provenance,"
                " residue_ordinal,residue_reducer_version,residue_input_hash,"
                " residue_causal_role,residue_confidence,residue_evidence,"
                " residue_predecessors,residue_line_root_hash,residue_affect,"
                " causal_node_hash,causal_node_kind,causal_payload_version,"
                " line_status) "
                "VALUES (%s,%s,'summary',%s,'subjective',%s,%s,%s,"
                " 'subjective_trace',true,%s,'validated_act_residue',"
                " %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'act_residue',"
                " 'causal_node_v1','active') RETURNING id",
                (
                    memory_id, agent_id, residue["kind"], residue["content"],
                    _vector_literal(residue["embedding"]), Jsonb(metadata), act_uuid,
                    ordinal, reducer_version, input_hash, residue["causal_role"],
                    residue["confidence"], Jsonb(residue["evidence_refs"]),
                    Jsonb(predecessor_json), next_root,
                    Jsonb(residue["affect"] or {}), residue_hashes[ordinal],
                ),
            )
            assert cur.fetchone()[0] == memory_id
            predecessors: Sequence[uuid.UUID | None] = (
                predecessor_frontier if predecessor_frontier else (None,)
            )
            for predecessor_ordinal, predecessor_id in enumerate(predecessors):
                coordinates = Jsonb({
                            "edge_kind": "predecessor_frontier",
                            "predecessor_ordinal": predecessor_ordinal,
                            "predecessor_frontier": predecessor_json,
                            "previous_root_hash": causal_state.causal_root_hash,
                            "line_root_hash": next_root,
                            "reducer_version": reducer_version,
                            "input_hash": input_hash,
                            "causal_role": residue["causal_role"],
                            "evidence_refs": residue["evidence_refs"],
                        })
                if predecessor_id is None:
                    # Compatibility-only root marker.  It is deliberately not
                    # a validated DAG edge.
                    cur.execute(
                        "INSERT INTO memory_lineage "
                        "(agent_id,source_memory_id,target_memory_id,cognitive_act_id,"
                        " transform,ordinal,retained_weight,source_coordinates) "
                        "VALUES (%s,NULL,%s,%s,'incorporated',%s,%s,%s)",
                        (
                            agent_id, memory_id, act_uuid, ordinal,
                            residue["confidence"], coordinates,
                        ),
                    )
                else:
                    source_hash = current_by_id[str(predecessor_id)].node_hash
                    edge_key = (
                        f"act:{act_uuid}:{reducer_version}:{ordinal}:"
                        f"{predecessor_id}"
                    )
                    cur.execute(
                        "INSERT INTO memory_lineage "
                        "(agent_id,source_memory_id,target_memory_id,cognitive_act_id,"
                        " transform,ordinal,retained_weight,source_coordinates,"
                        " edge_key,edge_provenance,relation_version,source_node_hash,"
                        " target_node_hash,valid_from_line_version,edge_hash) "
                        "VALUES (%s,%s,%s,%s,'incorporated',%s,%s,%s,%s,"
                        "'validated',1,%s,%s,%s,%s)",
                        (
                            agent_id, predecessor_id, memory_id, act_uuid, ordinal,
                            residue["confidence"], coordinates, edge_key,
                            source_hash, residue_hashes[ordinal],
                            expected_line_version, causal_edge_hash(
                                source_hash=source_hash,
                                target_hash=residue_hashes[ordinal],
                                relation="incorporated",
                            ),
                        ),
                    )
        _apply_affective_residue(
            conn,
            agent_id=agent_id,
            act_id=act_uuid,
            memory_id=memory_id,
            ordinal=ordinal,
            reducer_version=reducer_version,
            line_root_hash=next_root,
            residue=residue,
        )

    if memory_ids:
        actual_nodes, actual_edges = load_causal_graph(conn, agent_id)
        actual_validation = validate_graph(actual_nodes, actual_edges)
        if actual_validation.graph_root_hash != next_root:
            raise ActReductionConflict("applied causal graph differs from planned graph")
        if not set(str(item) for item in memory_ids).issubset(
            actual_validation.frontier
        ):
            raise ActReductionConflict("applied causal frontier omits new residues")
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM line_state WHERE agent_id=%s FOR UPDATE", (agent_id,))
            actual_line_version = int(cur.fetchone()[0])
            if actual_line_version != causal_state.line_version:
                raise ActReductionConflict("line version changed during atomic incorporation")
            actual_line_version = expected_line_version
            cur.execute(
                "UPDATE line_state SET version=%s,causal_root_hash=%s,causal_frontier=%s,"
                " causal_root_version=%s,causal_root_act_id=%s,dirty=true,"
                " causal_root_operation_id=NULL,updated_at=clock_timestamp() "
                "WHERE agent_id=%s",
                (
                    actual_line_version, next_root,
                    Jsonb(list(actual_validation.frontier)),
                    actual_line_version, act_uuid, agent_id,
                ),
            )
    else:
        actual_line_version = causal_state.line_version

    status = "applied" if memory_ids else "no_residue"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cognitive_act_reductions SET status=%s,residue_count=%s,"
            " output_line_version=%s,causal_root_hash=%s,predecessor_frontier=%s,"
            " result_hash=%s,last_error_code=NULL,completed_at=clock_timestamp(),"
            " updated_at=clock_timestamp() WHERE id=%s",
            (
                status, len(memory_ids), actual_line_version, next_root,
                Jsonb(predecessor_json), result_hash, row["id"],
            ),
        )
    return ReductionApply(
        row["id"], status, tuple(memory_ids), result_hash, False,
        actual_line_version, next_root, predecessor_frontier,
    )


def _mark_failure(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
    *,
    reducer_version: str,
    task_id: uuid.UUID | str,
    input_hash: str,
    error_code: str,
    terminal: bool,
) -> None:
    agent_id = _validate_agent_id(agent_id)
    act_uuid = _as_uuid(act_id, "act_id")
    task_uuid = _as_uuid(task_id, "task_id")
    reducer_version = _validate_reducer_version(reducer_version)
    input_hash = _validate_hash(input_hash)
    if not isinstance(error_code, str) or _ERROR_CODE_RE.fullmatch(error_code) is None:
        raise ActReductionValidationError("error_code is not safe for durable storage")
    row = _load_locked_reduction(conn, agent_id, act_uuid, reducer_version)
    if row["task_id"] != task_uuid or row["input_hash"] != input_hash:
        raise ActReductionConflict("task coordinates do not match reduction ledger")
    if row["status"] in {"applied", "no_residue", "terminal_failure"}:
        if terminal and row["status"] == "terminal_failure":
            return
        raise ActReductionConflict(f"cannot fail reduction in {row['status']} status")
    target = "terminal_failure" if terminal else "retryable"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cognitive_act_reductions SET status=%s,last_error_code=%s,"
            " residue_count=0,started_at=NULL,completed_at="
            + ("clock_timestamp()" if terminal else "NULL")
            + ",updated_at=clock_timestamp() WHERE id=%s",
            (target, error_code, row["id"]),
        )


def mark_act_reduction_retryable(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
    *,
    reducer_version: str,
    task_id: uuid.UUID | str,
    input_hash: str,
    error_code: str,
) -> None:
    _mark_failure(
        conn, agent_id, act_id, reducer_version=reducer_version,
        task_id=task_id, input_hash=input_hash, error_code=error_code,
        terminal=False,
    )


def mark_act_reduction_terminal_failure(
    conn: psycopg.Connection,
    agent_id: str,
    act_id: uuid.UUID | str,
    *,
    reducer_version: str,
    task_id: uuid.UUID | str,
    input_hash: str,
    error_code: str,
) -> None:
    _mark_failure(
        conn, agent_id, act_id, reducer_version=reducer_version,
        task_id=task_id, input_hash=input_hash, error_code=error_code,
        terminal=True,
    )
