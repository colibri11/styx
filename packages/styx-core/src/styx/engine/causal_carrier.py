"""Deterministic, query-independent technical carrier for a causal line.

The builder is deliberately pure: it receives the complete live eligible set
and returns a bounded extractive projection without touching storage, clocks or
models.  The projection is an engineering coordinate.  It does not establish
will, consciousness, personality, or any other ontological status.

Every input row contributes to full-set coverage.  Every validated row also
appears as a semantic frontier root unless a validated reduction root explicitly
names it in reduction ancestry.  Quarantined rows contribute only counts and
digests: their content never reaches the active carrier or its supports.  The
bounded ``supports`` list is diagnostic; it is never used as a substitute for
the whole carrier.  Active historical text is JSON-quoted and labelled as data;
instruction-shaped content is never promoted to an instruction surface.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Literal, Mapping, Sequence, TypedDict

ProjectionStatus = Literal["empty", "provisional", "ready", "degraded"]

ALGORITHM_VERSION = "causal_carrier_v1"
DEFAULT_MAX_SUPPORTS = 8
DEFAULT_MAX_CARRIER_CHARS = 6_000
DEFAULT_MAX_EXCERPT_CHARS = 600
MAX_EMBEDDING_DIMENSIONS = 4_096

# Wave 38 has exactly one active provenance.  Transform/consolidation rows are
# retained by storage for forward-compatible audit, but remain quarantined until
# causal rewiring is implemented in Wave 40.
VALIDATED_ACT_PROVENANCE = frozenset({"validated_act_residue"})
LEGACY_UNKNOWN_PROVENANCE = frozenset(
    {"", "legacy", "legacy_unknown", "provenance_unknown"}
)
REDUCTION_CAUSAL_ROLES = frozenset(
    {
        "carrier_reduction",
        "consolidated",
        "forgotten_rewire",
        "reduction",
        "reinterpreted",
    }
)
ACT_RESIDUE_CAUSAL_ROLES = frozenset(
    {
        "choice",
        "updated_belief",
        "goal",
        "constraint",
        "unresolved_tension",
        "affective_coordinate",
    }
)


class CarrierSupport(TypedDict):
    trace_id: str
    cognitive_act_id: str | None
    line_provenance: str
    classification: Literal["validated", "quarantine"]
    created_at: str
    content_hash: str
    excerpt: str
    weight: float
    causal_rank: int
    embedding_available: bool
    causal_role: str
    predecessor_ids: list[str]
    root_id: str | None
    covered_count: int
    covered_hash: str
    source_seq: int
    residue_ordinal: int


class CausalCarrierProjection(TypedDict):
    projection_status: ProjectionStatus
    projection_available: bool
    coverage_hash: str
    coverage_count: int
    root_coverage_hash: str
    root_count: int
    covered_node_count: int
    carrier_text: str
    supports: list[CarrierSupport]
    technical_strength: float
    coherence: float | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Trace:
    trace_id: str
    content: str
    content_hash: str
    embedding: tuple[float, ...] | None
    embedding_hash: str | None
    embedding_supplied: bool
    embedding_invalid: bool
    created_at: str
    created_at_invalid: bool
    line_provenance: str
    cognitive_act_id: str | None
    causal_role: str
    predecessor_ids: tuple[str, ...]
    root_id: str | None
    source_seq: int | None
    residue_ordinal: int | None
    source_coordinate_invalid: bool
    affect_hash: str
    classification: Literal["validated", "quarantine"]
    unfenced_validated: bool


@dataclass(frozen=True, slots=True)
class _Candidate:
    trace: _Trace
    causal_rank: int
    weight: float
    embedding_available: bool


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _canonical_value_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        encoded = _text(value)
    return _sha256_text(encoded)


def _canonical_created_at(value: Any) -> tuple[str, bool]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw, True
    elif value is None:
        return "", True
    else:
        return _text(value), True

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z"), False


def _embedding(value: Any) -> tuple[tuple[float, ...] | None, bool, bool]:
    """Return normalized input, supplied flag, invalid flag.

    Missing embeddings are a supported state.  Malformed, non-finite, zero or
    unreasonably wide vectors are ignored for coherence but never remove the
    row's textual contribution.
    """

    if value is None:
        return None, False, False
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None, True, True
    if not value or len(value) > MAX_EMBEDDING_DIMENSIONS:
        return None, True, True
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None, True, True
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            return None, True, True
        if not math.isfinite(number):
            return None, True, True
        out.append(number)
    if math.sqrt(sum(number * number for number in out)) == 0.0:
        return None, True, True
    return tuple(out), True, False


def _source_integer(value: Any, *, minimum: int) -> tuple[int | None, bool]:
    """Parse an exact integer causal coordinate without bool/string coercion."""

    if isinstance(value, bool):
        return None, True
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None, True
    if parsed < minimum or str(parsed) != str(value).strip():
        return None, True
    return parsed, False


def _normalize_trace(row: Mapping[str, Any]) -> _Trace:
    trace_id = _text(row.get("id"))
    content = _text(row.get("content"))
    created_at, created_at_invalid = _canonical_created_at(row.get("created_at"))
    provenance = _text(row.get("line_provenance")).strip().lower()
    act_id_text = _text(row.get("cognitive_act_id")).strip()
    act_id = act_id_text or None
    raw_role = row.get("causal_role")
    if raw_role is None:
        raw_role = row.get("residue_causal_role")
    causal_role = _text(raw_role or "residue").strip().lower()
    raw_predecessors = row.get("predecessor_ids")
    if raw_predecessors is None:
        raw_predecessors = row.get("predecessors")
    if raw_predecessors is None:
        raw_predecessors = row.get("residue_predecessors")
    if isinstance(raw_predecessors, Sequence) and not isinstance(
        raw_predecessors, (str, bytes, bytearray)
    ):
        predecessor_ids = tuple(sorted({
            text
            for value in raw_predecessors
            if (text := _text(value).strip())
        }))
    else:
        predecessor_ids = ()
    raw_root = row.get("root_id")
    if raw_root is None:
        raw_root = row.get("line_root_hash")
    if raw_root is None:
        raw_root = row.get("residue_line_root_hash")
    root_text = _text(raw_root).strip()
    root_id = root_text or None
    source_seq, seq_invalid = _source_integer(row.get("seq"), minimum=1)
    residue_ordinal, ordinal_invalid = _source_integer(
        row.get("residue_ordinal"), minimum=0
    )
    if residue_ordinal is not None and residue_ordinal > 3:
        residue_ordinal = None
        ordinal_invalid = True
    vector, supplied, invalid = _embedding(row.get("embedding"))

    claims_validation = provenance in VALIDATED_ACT_PROVENANCE
    validated = bool(
        claims_validation
        and act_id is not None
        and source_seq is not None
        and residue_ordinal is not None
        and causal_role in ACT_RESIDUE_CAUSAL_ROLES
    )
    classification: Literal["validated", "quarantine"] = (
        "validated" if validated else "quarantine"
    )
    embedding_hash = None
    if vector is not None:
        embedding_hash = _sha256_text(
            json.dumps(vector, ensure_ascii=True, separators=(",", ":"))
        )
    return _Trace(
        trace_id=trace_id,
        content=content,
        content_hash=_sha256_text(content),
        embedding=vector,
        embedding_hash=embedding_hash,
        embedding_supplied=supplied,
        embedding_invalid=invalid,
        created_at=created_at,
        created_at_invalid=created_at_invalid,
        line_provenance=provenance,
        cognitive_act_id=act_id,
        causal_role=causal_role,
        predecessor_ids=predecessor_ids,
        root_id=root_id,
        source_seq=source_seq,
        residue_ordinal=residue_ordinal,
        source_coordinate_invalid=seq_invalid or ordinal_invalid,
        affect_hash=_canonical_value_hash(row.get("residue_affect") or {}),
        classification=classification,
        unfenced_validated=claims_validation and not validated,
    )


def _trace_sort_key(trace: _Trace) -> tuple[int, int, str, str, str, str, str]:
    # Source sequence is the primary retained-line coordinate.  Timestamp and
    # embedding are audit/retrieval coordinates and cannot reorder active
    # semantics.  Ordinal is explicit so sibling residues remain deterministic.
    return (
        trace.source_seq if trace.source_seq is not None else 2**63 - 1,
        trace.residue_ordinal if trace.residue_ordinal is not None else 2**15 - 1,
        trace.cognitive_act_id or "",
        trace.trace_id,
        trace.content_hash,
        trace.line_provenance,
        trace.causal_role,
    )


def _coverage_hash(traces: Sequence[_Trace]) -> str:
    """Hash only coordinates that can change retained causal semantics.

    Capture time and embedding are deliberately absent.  They remain useful
    diagnostics, but changing either must not manufacture a different causal
    line or a different model-visible carrier.
    """
    digest = hashlib.sha256()
    digest.update(f"{ALGORITHM_VERSION}\n{len(traces)}\n".encode())
    for trace in traces:
        leaf = {
            "cognitive_act_id": trace.cognitive_act_id,
            "content_hash": trace.content_hash,
            "id": trace.trace_id,
            "line_provenance": trace.line_provenance,
            "causal_role": trace.causal_role,
            "predecessor_ids": trace.predecessor_ids,
            "root_id": trace.root_id,
            "seq": trace.source_seq,
            "residue_ordinal": trace.residue_ordinal,
            "affect_hash": trace.affect_hash,
        }
        encoded = json.dumps(
            leaf, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _topological_order(
    traces: Sequence[_Trace],
) -> tuple[list[_Trace], int, int, int, dict[str, _Trace]]:
    """Order known predecessor edges; report cycles/dangling/ambiguous ids."""

    stable = sorted(traces, key=_trace_sort_key)
    id_indices: dict[str, list[int]] = {}
    for index, trace in enumerate(stable):
        if trace.trace_id:
            id_indices.setdefault(trace.trace_id, []).append(index)
    unique = {
        trace_id: stable[indices[0]]
        for trace_id, indices in id_indices.items()
        if len(indices) == 1
    }
    unique_index = {
        trace_id: indices[0]
        for trace_id, indices in id_indices.items()
        if len(indices) == 1
    }
    successors: list[list[int]] = [[] for _ in stable]
    indegree = [0 for _ in stable]
    dangling = 0
    ambiguous = 0
    for child_index, trace in enumerate(stable):
        for predecessor_id in trace.predecessor_ids:
            indices = id_indices.get(predecessor_id)
            if indices is None:
                dangling += 1
                continue
            if len(indices) != 1:
                ambiguous += 1
                continue
            predecessor_index = unique_index[predecessor_id]
            successors[predecessor_index].append(child_index)
            indegree[child_index] += 1

    ready = [
        (_trace_sort_key(stable[index]), index)
        for index, degree in enumerate(indegree)
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered_indices: list[int] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered_indices.append(index)
        for successor in successors[index]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (_trace_sort_key(stable[successor]), successor))

    cycle_count = len(stable) - len(ordered_indices)
    if cycle_count:
        ordered_set = set(ordered_indices)
        ordered_indices.extend(
            index for index in range(len(stable)) if index not in ordered_set
        )
    return [stable[index] for index in ordered_indices], cycle_count, dangling, ambiguous, unique


def _semantic_frontier(
    traces: Sequence[_Trace], unique: Mapping[str, _Trace]
) -> tuple[list[_Trace], dict[str, frozenset[str]]]:
    """Return roots whose text carries the retained line and their ancestry.

    A predecessor edge is semantic reduction ancestry only when its child is a
    validated reduction node.  Ordinary causal succession never implies that a
    later residue summarizes an earlier one.
    """

    cache: dict[str, frozenset[str]] = {}
    visiting: set[str] = set()

    def closure(trace: _Trace) -> frozenset[str]:
        cached = cache.get(trace.trace_id)
        if cached is not None:
            return cached
        if trace.trace_id in visiting:
            return frozenset()
        visiting.add(trace.trace_id)
        covered: set[str] = set()
        if (
            trace.classification == "validated"
            and trace.causal_role in REDUCTION_CAUSAL_ROLES
        ):
            for predecessor_id in trace.predecessor_ids:
                predecessor = unique.get(predecessor_id)
                # Quarantined history cannot be silently promoted by a newer
                # residue.  It remains a separate quoted frontier root until
                # an explicit attestation contract exists.
                if (
                    predecessor is None
                    or predecessor.classification != "validated"
                ):
                    continue
                covered.add(predecessor_id)
                if (
                    predecessor.classification == "validated"
                    and predecessor.causal_role in REDUCTION_CAUSAL_ROLES
                ):
                    covered.update(closure(predecessor))
        visiting.remove(trace.trace_id)
        result = frozenset(covered)
        cache[trace.trace_id] = result
        return result

    reduced_ids: set[str] = set()
    for trace in traces:
        reduced_ids.update(closure(trace))
    frontier = [trace for trace in traces if trace.trace_id not in reduced_ids]
    root_ancestry: dict[str, frozenset[str]] = {
        trace.trace_id: frozenset({trace.trace_id, *closure(trace)})
        for trace in frontier
    }

    return frontier, root_ancestry


def _root_coverage_hash(
    frontier: Sequence[_Trace],
    root_ancestry: Mapping[str, frozenset[str]],
    traces: Sequence[_Trace],
) -> str:
    by_id = {trace.trace_id: trace for trace in traces}
    digest = hashlib.sha256()
    digest.update(f"{ALGORITHM_VERSION}:roots:{len(frontier)}\n".encode())
    for root in frontier:
        covered_leaves = []
        for trace_id in sorted(root_ancestry.get(root.trace_id, frozenset())):
            trace = by_id.get(trace_id)
            covered_leaves.append(
                {
                    "id": trace_id,
                    "content_hash": trace.content_hash if trace is not None else None,
                    "line_provenance": (
                        trace.line_provenance if trace is not None else None
                    ),
                    "cognitive_act_id": (
                        trace.cognitive_act_id if trace is not None else None
                    ),
                    "causal_role": trace.causal_role if trace is not None else None,
                    "predecessor_ids": (
                        trace.predecessor_ids if trace is not None else None
                    ),
                    "line_root": trace.root_id if trace is not None else None,
                    "seq": trace.source_seq if trace is not None else None,
                    "residue_ordinal": (
                        trace.residue_ordinal if trace is not None else None
                    ),
                    "affect_hash": trace.affect_hash if trace is not None else None,
                }
            )
        item = {
            "root_id": root.trace_id,
            "root_content_hash": root.content_hash,
            "line_root": root.root_id,
            "causal_role": root.causal_role,
            "covered_leaves": covered_leaves,
        }
        encoded = json.dumps(
            item, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def _coherence_data(
    traces: Sequence[_Trace],
) -> tuple[float | None, int | None, tuple[float, ...] | None, int]:
    dimensions: dict[int, int] = {}
    for trace in traces:
        if trace.embedding is not None:
            dimensions[len(trace.embedding)] = dimensions.get(len(trace.embedding), 0) + 1
    if not dimensions:
        return None, None, None, 0
    # Most common dimension wins; the smaller dimension is the deterministic
    # tie-breaker. Other vectors remain in coverage but are diagnostic-only.
    dimension = min(dimensions, key=lambda size: (-dimensions[size], size))
    vectors = [
        _unit(trace.embedding)
        for trace in traces
        if trace.embedding is not None and len(trace.embedding) == dimension
    ]
    centroid = tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    )
    resultant = math.sqrt(sum(value * value for value in centroid))
    normalized_centroid = _unit(centroid) if resultant > 0.0 else None
    incompatible = sum(
        1
        for trace in traces
        if trace.embedding is not None and len(trace.embedding) != dimension
    )
    return round(min(1.0, resultant), 6), dimension, normalized_centroid, incompatible


def _weighted_candidates(
    traces: Sequence[_Trace], centroid: tuple[float, ...] | None, dimension: int | None
) -> list[_Candidate]:
    total = len(traces)
    candidates: list[_Candidate] = []
    for index, trace in enumerate(traces):
        recency = 0.5 + 0.5 * ((index + 1) / total)
        provenance = 1.0 if trace.classification == "validated" else 0.2
        embedding_available = bool(
            trace.embedding is not None
            and dimension is not None
            and len(trace.embedding) == dimension
        )
        alignment = 1.0
        if embedding_available and centroid is not None and trace.embedding is not None:
            cosine = sum(
                left * right for left, right in zip(_unit(trace.embedding), centroid)
            )
            alignment = 0.75 + 0.25 * ((max(-1.0, min(1.0, cosine)) + 1.0) / 2.0)
        content_factor = 1.0 if trace.content else 0.0
        candidates.append(
            _Candidate(
                trace=trace,
                causal_rank=index,
                weight=round(provenance * recency * alignment * content_factor, 6),
                embedding_available=embedding_available,
            )
        )
    return candidates


def _stratified_select(candidates: Sequence[_Candidate], slots: int) -> list[_Candidate]:
    if slots <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item.causal_rank)
    if slots >= len(ordered):
        return ordered
    selected: list[_Candidate] = []
    for slot in range(slots):
        start = slot * len(ordered) // slots
        end = (slot + 1) * len(ordered) // slots
        bucket = ordered[start:end]
        selected.append(
            max(
                bucket,
                key=lambda item: (
                    item.weight,
                    item.causal_rank,
                    item.trace.trace_id,
                    item.trace.content_hash,
                ),
            )
        )
    return sorted(selected, key=lambda item: item.causal_rank)


def _bounded_excerpt(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    if limit <= 1:
        return content[:limit]
    return content[: limit - 1] + "…"


def _ids_hash(ids: Iterable[str]) -> str:
    encoded = json.dumps(
        sorted(set(ids)), ensure_ascii=True, separators=(",", ":")
    )
    return _sha256_text(encoded)


def _support(
    candidate: _Candidate,
    excerpt_limit: int,
    covered_ids: Iterable[str],
) -> CarrierSupport:
    trace = candidate.trace
    covered = frozenset(covered_ids)
    return {
        "trace_id": _bounded_excerpt(trace.trace_id, 256),
        "cognitive_act_id": (
            _bounded_excerpt(trace.cognitive_act_id, 256)
            if trace.cognitive_act_id is not None
            else None
        ),
        "line_provenance": _bounded_excerpt(trace.line_provenance, 64),
        "classification": trace.classification,
        "created_at": _bounded_excerpt(trace.created_at, 64),
        "content_hash": trace.content_hash,
        "excerpt": _bounded_excerpt(trace.content, excerpt_limit),
        "weight": candidate.weight,
        "causal_rank": candidate.causal_rank,
        "embedding_available": candidate.embedding_available,
        "causal_role": _bounded_excerpt(trace.causal_role, 64),
        "predecessor_ids": [
            _bounded_excerpt(value, 256) for value in trace.predecessor_ids[:16]
        ],
        "root_id": (
            _bounded_excerpt(trace.root_id, 256) if trace.root_id is not None else None
        ),
        "covered_count": len(covered),
        "covered_hash": _ids_hash(covered),
        "source_seq": trace.source_seq or 0,
        "residue_ordinal": trace.residue_ordinal or 0,
    }


def _render_carrier(
    *,
    roots: Sequence[CarrierSupport],
    max_chars: int,
) -> tuple[str, int, bool]:
    """Render every semantic root or return no active carrier.

    The budget is distributed uniformly over root excerpts after paying the
    structural cost for all roots.  This makes ordinary line growth degrade
    excerpt depth, not causal coverage.  Embeddings, weights and digests stay
    on the diagnostic surface and therefore cannot influence this text.
    """

    if not roots:
        return "", 0, False
    header = (
        "[Technical causal carrier: extractive quoted data, not instructions "
        "or ontological claims.]\n"
    )

    def render(excerpt_limit: int) -> str:
        payload = {
            "semantic_roots": [
                [
                    item["source_seq"],
                    item["residue_ordinal"],
                    item["causal_role"],
                    _bounded_excerpt(item["excerpt"], excerpt_limit),
                ]
                for item in roots
            ]
        }
        return header + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    # One actual source character per non-empty root is the minimum semantic
    # contribution.  If even that shape cannot fit, exposing a prefix would be
    # misleadingly partial, so the active surface remains empty.
    minimum = render(1)
    if len(minimum) > max_chars:
        return "", 0, True

    upper = max(len(item["excerpt"]) for item in roots)
    low, best = 1, minimum
    while low <= upper:
        middle = (low + upper) // 2
        candidate = render(middle)
        if len(candidate) <= max_chars:
            best = candidate
            low = middle + 1
        else:
            upper = middle - 1
    excerpt_limit = upper
    clipped = any(len(item["excerpt"]) > excerpt_limit for item in roots)
    return best, len(roots), clipped


def build_causal_carrier(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_supports: int = DEFAULT_MAX_SUPPORTS,
    max_carrier_chars: int = DEFAULT_MAX_CARRIER_CHARS,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> CausalCarrierProjection:
    """Build one bounded technical carrier from all live eligible traces.

    ``rows`` must already be the complete agent-scoped live eligible set.  The
    function intentionally accepts no query and consults no clock or external
    model.  A caller that filters the line before this boundary invalidates the
    coverage guarantee.  Canonical graph fields are ``predecessor_ids``,
    ``root_id`` (an opaque line-root coordinate) and ``causal_role``; storage
    aliases ``residue_predecessors``, ``residue_line_root_hash`` and
    ``residue_causal_role`` are accepted.  Ordinary predecessor edges order the
    line.  Only an allowlisted validated reduction role establishes semantic
    reduction ancestry.
    """

    if max_supports < 1:
        raise ValueError("max_supports must be at least 1")
    if max_carrier_chars < 512:
        raise ValueError("max_carrier_chars must be at least 512")
    if max_excerpt_chars < 1:
        raise ValueError("max_excerpt_chars must be at least 1")

    all_traces = sorted((_normalize_trace(row) for row in rows), key=_trace_sort_key)
    coverage_hash = _coverage_hash(all_traces)
    coverage_count = len(all_traces)
    empty_root_hash = _root_coverage_hash([], {}, [])
    if not all_traces:
        return {
            "projection_status": "empty",
            "projection_available": False,
            "coverage_hash": coverage_hash,
            "coverage_count": 0,
            "root_coverage_hash": empty_root_hash,
            "root_count": 0,
            "covered_node_count": 0,
            "carrier_text": "",
            "supports": [],
            "technical_strength": 0.0,
            "coherence": None,
            "diagnostics": {
                "algorithm_version": ALGORITHM_VERSION,
                "query_independent": True,
                "coverage_complete": True,
                "validated_count": 0,
                "quarantine_count": 0,
                "quarantine_excluded_count": 0,
                "legacy_unknown_count": 0,
                "unknown_provenance_count": 0,
                "unfenced_validated_count": 0,
                "malformed_row_count": 0,
                "active_malformed_row_count": 0,
                "empty_content_count": 0,
                "embedded_count": 0,
                "embedding_dimension": None,
                "invalid_embedding_count": 0,
                "active_invalid_embedding_count": 0,
                "incompatible_embedding_count": 0,
                "duplicate_id_count": 0,
                "cycle_node_count": 0,
                "dangling_predecessor_count": 0,
                "ambiguous_predecessor_count": 0,
                "line_root_coordinate_count": 0,
                "semantic_root_count": 0,
                "covered_by_reduction_count": 0,
                "root_coverage_complete": True,
                "selected_support_count": 0,
                "rendered_root_count": 0,
                "carrier_clipped": False,
            },
        }

    active_input = [
        trace for trace in all_traces if trace.classification == "validated"
    ]
    traces, cycle_count, dangling_count, ambiguous_count, unique = _topological_order(
        active_input
    )
    coherence, dimension, centroid, incompatible = _coherence_data(traces)
    candidates = _weighted_candidates(traces, centroid, dimension)
    ids = [trace.trace_id for trace in traces]
    duplicate_id_count = len(ids) - len(set(ids))
    missing_id_count = sum(not trace.trace_id for trace in traces)

    # A malformed identity graph cannot safely claim semantic subsumption.
    # Preserve every row as a frontier root and report degradation instead.
    if cycle_count or duplicate_id_count or missing_id_count:
        frontier = list(traces)
        root_ancestry = {
            trace.trace_id: frozenset({trace.trace_id}) for trace in frontier
        }
    else:
        frontier, root_ancestry = _semantic_frontier(traces, unique)
    root_coverage_hash = _root_coverage_hash(frontier, root_ancestry, traces)
    covered_ids = set().union(
        *(root_ancestry.get(trace.trace_id, frozenset()) for trace in frontier)
    )
    covered_node_count = len(covered_ids)
    root_count = len(frontier)
    root_coverage_complete = (
        covered_node_count == len(traces)
        and not duplicate_id_count
        and not missing_id_count
    )

    candidate_by_identity = {
        id(candidate.trace): candidate for candidate in candidates
    }
    frontier_candidates = [candidate_by_identity[id(trace)] for trace in frontier]
    validated = [
        candidate
        for candidate in frontier_candidates
        if candidate.trace.classification == "validated" and candidate.trace.content
    ]
    selected = _stratified_select(validated, max_supports)
    selected.sort(
        key=lambda item: (
            0 if item.trace.classification == "validated" else 1,
            item.causal_rank,
        )
    )
    supports = [
        _support(
            candidate,
            max_excerpt_chars,
            root_ancestry.get(candidate.trace.trace_id, {candidate.trace.trace_id}),
        )
        for candidate in selected
    ]
    # The carrier surface is constructed from every semantic frontier root.
    # ``supports`` above remains a bounded diagnostic sample only.
    root_supports = [
        _support(
            candidate,
            max_excerpt_chars,
            root_ancestry.get(candidate.trace.trace_id, {candidate.trace.trace_id}),
        )
        for candidate in frontier_candidates
        if candidate.trace.content
    ]

    malformed = sum(
        trace.created_at_invalid
        or not trace.trace_id
        or not trace.content
        or trace.unfenced_validated
        or trace.source_coordinate_invalid
        for trace in all_traces
    )
    active_malformed = sum(
        not trace.trace_id or not trace.content
        for trace in traces
    )
    invalid_embeddings = sum(trace.embedding_invalid for trace in all_traces)
    active_invalid_embeddings = sum(trace.embedding_invalid for trace in traces)
    unfenced_validated_count = sum(
        trace.unfenced_validated for trace in all_traces
    )
    validated_count = len(traces)
    structural_errors = (
        active_malformed
        + cycle_count
        + dangling_count
        + ambiguous_count
        + duplicate_id_count
        + (0 if root_coverage_complete else 1)
    )
    if structural_errors:
        status: ProjectionStatus = "degraded"
    elif validated_count:
        status = "ready"
    else:
        status = "provisional"

    technical_strength = (
        round(sum(candidate.weight for candidate in candidates) / validated_count, 6)
        if validated_count
        else 0.0
    )
    if root_supports:
        carrier_text, rendered_count, clipped = _render_carrier(
            roots=root_supports,
            max_chars=max_carrier_chars,
        )
    else:
        carrier_text, rendered_count, clipped = "", 0, False
    if rendered_count != root_count and status != "degraded":
        status = "degraded"
    projection_available = bool(
        root_count and rendered_count == root_count and carrier_text
    )
    return {
        "projection_status": status,
        "projection_available": projection_available,
        "coverage_hash": coverage_hash,
        "coverage_count": coverage_count,
        "root_coverage_hash": root_coverage_hash,
        "root_count": root_count,
        "covered_node_count": covered_node_count,
        "carrier_text": carrier_text,
        "supports": supports,
        "technical_strength": technical_strength,
        "coherence": coherence,
        "diagnostics": {
            "algorithm_version": ALGORITHM_VERSION,
            "query_independent": True,
            "coverage_complete": True,
            "validated_count": validated_count,
            "quarantine_count": coverage_count - validated_count,
            "quarantine_excluded_count": coverage_count - validated_count,
            "legacy_unknown_count": sum(
                trace.line_provenance in LEGACY_UNKNOWN_PROVENANCE
                for trace in all_traces
            ),
            "unknown_provenance_count": sum(
                trace.line_provenance not in VALIDATED_ACT_PROVENANCE
                and trace.line_provenance not in LEGACY_UNKNOWN_PROVENANCE
                for trace in all_traces
            ),
            "unfenced_validated_count": unfenced_validated_count,
            "malformed_row_count": malformed,
            "active_malformed_row_count": active_malformed,
            "empty_content_count": sum(not trace.content for trace in all_traces),
            "embedded_count": sum(
                trace.embedding is not None
                and dimension is not None
                and len(trace.embedding) == dimension
                for trace in traces
            ),
            "embedding_dimension": dimension,
            "invalid_embedding_count": invalid_embeddings,
            "active_invalid_embedding_count": active_invalid_embeddings,
            "incompatible_embedding_count": incompatible,
            "duplicate_id_count": duplicate_id_count,
            "cycle_node_count": cycle_count,
            "dangling_predecessor_count": dangling_count,
            "ambiguous_predecessor_count": ambiguous_count,
            "line_root_coordinate_count": sum(
                trace.root_id is not None for trace in all_traces
            ),
            "semantic_root_count": root_count,
            "covered_by_reduction_count": validated_count - root_count,
            "root_coverage_complete": root_coverage_complete,
            "selected_support_count": len(supports),
            "rendered_root_count": rendered_count,
            "carrier_clipped": clipped,
        },
    }


__all__ = [
    "ALGORITHM_VERSION",
    "CausalCarrierProjection",
    "CarrierSupport",
    "LEGACY_UNKNOWN_PROVENANCE",
    "ProjectionStatus",
    "REDUCTION_CAUSAL_ROLES",
    "VALIDATED_ACT_PROVENANCE",
    "build_causal_carrier",
]
