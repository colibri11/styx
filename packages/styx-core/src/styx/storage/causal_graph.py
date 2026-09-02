"""Transactional Wave 40 causal graph operations.

The caller owns the transaction.  Every operation is agent-scoped,
idempotent by ``operation_key`` and advances the semantic line once.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from styx.engine.causal_graph import (
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    CausalGraphError,
    GraphEdge,
    GraphNode,
    canonical_hash,
    causal_edge_hash,
    causal_node_hash,
    plan_forgetting,
    plan_transform,
    validate_graph,
)
from styx.storage.act_reduction import EMPTY_CAUSAL_ROOT, lock_agent_line
from styx.storage.cognition import redact_journal_metadata, redact_journal_text


_ERROR_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_OPERATION_KINDS = frozenset({
    "consolidate", "reinterpret", "forget", "rewire", "carrier_reduce",
})
_TRANSFORM_RELATION = {
    "consolidate": "consolidated",
    "reinterpret": "reinterpreted",
}


class CausalOperationConflict(ValueError):
    """An idempotency coordinate or frozen line snapshot changed."""


@dataclass(frozen=True, slots=True)
class CausalOperationResult:
    operation_id: uuid.UUID
    status: str
    output_line_version: int
    output_root_hash: str
    target_memory_ids: tuple[uuid.UUID, ...]
    duplicate: bool
    rewired_edge_count: int = 0
    tombstone_count: int = 0


def _uuid(value: uuid.UUID | str, field: str) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{field} must be a UUID") from exc


def _agent(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError("agent_id must contain 1..256 characters")
    return value


def _operation_key(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ValueError("operation_key must contain 1..512 characters")
    return value


def _vector_literal(values: Sequence[float] | None) -> str | None:
    if values is None:
        return None
    numbers = [float(item) for item in values]
    if not numbers or any(number != number or abs(number) == float("inf") for number in numbers):
        raise ValueError("embedding must contain finite numbers")
    return "[" + ",".join(str(number) for number in numbers) + "]"


def _line_state(conn: psycopg.Connection, agent_id: str) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO line_state(agent_id,version,dirty) VALUES (%s,0,true) "
            "ON CONFLICT (agent_id) DO NOTHING",
            (agent_id,),
        )
        cur.execute(
            "SELECT version,causal_root_hash FROM line_state "
            "WHERE agent_id=%s FOR UPDATE",
            (agent_id,),
        )
        row = cur.fetchone()
    assert row is not None
    return dict(row)


def load_causal_graph(
    conn: psycopg.Connection, agent_id: str,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Load the complete non-quarantined validated graph snapshot."""
    agent_id = _agent(agent_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,causal_node_hash,line_status FROM memories "
            "WHERE agent_id=%s AND line_provenance IN "
            "('validated_act_residue','validated_transform') "
            "AND line_status NOT IN ('forgotten','quarantined') "
            "ORDER BY causal_node_hash,id",
            (agent_id,),
        )
        rows = list(cur.fetchall())
        cur.execute(
            "SELECT id,source_memory_id,target_memory_id,transform,edge_hash "
            "FROM memory_lineage WHERE agent_id=%s "
            "AND edge_provenance='validated' "
            "AND valid_to_line_version IS NULL ORDER BY edge_key,id",
            (agent_id,),
        )
        edge_rows = list(cur.fetchall())
    nodes = [
        GraphNode(str(row["id"]), str(row["causal_node_hash"]), row["line_status"])
        for row in rows
    ]
    edges = [
        GraphEdge(
            str(row["id"]), str(row["source_memory_id"]),
            str(row["target_memory_id"]), str(row["transform"]),
            str(row["edge_hash"]),
        )
        for row in edge_rows
    ]
    return nodes, edges


def _existing_operation(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    operation_key: str,
    request_hash: str,
) -> CausalOperationResult | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,status,request_hash,output_line_version,output_root_hash," 
            "target_count FROM causal_operations "
            "WHERE agent_id=%s AND operation_key=%s FOR UPDATE",
            (agent_id, operation_key),
        )
        row = cur.fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise CausalOperationConflict(
                "operation_key was already used for a different request"
            )
        if row["status"] not in {"applied", "noop"}:
            raise CausalOperationConflict(
                f"operation retry found non-terminal status {row['status']}"
            )
        cur.execute(
            "SELECT id FROM memories WHERE agent_id=%s "
            "AND causal_operation_id=%s ORDER BY id",
            (agent_id, row["id"]),
        )
        targets = tuple(item["id"] for item in cur.fetchall())
        cur.execute(
            "SELECT count(*)::int AS count FROM memory_lineage "
            "WHERE agent_id=%s AND operation_id=%s "
            "AND transform='retained_rewire'",
            (agent_id, row["id"]),
        )
        rewired = int(cur.fetchone()["count"])
        cur.execute(
            "SELECT count(*)::int AS count FROM memory_tombstones "
            "WHERE agent_id=%s AND operation_id=%s",
            (agent_id, row["id"]),
        )
        tombstones = int(cur.fetchone()["count"])
    return CausalOperationResult(
        row["id"], row["status"], int(row["output_line_version"]),
        str(row["output_root_hash"]), targets, True, rewired, tombstones,
    )


def _insert_operation(
    conn: psycopg.Connection,
    *,
    operation_id: uuid.UUID,
    agent_id: str,
    operation_key: str,
    operation_kind: str,
    line_version: int,
    root_hash: str,
    request_hash: str,
    source_count: int,
    feature_coordinates: Mapping[str, Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO causal_operations ("
            "id,agent_id,operation_key,operation_kind,input_line_version,"
            "input_root_hash,request_hash,algorithm_name,algorithm_version,"
            "status,source_count,feature_coordinates) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'running',%s,%s)",
            (
                operation_id, agent_id, operation_key, operation_kind,
                line_version, root_hash, request_hash, ALGORITHM_NAME,
                ALGORITHM_VERSION, source_count,
                Jsonb(redact_journal_metadata(feature_coordinates or {})),
            ),
        )


def _insert_edge(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    operation_id: uuid.UUID | None,
    operation_key: str,
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    relation: str,
    source_hash: str,
    target_hash: str,
    line_version: int,
) -> None:
    edge_hash = causal_edge_hash(
        source_hash=source_hash,
        target_hash=target_hash,
        relation=relation,
    )
    edge_key = canonical_hash({
        "operation_key": operation_key,
        "relation": relation,
        "source_id": str(source_id),
        "target_id": str(target_id),
    })
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memory_lineage ("
            "agent_id,source_memory_id,target_memory_id,transform,ordinal,"
            "source_coordinates,edge_key,edge_provenance,operation_id,"
            "relation_version,source_node_hash,target_node_hash,"
            "valid_from_line_version,edge_hash) "
            "VALUES (%s,%s,%s,%s,0,'{}'::jsonb,%s,'validated',%s,1,%s,%s,%s,%s)",
            (
                agent_id, source_id, target_id, relation, edge_key,
                operation_id, source_hash, target_hash, line_version, edge_hash,
            ),
        )


def _finish_operation(
    conn: psycopg.Connection,
    *,
    operation_id: uuid.UUID,
    agent_id: str,
    input_line_version: int,
    input_root_hash: str,
    output_line_version: int,
    output_root_hash: str,
    frontier: Sequence[str],
    target_count: int,
) -> None:
    if len(frontier) > 64:
        raise CausalOperationConflict("causal frontier exceeds storage bound")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE line_state SET version=%s,causal_root_hash=%s,"
            "causal_frontier=%s,causal_root_version=%s,causal_root_act_id=NULL,"
            "causal_root_operation_id=%s,dirty=true,updated_at=clock_timestamp() "
            "WHERE agent_id=%s AND version=%s AND causal_root_hash=%s",
            (
                output_line_version, output_root_hash, Jsonb(list(frontier)),
                output_line_version, operation_id, agent_id,
                input_line_version, input_root_hash,
            ),
        )
        if cur.rowcount != 1:
            raise CausalOperationConflict("stale line root CAS")
        cur.execute(
            "UPDATE causal_operations SET status='applied',"
            "output_line_version=%s,output_root_hash=%s,target_count=%s,"
            "error_code=NULL,applied_at=clock_timestamp() WHERE id=%s",
            (output_line_version, output_root_hash, target_count, operation_id),
        )


def apply_causal_transform(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    operation_key: str,
    operation_kind: Literal["consolidate", "reinterpret"],
    source_memory_ids: Sequence[uuid.UUID | str],
    content: str,
    embedding: Sequence[float] | None,
    kind: str,
    visibility: str,
    metadata: Mapping[str, Any] | None = None,
    expected_line_version: int | None = None,
    expected_root_hash: str | None = None,
) -> CausalOperationResult:
    """Create one immutable transform node and atomically rewire successors."""
    agent_id = _agent(agent_id)
    operation_key = _operation_key(operation_key)
    if operation_kind not in _TRANSFORM_RELATION:
        raise ValueError("operation_kind must be consolidate or reinterpret")
    source_ids = tuple(_uuid(value, "source_memory_id") for value in source_memory_ids)
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("source memories must be non-empty and unique")
    if not isinstance(content, str) or not 1 <= len(content) <= 2400:
        raise ValueError("content must contain 1..2400 characters")
    if kind not in {"fact", "episode", "decision", "concept", "note"}:
        raise ValueError("kind is invalid")
    if visibility not in {"shared", "private"}:
        raise ValueError("visibility is invalid")
    safe_content = redact_journal_text(content, limit=2400)
    vector_literal = _vector_literal(embedding)
    safe_metadata = redact_journal_metadata(metadata or {})
    request = {
        "content": safe_content,
        "embedding_hash": (
            hashlib.sha256(vector_literal.encode("utf-8")).hexdigest()
            if vector_literal is not None else None
        ),
        "kind": kind,
        "metadata": safe_metadata,
        "operation_kind": operation_kind,
        "source_memory_ids": [str(value) for value in source_ids],
        "visibility": visibility,
    }
    request_hash = canonical_hash(request)

    lock_agent_line(conn, agent_id)
    duplicate = _existing_operation(
        conn, agent_id=agent_id, operation_key=operation_key,
        request_hash=request_hash,
    )
    if duplicate is not None:
        return duplicate
    line = _line_state(conn, agent_id)
    if expected_line_version is not None and line["version"] != expected_line_version:
        raise CausalOperationConflict("stale input_line_version")
    if expected_root_hash is not None and line["causal_root_hash"] != expected_root_hash:
        raise CausalOperationConflict("stale input_root_hash")

    nodes, edges = load_causal_graph(conn, agent_id)
    validate_graph(nodes, edges)
    source_set = {str(value) for value in source_ids}
    by_id = {node.node_id: node for node in nodes}
    if any(value not in by_id for value in source_set):
        raise CausalOperationConflict("transform source is absent from validated graph")
    target_id = uuid.uuid4()
    node_kind = (
        "consolidation" if operation_kind == "consolidate" else "reinterpretation"
    )
    role = "consolidated" if operation_kind == "consolidate" else "reinterpreted"
    target_hash = causal_node_hash(
        node_kind=node_kind,
        content=safe_content,
        causal_role=role,
        predecessor_hashes=[by_id[value].node_hash for value in source_set],
    )
    target = GraphNode(str(target_id), target_hash, "active")
    plan = plan_transform(
        nodes, edges, source_node_ids=source_set, target=target,
        relation=_TRANSFORM_RELATION[operation_kind],  # type: ignore[arg-type]
    )
    operation_id = uuid.uuid4()
    new_version = int(line["version"]) + 1
    _insert_operation(
        conn, operation_id=operation_id, agent_id=agent_id,
        operation_key=operation_key, operation_kind=operation_kind,
        line_version=int(line["version"]), root_hash=str(line["causal_root_hash"]),
        request_hash=request_hash, source_count=len(source_ids),
    )
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('styx.causal_operation','1',true)")
        cur.execute(
            "INSERT INTO memories ("
            "id,agent_id,role,visibility,kind,kind_src,content,embedding,metadata,"
            "memory_domain,line_eligible,line_provenance,causal_node_hash,"
            "causal_node_kind,causal_payload_version,causal_operation_id,line_status) "
            "VALUES (%s,%s,'summary',%s,%s,'subjective',%s,%s,%s,"
            "'subjective_trace',true,'validated_transform',%s,%s,"
            "'causal_node_v1',%s,'active')",
            (
                target_id, agent_id, visibility, kind, safe_content,
                vector_literal, Jsonb(safe_metadata),
                target_hash, node_kind, operation_id,
            ),
        )
        if plan.close_edge_ids:
            cur.execute(
                "UPDATE memory_lineage SET valid_to_line_version=%s "
                "WHERE agent_id=%s AND id=ANY(%s::bigint[]) "
                "AND valid_to_line_version IS NULL",
                (new_version, agent_id, [int(value) for value in plan.close_edge_ids]),
            )
            if cur.rowcount != len(plan.close_edge_ids):
                raise CausalOperationConflict("transform edge set changed during apply")
        cur.execute(
            "UPDATE memories SET line_status='superseded',superseded_by=%s,"
            "updated_at=clock_timestamp() WHERE agent_id=%s AND id=ANY(%s) "
            "AND line_status='active'",
            (target_id, agent_id, list(source_ids)),
        )
        if cur.rowcount != len(source_ids):
            raise CausalOperationConflict("transform source set changed during apply")

    node_hashes = {node.node_id: node.node_hash for node in nodes}
    node_hashes[str(target_id)] = target_hash
    for edge in plan.new_edges:
        _insert_edge(
            conn, agent_id=agent_id, operation_id=operation_id,
            operation_key=operation_key, source_id=uuid.UUID(edge.source_id),
            target_id=uuid.UUID(edge.target_id), relation=edge.relation,
            source_hash=node_hashes[edge.source_id],
            target_hash=node_hashes[edge.target_id], line_version=new_version,
        )
    output_nodes, output_edges = load_causal_graph(conn, agent_id)
    validation = validate_graph(output_nodes, output_edges)
    if validation.graph_root_hash != plan.output_root_hash:
        raise CausalOperationConflict("applied transform graph differs from plan")
    _finish_operation(
        conn, operation_id=operation_id, agent_id=agent_id,
        input_line_version=int(line["version"]),
        input_root_hash=str(line["causal_root_hash"]),
        output_line_version=new_version,
        output_root_hash=validation.graph_root_hash,
        frontier=validation.frontier, target_count=1,
    )
    return CausalOperationResult(
        operation_id, "applied", new_version, validation.graph_root_hash,
        (target_id,), False,
        sum(edge.relation == "retained_rewire" for edge in plan.new_edges), 0,
    )


def apply_causal_forgetting(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    operation_key: str,
    memory_ids: Sequence[uuid.UUID | str],
    reason_code: str,
    feature_coordinates: Mapping[str, Any] | None = None,
    expected_line_version: int | None = None,
    expected_root_hash: str | None = None,
) -> CausalOperationResult:
    """Tombstone active nodes and atomically install nearest-retained edges."""
    agent_id = _agent(agent_id)
    operation_key = _operation_key(operation_key)
    if not isinstance(reason_code, str) or _ERROR_CODE.fullmatch(reason_code) is None:
        raise ValueError("reason_code is invalid")
    ids = tuple(_uuid(value, "memory_id") for value in memory_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("memory_ids must be non-empty and unique")
    safe_features = redact_journal_metadata(feature_coordinates or {})
    request_hash = canonical_hash({
        "feature_coordinates": safe_features,
        "memory_ids": [str(value) for value in ids],
        "operation_kind": "forget",
        "reason_code": reason_code,
    })
    lock_agent_line(conn, agent_id)
    duplicate = _existing_operation(
        conn, agent_id=agent_id, operation_key=operation_key,
        request_hash=request_hash,
    )
    if duplicate is not None:
        return duplicate
    line = _line_state(conn, agent_id)
    if expected_line_version is not None and line["version"] != expected_line_version:
        raise CausalOperationConflict("stale input_line_version")
    if expected_root_hash is not None and line["causal_root_hash"] != expected_root_hash:
        raise CausalOperationConflict("stale input_root_hash")
    nodes, edges = load_causal_graph(conn, agent_id)
    validate_graph(nodes, edges)
    plan = plan_forgetting(nodes, edges, [str(value) for value in ids])
    by_id = {node.node_id: node for node in nodes}
    incoming: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    outgoing: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    for edge in edges:
        incoming[edge.target_id].append(edge.source_id)
        outgoing[edge.source_id].append(edge.target_id)
    operation_id = uuid.uuid4()
    new_version = int(line["version"]) + 1
    _insert_operation(
        conn, operation_id=operation_id, agent_id=agent_id,
        operation_key=operation_key, operation_kind="forget",
        line_version=int(line["version"]), root_hash=str(line["causal_root_hash"]),
        request_hash=request_hash, source_count=len(ids),
        feature_coordinates=safe_features,
    )
    rewire_hash = canonical_hash({
        "closed": list(plan.close_edge_ids),
        "replacement": [
            [edge.source_id, edge.target_id, edge.relation]
            for edge in plan.replacement_edges
        ],
        "version": ALGORITHM_VERSION,
    })
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT set_config('styx.causal_operation','1',true)")
        if plan.close_edge_ids:
            cur.execute(
                "UPDATE memory_lineage SET valid_to_line_version=%s "
                "WHERE agent_id=%s AND id=ANY(%s::bigint[]) "
                "AND valid_to_line_version IS NULL",
                (new_version, agent_id, [int(value) for value in plan.close_edge_ids]),
            )
            if cur.rowcount != len(plan.close_edge_ids):
                raise CausalOperationConflict("forget edge set changed during apply")
        cur.execute(
            "SELECT id,content,causal_node_hash FROM memories "
            "WHERE agent_id=%s AND id=ANY(%s) AND line_status='active' "
            "ORDER BY id FOR UPDATE",
            (agent_id, list(ids)),
        )
        source_rows = list(cur.fetchall())
        if len(source_rows) != len(ids):
            raise CausalOperationConflict("forget source set changed during apply")
        for row in source_rows:
            node_id = str(row["id"])
            predecessor_hashes = sorted(
                by_id[value].node_hash for value in incoming[node_id]
                if value in by_id
            )
            successor_hashes = sorted(
                by_id[value].node_hash for value in outgoing[node_id]
                if value in by_id
            )
            cur.execute(
                "INSERT INTO memory_tombstones ("
                "agent_id,memory_id,causal_node_hash,content_hash,operation_id,"
                "reason_code,removed_line_version,predecessor_hashes,"
                "successor_hashes,rewire_hash) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    agent_id, row["id"], row["causal_node_hash"],
                    hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
                    operation_id, reason_code, new_version,
                    Jsonb(predecessor_hashes), Jsonb(successor_hashes), rewire_hash,
                ),
            )

    for edge in plan.replacement_edges:
        _insert_edge(
            conn, agent_id=agent_id, operation_id=operation_id,
            operation_key=operation_key, source_id=uuid.UUID(edge.source_id),
            target_id=uuid.UUID(edge.target_id), relation=edge.relation,
            source_hash=by_id[edge.source_id].node_hash,
            target_hash=by_id[edge.target_id].node_hash,
            line_version=new_version,
        )
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET line_status='forgotten',"
            "updated_at=clock_timestamp() WHERE agent_id=%s AND id=ANY(%s) "
            "AND line_status='active'",
            (agent_id, list(ids)),
        )
        if cur.rowcount != len(ids):
            raise CausalOperationConflict("forget source status changed during apply")
    output_nodes, output_edges = load_causal_graph(conn, agent_id)
    validation = validate_graph(output_nodes, output_edges)
    if validation.graph_root_hash != plan.output_root_hash:
        raise CausalOperationConflict("applied forgetting graph differs from plan")
    _finish_operation(
        conn, operation_id=operation_id, agent_id=agent_id,
        input_line_version=int(line["version"]),
        input_root_hash=str(line["causal_root_hash"]),
        output_line_version=new_version,
        output_root_hash=validation.graph_root_hash,
        frontier=validation.frontier, target_count=0,
    )
    return CausalOperationResult(
        operation_id, "applied", new_version, validation.graph_root_hash,
        (), False, len(plan.replacement_edges), len(ids),
    )


def causal_graph_stats(conn: psycopg.Connection, agent_id: str) -> dict[str, Any]:
    """Content-free observability for one agent's causal graph."""
    agent_id = _agent(agent_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT count(*) FILTER (WHERE line_status='active')::int active_nodes,"
            "count(*) FILTER (WHERE line_status='superseded')::int superseded_nodes,"
            "count(*) FILTER (WHERE line_status='forgotten')::int forgotten_nodes "
            "FROM memories WHERE agent_id=%s AND line_provenance IN "
            "('validated_act_residue','validated_transform')",
            (agent_id,),
        )
        nodes = cur.fetchone()
        cur.execute(
            "SELECT count(*)::int active_roots FROM memories node "
            "WHERE node.agent_id=%s AND node.line_provenance IN "
            "('validated_act_residue','validated_transform') "
            "AND node.line_status='active' AND NOT EXISTS ("
            " SELECT 1 FROM memory_lineage edge JOIN memories source "
            " ON source.agent_id=edge.agent_id "
            " AND source.id=edge.source_memory_id "
            " AND source.line_status='active' "
            " WHERE edge.agent_id=node.agent_id "
            " AND edge.target_memory_id=node.id "
            " AND edge.edge_provenance='validated' "
            " AND edge.valid_to_line_version IS NULL)",
            (agent_id,),
        )
        roots = cur.fetchone()
        cur.execute(
            "SELECT count(*) FILTER (WHERE valid_to_line_version IS NULL)::int "
            "active_edges,count(*) FILTER (WHERE transform='retained_rewire')::int "
            "rewired_edges FROM memory_lineage WHERE agent_id=%s "
            "AND edge_provenance='validated'",
            (agent_id,),
        )
        edges = cur.fetchone()
        cur.execute(
            "SELECT count(*)::int tombstones FROM memory_tombstones "
            "WHERE agent_id=%s",
            (agent_id,),
        )
        tombstones = cur.fetchone()
        cur.execute(
            "SELECT count(*) FILTER (WHERE status IN "
            "('pending','running','retryable'))::int pending_operations,"
            "count(*) FILTER (WHERE status='terminal_failure')::int "
            "failed_operations FROM causal_operations WHERE agent_id=%s",
            (agent_id,),
        )
        operations = cur.fetchone()
    return {
        **dict(nodes), **dict(roots), **dict(edges),
        **dict(tombstones), **dict(operations),
    }


__all__ = [
    "CausalOperationConflict", "CausalOperationResult",
    "apply_causal_forgetting", "apply_causal_transform",
    "causal_graph_stats", "load_causal_graph",
]
