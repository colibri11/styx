"""Pure planner and validator for the versioned causal graph (Wave 40).

IDs address rows, while semantic hashes deliberately exclude UUIDs, clocks and
embeddings.  The module has no database, model or wall-clock dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence


ALGORITHM_NAME = "nearest_retained_causal_graph"
ALGORITHM_VERSION = "causal_graph_v1"
EMPTY_GRAPH_ROOT = "0" * 64
LineStatus = Literal["active", "superseded", "forgotten", "quarantined"]


class CausalGraphError(ValueError):
    """The proposed graph cannot be represented as a validated DAG."""


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_hash: str
    line_status: LineStatus = "active"


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    edge_hash: str | None = None


@dataclass(frozen=True, slots=True)
class GraphValidation:
    topological_order: tuple[str, ...]
    roots: tuple[str, ...]
    frontier: tuple[str, ...]
    graph_root_hash: str


@dataclass(frozen=True, slots=True)
class ReplacementEdge:
    source_id: str
    target_id: str
    relation: Literal[
        "retained_rewire", "consolidated", "reinterpreted"
    ] = "retained_rewire"


@dataclass(frozen=True, slots=True)
class ForgetPlan:
    removed_node_ids: tuple[str, ...]
    close_edge_ids: tuple[str, ...]
    replacement_edges: tuple[ReplacementEdge, ...]
    predecessor_ids: tuple[str, ...]
    successor_ids: tuple[str, ...]
    output_root_hash: str


@dataclass(frozen=True, slots=True)
class TransformPlan:
    source_node_ids: tuple[str, ...]
    target_node_id: str
    close_edge_ids: tuple[str, ...]
    new_edges: tuple[ReplacementEdge, ...]
    output_root_hash: str


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def causal_node_hash(
    *,
    node_kind: str,
    content: str,
    causal_role: str,
    predecessor_hashes: Sequence[str],
    payload_version: str = "causal_node_v1",
) -> str:
    """Hash semantic payload only; IDs, clocks and embeddings are excluded."""
    return canonical_hash({
        "causal_role": causal_role,
        "content": content,
        "node_kind": node_kind,
        "payload_version": payload_version,
        "predecessor_hashes": sorted(predecessor_hashes),
    })


def causal_edge_hash(
    *, source_hash: str, target_hash: str, relation: str,
    relation_version: int = 1,
) -> str:
    # Keep the wire format deliberately small so migration backfills can
    # reproduce it byte-for-byte in PostgreSQL as well as Python.
    material = "\x1f".join((
        "causal_edge_v1", str(relation_version), source_hash,
        target_hash, relation,
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_hash(value: str, field: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise CausalGraphError(f"{field} must be a lowercase sha256")


def validate_graph(
    nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
) -> GraphValidation:
    node_list = list(nodes)
    edge_list = list(edges)
    by_id: dict[str, GraphNode] = {}
    for node in node_list:
        if not node.node_id or node.node_id in by_id:
            raise CausalGraphError("graph node ids must be non-empty and unique")
        _validate_hash(node.node_hash, "node_hash")
        if node.line_status not in {
            "active", "superseded", "forgotten", "quarantined",
        }:
            raise CausalGraphError("graph node status is invalid")
        by_id[node.node_id] = node

    live_ids = {
        node.node_id
        for node in node_list
        if node.line_status not in {"forgotten", "quarantined"}
    }
    incoming = {node_id: set() for node_id in live_ids}
    outgoing = {node_id: set() for node_id in live_ids}
    pairs: set[tuple[str, str, str]] = set()
    edge_semantics: list[dict[str, Any]] = []
    for edge in edge_list:
        if edge.source_id == edge.target_id:
            raise CausalGraphError("self edge is forbidden")
        if edge.source_id not in by_id or edge.target_id not in by_id:
            raise CausalGraphError("active edge is dangling")
        if edge.source_id not in live_ids or edge.target_id not in live_ids:
            raise CausalGraphError("active edge references a non-live node")
        pair = (edge.source_id, edge.target_id, edge.relation)
        if pair in pairs:
            raise CausalGraphError("duplicate active edge is forbidden")
        pairs.add(pair)
        outgoing[edge.source_id].add(edge.target_id)
        incoming[edge.target_id].add(edge.source_id)
        semantic = {
            "relation": edge.relation,
            "source_hash": by_id[edge.source_id].node_hash,
            "target_hash": by_id[edge.target_id].node_hash,
        }
        if edge.edge_hash is not None:
            _validate_hash(edge.edge_hash, "edge_hash")
            expected_edge_hash = causal_edge_hash(
                source_hash=by_id[edge.source_id].node_hash,
                target_hash=by_id[edge.target_id].node_hash,
                relation=edge.relation,
            )
            if edge.edge_hash != expected_edge_hash:
                raise CausalGraphError("edge_hash does not match edge semantics")
        edge_semantics.append(semantic)

    ready = sorted(node_id for node_id in live_ids if not incoming[node_id])
    indegree = {node_id: len(incoming[node_id]) for node_id in live_ids}
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for target in sorted(outgoing[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(ordered) != len(live_ids):
        raise CausalGraphError("causal graph contains a cycle")

    active_ids = {
        node.node_id for node in node_list if node.line_status == "active"
    }
    active_incoming = {node_id: set() for node_id in active_ids}
    active_outgoing = {node_id: set() for node_id in active_ids}
    for edge in edge_list:
        if edge.source_id in active_ids and edge.target_id in active_ids:
            active_outgoing[edge.source_id].add(edge.target_id)
            active_incoming[edge.target_id].add(edge.source_id)
    roots = tuple(sorted(
        node_id for node_id in active_ids if not active_incoming[node_id]
    ))
    frontier = tuple(
        sorted(node_id for node_id in active_ids if not active_outgoing[node_id])
    )
    root_hash = graph_root_hash(node_list, edge_list)
    return GraphValidation(tuple(ordered), roots, frontier, root_hash)


def graph_root_hash(
    nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
) -> str:
    active_nodes = [node for node in nodes if node.line_status == "active"]
    if not active_nodes:
        return EMPTY_GRAPH_ROOT
    by_id = {node.node_id: node for node in active_nodes}
    semantic_lines = [
        "\x1f".join(("N", node.node_hash, "active"))
        for node in active_nodes
    ]
    semantic_lines.extend(
        "\x1f".join((
            "E", by_id[edge.source_id].node_hash,
            by_id[edge.target_id].node_hash, edge.relation,
        ))
        for edge in edges
        if edge.source_id in by_id and edge.target_id in by_id
    )
    material = ALGORITHM_VERSION + "\n" + "\n".join(sorted(semantic_lines))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def plan_forgetting(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    remove_node_ids: Iterable[str],
) -> ForgetPlan:
    """Plan induced-subgraph removal with nearest retained predecessors."""
    by_id = {node.node_id: node for node in nodes}
    remove = set(remove_node_ids)
    if not remove or any(node_id not in by_id for node_id in remove):
        raise CausalGraphError("forget set must contain known nodes")
    if any(by_id[node_id].line_status != "active" for node_id in remove):
        raise CausalGraphError("only active nodes can be forgotten")
    active_count = sum(node.line_status == "active" for node in nodes)
    if len(remove) >= active_count:
        raise CausalGraphError("forgetting the last active node is forbidden")

    predecessors: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    successors: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    for edge in edges:
        predecessors[edge.target_id].add(edge.source_id)
        successors[edge.source_id].add(edge.target_id)

    boundary_successors = {
        target
        for removed in remove
        for target in successors[removed]
        if target not in remove and by_id[target].line_status == "active"
    }

    def nearest_retained(start: str) -> set[str]:
        found: set[str] = set()
        stack = [start]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for predecessor in predecessors[current]:
                if predecessor in remove:
                    stack.append(predecessor)
                elif by_id[predecessor].line_status == "active":
                    found.add(predecessor)
        return found

    replacement_pairs = {
        (predecessor, successor)
        for successor in boundary_successors
        for removed in remove
        if successor in successors[removed]
        for predecessor in nearest_retained(removed)
        if predecessor != successor
    }
    close = tuple(sorted(
        edge.edge_id
        for edge in edges
        if edge.source_id in remove or edge.target_id in remove
    ))
    retained_nodes = [
        GraphNode(
            node.node_id,
            node.node_hash,
            "forgotten" if node.node_id in remove else node.line_status,
        )
        for node in nodes
    ]
    retained_edges = [edge for edge in edges if edge.edge_id not in set(close)]
    retained_edges.extend(
        GraphEdge(
            edge_id=f"planned:{source}:{target}",
            source_id=source,
            target_id=target,
            relation="retained_rewire",
        )
        for source, target in sorted(replacement_pairs)
    )
    validation = validate_graph(retained_nodes, retained_edges)
    boundary_predecessors = {
        source
        for removed in remove
        for source in predecessors[removed]
        if source not in remove and by_id[source].line_status == "active"
    }
    return ForgetPlan(
        removed_node_ids=tuple(sorted(remove)),
        close_edge_ids=close,
        replacement_edges=tuple(
            ReplacementEdge(source, target)
            for source, target in sorted(replacement_pairs)
        ),
        predecessor_ids=tuple(sorted(boundary_predecessors)),
        successor_ids=tuple(sorted(boundary_successors)),
        output_root_hash=validation.graph_root_hash,
    )


def plan_transform(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    source_node_ids: Iterable[str],
    target: GraphNode,
    relation: Literal["consolidated", "reinterpreted"],
) -> TransformPlan:
    """Plan append-only consolidation/reinterpretation and successor transfer."""
    by_id = {node.node_id: node for node in nodes}
    sources = set(source_node_ids)
    if not sources or any(node_id not in by_id for node_id in sources):
        raise CausalGraphError("transform sources must be known")
    if target.node_id in by_id or target.node_id in sources:
        raise CausalGraphError("transform target must be a new node")
    if target.line_status != "active":
        raise CausalGraphError("transform target must be active")
    if any(by_id[node_id].line_status != "active" for node_id in sources):
        raise CausalGraphError("transform sources must be active")

    outward = {
        edge.target_id
        for edge in edges
        if edge.source_id in sources and edge.target_id not in sources
    }
    close = tuple(sorted(
        edge.edge_id
        for edge in edges
        if edge.source_id in sources and edge.target_id not in sources
    ))
    edge_specs: list[ReplacementEdge] = [
        ReplacementEdge(source, target.node_id, relation)
        for source in sorted(sources)
    ]
    edge_specs.extend(
        ReplacementEdge(target.node_id, successor, "retained_rewire")
        for successor in sorted(outward)
    )

    transformed_nodes = [
        GraphNode(
            node.node_id,
            node.node_hash,
            "superseded" if node.node_id in sources else node.line_status,
        )
        for node in nodes
    ] + [target]
    retained_edges = [edge for edge in edges if edge.edge_id not in set(close)]
    retained_edges.extend(
        GraphEdge(
            edge_id=f"planned:{item.source_id}:{item.target_id}:{item.relation}",
            source_id=item.source_id,
            target_id=item.target_id,
            relation=item.relation,
        )
        for item in edge_specs
    )
    validation = validate_graph(transformed_nodes, retained_edges)
    return TransformPlan(
        source_node_ids=tuple(sorted(sources)),
        target_node_id=target.node_id,
        close_edge_ids=close,
        new_edges=tuple(edge_specs),
        output_root_hash=validation.graph_root_hash,
    )


__all__ = [
    "ALGORITHM_NAME", "ALGORITHM_VERSION", "CausalGraphError",
    "EMPTY_GRAPH_ROOT", "ForgetPlan", "GraphEdge", "GraphNode",
    "GraphValidation", "ReplacementEdge", "TransformPlan", "canonical_hash",
    "causal_edge_hash", "causal_node_hash", "graph_root_hash",
    "plan_forgetting", "plan_transform", "validate_graph",
]
