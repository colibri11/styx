"""Deterministic Wave 40 causal graph planner tests."""

from __future__ import annotations

import pytest

from styx.engine.causal_carrier import build_causal_carrier
from styx.engine.causal_graph import (
    CausalGraphError,
    GraphEdge,
    GraphNode,
    causal_edge_hash,
    causal_node_hash,
    plan_forgetting,
    plan_transform,
    validate_graph,
)


def _node(name: str, *, status: str = "active", content: str | None = None):
    return GraphNode(
        name,
        causal_node_hash(
            node_kind="act_residue",
            content=content or name,
            causal_role="choice",
            predecessor_hashes=[],
        ),
        status,  # type: ignore[arg-type]
    )


def _edge(name: str, source: str, target: str, relation: str = "incorporated"):
    return GraphEdge(name, source, target, relation)


def test_validate_rejects_dangling_self_duplicate_and_cycle() -> None:
    nodes = [_node("a"), _node("b")]
    with pytest.raises(CausalGraphError, match="dangling"):
        validate_graph(nodes, [_edge("x", "a", "missing")])
    with pytest.raises(CausalGraphError, match="self"):
        validate_graph(nodes, [_edge("x", "a", "a")])
    with pytest.raises(CausalGraphError, match="duplicate"):
        validate_graph(nodes, [_edge("x", "a", "b"), _edge("y", "a", "b")])
    with pytest.raises(CausalGraphError, match="cycle"):
        validate_graph(nodes, [_edge("x", "a", "b"), _edge("y", "b", "a")])


def test_graph_root_ignores_ids_but_preserves_semantic_structure() -> None:
    first = validate_graph(
        [_node("a", content="left"), _node("b", content="right")],
        [_edge("edge-a", "a", "b")],
    )
    second = validate_graph(
        [_node("x", content="left"), _node("y", content="right")],
        [_edge("edge-b", "x", "y")],
    )
    assert first.graph_root_hash == second.graph_root_hash
    assert first.roots == ("a",)
    assert first.frontier == ("b",)


@pytest.mark.parametrize(
    ("removed", "expected_close", "expected_rewire"),
    [
        ({"b"}, {"ab", "bc"}, {("a", "c")}),
        ({"a"}, {"ab"}, set()),
        ({"d"}, {"cd"}, set()),
        ({"b", "c"}, {"ab", "bc", "cd"}, {("a", "d")}),
    ],
)
def test_forget_middle_root_leaf_and_connected_set(
    removed: set[str],
    expected_close: set[str],
    expected_rewire: set[tuple[str, str]],
) -> None:
    nodes = [_node(name) for name in "abcd"]
    edges = [
        _edge("ab", "a", "b"),
        _edge("bc", "b", "c"),
        _edge("cd", "c", "d"),
    ]
    plan = plan_forgetting(nodes, edges, removed)
    assert set(plan.close_edge_ids) == expected_close
    assert {
        (edge.source_id, edge.target_id) for edge in plan.replacement_edges
    } == expected_rewire
    assert len(plan.output_root_hash) == 64


def test_forgetting_last_active_node_is_rejected() -> None:
    with pytest.raises(CausalGraphError, match="last active"):
        plan_forgetting([_node("a")], [], ["a"])


def test_reinterpret_creates_successor_and_transfers_outgoing_edges() -> None:
    nodes = [_node("a"), _node("b"), _node("c")]
    edges = [_edge("ab", "a", "b"), _edge("bc", "b", "c")]
    target = _node("b-prime", content="new b")
    plan = plan_transform(
        nodes,
        edges,
        source_node_ids=["b"],
        target=target,
        relation="reinterpreted",
    )
    assert plan.close_edge_ids == ("bc",)
    assert {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in plan.new_edges
    } == {
        ("b", "b-prime", "reinterpreted"),
        ("b-prime", "c", "retained_rewire"),
    }


def test_consolidation_requires_complete_active_sources() -> None:
    nodes = [_node("a"), _node("b", status="superseded")]
    with pytest.raises(CausalGraphError, match="must be active"):
        plan_transform(
            nodes,
            [],
            source_node_ids=["a", "b"],
            target=_node("c"),
            relation="consolidated",
        )


def test_wave40_counterfactual_retained_effect_and_unique_loss() -> None:
    """The choice follows retained meaning, not mere node persistence."""

    def rows(nodes: list[GraphNode], content: dict[str, str]):
        return [
            {
                "id": node.node_id,
                "seq": index + 1,
                "content": content[node.node_id],
                "embedding": [float(index + 1), 1.0],
                "created_at": f"2026-09-0{index + 1}T00:00:00Z",
                "line_provenance": "validated_act_residue",
                "cognitive_act_id": f"act-{node.node_id}",
                "residue_ordinal": 0,
                "residue_causal_role": "choice",
                "causal_node_hash": node.node_hash,
                "causal_node_kind": "act_residue",
                "line_status": "active",
            }
            for index, node in enumerate(nodes)
            if node.line_status == "active"
        ]

    def edge_rows(edges: list[GraphEdge], nodes: list[GraphNode]):
        hashes = {node.node_id: node.node_hash for node in nodes}
        return [
            {
                "source_memory_id": edge.source_id,
                "target_memory_id": edge.target_id,
                "transform": edge.relation,
                "source_node_hash": hashes[edge.source_id],
                "target_node_hash": hashes[edge.target_id],
                "edge_hash": causal_edge_hash(
                    source_hash=hashes[edge.source_id],
                    target_hash=hashes[edge.target_id],
                    relation=edge.relation,
                ),
            }
            for edge in edges
        ]

    def decision(carrier: dict) -> str:
        text = str(carrier["carrier_text"])
        return "block" if "block deployment" in text else "deploy"

    held_content = {
        "a": "deployment requested",
        "b": "risk examined",
        "c": "deploy; successor retains the risk resolution",
    }
    held_nodes = [_node(name, content=held_content[name]) for name in "abc"]
    held_edges = [_edge("ab", "a", "b"), _edge("bc", "b", "c")]
    before_held = build_causal_carrier(
        rows(held_nodes, held_content), edges=edge_rows(held_edges, held_nodes),
    )
    held_plan = plan_forgetting(held_nodes, held_edges, ["b"])
    after_held_nodes = [held_nodes[0], held_nodes[2]]
    after_held_edges = [
        _edge("ac", edge.source_id, edge.target_id, edge.relation)
        for edge in held_plan.replacement_edges
    ]
    after_held = build_causal_carrier(
        rows(after_held_nodes, held_content),
        edges=edge_rows(after_held_edges, after_held_nodes),
    )
    assert decision(before_held) == decision(after_held) == "deploy"

    unique_content = {
        "a": "deployment requested",
        "b": "block deployment due to unique evidence",
        "c": "deploy after routine checks",
    }
    unique_nodes = [_node(name, content=unique_content[name]) for name in "abc"]
    unique_edges = [_edge("ac", "a", "c")]
    before_unique = build_causal_carrier(
        rows(unique_nodes, unique_content),
        edges=edge_rows(unique_edges, unique_nodes),
    )
    unique_plan = plan_forgetting(unique_nodes, unique_edges, ["b"])
    after_unique_nodes = [unique_nodes[0], unique_nodes[2]]
    after_unique = build_causal_carrier(
        rows(after_unique_nodes, unique_content),
        edges=edge_rows(unique_edges, after_unique_nodes),
    )
    assert unique_plan.replacement_edges == ()
    assert decision(before_unique) == "block"
    assert decision(after_unique) == "deploy"
