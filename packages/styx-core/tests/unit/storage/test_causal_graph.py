"""PostgreSQL contract for Wave 40 causal graph operations."""

from __future__ import annotations

import concurrent.futures
import threading
import uuid

import psycopg
import pytest

from styx.engine.causal_graph import (
    CausalGraphError,
    GraphEdge,
    GraphNode,
    causal_edge_hash,
    causal_node_hash,
    validate_graph,
)
from styx.storage.causal_graph import (
    CausalOperationConflict,
    apply_causal_forgetting,
    apply_causal_transform,
    causal_graph_stats,
    load_causal_graph,
)
from styx.storage.queries import AgentScopedQueries


def _seed_graph(
    conn: psycopg.Connection,
    agent_id: str,
    labels: str,
    edge_pairs: list[tuple[str, str]],
) -> dict[str, uuid.UUID]:
    ids = {label: uuid.uuid4() for label in labels}
    hashes = {
        label: causal_node_hash(
            node_kind="act_residue",
            content=f"memory-{label}",
            causal_role="choice",
            predecessor_hashes=[],
        )
        for label in labels
    }
    with conn.cursor() as cur:
        # Fixture setup is one synthetic frozen snapshot, not a semantic
        # operation.  Suppress per-row line increments and leave the initial
        # zero coordinate for the first real operation to replace by CAS.
        cur.execute("SELECT set_config('styx.causal_operation','1',true)")
        for label in labels:
            act_id = uuid.uuid4()
            cur.execute(
                "INSERT INTO cognitive_acts ("
                "id,agent_id,host_key,status,channel_input,channel_output,completed_at) "
                "VALUES (%s,%s,%s,'completed','{}'::jsonb,'{}'::jsonb,"
                "clock_timestamp())",
                (act_id, agent_id, f"fixture-{label}-{act_id}"),
            )
            cur.execute(
                "INSERT INTO memories ("
                "id,agent_id,role,visibility,kind,kind_src,content,metadata,"
                "memory_domain,line_eligible,line_provenance,causal_node_hash,"
                "causal_node_kind,causal_payload_version,line_status,"
                "cognitive_act_id,residue_ordinal,residue_reducer_version,"
                "residue_input_hash,residue_causal_role,residue_confidence,"
                "residue_evidence,residue_predecessors,residue_line_root_hash) "
                "VALUES (%s,%s,'summary','shared','decision','subjective',%s,"
                "'{}'::jsonb,'subjective_trace',true,'validated_act_residue',"
                "%s,'act_residue','causal_node_v1','active',%s,0,'fixture_v1',"
                "%s,'choice',1.0,'[{\"source\":\"channel_output\","
                "\"key\":\"assistant_response\"}]'::jsonb,'[]'::jsonb,%s)",
                (
                    ids[label], agent_id, f"memory-{label}", hashes[label],
                    act_id, "f" * 64, "0" * 64,
                ),
            )
        for ordinal, (source, target) in enumerate(edge_pairs):
            relation = "incorporated"
            cur.execute(
                "INSERT INTO memory_lineage ("
                "agent_id,source_memory_id,target_memory_id,transform,ordinal,"
                "edge_key,edge_provenance,relation_version,source_node_hash,"
                "target_node_hash,valid_from_line_version,edge_hash) "
                "VALUES (%s,%s,%s,%s,%s,%s,'validated',1,%s,%s,0,%s)",
                (
                    agent_id, ids[source], ids[target], relation, ordinal,
                    f"fixture:{agent_id}:{source}:{target}", hashes[source],
                    hashes[target], causal_edge_hash(
                        source_hash=hashes[source],
                        target_hash=hashes[target], relation=relation,
                    ),
                ),
            )
    return ids


def test_reinterpret_is_append_only_atomic_and_exactly_idempotent(
    migrated_db: str,
) -> None:
    agent = f"causal-transform-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn, conn.transaction():
        ids = _seed_graph(conn, agent, "abc", [("a", "b"), ("b", "c")])
        result = apply_causal_transform(
            conn,
            agent,
            operation_key="reinterpret:b:v1",
            operation_kind="reinterpret",
            source_memory_ids=[ids["b"]],
            content="memory-b understood under later evidence",
            embedding=None,
            kind="decision",
            visibility="shared",
            metadata={"Authorization": "Bearer must-not-survive"},
            expected_line_version=0,
            expected_root_hash="0" * 64,
        )
        duplicate = apply_causal_transform(
            conn,
            agent,
            operation_key="reinterpret:b:v1",
            operation_kind="reinterpret",
            source_memory_ids=[ids["b"]],
            content="memory-b understood under later evidence",
            embedding=None,
            kind="decision",
            visibility="shared",
            metadata={"Authorization": "Bearer must-not-survive"},
        )

        assert result.status == "applied"
        assert result.output_line_version == 1
        assert len(result.target_memory_ids) == 1
        assert result.rewired_edge_count == 1
        assert duplicate.duplicate is True
        assert duplicate.operation_id == result.operation_id
        assert duplicate.target_memory_ids == result.target_memory_ids

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,line_status,content,metadata FROM memories "
                "WHERE agent_id=%s ORDER BY seq", (agent,),
            )
            rows = cur.fetchall()
            assert [row[1] for row in rows] == ["active", "superseded", "active", "active"]
            assert rows[-1][2] == "memory-b understood under later evidence"
            assert "must-not-survive" not in str(rows[-1][3])
            cur.execute(
                "SELECT transform,valid_to_line_version FROM memory_lineage "
                "WHERE agent_id=%s ORDER BY id", (agent,),
            )
            edges = cur.fetchall()
            assert ("incorporated", 1) in edges
            assert ("reinterpreted", None) in edges
            assert ("retained_rewire", None) in edges
            cur.execute(
                "SELECT version,causal_root_hash,causal_root_operation_id "
                "FROM line_state WHERE agent_id=%s", (agent,),
            )
            line = cur.fetchone()
            assert line == (1, result.output_root_hash, result.operation_id)

        lineage = AgentScopedQueries(
            conn, agent_id=agent,
        ).explain_causal_lineage(memory_id=result.target_memory_ids[0])
        assert lineage is not None
        assert lineage["node"]["line_status"] == "active"
        assert lineage["operation"]["operation_kind"] == "reinterpret"
        assert lineage["operation"]["status"] == "applied"
        assert {edge["transform"] for edge in lineage["edges"]} == {
            "reinterpreted", "retained_rewire",
        }
        assert "under later evidence" not in str(lineage)

        with pytest.raises(CausalOperationConflict, match="different request"):
            apply_causal_transform(
                conn,
                agent,
                operation_key="reinterpret:b:v1",
                operation_kind="reinterpret",
                source_memory_ids=[ids["b"]],
                content="changed payload",
                embedding=None,
                kind="decision",
                visibility="shared",
            )


def test_canonical_semantic_payload_is_immutable(migrated_db: str) -> None:
    agent = f"causal-immutable-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn:
        ids = _seed_graph(conn, agent, "ab", [("a", "b")])
        conn.commit()
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memories SET content='mutated' WHERE id=%s",
                        (ids["a"],),
                    )
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM memories WHERE id=%s", (ids["a"],))
            assert cur.fetchone()[0] == "memory-a"


@pytest.mark.parametrize(
    ("forgotten", "expected_rewires"),
    [(["b"], 1), (["a"], 0), (["c"], 1), (["b", "c"], 1)],
)
def test_forget_writes_tombstones_before_status_and_rewires_nearest_retained(
    migrated_db: str,
    forgotten: list[str],
    expected_rewires: int,
) -> None:
    agent = f"causal-forget-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn, conn.transaction():
        ids = _seed_graph(conn, agent, "abcd", [("a", "b"), ("b", "c"), ("c", "d")])
        result = apply_causal_forgetting(
            conn,
            agent,
            operation_key="forget:" + "-".join(forgotten),
            memory_ids=[ids[label] for label in forgotten],
            reason_code="bounded_relevance_v1",
            feature_coordinates={"relevance_ceiling": 0.1, "raw": "not retained"},
            expected_line_version=0,
            expected_root_hash="0" * 64,
        )
        assert result.output_line_version == 1
        assert result.tombstone_count == len(forgotten)
        assert result.rewired_edge_count == expected_rewires
        with conn.cursor() as cur:
            cur.execute(
                "SELECT memory_id,content_hash,predecessor_hashes,successor_hashes "
                "FROM memory_tombstones WHERE agent_id=%s ORDER BY memory_id",
                (agent,),
            )
            tombstones = cur.fetchall()
            assert len(tombstones) == len(forgotten)
            assert all(len(row[1]) == 64 for row in tombstones)
            cur.execute(
                "SELECT id,line_status FROM memories WHERE agent_id=%s", (agent,),
            )
            statuses = dict(cur.fetchall())
            assert all(statuses[ids[label]] == "forgotten" for label in forgotten)
        nodes, edges = load_causal_graph(conn, agent)
        validation = validate_graph(nodes, edges)
        assert validation.graph_root_hash == result.output_root_hash
        if forgotten == ["b"]:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE id=%s", (ids["b"],))
                cur.execute(
                    "SELECT count(*) FROM memory_tombstones "
                    "WHERE agent_id=%s AND memory_id=%s", (agent, ids["b"]),
                )
                assert cur.fetchone()[0] == 1


def test_stale_snapshot_and_last_node_leave_no_partial_operation(
    migrated_db: str,
) -> None:
    agent = f"causal-rollback-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn:
        ids = _seed_graph(conn, agent, "a", [])
        conn.commit()
        with pytest.raises(CausalOperationConflict, match="stale input_line_version"):
            with conn.transaction():
                apply_causal_forgetting(
                    conn, agent, operation_key="stale", memory_ids=[ids["a"]],
                    reason_code="test", expected_line_version=99,
                )
        with pytest.raises(CausalGraphError, match="last active"):
            with conn.transaction():
                apply_causal_forgetting(
                    conn, agent, operation_key="last", memory_ids=[ids["a"]],
                    reason_code="test",
                )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM causal_operations WHERE agent_id=%s", (agent,),
            )
            assert cur.fetchone()[0] == 0
            cur.execute(
                "SELECT count(*) FROM memory_tombstones WHERE agent_id=%s", (agent,),
            )
            assert cur.fetchone()[0] == 0


def test_observability_is_content_free(migrated_db: str) -> None:
    agent = f"causal-stats-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn, conn.transaction():
        _seed_graph(conn, agent, "ab", [("a", "b")])
        stats = causal_graph_stats(conn, agent)
    assert stats == {
        "active_nodes": 2,
        "active_roots": 1,
        "superseded_nodes": 0,
        "forgotten_nodes": 0,
        "active_edges": 1,
        "rewired_edges": 0,
        "tombstones": 0,
        "pending_operations": 0,
        "failed_operations": 0,
    }
    assert "memory" not in str(stats)


def test_storage_rejects_self_cross_agent_and_duplicate_active_edges(
    migrated_db: str,
) -> None:
    agent = f"causal-edge-guard-{uuid.uuid4()}"
    other = f"causal-edge-other-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn:
        ids = _seed_graph(conn, agent, "ab", [("a", "b")])
        other_ids = _seed_graph(conn, other, "x", [])
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,causal_node_hash FROM memories WHERE agent_id=%s",
                (agent,),
            )
            hashes = dict(cur.fetchall())
            cur.execute(
                "SELECT causal_node_hash FROM memories WHERE id=%s",
                (other_ids["x"],),
            )
            other_hash = cur.fetchone()[0]

        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_lineage (agent_id,source_memory_id,"
                    "target_memory_id,transform,edge_key,edge_provenance,"
                    "relation_version,source_node_hash,target_node_hash,"
                    "valid_from_line_version,edge_hash) VALUES ("
                    "%s,%s,%s,'incorporated',%s,'validated',1,%s,%s,0,%s)",
                    (
                        agent, ids["a"], ids["a"], f"self:{uuid.uuid4()}",
                        hashes[ids["a"]], hashes[ids["a"]],
                        causal_edge_hash(
                            source_hash=hashes[ids["a"]],
                            target_hash=hashes[ids["a"]],
                            relation="incorporated",
                        ),
                    ),
                )
                cur.execute("SET CONSTRAINTS ALL IMMEDIATE")

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_lineage (agent_id,source_memory_id,"
                    "target_memory_id,transform,edge_key,edge_provenance,"
                    "relation_version,source_node_hash,target_node_hash,"
                    "valid_from_line_version,edge_hash) VALUES ("
                    "%s,%s,%s,'incorporated',%s,'validated',1,%s,%s,0,%s)",
                    (
                        agent, ids["a"], other_ids["x"],
                        f"foreign:{uuid.uuid4()}", hashes[ids["a"]],
                        other_hash, causal_edge_hash(
                            source_hash=hashes[ids["a"]],
                            target_hash=other_hash, relation="incorporated",
                        ),
                    ),
                )
                cur.execute("SET CONSTRAINTS ALL IMMEDIATE")

        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_lineage (agent_id,source_memory_id,"
                    "target_memory_id,transform,edge_key,edge_provenance,"
                    "relation_version,source_node_hash,target_node_hash,"
                    "valid_from_line_version,edge_hash) VALUES ("
                    "%s,%s,%s,'incorporated',%s,'validated',1,%s,%s,0,%s)",
                    (
                        agent, ids["a"], ids["b"],
                        f"duplicate:{uuid.uuid4()}", hashes[ids["a"]],
                        hashes[ids["b"]], causal_edge_hash(
                            source_hash=hashes[ids["a"]],
                            target_hash=hashes[ids["b"]],
                            relation="incorporated",
                        ),
                    ),
                )


def test_concurrent_writers_serialize_and_stale_writer_rolls_back(
    migrated_db: str,
) -> None:
    agent = f"causal-concurrent-{uuid.uuid4()}"
    with psycopg.connect(migrated_db) as conn:
        ids = _seed_graph(conn, agent, "abc", [("a", "b"), ("b", "c")])
        conn.commit()

    barrier = threading.Barrier(2)

    def apply(label: str) -> str:
        with psycopg.connect(migrated_db) as conn:
            barrier.wait(timeout=5)
            try:
                with conn.transaction():
                    apply_causal_transform(
                        conn, agent,
                        operation_key=f"concurrent:{label}",
                        operation_kind="reinterpret",
                        source_memory_ids=[ids[label]],
                        content=f"concurrent replacement {label}",
                        embedding=None, kind="decision", visibility="shared",
                        expected_line_version=0,
                        expected_root_hash="0" * 64,
                    )
            except CausalOperationConflict as exc:
                return str(exc)
        return "applied"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(apply, ("b", "c")))

    assert sorted(outcome == "applied" for outcome in outcomes) == [False, True]
    assert any("stale input_line_version" in outcome for outcome in outcomes)
    with psycopg.connect(migrated_db) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*),min(status),max(status) FROM causal_operations "
            "WHERE agent_id=%s", (agent,),
        )
        assert cur.fetchone() == (1, "applied", "applied")
        cur.execute("SELECT version FROM line_state WHERE agent_id=%s", (agent,))
        assert cur.fetchone()[0] == 1
