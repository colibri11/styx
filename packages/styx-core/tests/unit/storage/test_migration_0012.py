"""Upgrade contract for the Wave 40 causal graph schema."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from styx.engine.causal_graph import (
    GraphEdge,
    GraphNode,
    causal_edge_hash,
    validate_graph,
)
from styx.storage import migrate


@pytest.fixture
def pre_wave40_db(clean_db: str):
    migrations = migrate.discover_migrations()
    with psycopg.connect(clean_db) as conn:
        migrate.apply(conn, [item for item in migrations if item.name < "0012"])
    yield clean_db


def _apply_0012(conn: psycopg.Connection) -> None:
    migrate.apply(
        conn,
        [
            item for item in migrate.discover_migrations()
            if item.name == "0012_causal_rewiring.sql"
        ],
    )


def test_upgrade_attests_wave38_residues_but_quarantines_old_transform(
    pre_wave40_db: str,
) -> None:
    agent = "upgrade-agent"
    act_ids = [uuid.uuid4(), uuid.uuid4()]
    residue_ids = [uuid.uuid4(), uuid.uuid4()]
    with psycopg.connect(pre_wave40_db) as conn, conn.cursor() as cur:
        for index, act_id in enumerate(act_ids):
            cur.execute(
                "INSERT INTO cognitive_acts "
                "(id,agent_id,host_key,status,completed_at) "
                "VALUES (%s,%s,%s,'completed',clock_timestamp())",
                (act_id, agent, f"turn-{index}"),
            )
            cur.execute(
                "INSERT INTO memories ("
                "id,agent_id,role,kind,kind_src,content,memory_domain,line_eligible,"
                "cognitive_act_id,line_provenance,residue_ordinal,"
                "residue_reducer_version,residue_input_hash,residue_causal_role,"
                "residue_confidence,residue_evidence,residue_predecessors,"
                "residue_line_root_hash) VALUES ("
                "%s,%s,'summary','decision','subjective',%s,'subjective_trace',"
                "true,%s,'validated_act_residue',0,'v1',%s,'choice',1.0,"
                "'[{\"source\":\"channel_output\",\"key\":"
                "\"assistant_response\"}]'::jsonb,%s,%s)",
                (
                    residue_ids[index], agent, f"residue-{index}", act_id,
                    "a" * 64,
                    Jsonb([str(residue_ids[0])] if index else []), "b" * 64,
                ),
            )
        cur.execute(
            "INSERT INTO memory_lineage (agent_id,source_memory_id,"
            "target_memory_id,cognitive_act_id,transform) "
            "VALUES (%s,%s,%s,%s,'incorporated')",
            (agent, residue_ids[0], residue_ids[1], act_ids[1]),
        )
        cur.execute(
            "INSERT INTO memories (agent_id,role,kind,kind_src,content,"
            "memory_domain,line_eligible,line_provenance) VALUES ("
            "%s,'summary','note','subjective','old transform',"
            "'subjective_trace',true,'validated_transform') RETURNING id",
            (agent,),
        )
        transform_id = cur.fetchone()[0]
        conn.commit()

        _apply_0012(conn)
        cur.execute(
            "SELECT causal_node_hash,line_status FROM memories "
            "WHERE id=ANY(%s) ORDER BY id", (residue_ids,),
        )
        residues = cur.fetchall()
        assert len(residues) == 2
        assert all(len(row[0]) == 64 and row[1] == "active" for row in residues)
        cur.execute(
            "SELECT causal_node_hash,line_status FROM memories WHERE id=%s",
            (transform_id,),
        )
        assert cur.fetchone() == (None, "quarantined")
        cur.execute(
            "SELECT source_node_hash,target_node_hash,transform,edge_hash "
            "FROM memory_lineage WHERE source_memory_id=%s",
            (residue_ids[0],),
        )
        source_hash, target_hash, relation, edge_hash = cur.fetchone()
        assert edge_hash == causal_edge_hash(
            source_hash=source_hash, target_hash=target_hash,
            relation=relation,
        )
        cur.execute(
            "SELECT id::text,causal_node_hash,line_status FROM memories "
            "WHERE id=ANY(%s) ORDER BY id::text", (residue_ids,),
        )
        graph_nodes = [GraphNode(*row) for row in cur.fetchall()]
        cur.execute(
            "SELECT id::text,source_memory_id::text,target_memory_id::text,"
            "transform,edge_hash FROM memory_lineage "
            "WHERE edge_provenance='validated' "
            "AND valid_to_line_version IS NULL AND agent_id=%s",
            (agent,),
        )
        graph_edges = [GraphEdge(*row) for row in cur.fetchall()]
        validation = validate_graph(graph_nodes, graph_edges)
        cur.execute(
            "SELECT causal_root_hash,causal_frontier,causal_root_version,"
            "version,dirty FROM line_state WHERE agent_id=%s", (agent,),
        )
        root_hash, frontier, root_version, version, dirty = cur.fetchone()
        assert root_hash == validation.graph_root_hash
        assert tuple(frontier) == validation.frontier
        assert root_version == version
        assert dirty is True


def test_upgrade_guards_canonical_update_and_delete(pre_wave40_db: str) -> None:
    agent = "guard-agent"
    act_id = uuid.uuid4()
    memory_id = uuid.uuid4()
    with psycopg.connect(pre_wave40_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cognitive_acts "
            "(id,agent_id,host_key,status,completed_at) "
            "VALUES (%s,%s,'turn','completed',clock_timestamp())",
            (act_id, agent),
        )
        cur.execute(
            "INSERT INTO memories (id,agent_id,role,kind,kind_src,content,"
            "memory_domain,line_eligible,cognitive_act_id,line_provenance,"
            "residue_ordinal,residue_reducer_version,residue_input_hash,"
            "residue_causal_role,residue_confidence,residue_evidence,"
            "residue_predecessors,residue_line_root_hash) VALUES ("
            "%s,%s,'summary','note','subjective','immutable',"
            "'subjective_trace',true,%s,'validated_act_residue',0,'v1',%s,"
            "'choice',1.0,'[{\"source\":\"channel_output\","
            "\"key\":\"assistant_response\"}]'::jsonb,'[]'::jsonb,%s)",
            (memory_id, agent, act_id, "a" * 64, "b" * 64),
        )
        conn.commit()
        _apply_0012(conn)
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            with conn.transaction():
                cur.execute(
                    "UPDATE memories SET content='changed' WHERE id=%s",
                    (memory_id,),
                )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            with conn.transaction():
                cur.execute("DELETE FROM memories WHERE id=%s", (memory_id,))
