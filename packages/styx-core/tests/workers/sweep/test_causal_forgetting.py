"""Conservative automatic forgetting gates and PostgreSQL effects."""

from __future__ import annotations

import datetime as dt
import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from styx import turn_state
from styx.engine.causal_graph import causal_edge_hash, causal_node_hash
from styx.storage import migrate
from styx.storage.cognition import ensure_will_projection
from styx.workers.sweep.causal_forgetting import (
    CausalForgettingConfig,
    run_causal_forgetting_sweep,
)


@pytest.fixture
def db(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        yield conn


@pytest.fixture(autouse=True)
def reset_turn_state():
    turn_state.reset()
    yield
    turn_state.reset()


def _seed(
    conn: psycopg.Connection,
    agent: str,
    *,
    middle_embedding: bool = True,
) -> dict[str, uuid.UUID]:
    labels = "abc"
    ids = {label: uuid.uuid4() for label in labels}
    hashes = {
        label: causal_node_hash(
            node_kind="act_residue", content=f"memory-{label}",
            causal_role="choice", predecessor_hashes=[],
        )
        for label in labels
    }
    old = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=120)
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('styx.causal_operation','1',true)")
        for index, label in enumerate(labels):
            act_id = uuid.uuid4()
            cur.execute(
                "INSERT INTO cognitive_acts "
                "(id,agent_id,host_key,status,completed_at) "
                "VALUES (%s,%s,%s,'completed',clock_timestamp())",
                (act_id, agent, f"forget-test-{act_id}"),
            )
            embedding = None if label == "b" and not middle_embedding else (
                "[" + ",".join(str(value) for value in (
                    [1.0, index * 0.01] + [0.0] * 766
                )) + "]"
            )
            importance = 0.01 if label == "b" else 0.8
            cur.execute(
                "INSERT INTO memories ("
                "id,agent_id,role,visibility,kind,kind_src,content,embedding,"
                "created_at,last_accessed_at,importance_provisional,"
                "memory_domain,line_eligible,line_provenance,cognitive_act_id,"
                "residue_ordinal,residue_reducer_version,residue_input_hash,"
                "residue_causal_role,residue_confidence,residue_evidence,"
                "residue_predecessors,residue_line_root_hash,causal_node_hash,"
                "causal_node_kind,causal_payload_version,line_status) VALUES ("
                "%s,%s,'summary','shared','note','subjective',%s,%s,%s,%s,%s,"
                "'subjective_trace',true,'validated_act_residue',%s,0,'test_v1',"
                "%s,'choice',1.0,'[{\"source\":\"channel_output\","
                "\"key\":\"assistant_response\"}]'::jsonb,'[]'::jsonb,%s,%s,"
                "'act_residue','causal_node_v1','active')",
                (
                    ids[label], agent, f"memory-{label}", embedding, old, old,
                    importance, act_id, "a" * 64, "0" * 64, hashes[label],
                ),
            )
        for ordinal, (source, target) in enumerate((("a", "b"), ("b", "c"))):
            cur.execute(
                "INSERT INTO memory_lineage ("
                "agent_id,source_memory_id,target_memory_id,transform,ordinal,"
                "edge_key,edge_provenance,relation_version,source_node_hash,"
                "target_node_hash,valid_from_line_version,edge_hash) VALUES ("
                "%s,%s,%s,'incorporated',%s,%s,'validated',1,%s,%s,0,%s)",
                (
                    agent, ids[source], ids[target], ordinal,
                    f"forget-fixture:{source}:{target}", hashes[source],
                    hashes[target], causal_edge_hash(
                        source_hash=hashes[source], target_hash=hashes[target],
                        relation="incorporated",
                    ),
                ),
            )
    ensure_will_projection(conn, agent)
    conn.commit()
    return ids


def test_disabled_policy_is_noop(db) -> None:
    ids = _seed(db, "alpha")
    summary = run_causal_forgetting_sweep(
        db, config=CausalForgettingConfig(enabled=False),
    )
    assert summary.applied_operations == 0
    with db.cursor() as cur:
        cur.execute("SELECT line_status FROM memories WHERE id=%s", (ids["b"],))
        assert cur.fetchone()[0] == "active"


def test_eligible_middle_node_is_tombstoned_and_rewired(db) -> None:
    ids = _seed(db, "alpha")
    summary = run_causal_forgetting_sweep(
        db,
        config=CausalForgettingConfig(
            enabled=True, min_age_days=90, min_idle_days=30,
            relevance_ceiling=0.1, max_batch=1,
        ),
    )
    assert summary.applied_operations == 1
    assert summary.forgotten_nodes == 1
    assert summary.rewired_edges == 1
    with db.cursor() as cur:
        cur.execute("SELECT line_status FROM memories WHERE id=%s", (ids["b"],))
        assert cur.fetchone()[0] == "forgotten"
        cur.execute(
            "SELECT transform FROM memory_lineage WHERE source_memory_id=%s "
            "AND target_memory_id=%s AND valid_to_line_version IS NULL",
            (ids["a"], ids["c"]),
        )
        assert cur.fetchone()[0] == "retained_rewire"
        cur.execute(
            "SELECT feature_coordinates FROM causal_operations "
            "WHERE agent_id='alpha' AND operation_kind='forget'",
        )
        coordinates = cur.fetchone()[0]
        assert coordinates["policy_version"] == "causal_forgetting_v1"
        assert "memory" not in str(coordinates)


def test_missing_embedding_and_unready_carrier_are_not_forgetting_evidence(db) -> None:
    ids = _seed(db, "alpha", middle_embedding=False)
    summary = run_causal_forgetting_sweep(
        db, config=CausalForgettingConfig(enabled=True, relevance_ceiling=0.1),
    )
    assert summary.applied_operations == 0
    with db.cursor() as cur:
        cur.execute("UPDATE line_state SET dirty=true WHERE agent_id='alpha'")
    db.commit()
    summary = run_causal_forgetting_sweep(
        db, config=CausalForgettingConfig(enabled=True, relevance_ceiling=1.0),
    )
    assert summary.applied_operations == 0
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories WHERE id=ANY(%s) AND line_status='active'",
            (list(ids.values()),),
        )
        assert cur.fetchone()[0] == 3


def test_active_agent_is_not_changed(db) -> None:
    ids = _seed(db, "alpha")
    turn_state.observe("alpha")
    summary = run_causal_forgetting_sweep(
        db, config=CausalForgettingConfig(enabled=True, relevance_ceiling=0.1),
    )
    assert summary.applied_operations == 0
    assert summary.skipped_agents == 1
    with db.cursor() as cur:
        cur.execute("SELECT line_status FROM memories WHERE id=%s", (ids["b"],))
        assert cur.fetchone()[0] == "active"


def test_active_emotional_cause_lease_protects_its_residue(db) -> None:
    ids = _seed(db, "alpha")
    now = dt.datetime.now(tz=dt.timezone.utc)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_events (agent_id,occurred_at,source_kind,"
            "cause_status,metadata) VALUES (%s,%s,'cognitive_act_residue',"
            "'active',%s) RETURNING id",
            ("alpha", now, Jsonb({"residue_memory_id": str(ids["b"])})),
        )
        event_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO emotional_cause_status (agent_id,cause_event_id,at,"
            "status,lease_expires_at) VALUES (%s,%s,%s,'active',%s)",
            ("alpha", event_id, now, now + dt.timedelta(hours=1)),
        )
    db.commit()

    summary = run_causal_forgetting_sweep(
        db,
        config=CausalForgettingConfig(
            enabled=True, min_age_days=90, min_idle_days=30,
            relevance_ceiling=0.1,
        ),
        now=now,
    )

    assert summary.applied_operations == 0
    with db.cursor() as cur:
        cur.execute("SELECT line_status FROM memories WHERE id=%s", (ids["b"],))
        assert cur.fetchone()[0] == "active"
