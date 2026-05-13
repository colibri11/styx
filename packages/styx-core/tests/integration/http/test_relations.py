"""POST /relations/query, /graph/traverse, /link integration (волна 21).

TestClient + real Postgres. Использует тот же stack-фикстуру что
test_memory_store.py.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from styx.config import StyxConfig, load as load_config
from styx.http import registry
from styx.http.app import create_app
from styx.providers.memory import StyxMemoryCore
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


@pytest.fixture
def stack(clean_db: str):
    migrate.run(clean_db)
    cfg: StyxConfig = load_config()
    from dataclasses import replace
    cfg = replace(cfg, database_url=clean_db, http_token=None)

    agent = "alpha"
    core = StyxMemoryCore(agent_id=agent)
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    registry.reset_all()
    registry.register(agent_id=agent, core=core)

    app = create_app(cfg)
    client = TestClient(app)

    yield client, agent, clean_db

    core.shutdown()
    registry.reset_all()


def test_link_creates_idempotent(stack) -> None:
    client, agent, dsn = stack
    src, tgt = str(uuid.uuid4()), str(uuid.uuid4())
    payload = {
        "agent_id": agent,
        "source_type": "memory", "source_id": src,
        "target_type": "memory", "target_id": tgt,
        "relation": "custom",
    }
    resp1 = client.post("/link", json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["created"] is True

    resp2 = client.post("/link", json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["created"] is False


def test_relations_query_filter_by_relation(stack) -> None:
    client, agent, dsn = stack
    src = str(uuid.uuid4())
    tgt1, tgt2 = str(uuid.uuid4()), str(uuid.uuid4())
    client.post("/link", json={
        "agent_id": agent, "source_type": "memory", "source_id": src,
        "target_type": "memory", "target_id": tgt1, "relation": "related_to",
    })
    client.post("/link", json={
        "agent_id": agent, "source_type": "memory", "source_id": src,
        "target_type": "memory", "target_id": tgt2, "relation": "custom",
    })

    resp = client.post("/relations/query", json={
        "agent_id": agent, "source_id": src, "relation": "related_to",
    })
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["relation"] == "related_to"
    assert rows[0]["target_id"] == tgt1


def test_graph_traverse_finds_neighbors(stack) -> None:
    client, agent, dsn = stack
    # Pre-seed memories через прямой queries call.
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        a = q.insert_memory(role="summary", content="root node",
                            kind="note", kind_src="subjective")
        b = q.insert_memory(role="summary", content="child B",
                            kind="note", kind_src="subjective")
        c = q.insert_memory(role="summary", content="grandchild C",
                            kind="note", kind_src="subjective")
        conn.commit()

    # a → b → c
    client.post("/link", json={
        "agent_id": agent, "source_type": "memory", "source_id": str(a),
        "target_type": "memory", "target_id": str(b), "relation": "related_to",
    })
    client.post("/link", json={
        "agent_id": agent, "source_type": "memory", "source_id": str(b),
        "target_type": "memory", "target_id": str(c), "relation": "related_to",
    })

    # depth=2 → видим обоих b и c.
    resp = client.post("/graph/traverse", json={
        "agent_id": agent, "entity_id": str(a),
        "depth": 2, "limit": 20,
    })
    assert resp.status_code == 200
    body = resp.json()
    ids = {n["id"] for n in body["nodes"]}
    assert str(b) in ids
    assert str(c) in ids
    assert body["root"]["id"] == str(a)


def test_graph_traverse_404_unknown_entity(stack) -> None:
    client, agent, _ = stack
    resp = client.post("/graph/traverse", json={
        "agent_id": agent, "entity_id": str(uuid.uuid4()),
    })
    assert resp.status_code == 404


def test_graph_traverse_relation_filter(stack) -> None:
    """relation_filter в каждой ветке CTE: не пропускает через другой
    relation type."""
    client, agent, dsn = stack
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        a = q.insert_memory(role="summary", content="A",
                            kind="note", kind_src="subjective")
        b = q.insert_memory(role="summary", content="B",
                            kind="note", kind_src="subjective")
        c = q.insert_memory(role="summary", content="C",
                            kind="note", kind_src="subjective")
        conn.commit()

    # a --related_to--> b --custom--> c
    client.post("/link", json={
        "agent_id": agent, "source_type": "memory", "source_id": str(a),
        "target_type": "memory", "target_id": str(b), "relation": "related_to",
    })
    client.post("/link", json={
        "agent_id": agent, "source_type": "memory", "source_id": str(b),
        "target_type": "memory", "target_id": str(c), "relation": "custom",
    })

    resp = client.post("/graph/traverse", json={
        "agent_id": agent, "entity_id": str(a),
        "depth": 3, "relation_filter": "related_to",
    })
    assert resp.status_code == 200
    ids = {n["id"] for n in resp.json()["nodes"]}
    assert str(b) in ids
    assert str(c) not in ids  # не дотянулись через 'custom' ребро
