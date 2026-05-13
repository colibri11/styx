"""GET /analytics integration (волна 25).

Caller-scoped: один агент в response.agents. agent_id-isolation;
503 при disabled.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from styx.config import StyxConfig, load as load_config
from styx.http import registry
from styx.http.app import create_app
from styx.providers.memory import StyxMemoryCore
from styx.storage import migrate


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


@pytest.fixture
def stack(clean_db: str):
    migrate.run(clean_db)
    cfg: StyxConfig = load_config()
    cfg = replace(cfg, database_url=clean_db, http_token=None)
    agent = "alpha"
    core = StyxMemoryCore(agent_id=agent)
    core._config = cfg
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    registry.reset_all()
    registry.register(agent_id=agent, core=core)
    app = create_app(cfg)
    client = TestClient(app)
    yield client, agent, clean_db, core, cfg
    core.shutdown()
    registry.reset_all()


@pytest.fixture
def stack_disabled(clean_db: str, monkeypatch: pytest.MonkeyPatch):
    migrate.run(clean_db)
    monkeypatch.setenv("STYX_EXPLAIN_API_ENABLED", "0")
    cfg: StyxConfig = replace(
        load_config(), database_url=clean_db, http_token=None,
    )
    assert cfg.explain_api_enabled is False
    agent = "alpha"
    core = StyxMemoryCore(agent_id=agent)
    core._config = cfg
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    core._config = cfg
    registry.reset_all()
    registry.register(agent_id=agent, core=core)
    app = create_app(cfg)
    client = TestClient(app)
    yield client, agent
    core.shutdown()
    registry.reset_all()


def test_analytics_happy(stack) -> None:
    client, agent, _, core, _ = stack
    # Прямой insert через provider's queries — minёт gatekeeper и
    # обеспечивает детерминированный набор kind'ов.
    core._queries.insert_memory(
        role="summary", content="fact alpha unique", kind="fact",
        kind_src="subjective",
    )
    core._queries.insert_memory(
        role="summary", content="fact beta separate topic", kind="fact",
        kind_src="subjective",
    )
    core._queries.insert_memory(
        role="summary", content="note xi different content", kind="note",
        kind_src="subjective",
    )
    core._queries.insert_message(role="user", content="dialogue ping")
    core._conn.commit()

    resp = client.get(f"/analytics?agent_id={agent}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "agents" in body and len(body["agents"]) == 1
    a = body["agents"][0]
    assert a["agent_id"] == agent
    assert a["display_name"] is None
    assert a["memories_count"] == 4  # 3 subjective + 1 dialogue
    by_kind = a["memories_by_kind"]
    assert by_kind["fact"] == 2
    assert by_kind["note"] == 1
    assert by_kind["episode"] == 1  # dialogue → kind=episode default
    assert a["dialogue_messages_count"] == 1

    g = body["global"]
    assert g["total_memories"] == 4
    assert g["total_dialogue_messages"] == 1
    # database_size_bytes — int либо None (best-effort)
    assert g["database_size_bytes"] is None or isinstance(
        g["database_size_bytes"], int
    )

    pi = body["pending_indexing"]
    assert "memories" in pi and "chunks" in pi
    assert pi["dialogue_messages"] == 0  # Styx без dialogue_messages-таблицы


def test_analytics_agent_isolation(stack) -> None:
    client, agent, _, core, _ = stack

    cfg = client.app.state.config
    beta = "beta"
    core_b = StyxMemoryCore(agent_id=beta)
    core_b._config = cfg
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity=beta)
    registry.register(agent_id=beta, core=core_b)
    try:
        # Прямой insert через provider queries (минуя gatekeeper).
        core_b._queries.insert_memory(
            role="summary", content="beta-private only fact",
            kind="fact", kind_src="subjective",
        )
        core_b._conn.commit()
        core._queries.insert_memory(
            role="summary", content="alpha-private only fact",
            kind="fact", kind_src="subjective",
        )
        core._conn.commit()

        resp_a = client.get(f"/analytics?agent_id={agent}")
        resp_b = client.get(f"/analytics?agent_id={beta}")
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        # alpha видит только своих
        assert resp_a.json()["agents"][0]["memories_count"] == 1
        assert resp_b.json()["agents"][0]["memories_count"] == 1
    finally:
        core_b.shutdown()


def test_analytics_disabled_503(stack_disabled) -> None:
    client, agent = stack_disabled
    resp = client.get(f"/analytics?agent_id={agent}")
    assert resp.status_code == 503


def test_analytics_missing_agent_id_422(stack) -> None:
    client, _, _, _, _ = stack
    resp = client.get("/analytics")
    assert resp.status_code == 422


def test_analytics_unknown_agent_404_or_500(stack) -> None:
    """Unknown agent_id → registry.get raises (registry-зависимый код).

    Принципиально что не 200 (без leak'а данных)."""
    client, _, _, _, _ = stack
    resp = client.get("/analytics?agent_id=nonexistent")
    assert resp.status_code != 200
