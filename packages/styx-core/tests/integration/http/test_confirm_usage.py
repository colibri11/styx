"""POST /confirm_usage integration (волна 25).

Cross-agent guard, idempotency, dedupe input. 503 при disabled.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import psycopg
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


def _trigger_recall(
    client: TestClient, agent: str, query: str, core: StyxMemoryCore,
) -> str:
    """Записывает memory + recall_event напрямую через provider's queries.

    Прямой insert (минуя /memory_store + gatekeeper) делает test'ы
    детерминированными — gatekeeper мог бы skip'нуть похожие content'ы
    как duplicates. Прямой record_recall_event обходит зависимость на
    /recall (Ollama availability).
    """
    mid = core._queries.insert_memory(
        role="summary",
        content=query,
        kind="fact",
        kind_src="subjective",
    )
    core._conn.commit()
    qhash = (query.encode("utf-8") + b"\x00" * 32)[:32]
    core._queries.record_recall_event(
        memory_id=mid,
        query_hash=qhash,
        match_score=0.5,
    )
    core._conn.commit()
    return str(mid)


def test_confirm_usage_happy_flips_used_in_output(stack) -> None:
    client, agent, dsn, core, _ = stack
    mid = _trigger_recall(client, agent, "confirm me", core)

    resp = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": [mid]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["updated"] == 1
    assert body["requested"] == 1
    assert body["missing"] == []

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT used_in_output FROM recall_events "
                "WHERE memory_id = %s ORDER BY matched_at DESC LIMIT 1",
                (mid,),
            )
            assert cur.fetchone()[0] is True


def test_confirm_usage_idempotent(stack) -> None:
    client, agent, _, core, _ = stack
    mid = _trigger_recall(client, agent, "idempotent", core)

    body1 = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": [mid]},
    ).json()
    body2 = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": [mid]},
    ).json()
    assert body1["updated"] == body2["updated"] == 1
    assert body2["missing"] == []


def test_confirm_usage_cross_agent_guard(stack) -> None:
    """memory_id чужого агента → попадает в `missing`, не updated."""
    client, agent, dsn, _, _ = stack
    cfg = client.app.state.config
    beta = "beta"
    core_b = StyxMemoryCore(agent_id=beta)
    core_b._config = cfg
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity=beta)
    registry.register(agent_id=beta, core=core_b)
    try:
        # beta создаёт recall_event для своей memory
        mid_beta = _trigger_recall(client, beta, "beta secret", core_b)

        # alpha пробует confirm_usage на чужой memory_id
        resp = client.post(
            "/confirm_usage",
            json={"agent_id": agent, "memory_ids": [mid_beta]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["updated"] == 0
        assert body["requested"] == 1
        assert mid_beta in body["missing"]

        # beta-recall_event остался used_in_output=false
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT bool_or(used_in_output) FROM recall_events "
                    "WHERE memory_id = %s",
                    (mid_beta,),
                )
                used = cur.fetchone()[0]
                assert used is False
    finally:
        core_b.shutdown()


def test_confirm_usage_duplicates_collapsed(stack) -> None:
    client, agent, _, core, _ = stack
    mid = _trigger_recall(client, agent, "dedupe me", core)

    resp = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": [mid, mid, mid]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["requested"] == 1  # collapsed
    assert body["updated"] == 1


def test_confirm_usage_missing_memory(stack) -> None:
    client, agent, _, _, _ = stack
    fake = str(uuid.uuid4())
    resp = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": [fake]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["updated"] == 0
    assert body["missing"] == [fake]


def test_confirm_usage_empty_input_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": []},
    )
    assert resp.status_code == 422


def test_confirm_usage_too_many_ids_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/confirm_usage",
        json={
            "agent_id": agent,
            "memory_ids": [str(uuid.uuid4()) for _ in range(101)],
        },
    )
    assert resp.status_code == 422


def test_confirm_usage_bad_uuid_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": ["not-a-uuid"]},
    )
    assert resp.status_code == 422


def test_confirm_usage_disabled_503(stack_disabled) -> None:
    client, agent = stack_disabled
    resp = client.post(
        "/confirm_usage",
        json={"agent_id": agent, "memory_ids": [str(uuid.uuid4())]},
    )
    assert resp.status_code == 503
