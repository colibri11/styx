"""POST /reinterpret integration: TestClient + real Postgres (волна 22).

LLM не дёргается — route enqueue-only. Тестируем status codes + body
shape для всех веток.
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
    core._config = cfg
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    registry.reset_all()
    registry.register(agent_id=agent, core=core)

    app = create_app(cfg)
    client = TestClient(app)

    yield client, agent, clean_db, core

    core.shutdown()
    registry.reset_all()


def _seed_memory(
    dsn: str, agent: str, *, content: str = "старая мысль",
) -> uuid.UUID:
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        mid = q.insert_memory(
            role="summary", content=content, kind="note",
            kind_src="subjective",
            embedding=[1.0, 0.0] + [0.0] * 766,
        )
        conn.commit()
    return mid


def test_reinterpret_returns_queued(stack) -> None:
    client, agent, dsn, _ = stack
    mid = _seed_memory(dsn, agent)
    resp = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": str(mid),
            "new_understanding_text": "новое понимание",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert body["memory_id"] == str(mid)
    assert body["task_id"] is not None
    assert body["application_id"] is not None
    assert "30-60s" in body["message"]

    # Проверим что pending-row создан в БД.
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM reinterpret_applications "
                " WHERE memory_id=%s AND status='pending_sleep'",
                (mid,),
            )
            assert cur.fetchone()[0] == 1


def test_reinterpret_returns_404_unknown_memory(stack) -> None:
    client, agent, _, _ = stack
    resp = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": str(uuid.uuid4()),
            "new_understanding_text": "x",
        },
    )
    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert body["detail"]["status"] == "memory_not_found"


def test_reinterpret_returns_409_already_pending(stack) -> None:
    client, agent, dsn, _ = stack
    mid = _seed_memory(dsn, agent)
    # Первый раз — queued.
    r1 = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": str(mid),
            "new_understanding_text": "первое",
        },
    )
    assert r1.status_code == 200
    # Второй раз на ту же memory → 409 already_pending.
    r2 = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": str(mid),
            "new_understanding_text": "второе",
        },
    )
    assert r2.status_code == 409, r2.text
    body = r2.json()
    assert body["detail"]["status"] == "already_pending"
    assert body["detail"]["pending_application_id"] is not None


def test_reinterpret_returns_409_cooldown(stack) -> None:
    """Если в memory_reinterpretations есть свежая revision → 409 cooldown."""
    client, agent, dsn, _ = stack
    mid = _seed_memory(dsn, agent)
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        q.insert_memory_reinterpretation(
            memory_id=mid, previous_text="a",
            new_understanding_text="b", merged_text="c",
            previous_embedding=[1.0, 0.0] + [0.0] * 766,
            merged_embedding=[0.7, 0.7] + [0.0] * 766,
            weight_applied=0.5,
        )
        conn.commit()
    resp = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": str(mid),
            "new_understanding_text": "новое",
        },
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["detail"]["status"] == "cooldown"
    assert body["detail"]["next_available_at"] is not None
    assert body["detail"]["last_reinterpreted_at"] is not None


def test_reinterpret_validates_input(stack) -> None:
    client, agent, _, _ = stack
    # Пустой text — 422 (Pydantic).
    resp = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": str(uuid.uuid4()),
            "new_understanding_text": "",
        },
    )
    assert resp.status_code == 422


def test_reinterpret_invalid_uuid_returns_422(stack) -> None:
    client, agent, _, _ = stack
    resp = client.post(
        "/reinterpret",
        json={
            "agent_id": agent,
            "memory_id": "not-a-uuid",
            "new_understanding_text": "test",
        },
    )
    # core.reinterpret_enqueue raise ValueError → 422.
    assert resp.status_code == 422


def test_reinterpret_isolates_agents(stack, clean_db: str) -> None:
    """agent A не может reinterpret memory agent B."""
    client, agent_a, dsn, core_a = stack
    # Зарегистрируем второго агента.
    cfg = StyxConfig(database_url=dsn)
    from dataclasses import replace
    cfg = replace(load_config(), database_url=dsn, http_token=None)
    core_b = StyxMemoryCore(agent_id="beta")
    core_b._config = cfg
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity="beta")
    registry.register(agent_id="beta", core=core_b)
    try:
        with psycopg.connect(dsn) as conn:
            qb = AgentScopedQueries(conn, "beta")
            mid_b = qb.insert_memory(
                role="summary", content="beta-only",
                kind="note", kind_src="subjective",
                embedding=[1.0, 0.0] + [0.0] * 766,
            )
            conn.commit()
        # alpha запрашивает reinterpret чужой memory — must 404.
        resp = client.post(
            "/reinterpret",
            json={
                "agent_id": agent_a,
                "memory_id": str(mid_b),
                "new_understanding_text": "alpha hijack attempt",
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"]["status"] == "memory_not_found"
    finally:
        core_b.shutdown()
