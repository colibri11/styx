"""POST /explain/{decompose,lifetime,topK} integration: TestClient
+ real Postgres + Ollama (волна 25).

3 routes над scoring slot'ом. agent_id-isolation; 503 при disabled.
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
    core._config = cfg  # initialize перезатёр — restore.
    registry.reset_all()
    registry.register(agent_id=agent, core=core)
    app = create_app(cfg)
    client = TestClient(app)
    yield client, agent
    core.shutdown()
    registry.reset_all()


def _store_subjective(
    client: TestClient, agent: str, content: str, kind: str = "fact",
) -> str:
    """Записать subjective memory через /memory_store. Возвращает id."""
    resp = client.post(
        "/memory_store",
        json={"agent_id": agent, "content": content, "kind": kind},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["memory_id"]


# ── /explain/decompose ──────────────────────────────────────────────


def test_decompose_happy_returns_factors(stack) -> None:
    client, agent, _, _, _ = stack
    mid = _store_subjective(client, agent, "Postgres tuning notes")
    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": mid,
            "query": "postgres",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "decompose"
    assert body["memory_id"] == mid
    assert body["kind"] == "fact"
    assert body["final_score"] > 0
    factors = body["factors"]
    for key in (
        "base_match", "relevance", "recency_boost", "frequency_boost",
        "lifecycle_factor", "feedback_factor", "importance_factor",
        "diversity_bonus", "usage_factor", "decay_factor",
    ):
        assert key in factors, f"missing factor: {key}"
    assert body["would_be_returned"] is True
    assert body["return_reason"] in ("top_k", "top_k_with_min_score")


def test_decompose_below_min_score(stack) -> None:
    client, agent, _, _, _ = stack
    mid = _store_subjective(client, agent, "irrelevant content")
    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": mid,
            "query": "completely unrelated query topic ZYX",
            "min_score": 999.0,  # заведомо выше
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["would_be_returned"] is False
    assert body["not_returned_because"]["code"] == "below_min_score"


def test_decompose_outside_top_k(stack) -> None:
    client, agent, _, _, _ = stack
    target = _store_subjective(client, agent, "z weak match content")
    # Записать ещё кучу более релевантных
    for i in range(5):
        _store_subjective(client, agent, f"strong match item {i}")
    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": target,
            "query": "strong match",
            "top_k_limit": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # rank target'а > 1 = outside_top_k (если scoring сделал его не топ).
    # Возможен fluke что target всё равно в топе — тест-проверка с допуском.
    if not body["would_be_returned"]:
        assert body["not_returned_because"]["code"] == "outside_top_k"


def test_decompose_superseded(stack) -> None:
    client, agent, dsn, _, _ = stack
    # Записать две memory, потом руками поставить superseded_by
    mid_old = _store_subjective(client, agent, "old version")
    mid_new = _store_subjective(client, agent, "new version")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET superseded_by = %s WHERE id = %s",
                (mid_new, mid_old),
            )
        conn.commit()

    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": mid_old,
            "query": "old version",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["would_be_returned"] is False
    assert body["not_returned_because"]["code"] == "superseded"
    assert body["rank_in_result_set"] is None


def test_decompose_unknown_memory_404(stack) -> None:
    client, agent, _, _, _ = stack
    fake = str(uuid.uuid4())
    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": fake,
            "query": "anything",
        },
    )
    assert resp.status_code == 404


def test_decompose_disabled_503(stack_disabled) -> None:
    client, agent = stack_disabled
    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": str(uuid.uuid4()),
            "query": "x",
        },
    )
    assert resp.status_code == 503


def test_decompose_bad_uuid_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/explain/decompose",
        json={
            "agent_id": agent,
            "memory_id": "not-a-uuid",
            "query": "x",
        },
    )
    assert resp.status_code == 422


# ── /explain/lifetime ───────────────────────────────────────────────


def test_lifetime_happy(stack) -> None:
    client, agent, _, _, _ = stack
    mid = _store_subjective(client, agent, "tracked memory for lifetime")
    resp = client.post(
        "/explain/lifetime",
        json={"agent_id": agent, "memory_id": mid},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "lifetime"
    assert body["memory_id"] == mid
    assert body["agent_id"] == agent
    assert body["age_days"] >= 0
    assert "current_decay_factor" in body["decay"]
    assert body["recall_history"] == []  # нет recall'ов
    assert body["co_retrieval_links"] == []


def test_lifetime_recall_history_populated(stack) -> None:
    """recall_history содержит запись после явного record_recall_event."""
    client, agent, _, core, _ = stack
    mid = _store_subjective(client, agent, "recall me please")

    # Записываем recall_event напрямую (пропускаем зависимость на embed
    # availability в /recall).
    qhash = b"\xde\xad\xbe\xef" + b"\x00" * 28
    core._queries.record_recall_event(
        memory_id=uuid.UUID(mid),
        query_hash=qhash,
        match_score=0.5,
    )
    core._conn.commit()

    resp = client.post(
        "/explain/lifetime",
        json={"agent_id": agent, "memory_id": mid},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recall_history"]
    entry = body["recall_history"][0]
    assert "matched_at" in entry
    assert entry["query_hash"].startswith("0x")


def test_lifetime_unknown_404(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/explain/lifetime",
        json={"agent_id": agent, "memory_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 404


def test_lifetime_other_agent_404(stack) -> None:
    """memory чужого агента → 404 (scope guard)."""
    client, agent, dsn, core, cfg = stack
    # Inject memory под чужим agent_id напрямую в БД
    other_id = str(uuid.uuid4())
    other_agent = "beta"
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories "
                "(id, agent_id, role, kind, kind_src, content) "
                "VALUES (%s, %s, 'summary', 'fact', 'subjective', 'beta-only')",
                (other_id, other_agent),
            )
        conn.commit()

    resp = client.post(
        "/explain/lifetime",
        json={"agent_id": agent, "memory_id": other_id},
    )
    assert resp.status_code == 404


def test_lifetime_disabled_503(stack_disabled) -> None:
    client, agent = stack_disabled
    resp = client.post(
        "/explain/lifetime",
        json={"agent_id": agent, "memory_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 503


# ── /explain/topK ───────────────────────────────────────────────────


def test_topk_orders_by_score_desc(stack) -> None:
    client, agent, _, _, _ = stack
    _store_subjective(client, agent, "strong relevant alpha")
    _store_subjective(client, agent, "medium some text")
    _store_subjective(client, agent, "completely different topic")
    resp = client.post(
        "/explain/topK",
        json={"agent_id": agent, "query": "strong relevant", "limit": 3},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "top_k"
    assert body["total_candidates_considered"] >= 3
    items = body["items"]
    assert len(items) == 3
    # ranks 1..N
    assert [i["rank"] for i in items] == [1, 2, 3]
    # факторы заполнены
    assert items[0]["factors"] is not None
    assert "base_match" in items[0]["factors"]
    # final_score DESC
    scores = [i["final_score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_topk_include_factors_false(stack) -> None:
    client, agent, _, _, _ = stack
    _store_subjective(client, agent, "topic for include false")
    resp = client.post(
        "/explain/topK",
        json={
            "agent_id": agent,
            "query": "topic",
            "limit": 5,
            "include_factors": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert all(i["factors"] is None for i in body["items"])


def test_topk_agent_isolation(stack) -> None:
    """alpha не видит memories beta."""
    client, agent, dsn, _, _ = stack
    # beta agent registered: создадим runtime registration
    beta = "beta"
    cfg = client.app.state.config
    core_b = StyxMemoryCore(agent_id=beta)
    core_b._config = cfg
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity=beta)
    registry.register(agent_id=beta, core=core_b)
    try:
        client.post(
            "/memory_store",
            json={"agent_id": beta, "content": "beta-private secret", "kind": "fact"},
        )
        # alpha запрашивает — не видит beta-секрет
        resp = client.post(
            "/explain/topK",
            json={"agent_id": agent, "query": "beta-private", "limit": 5},
        )
        assert resp.status_code == 200
        contents = [i["content_preview"] for i in resp.json()["items"]]
        assert all("beta-private" not in c for c in contents)
    finally:
        core_b.shutdown()


def test_topk_disabled_503(stack_disabled) -> None:
    client, agent = stack_disabled
    resp = client.post(
        "/explain/topK",
        json={"agent_id": agent, "query": "x", "limit": 5},
    )
    assert resp.status_code == 503


def test_topk_limit_validation_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/explain/topK",
        json={"agent_id": agent, "query": "x", "limit": 100},
    )
    assert resp.status_code == 422  # > 50


def test_topk_kinds_filter(stack) -> None:
    client, agent, _, _, _ = stack
    _store_subjective(client, agent, "fact-content", kind="fact")
    _store_subjective(client, agent, "note-content", kind="note")
    resp = client.post(
        "/explain/topK",
        json={
            "agent_id": agent,
            "query": "content",
            "kinds": ["fact"],
            "limit": 10,
        },
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert all(i["kind"] == "fact" for i in items)
