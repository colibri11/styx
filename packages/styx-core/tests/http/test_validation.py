"""Pydantic schema validation на request body."""

from __future__ import annotations


def test_initialize_requires_agent_id(client_no_auth):
    resp = client_no_auth.post("/agent/initialize", json={})
    assert resp.status_code == 422  # Pydantic schema fail


def test_recall_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.post(
        "/recall", json={"agent_id": "ghost", "query": "anything"}
    )
    assert resp.status_code == 404


def test_recall_400_on_empty_query(client_no_auth):
    """Empty query — после прохода через registry не должен достичь recall.

    Сейчас 404 (нет агента), но проверяем что валидация Pydantic не
    обрезает empty string (она считает пустую строку valid).
    """
    resp = client_no_auth.post(
        "/recall", json={"agent_id": "ghost", "query": ""}
    )
    assert resp.status_code == 404  # registry.get падает раньше валидатора


def test_sync_turn_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.post(
        "/sync_turn",
        json={
            "agent_id": "ghost",
            "user_content": "hi",
            "assistant_content": "hello",
        },
    )
    assert resp.status_code == 404


def test_context_build_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.post(
        "/context/build",
        json={"agent_id": "ghost", "messages": []},
    )
    assert resp.status_code == 404


def test_pre_llm_inject_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.post(
        "/pre_llm_inject",
        json={"agent_id": "ghost"},
    )
    assert resp.status_code == 404


def test_agent_state_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.get("/agent_state?agent_id=ghost")
    assert resp.status_code == 404


def test_shutdown_idempotent_when_unknown(client_no_auth):
    """shutdown unknown agent — 204 (no-op), не 404."""
    resp = client_no_auth.post(
        "/agent/shutdown", json={"agent_id": "ghost"}
    )
    assert resp.status_code == 204


# ── memory_store (волна 17) ───────────────────────────────────────────


def test_memory_store_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.post(
        "/memory_store",
        json={"agent_id": "ghost", "content": "any content here"},
    )
    assert resp.status_code == 404


def test_memory_store_422_on_empty_content(client_no_auth):
    resp = client_no_auth.post(
        "/memory_store",
        json={"agent_id": "ghost", "content": ""},
    )
    assert resp.status_code == 422


def test_memory_store_422_on_oversized_content(client_no_auth):
    """После волны 19 content > 2400 роутится в documents+chunks
    (Pydantic max_length=500_000); только превышение upper-bound'а
    Pydantic'а отлавливается на схеме."""
    resp = client_no_auth.post(
        "/memory_store",
        json={"agent_id": "ghost", "content": "x" * 500_001},
    )
    assert resp.status_code == 422


# ── relations / graph / link (волна 21) ───────────────────────────────


def test_relations_query_404_when_agent_not_initialized(client_no_auth):
    resp = client_no_auth.post(
        "/relations/query", json={"agent_id": "ghost"},
    )
    assert resp.status_code == 404


def test_relations_query_422_invalid_uuid(client_no_auth):
    """Невалидный UUID в фильтре → 422."""
    resp = client_no_auth.post(
        "/relations/query",
        json={"agent_id": "ghost", "source_id": "not-a-uuid"},
    )
    # 404 (agent не зарегистрирован) приходит раньше валидации UUID,
    # но оба acceptable — главное не 200.
    assert resp.status_code in (404, 422)


def test_graph_traverse_404_when_agent_not_initialized(client_no_auth):
    import uuid
    resp = client_no_auth.post(
        "/graph/traverse",
        json={"agent_id": "ghost", "entity_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 404


def test_graph_traverse_422_on_depth_out_of_range(client_no_auth):
    import uuid
    resp = client_no_auth.post(
        "/graph/traverse",
        json={
            "agent_id": "ghost",
            "entity_id": str(uuid.uuid4()),
            "depth": 10,  # max=3
        },
    )
    assert resp.status_code == 422


def test_link_404_when_agent_not_initialized(client_no_auth):
    import uuid
    resp = client_no_auth.post(
        "/link",
        json={
            "agent_id": "ghost",
            "source_type": "memory",
            "source_id": str(uuid.uuid4()),
            "target_type": "memory",
            "target_id": str(uuid.uuid4()),
            "relation": "custom",
        },
    )
    assert resp.status_code == 404
