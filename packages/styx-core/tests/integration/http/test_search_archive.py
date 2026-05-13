"""POST /search_archive integration: TestClient + real Postgres + Ollama.

Проверяет HTTP layer: scope dispatch, response shape, agent isolation,
422 на invalid scope. Pre-seed данных через AgentScopedQueries
(synthetic embeddings — детерминированно), embedder из core._embedding
обслуживает только embed(query).

Требует ``STYX_TEST_DATABASE_URL`` + Ollama (как test_memory_store).
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

    yield client, agent, clean_db

    core.shutdown()
    registry.reset_all()


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    return base


def _seed_archive(dsn: str, agent: str) -> uuid.UUID:
    """Вставляет 1 document с 2 chunks + 2 dialogue реплики."""
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        doc_id = q.insert_document(source="memory_store", char_count=50)
        q.insert_chunks_batch(doc_id, [
            (0, "первый чанк документа", _embed(1.0), 0, 22),
            (1, "второй чанк документа", _embed(0.95), 22, 44),
        ])
        q.insert_message(
            role="user", content="вопрос пользователя",
            embedding=_embed(1.0),
        )
        q.insert_message(
            role="assistant", content="ответ ассистента",
            embedding=_embed(0.9),
        )
        conn.commit()
        return doc_id


def test_search_archive_invalid_scope_returns_422(stack) -> None:
    client, agent, dsn = stack
    resp = client.post("/search_archive", json={
        "agent_id": agent, "query": "x", "scope": "invalid",
    })
    assert resp.status_code == 422


def test_search_archive_chunks_returns_individual(stack) -> None:
    client, agent, dsn = stack
    doc_id = _seed_archive(dsn, agent)
    resp = client.post("/search_archive", json={
        "agent_id": agent, "query": "чанк", "scope": "chunks", "limit": 10,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_matched"] >= 2
    assert all(r["scope"] == "chunk" for r in body["results"])
    positions = sorted(r["chunk_position"] for r in body["results"])
    assert positions == [0, 1]
    assert body["results"][0]["document_id"] == str(doc_id)


def test_search_archive_documents_returns_stitched(stack) -> None:
    client, agent, dsn = stack
    doc_id = _seed_archive(dsn, agent)
    resp = client.post("/search_archive", json={
        "agent_id": agent, "query": "чанк документа",
        "scope": "documents", "limit": 5,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_matched"] == 1
    region = body["results"][0]
    assert region["scope"] == "document"
    assert region["document_id"] == str(doc_id)
    assert region["chunk_positions"] == [0, 1]
    assert "первый" in region["text"]


def test_search_archive_dialogue_returns_role_filtered(stack) -> None:
    client, agent, dsn = stack
    _seed_archive(dsn, agent)
    resp = client.post("/search_archive", json={
        "agent_id": agent, "query": "ответ вопрос",
        "scope": "dialogue", "limit": 10,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_matched"] == 2
    roles = sorted(r["role"] for r in body["results"])
    assert roles == ["assistant", "user"]


def test_search_archive_all_interleaves(stack) -> None:
    client, agent, dsn = stack
    _seed_archive(dsn, agent)
    resp = client.post("/search_archive", json={
        "agent_id": agent, "query": "чанк ответ", "scope": "all", "limit": 10,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    scopes = {r["scope"] for r in body["results"]}
    assert "document" in scopes
    assert "dialogue" in scopes


def test_search_archive_agent_isolation(stack, clean_db: str) -> None:
    client, agent_a, dsn = stack
    _seed_archive(dsn, agent_a)

    # Регистрируем второго агента в том же daemon'е для isolation теста.
    agent_b = "beta"
    cfg = load_config()
    from dataclasses import replace
    cfg_b = replace(cfg, database_url=dsn, http_token=None)
    core_b = StyxMemoryCore(agent_id=agent_b)
    core_b._config = cfg_b
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity=agent_b)
    registry.register(agent_id=agent_b, core=core_b)
    try:
        resp = client.post("/search_archive", json={
            "agent_id": agent_b, "query": "чанк ответ",
            "scope": "all", "limit": 10,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # У beta нет seed'а — пусто.
        assert body["total_matched"] == 0
    finally:
        core_b.shutdown()


def test_search_archive_unknown_agent_503(stack) -> None:
    client, agent, dsn = stack
    resp = client.post("/search_archive", json={
        "agent_id": "ghost", "query": "x", "scope": "all",
    })
    # registry.get для незарегистрированного — KeyError → 503/500.
    assert resp.status_code in (404, 500, 503)
