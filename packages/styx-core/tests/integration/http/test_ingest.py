"""POST /ingest_experience integration: TestClient + real Postgres +
Ollama (волна 23).

Embedding идёт через core._embedding (реальный Ollama в Docker).
Тестируем UPSERT pattern: повторный hash → deduplicated; разная
pipeline_version → новый ряд; без hash → не идемпотентно;
agent_id isolation; STYX_INGEST_API_ENABLED=0 → 503; 422 на content
> 2400.
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


def _count_memories(dsn: str, agent: str, *, content_hash: str | None = None):
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            if content_hash is None:
                cur.execute(
                    "SELECT count(*) FROM memories WHERE agent_id=%s",
                    (agent,),
                )
            else:
                cur.execute(
                    "SELECT count(*) FROM memories "
                    " WHERE agent_id=%s AND content_hash=%s",
                    (agent, content_hash),
                )
            return cur.fetchone()[0]


# ── happy path ──────────────────────────────────────────────────────


def test_ingest_explicit_hash_returns_memory_id(stack) -> None:
    client, agent, dsn, _, _ = stack
    resp = client.post(
        "/ingest_experience",
        json={
            "agent_id": agent,
            "content": "запись из аудиоканала",
            "kind": "fact",
            "content_hash": "a" * 64,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory_id"]
    assert body["deduplicated"] is False
    assert body["content_hash"] == "a" * 64
    assert _count_memories(dsn, agent, content_hash="a" * 64) == 1


def test_ingest_auto_compute_hash_from_pipeline_triplet(stack) -> None:
    client, agent, dsn, _, _ = stack
    resp = client.post(
        "/ingest_experience",
        json={
            "agent_id": agent,
            "content": "транскрипт записи",
            "kind": "episode",
            "pipeline_id": "audiobox",
            "pipeline_version": "v1.0",
            "content_ref": {"file_path": "/recordings/a.wav"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory_id"]
    assert body["deduplicated"] is False
    # Auto-computed sha256 — 64 hex chars.
    h = body["content_hash"]
    assert h is not None and len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ── идемпотентность ─────────────────────────────────────────────────


def test_ingest_repeat_same_hash_returns_deduplicated(stack) -> None:
    """Повторный ingest того же payload → тот же memory_id, dedup=True."""
    client, agent, dsn, _, _ = stack
    payload = {
        "agent_id": agent,
        "content": "первый ingest",
        "kind": "fact",
        "pipeline_id": "audiobox",
        "pipeline_version": "v1.0",
        "content_ref": {"file_path": "/a.wav"},
    }
    r1 = client.post("/ingest_experience", json=payload)
    assert r1.status_code == 200
    first_id = r1.json()["memory_id"]
    first_hash = r1.json()["content_hash"]

    # Повтор того же payload (даже с другим content/metadata —
    # existing ряд возвращается как есть).
    payload2 = {**payload, "content": "другой текст того же источника"}
    r2 = client.post("/ingest_experience", json=payload2)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["memory_id"] == first_id
    assert body2["deduplicated"] is True
    assert body2["content_hash"] == first_hash
    # Только один ряд в БД.
    assert _count_memories(dsn, agent, content_hash=first_hash) == 1


def test_ingest_repeat_different_pipeline_version_creates_new(stack) -> None:
    client, agent, dsn, _, _ = stack
    base = {
        "agent_id": agent,
        "content": "транскрипт",
        "kind": "episode",
        "pipeline_id": "audiobox",
        "content_ref": {"file_path": "/a.wav"},
    }
    r1 = client.post(
        "/ingest_experience",
        json={**base, "pipeline_version": "v1.0"},
    )
    r2 = client.post(
        "/ingest_experience",
        json={**base, "pipeline_version": "v2.0"},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["memory_id"] != r2.json()["memory_id"]
    assert r1.json()["content_hash"] != r2.json()["content_hash"]
    assert r1.json()["deduplicated"] is False
    assert r2.json()["deduplicated"] is False


def test_ingest_without_hash_not_idempotent(stack) -> None:
    """Без content_hash и без полного triplet — каждый INSERT новый ряд."""
    client, agent, dsn, _, _ = stack
    payload = {
        "agent_id": agent,
        "content": "одинаковый текст",
        "kind": "note",
    }
    r1 = client.post("/ingest_experience", json=payload)
    r2 = client.post("/ingest_experience", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["memory_id"] != r2.json()["memory_id"]
    assert r1.json()["deduplicated"] is False
    assert r2.json()["deduplicated"] is False
    assert r1.json()["content_hash"] is None
    assert r2.json()["content_hash"] is None


def test_ingest_empty_content_ref_disables_idempotency(stack) -> None:
    """content_ref={} семантически = «нет ссылки» → hash не вычисляется."""
    client, agent, _, _, _ = stack
    payload = {
        "agent_id": agent,
        "content": "test",
        "kind": "note",
        "pipeline_id": "p",
        "pipeline_version": "v1",
        "content_ref": {},
    }
    r1 = client.post("/ingest_experience", json=payload)
    r2 = client.post("/ingest_experience", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["content_hash"] is None
    assert r2.json()["content_hash"] is None
    assert r1.json()["memory_id"] != r2.json()["memory_id"]


# ── isolation ──────────────────────────────────────────────────────


def test_ingest_agent_id_isolation(stack) -> None:
    """Один и тот же hash от разных агентов — два независимых ряда."""
    client, agent_a, dsn, _, cfg = stack
    core_b = StyxMemoryCore(agent_id="beta")
    core_b._config = cfg
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity="beta")
    registry.register(agent_id="beta", core=core_b)
    try:
        h = "b" * 64
        r1 = client.post(
            "/ingest_experience",
            json={
                "agent_id": agent_a,
                "content": "alpha view",
                "kind": "note",
                "content_hash": h,
            },
        )
        r2 = client.post(
            "/ingest_experience",
            json={
                "agent_id": "beta",
                "content": "beta view",
                "kind": "note",
                "content_hash": h,
            },
        )
        assert r1.status_code == 200 and r2.status_code == 200
        # Разные memory_id под одним hash — partial UNIQUE per (agent_id, hash).
        assert r1.json()["memory_id"] != r2.json()["memory_id"]
        assert r1.json()["deduplicated"] is False
        assert r2.json()["deduplicated"] is False
    finally:
        core_b.shutdown()


# ── feature flag + validation ──────────────────────────────────────


def test_ingest_disabled_returns_503(
    stack, clean_db: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, agent, dsn, core, cfg = stack
    # initialize() перечитывает StyxConfig через load_config() — поэтому
    # flag нужно ставить через ENV, не через replace(cfg, ...).
    core.shutdown()
    registry.reset_all()
    monkeypatch.setenv("STYX_INGEST_API_ENABLED", "0")
    cfg2 = replace(load_config(), database_url=clean_db, http_token=None)
    assert cfg2.ingest_api_enabled is False
    new_core = StyxMemoryCore(agent_id=agent)
    new_core._config = cfg2
    new_core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    # initialize перезатёр _config — ставим обратно после.
    new_core._config = cfg2
    registry.register(agent_id=agent, core=new_core)
    app2 = create_app(cfg2)
    client2 = TestClient(app2)
    try:
        resp = client2.post(
            "/ingest_experience",
            json={"agent_id": agent, "content": "x", "kind": "note"},
        )
        assert resp.status_code == 503, resp.text
        assert "disabled" in resp.json()["detail"].lower()
    finally:
        new_core.shutdown()


def test_ingest_content_too_long_returns_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/ingest_experience",
        json={
            "agent_id": agent,
            "content": "x" * 2401,  # > 2400
            "kind": "note",
        },
    )
    assert resp.status_code == 422


def test_ingest_invalid_kind_returns_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/ingest_experience",
        json={
            "agent_id": agent,
            "content": "valid content",
            "kind": "weird_kind_not_in_enum",
        },
    )
    # core.ingest_experience raise ValueError → 422.
    assert resp.status_code == 422


def test_ingest_metadata_includes_pipeline_source(stack) -> None:
    """Pipeline_id / version / content_ref enrich'аются в metadata."""
    client, agent, dsn, _, _ = stack
    resp = client.post(
        "/ingest_experience",
        json={
            "agent_id": agent,
            "content": "test",
            "kind": "note",
            "pipeline_id": "audiobox",
            "pipeline_version": "v1.0",
            "content_ref": {"file_path": "/a.wav"},
            "metadata": {"user_field": "user_value"},
        },
    )
    assert resp.status_code == 200
    mid = resp.json()["memory_id"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM memories WHERE id=%s", (mid,)
            )
            meta = cur.fetchone()[0]
    assert meta["user_field"] == "user_value"
    assert meta["source"]["pipeline_id"] == "audiobox"
    assert meta["source"]["pipeline_version"] == "v1.0"
    assert meta["content_ref"]["file_path"] == "/a.wav"
