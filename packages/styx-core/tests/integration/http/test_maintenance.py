"""POST /maintenance/reembed integration: TestClient + real Postgres +
Ollama (волна 31).

Эндпоинт строит собственный embed-клиент из ``config.ollama_url`` и
короткоживущий ``psycopg.connect`` — поэтому тестируется как полноценный
integration (нужны Postgres + Ollama, как у остальных в этой папке).

Кейсы:
- auth required (401 без Bearer при заданном http_token);
- валидация body (422 на rate_per_second=0);
- dry_run возвращает would_process без UPDATE;
- реальный backfill: memory с embedding=NULL → reembed → embedding NOT NULL,
  processed≥1;
- идемпотентность: повторный вызов processed=0;
- advisory-lock skip: при удержанном локе из другого соединения → skipped.
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
from styx.http.routes.maintenance import REEMBED_ADVISORY_LOCK_KEY
from styx.providers.memory import StyxMemoryCore
from styx.storage import migrate


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


def _make_stack(clean_db: str, *, http_token: str | None = None):
    migrate.run(clean_db)
    cfg: StyxConfig = load_config()
    cfg = replace(cfg, database_url=clean_db, http_token=http_token)
    agent = "alpha"
    core = StyxMemoryCore(agent_id=agent)
    core._config = cfg
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    registry.reset_all()
    registry.register(agent_id=agent, core=core)
    app = create_app(cfg)
    client = TestClient(app)
    return client, agent, clean_db, core, cfg


@pytest.fixture
def stack(clean_db: str):
    client, agent, dsn, core, cfg = _make_stack(clean_db)
    yield client, agent, dsn, core, cfg
    core.shutdown()
    registry.reset_all()


@pytest.fixture
def stack_with_token(clean_db: str):
    client, agent, dsn, core, cfg = _make_stack(clean_db, http_token="s3cr3t")
    yield client, agent, dsn, core, cfg
    core.shutdown()
    registry.reset_all()


def _insert_null_embedding_memory(core: StyxMemoryCore, content: str) -> str:
    """Прямой insert через provider queries; embedding не передан → NULL."""
    mid = core._queries.insert_memory(
        role="summary", content=content, kind="fact", kind_src="subjective",
    )
    core._conn.commit()
    return str(mid)


def _embedding_is_null(dsn: str, memory_id: str) -> bool:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding IS NULL FROM memories WHERE id=%s",
                (memory_id,),
            )
            return bool(cur.fetchone()[0])


def _count_null_embeddings(dsn: str, agent: str) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories "
                " WHERE agent_id=%s AND embedding IS NULL "
                "   AND superseded_by IS NULL",
                (agent,),
            )
            return int(cur.fetchone()[0])


# ── auth ────────────────────────────────────────────────────────────


def test_reembed_requires_bearer_when_token_set(stack_with_token) -> None:
    client, agent, _, _, _ = stack_with_token
    resp = client.post("/maintenance/reembed", json={})
    assert resp.status_code == 401, resp.text


def test_reembed_accepts_valid_bearer(stack_with_token) -> None:
    client, agent, _, _, _ = stack_with_token
    resp = client.post(
        "/maintenance/reembed",
        json={"dry_run": True},
        headers={"Authorization": "Bearer s3cr3t"},
    )
    assert resp.status_code == 200, resp.text


# ── validation ──────────────────────────────────────────────────────


def test_reembed_invalid_rate_returns_422(stack) -> None:
    client, _, _, _, _ = stack
    resp = client.post("/maintenance/reembed", json={"rate_per_second": 0})
    assert resp.status_code == 422


def test_reembed_invalid_mode_returns_422(stack) -> None:
    client, _, _, _, _ = stack
    resp = client.post("/maintenance/reembed", json={"mode": "garbage"})
    assert resp.status_code == 422


def test_reembed_negative_limit_returns_422(stack) -> None:
    client, _, _, _, _ = stack
    resp = client.post("/maintenance/reembed", json={"limit": -1})
    assert resp.status_code == 422


# ── dry_run ─────────────────────────────────────────────────────────


def test_reembed_dry_run_counts_without_update(stack) -> None:
    client, agent, dsn, core, _ = stack
    mid = _insert_null_embedding_memory(core, "факт без эмбеддинга один")

    resp = client.post(
        "/maintenance/reembed",
        json={"mode": "null_only", "agent_id": agent, "dry_run": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    assert body["would_process"] >= 1
    assert body["processed"] == 0
    assert body["failed"] == 0
    assert body["skipped"] is False
    # UPDATE не выполнялся — embedding всё ещё NULL.
    assert _embedding_is_null(dsn, mid) is True


# ── mode=all ────────────────────────────────────────────────────────


def test_reembed_mode_all_counts_already_embedded(stack) -> None:
    """mode=all считает ВСЕ не-superseded ряды (включая уже заэмбеженные) —
    в отличие от null_only, который видит только embedding IS NULL."""
    client, agent, dsn, core, _ = stack
    _insert_null_embedding_memory(core, "факт для режима all")

    # Добиваем NULL → ряд становится заэмбеженным.
    r0 = client.post(
        "/maintenance/reembed", json={"mode": "null_only", "agent_id": agent}
    )
    assert r0.status_code == 200, r0.text
    assert _count_null_embeddings(dsn, agent) == 0

    # null_only теперь видит 0 (добивать нечего)...
    r_null = client.post(
        "/maintenance/reembed",
        json={"mode": "null_only", "agent_id": agent, "dry_run": True},
    )
    assert r_null.status_code == 200, r_null.text
    assert r_null.json()["would_process"] == 0

    # ...а all считает уже-заэмбеженный ряд (re-embed после смены модели).
    r_all = client.post(
        "/maintenance/reembed",
        json={"mode": "all", "agent_id": agent, "dry_run": True},
    )
    assert r_all.status_code == 200, r_all.text
    body = r_all.json()
    assert body["dry_run"] is True
    assert body["would_process"] >= 1
    assert body["processed"] == 0


# ── реальный backfill + идемпотентность ─────────────────────────────


def test_reembed_backfills_null_embedding(stack) -> None:
    client, agent, dsn, core, _ = stack
    mid = _insert_null_embedding_memory(core, "факт для бэкфилла вектора")
    assert _embedding_is_null(dsn, mid) is True

    resp = client.post(
        "/maintenance/reembed",
        json={"mode": "null_only", "agent_id": agent},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["processed"] >= 1
    assert body["failed"] == 0
    assert body["would_process"] == 0
    assert body["dry_run"] is False
    assert body["skipped"] is False
    assert isinstance(body["elapsed_ms"], int)
    # embedding теперь NOT NULL в БД.
    assert _embedding_is_null(dsn, mid) is False


def test_reembed_idempotent_second_call_processed_zero(stack) -> None:
    client, agent, dsn, core, _ = stack
    _insert_null_embedding_memory(core, "факт для идемпотентного прогона")

    r1 = client.post(
        "/maintenance/reembed",
        json={"mode": "null_only", "agent_id": agent},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["processed"] >= 1
    assert _count_null_embeddings(dsn, agent) == 0

    # Второй прогон в null_only режиме: добивать нечего.
    r2 = client.post(
        "/maintenance/reembed",
        json={"mode": "null_only", "agent_id": agent},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["processed"] == 0
    assert r2.json()["failed"] == 0


# ── advisory-lock skip ──────────────────────────────────────────────


def test_reembed_skips_when_lock_held(stack) -> None:
    client, agent, dsn, core, _ = stack
    _insert_null_embedding_memory(core, "факт под удержанным локом")

    # Удерживаем session-level lock из отдельного соединения — endpoint
    # не должен запустить backfill.
    holder = psycopg.connect(dsn)
    try:
        with holder.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s::bigint)",
                (REEMBED_ADVISORY_LOCK_KEY,),
            )
            got = cur.fetchone()[0]
        holder.commit()
        assert got is True

        resp = client.post(
            "/maintenance/reembed",
            json={"mode": "null_only", "agent_id": agent},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["skipped"] is True
        assert body["processed"] == 0
        assert body["would_process"] == 0
        # Backfill не запускался — embedding всё ещё NULL.
        assert _count_null_embeddings(dsn, agent) == 1
    finally:
        with holder.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(%s::bigint)",
                (REEMBED_ADVISORY_LOCK_KEY,),
            )
        holder.commit()
        holder.close()
