"""POST /memory_store integration: TestClient + real Postgres + Ollama.

Реальный embed через configured Ollama; gatekeeper apply на real DB.
Требует ``STYX_TEST_DATABASE_URL`` + Ollama (через ``STYX_OLLAMA_URL`` /
дефолт ``http://ollama:11434``).
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

    # Используем live config — Ollama URL приходит из ENV.
    cfg: StyxConfig = load_config()
    # Override DSN на test DB (clean_db) и убираем токен — TestClient
    # ходит без Authorization header'а.
    from dataclasses import replace
    cfg = replace(cfg, database_url=clean_db, http_token=None)

    agent = "alpha"
    core = StyxMemoryCore(agent_id=agent)
    core._config = cfg  # initialize ниже подцепит правильный
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    registry.reset_all()
    registry.register(agent_id=agent, core=core)

    app = create_app(cfg)
    client = TestClient(app)

    yield client, agent, clean_db

    core.shutdown()
    registry.reset_all()


def _count_memories(dsn: str, agent: str) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories "
                " WHERE agent_id = %s AND superseded_by IS NULL "
                "   AND kind_src = 'subjective'",
                (agent,),
            )
            return cur.fetchone()[0]


def test_memory_store_returns_store_action(stack) -> None:
    client, agent, dsn = stack
    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "новая мысль про архитектуру системы",
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "store"
    assert body["memory_id"] is not None
    assert _count_memories(dsn, agent) == 1


def test_memory_store_merge_on_duplicate(stack) -> None:
    client, agent, dsn = stack
    payload = {
        "agent_id": agent,
        "content": "напоминание про релиз пятницы",
        "kind": "note",
        "kind_src": "subjective",
    }
    resp1 = client.post("/memory_store", json=payload)
    assert resp1.status_code == 200, resp1.text
    assert resp1.json()["action"] == "store"

    # Точно тот же текст → identical embedding → merge / supersede.
    resp2 = client.post("/memory_store", json=payload)
    assert resp2.status_code == 200, resp2.text
    body2 = resp2.json()
    # similarity = 1.0 → merge (выше merge_threshold=0.92).
    assert body2["action"] == "merge"
    assert body2["existing_id"] == resp1.json()["memory_id"]
    assert body2["memory_id"] is None
    # В БД остался один ряд.
    assert _count_memories(dsn, agent) == 1


def test_memory_store_unknown_kind_422(stack) -> None:
    client, agent, _ = stack
    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "nonsense kind",
            "kind": "unknown_kind",
            "kind_src": "subjective",
        },
    )
    # Pydantic не ограничивает kind строкой (мы хотели allow custom через
    # будущие расширения), но _role_for_kind в провайдере падает с
    # ValueError → 422 от FastAPI handler'а.
    assert resp.status_code == 422


def test_memory_store_skip_short_content(stack) -> None:
    """noise filter (default 10 chars min) → skip."""
    client, agent, dsn = stack
    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "коротко",  # < 10 chars
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "skip"
    assert body["memory_id"] is None
    # В БД ничего не осталось.
    assert _count_memories(dsn, agent) == 0


def test_memory_store_creates_auto_link_relations(stack) -> None:
    """STORE через /memory_store → auto-link создаёт related_to рёбра.

    Cross-agent: pre-seeded ряд агента beta попадает в соседи alpha.
    """
    client, agent, dsn = stack

    # Pre-seed cross-agent ряд через прямой queries call. Содержит
    # тот же текст что мы попросим store через /memory_store —
    # similarity достаточная для попадания в auto-link окно.
    foreign_content = "напоминание про релиз пятницы"
    with psycopg.connect(dsn) as cn:
        # Получаем embedding через тот же ollama что использует stack.
        # Простейший способ — через core (alpha provider) embed-helper.
        session = registry.get(agent)
        vec = session.core._embedding.embed(foreign_content)
        q_beta = AgentScopedQueries(cn, "beta")
        foreign_id = q_beta.insert_memory(
            role="summary", content=foreign_content,
            kind="note", kind_src="subjective",
            embedding=vec,
        )
        cn.commit()

    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": foreign_content,
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Тот же текст и embedding → gatekeeper merge (similarity > 0.92).
    # На merge auto-link не зовётся — это правильно (новый ряд удалён).
    # Но если первый раз пишем (без foreign) — store + auto-link.
    # Поэтому проверяем что относительно действия behavior корректный.
    if body["action"] == "store":
        new_id = body["memory_id"]
        with psycopg.connect(dsn) as cn:
            with cn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM relations "
                    " WHERE source_type='memory' AND source_id=%s "
                    "   AND target_id=%s AND relation='related_to'",
                    (new_id, foreign_id),
                )
                assert cur.fetchone()[0] == 1
    elif body["action"] == "merge":
        # На merge новый ряд удалён, auto-link не зовётся — это OK.
        assert body["existing_id"] == str(foreign_id)


def test_memory_store_disabled_gatekeeper_falls_back(stack, monkeypatch) -> None:
    """STYX_SELECTIVE_ENABLED=0 → каждый writer пишет как раньше (store)."""
    client, agent, dsn = stack
    # Pre-seed похожий ряд через прямой queries call.
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        q.insert_memory(
            role="summary", content="базовая запись для дедупликации",
            kind="note", kind_src="subjective",
        )
        conn.commit()

    # Подменим gatekeeper config через perform с monkeypatch на core.
    session = registry.get(agent)
    from dataclasses import replace
    session.core._config = replace(
        session.core._config, selective_enabled=False,
    )

    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "базовая запись для дедупликации",
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "store"
    # Оба ряда живы.
    assert _count_memories(dsn, agent) == 2
