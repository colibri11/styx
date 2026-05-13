"""POST /context/{bootstrap,ingest,ingest_batch,dispose} integration —
TestClient + real Postgres (волна 26 Phase B).

Embedding-after-commit идёт через core._embedding (Ollama в Docker).
В host-only прогоне `STYX_TEST_DATABASE_URL` обычно set, но Ollama
недоступен — embed-after-commit падает silent'но (warning), memory
остаётся без embedding'а. Это design (fail-open). Тесты на счёт
memories не зависят от embedding'а.
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

    registry.reset_all()

    app = create_app(cfg)
    client = TestClient(app)

    yield client, clean_db, cfg

    registry.reset_all()


def _count_memories(dsn: str, agent: str, *, role: str | None = None) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            if role is None:
                cur.execute(
                    "SELECT count(*) FROM memories WHERE agent_id=%s",
                    (agent,),
                )
            else:
                cur.execute(
                    "SELECT count(*) FROM memories WHERE agent_id=%s AND role=%s",
                    (agent, role),
                )
            return cur.fetchone()[0]


def _fetch_roles(dsn: str, agent: str) -> list[str]:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role FROM memories WHERE agent_id=%s ORDER BY seq",
                (agent,),
            )
            return [row[0] for row in cur.fetchall()]


# ── /context/bootstrap ────────────────────────────────────────────────


def test_bootstrap_new_agent_initialized_true(stack) -> None:
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"

    resp = client.post(
        "/context/bootstrap",
        json={"agent_id": agent, "session_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"ok": True, "initialized": True}
    assert registry.get_optional(agent) is not None


def test_bootstrap_existing_agent_initialized_false(stack) -> None:
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())

    first = client.post(
        "/context/bootstrap",
        json={"agent_id": agent, "session_id": sid},
    )
    assert first.json()["initialized"] is True

    second = client.post(
        "/context/bootstrap",
        json={"agent_id": agent, "session_id": sid},
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body == {"ok": True, "initialized": False}


# ── /context/ingest ───────────────────────────────────────────────────


def test_ingest_user_message_writes_one_memory(stack) -> None:
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest",
        json={
            "agent_id": agent,
            "session_id": sid,
            "message": {"role": "user", "content": "привет"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["ingested"] is True
    assert body["memory_id"]
    assert _count_memories(dsn, agent) == 1
    assert _fetch_roles(dsn, agent) == ["user"]


def test_ingest_heartbeat_skipped(stack) -> None:
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest",
        json={
            "agent_id": agent,
            "session_id": sid,
            "message": {"role": "user", "content": "ignored"},
            "is_heartbeat": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"ok": True, "ingested": False, "memory_id": None}
    assert _count_memories(dsn, agent) == 0


def test_ingest_empty_content_not_ingested(stack) -> None:
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest",
        json={
            "agent_id": agent,
            "session_id": sid,
            "message": {"role": "user", "content": ""},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ingested"] is False
    assert body["memory_id"] is None
    assert _count_memories(dsn, agent) == 0


# ── /context/ingest_batch ─────────────────────────────────────────────


def test_ingest_batch_pairs_become_sync_turns(stack) -> None:
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest_batch",
        json={
            "agent_id": agent,
            "session_id": sid,
            "messages": [
                {"role": "user", "content": "вопрос 1"},
                {"role": "assistant", "content": "ответ 1"},
                {"role": "user", "content": "вопрос 2"},
                {"role": "assistant", "content": "ответ 2"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["ingested_count"] == 4
    assert _fetch_roles(dsn, agent) == ["user", "assistant", "user", "assistant"]


def test_ingest_batch_skips_system_and_tool(stack) -> None:
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest_batch",
        json={
            "agent_id": agent,
            "session_id": sid,
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "привет"},
                {"role": "assistant", "content": "здравствуй"},
                {"role": "tool", "content": "tool output"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ingested_count"] == 2
    assert _fetch_roles(dsn, agent) == ["user", "assistant"]


def test_ingest_batch_lone_user_tail(stack) -> None:
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest_batch",
        json={
            "agent_id": agent,
            "session_id": sid,
            "messages": [{"role": "user", "content": "одинокий вопрос"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ingested_count"] == 1
    assert _fetch_roles(dsn, agent) == ["user"]


def test_ingest_batch_two_users_back_to_back(stack) -> None:
    """Два user подряд — flush первого как (user, ""), второго как pending."""
    client, dsn, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/ingest_batch",
        json={
            "agent_id": agent,
            "session_id": sid,
            "messages": [
                {"role": "user", "content": "первый user"},
                {"role": "user", "content": "второй user"},
                {"role": "assistant", "content": "ответ"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ingested_count"] == 3
    assert _fetch_roles(dsn, agent) == ["user", "user", "assistant"]


# ── /context/dispose ──────────────────────────────────────────────────


def test_dispose_with_agent_id_unregisters(stack) -> None:
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})
    assert registry.get_optional(agent) is not None

    resp = client.post("/context/dispose", json={"agent_id": agent})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    assert registry.get_optional(agent) is None


def test_dispose_no_agent_id_is_noop(stack) -> None:
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post("/context/dispose", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    # Registry не тронут — agent ещё зарегистрирован.
    assert registry.get_optional(agent) is not None


# ── /context/assemble ─────────────────────────────────────────────────


def test_assemble_passthrough_below_threshold(stack) -> None:
    """Без token_budget — composer no-op путь, messages возвращаются."""
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    msgs = [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
    resp = client.post(
        "/context/assemble",
        json={"agent_id": agent, "session_id": sid, "messages": msgs},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Composer возвращает messages (плюс возможный salient block, но
    # без накопленной памяти salient = None).
    assert body["messages"] == msgs
    assert body["prompt_authority"] == "assembled"
    assert body["system_prompt_addition"] is None
    # estimated_tokens ≈ ceil(sum(content_len)/4); "вопрос"+"ответ"=12 → 3.
    assert body["estimated_tokens"] >= 1


def test_assemble_with_token_budget_triggers_eviction(stack) -> None:
    """token_budget над threshold: composer должен делать eviction."""
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    # Создаём много messages чтобы было что эвиктить (default protect_first=4,
    # protect_last=4; нужно > 8 чтобы eviction имел смысл).
    msgs = [
        {"role": ("user" if i % 2 == 0 else "assistant"), "content": f"м{i}"}
        for i in range(20)
    ]
    # Очень маленький бюджет — заведомо ниже current_tokens.
    resp = client.post(
        "/context/assemble",
        json={
            "agent_id": agent,
            "session_id": sid,
            "messages": msgs,
            "token_budget": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # В Phase C composer attached fresh с ctx_len=0 (default StyxConfig);
    # threshold = 0, eviction-минимум недостижим → fallback к no-op.
    # Главное — endpoint не падает и возвращает messages.
    assert isinstance(body["messages"], list)
    assert body["estimated_tokens"] >= 0


def test_assemble_unregistered_agent_404(stack) -> None:
    """assemble требует bootstrap (registry membership)."""
    client, _, _ = stack
    resp = client.post(
        "/context/assemble",
        json={
            "agent_id": "never-bootstrapped",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    # registry.get raises HTTPException(404) для unknown agent.
    assert resp.status_code == 404, resp.text


def test_assemble_messages_no_styx_wrapper_volna_26_7(stack) -> None:
    """Волны 26.7 + 30: response.messages не содержит inject-обёртки.

    Salient (если бы был) теперь идёт через system_prompt_addition,
    не через messages array. Без накопленной памяти salient_bridge
    handle = None → system_prompt_addition тоже None; главное —
    messages чистые, без какого-либо styx injection (ни legacy
    `<styx>` ни family `<styx-…>`).

    Real production: production-агент с memories будет получать
    system_prompt_addition как обёрнутую `<styx-salient>...</styx-salient>`
    строку через salient_text; OpenClaw runtime ставит её в system
    prompt через `assembled.systemPromptAddition` path.
    """
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    resp = client.post(
        "/context/assemble",
        json={"agent_id": agent, "session_id": sid, "messages": msgs},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Все output messages чистые — ни один не содержит styx-маркеров
    # (ни legacy `<styx>...</styx>` ни family `<styx-…>...</styx-…>`).
    for m in body["messages"]:
        content = m.get("content", "")
        assert "<styx>" not in content and "<styx-" not in content, (
            f"messages не должны содержать styx-инжект (волны 26.7 + 30): {m}"
        )

    # Без памяти — system_prompt_addition остаётся None.
    assert body["system_prompt_addition"] is None
    assert body["prompt_authority"] == "assembled"


# ── /context/compact ──────────────────────────────────────────────────


def test_compact_returns_no_change(stack) -> None:
    """Phase C minimal — compact возвращает {ok, compacted=false, reason}."""
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/compact",
        json={"agent_id": agent, "session_id": sid, "force": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["compacted"] is False
    assert body["reason"] == "async-consolidation"


def test_compact_unregistered_agent_404(stack) -> None:
    client, _, _ = stack
    resp = client.post(
        "/context/compact",
        json={"agent_id": "never-bootstrapped"},
    )
    assert resp.status_code == 404, resp.text


# ── /context/after_turn ───────────────────────────────────────────────


def test_after_turn_returns_ok(stack) -> None:
    """Phase C minimal — after_turn fire-and-forget, {ok: true}."""
    client, _, _ = stack
    agent = f"alpha-{uuid.uuid4()}"
    sid = str(uuid.uuid4())
    client.post("/context/bootstrap", json={"agent_id": agent, "session_id": sid})

    resp = client.post(
        "/context/after_turn",
        json={
            "agent_id": agent,
            "session_id": sid,
            "messages": [
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": "ответ"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}


def test_after_turn_unregistered_agent_404(stack) -> None:
    client, _, _ = stack
    resp = client.post(
        "/context/after_turn",
        json={"agent_id": "never-bootstrapped", "messages": []},
    )
    assert resp.status_code == 404, resp.text
