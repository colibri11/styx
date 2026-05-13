"""POST /dialogue/* integration: TestClient + real Postgres + Ollama
(волна 24).

5 routes (save / search / recent / sessions / prepare_summary). Все
читают/пишут `memories WHERE role IN ('user','assistant')` с
agent_id-isolation. Hermes wrapper отсутствует (D10).
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


def _count_memories(dsn: str, agent: str) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories "
                "WHERE agent_id=%s AND role IN ('user','assistant')",
                (agent,),
            )
            return cur.fetchone()[0]


# ── /dialogue/save ──────────────────────────────────────────────────


def test_save_returns_memory_id(stack) -> None:
    client, agent, dsn, _, _ = stack
    resp = client.post(
        "/dialogue/save",
        json={
            "agent_id": agent,
            "role": "user",
            "content": "явно сохранённая реплика",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory_id"]
    assert _count_memories(dsn, agent) == 1


def test_save_with_session_id_creates_session_row(stack) -> None:
    client, agent, dsn, _, _ = stack
    sid = str(uuid.uuid4())
    resp = client.post(
        "/dialogue/save",
        json={
            "agent_id": agent,
            "role": "assistant",
            "content": "ответ",
            "session_id": sid,
        },
    )
    assert resp.status_code == 200, resp.text
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM sessions WHERE id=%s AND agent_id=%s",
                (sid, agent),
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT session_id, role, content FROM memories "
                "WHERE id=%s",
                (body_memory_id := resp.json()["memory_id"],),
            )
            row = cur.fetchone()
            assert str(row[0]) == sid
            assert row[1] == "assistant"
            assert row[2] == "ответ"


def test_save_invalid_role_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/dialogue/save",
        json={
            "agent_id": agent,
            "role": "system",
            "content": "x",
        },
    )
    assert resp.status_code == 422


def test_save_content_too_long_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/dialogue/save",
        json={
            "agent_id": agent,
            "role": "user",
            "content": "x" * 2401,
        },
    )
    assert resp.status_code == 422


# ── /dialogue/search ────────────────────────────────────────────────


def test_search_hybrid_finds_matching_content(stack) -> None:
    client, agent, _, _, _ = stack
    sid = str(uuid.uuid4())
    for content in (
        "postgres performance tuning",
        "погода сегодня хорошая",
        "выходные на даче",
    ):
        client.post(
            "/dialogue/save",
            json={
                "agent_id": agent,
                "role": "user",
                "content": content,
                "session_id": sid,
            },
        )

    resp = client.post(
        "/dialogue/search",
        json={
            "agent_id": agent,
            "query": "postgres",
            "limit": 5,
        },
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) >= 1
    assert results[0]["content"] == "postgres performance tuning"
    assert 0.0 <= results[0]["score"]


def test_search_semantic_only_pure_vector(stack) -> None:
    client, agent, _, _, _ = stack
    sid = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "обсудили рабочий проект", "session_id": sid},
    )
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "assistant",
              "content": "что-то совсем другое", "session_id": sid},
    )

    resp = client.post(
        "/dialogue/search",
        json={
            "agent_id": agent,
            "query": "проект и работа",
            "semantic_only": True,
            "limit": 5,
        },
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 2
    # Pure-vector score = 1 - distance ∈ [0..1]
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_search_session_filter(stack) -> None:
    client, agent, _, _, _ = stack
    s1 = str(uuid.uuid4())
    s2 = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "сессия один", "session_id": s1},
    )
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "сессия два", "session_id": s2},
    )

    resp = client.post(
        "/dialogue/search",
        json={
            "agent_id": agent,
            "query": "сессия",
            "session_id": s1,
            "limit": 10,
        },
    )
    assert resp.status_code == 200
    contents = {r["content"] for r in resp.json()["results"]}
    assert contents == {"сессия один"}


def test_search_agent_isolation(stack, clean_db: str) -> None:
    client, agent_a, dsn, _, cfg = stack
    core_b = StyxMemoryCore(agent_id="beta")
    core_b._config = cfg
    core_b.initialize(session_id=str(uuid.uuid4()), agent_identity="beta")
    registry.register(agent_id="beta", core=core_b)
    try:
        client.post(
            "/dialogue/save",
            json={"agent_id": agent_a, "role": "user",
                  "content": "alpha private content"},
        )
        client.post(
            "/dialogue/save",
            json={"agent_id": "beta", "role": "user",
                  "content": "beta private content"},
        )

        resp_a = client.post(
            "/dialogue/search",
            json={"agent_id": agent_a, "query": "private", "limit": 10},
        )
        resp_b = client.post(
            "/dialogue/search",
            json={"agent_id": "beta", "query": "private", "limit": 10},
        )
        contents_a = {r["content"] for r in resp_a.json()["results"]}
        contents_b = {r["content"] for r in resp_b.json()["results"]}
        assert "alpha private content" in contents_a
        assert "beta private content" not in contents_a
        assert "beta private content" in contents_b
        assert "alpha private content" not in contents_b
    finally:
        core_b.shutdown()


def test_search_empty_query_422(stack) -> None:
    client, agent, _, _, _ = stack
    resp = client.post(
        "/dialogue/search",
        json={"agent_id": agent, "query": ""},
    )
    assert resp.status_code == 422


# ── /dialogue/recent ────────────────────────────────────────────────


def test_recent_chronological_order(stack) -> None:
    client, agent, _, _, _ = stack
    sid = str(uuid.uuid4())
    for content in ("первая", "вторая", "третья"):
        client.post(
            "/dialogue/save",
            json={"agent_id": agent, "role": "user",
                  "content": content, "session_id": sid},
        )

    resp = client.post(
        "/dialogue/recent",
        json={"agent_id": agent, "limit": 10},
    )
    assert resp.status_code == 200, resp.text
    rows = resp.json()["rows"]
    assert [r["content"] for r in rows] == ["первая", "вторая", "третья"]


def test_recent_filters_non_dialogue_roles(stack) -> None:
    client, agent, dsn, _, _ = stack
    sid = str(uuid.uuid4())
    # Через API можем сохранить только user/assistant. Запишем system
    # напрямую через psycopg для теста фильтра.
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (sid, agent),
            )
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, session_id) "
                "VALUES (%s, 'system', 'system row', %s)",
                (agent, sid),
            )
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, session_id) "
                "VALUES (%s, 'tool', 'tool row', %s)",
                (agent, sid),
            )
            conn.commit()
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "dialog row", "session_id": sid},
    )

    resp = client.post(
        "/dialogue/recent",
        json={"agent_id": agent, "limit": 10},
    )
    rows = resp.json()["rows"]
    contents = {r["content"] for r in rows}
    assert "dialog row" in contents
    assert "system row" not in contents
    assert "tool row" not in contents


def test_recent_session_filter(stack) -> None:
    client, agent, _, _, _ = stack
    s1 = str(uuid.uuid4())
    s2 = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "s1", "session_id": s1},
    )
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "s2", "session_id": s2},
    )

    resp = client.post(
        "/dialogue/recent",
        json={"agent_id": agent, "session_id": s1, "limit": 10},
    )
    rows = resp.json()["rows"]
    assert [r["content"] for r in rows] == ["s1"]


# ── /dialogue/sessions ──────────────────────────────────────────────


def test_sessions_list_with_counts(stack) -> None:
    client, agent, _, _, _ = stack
    s1 = str(uuid.uuid4())
    s2 = str(uuid.uuid4())
    for c in ("a", "b"):
        client.post(
            "/dialogue/save",
            json={"agent_id": agent, "role": "user",
                  "content": c, "session_id": s1},
        )
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "x", "session_id": s2},
    )

    resp = client.post(
        "/dialogue/sessions",
        json={"agent_id": agent, "limit": 10},
    )
    assert resp.status_code == 200, resp.text
    sessions = resp.json()["sessions"]
    by_id = {s["session_id"]: s["message_count"] for s in sessions}
    assert by_id == {s1: 2, s2: 1}


def test_sessions_skips_null_session(stack) -> None:
    client, agent, _, _, _ = stack
    sid = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "with session", "session_id": sid},
    )
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "no session"},  # session_id=None
    )

    resp = client.post(
        "/dialogue/sessions",
        json={"agent_id": agent, "limit": 10},
    )
    sessions = resp.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == sid
    assert sessions[0]["message_count"] == 1


# ── /dialogue/prepare_summary ───────────────────────────────────────


def test_prepare_summary_transcript_format(stack) -> None:
    client, agent, _, _, _ = stack
    sid = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "привет", "session_id": sid},
    )
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "assistant",
              "content": "привет в ответ", "session_id": sid},
    )

    resp = client.post(
        "/dialogue/prepare_summary",
        json={"agent_id": agent, "session_id": sid},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == sid
    assert body["message_count"] == 2
    assert body["first_message_at"]
    assert body["last_message_at"]
    transcript = body["transcript"]
    lines = transcript.split("\n")
    assert len(lines) == 2
    # [YYYY-MM-DD HH:MM:SS] Speaker: content
    import re
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Human: привет$",
        lines[0],
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Agent: привет в ответ$",
        lines[1],
    )


def test_prepare_summary_empty_session(stack) -> None:
    client, agent, dsn, _, _ = stack
    sid = str(uuid.uuid4())
    # Создаём session напрямую (без реплик).
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES (%s, %s)",
                (sid, agent),
            )
            conn.commit()

    resp = client.post(
        "/dialogue/prepare_summary",
        json={"agent_id": agent, "session_id": sid},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message_count"] == 0
    assert body["transcript"] == ""
    assert body["first_message_at"] is None
    assert body["last_message_at"] is None


def test_prepare_summary_filters_non_dialogue_roles(stack) -> None:
    client, agent, dsn, _, _ = stack
    sid = str(uuid.uuid4())
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES (%s, %s)",
                (sid, agent),
            )
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, session_id) "
                "VALUES (%s, 'system', 'system msg', %s)",
                (agent, sid),
            )
            conn.commit()
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "user msg", "session_id": sid},
    )

    resp = client.post(
        "/dialogue/prepare_summary",
        json={"agent_id": agent, "session_id": sid},
    )
    body = resp.json()
    assert body["message_count"] == 1
    assert "user msg" in body["transcript"]
    assert "system msg" not in body["transcript"]


# ── feature flag ────────────────────────────────────────────────────


# ── handle_tool_call (Hermes wrapper in-process путь) ──────────────


def test_handle_tool_call_dialogue_search_in_process(stack) -> None:
    """styx_dialogue_search через core.handle_tool_call — in-process путь.

    Hermes plugin ходит по HTTP через client.dialogue_search; in-process
    callers (OpenClaw plugin same-process) — через handle_tool_call.
    """
    import json as _json

    client, agent, _, core, _ = stack
    sid = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "tool call test reply", "session_id": sid},
    )

    raw = core.handle_tool_call(
        "styx_dialogue_search",
        {"query": "tool call test", "limit": 5},
    )
    out = _json.loads(raw)
    assert "results" in out
    assert len(out["results"]) >= 1
    assert "tool call test reply" in out["results"][0]["content"]


def test_handle_tool_call_dialogue_prepare_summary_in_process(stack) -> None:
    import json as _json

    client, agent, _, core, _ = stack
    sid = str(uuid.uuid4())
    client.post(
        "/dialogue/save",
        json={"agent_id": agent, "role": "user",
              "content": "tool reply", "session_id": sid},
    )

    raw = core.handle_tool_call(
        "styx_dialogue_prepare_summary",
        {"session_id": sid},
    )
    out = _json.loads(raw)
    assert out["session_id"] == sid
    assert out["message_count"] == 1
    assert "Human: tool reply" in out["transcript"]


def test_handle_tool_call_dialogue_recent_empty(stack) -> None:
    import json as _json
    _, agent, _, core, _ = stack
    raw = core.handle_tool_call("styx_dialogue_recent", {})
    out = _json.loads(raw)
    assert out["rows"] == []


def test_dialogue_disabled_returns_503(
    stack, clean_db: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, agent, _, core, _ = stack
    core.shutdown()
    registry.reset_all()
    monkeypatch.setenv("STYX_DIALOGUE_API_ENABLED", "0")
    cfg2 = replace(load_config(), database_url=clean_db, http_token=None)
    assert cfg2.dialogue_api_enabled is False
    new_core = StyxMemoryCore(agent_id=agent)
    new_core._config = cfg2
    new_core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    new_core._config = cfg2  # initialize перезатёр.
    registry.register(agent_id=agent, core=new_core)
    app2 = create_app(cfg2)
    client2 = TestClient(app2)
    try:
        for path, payload in [
            ("/dialogue/save",
             {"agent_id": agent, "role": "user", "content": "x"}),
            ("/dialogue/search",
             {"agent_id": agent, "query": "x"}),
            ("/dialogue/recent",
             {"agent_id": agent}),
            ("/dialogue/sessions",
             {"agent_id": agent}),
            ("/dialogue/prepare_summary",
             {"agent_id": agent, "session_id": str(uuid.uuid4())}),
        ]:
            resp = client2.post(path, json=payload)
            assert resp.status_code == 503, (path, resp.text)
            assert "disabled" in resp.json()["detail"].lower()
    finally:
        new_core.shutdown()
