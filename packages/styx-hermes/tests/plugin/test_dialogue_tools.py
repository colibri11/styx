"""Тесты styx_dialogue_* handle_tool_call в HTTP-wrapper (волна 24 follow-up).

3 read-only tools: search / recent / prepare_summary. Pattern зеркалит
test_search_archive_tool.py: mock StyxCoreClient, проверяем что
Hermes plugin корректно адаптирует tool call в HTTP-вызовы.
End-to-end (real БД) — в core integration tests + Docker batch suite.

styx_dialogue_save и styx_dialogue_sessions сознательно НЕ имеют
Hermes wrapper'ов: save — для plugin-канала (как ingest_experience),
sessions — administration без LLM use case.
"""

from __future__ import annotations

import json

import pytest

from styx_hermes import _agent_session
from styx_hermes.providers.memory import StyxMemoryProvider


@pytest.fixture(autouse=True)
def _reset_session():
    yield
    _agent_session.clear_session()


class _FakeClient:
    def __init__(
        self,
        search_response: dict | None = None,
        recent_response: dict | None = None,
        prepare_response: dict | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._search_response = search_response or {"results": []}
        self._recent_response = recent_response or {"rows": []}
        self._prepare_response = prepare_response or {
            "session_id": "sid",
            "message_count": 0,
            "first_message_at": None,
            "last_message_at": None,
            "transcript": "",
        }
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        self.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {
            "agent_id": agent_id,
            "tools": [
                {"name": "styx_dialogue_search", "description": "...",
                 "parameters": {"type": "object", "properties": {},
                                "required": []}},
            ],
        }

    def shutdown_agent(self, agent_id):
        self.calls.append(("shutdown_agent", (agent_id,), {}))

    def dialogue_search(self, agent_id, query, **kwargs):
        self.calls.append(("dialogue_search", (agent_id, query), kwargs))
        return self._search_response

    def dialogue_recent(self, agent_id, **kwargs):
        self.calls.append(("dialogue_recent", (agent_id,), kwargs))
        return self._recent_response

    def dialogue_prepare_summary(self, agent_id, session_id, **kwargs):
        self.calls.append(
            ("dialogue_prepare_summary", (agent_id, session_id), kwargs)
        )
        return self._prepare_response

    def sync_turn(self, *a, **kw):
        return {"memory_ids": [], "recall_event_ids": []}

    def close(self):
        self.closed = True


def _provider_with_fake(monkeypatch, fake_client) -> StyxMemoryProvider:
    monkeypatch.setattr(
        "styx_hermes.providers.memory.StyxCoreClient",
        lambda *a, **kw: fake_client,
    )
    p = StyxMemoryProvider()
    p.initialize(session_id="sid-test", agent_identity="alpha")
    return p


# ── schema discovery ─────────────────────────────────────────────────


def test_get_tool_schemas_includes_three_dialogue_tools(monkeypatch) -> None:
    """3 read-only dialogue tools обнаружены через initialize."""
    fake = _FakeClient()

    def _init(agent_id, **kwargs):
        fake.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {
            "agent_id": agent_id,
            "tools": [
                {"name": "styx_dialogue_search", "description": "",
                 "parameters": {"type": "object", "properties": {},
                                "required": ["query"]}},
                {"name": "styx_dialogue_recent", "description": "",
                 "parameters": {"type": "object", "properties": {},
                                "required": []}},
                {"name": "styx_dialogue_prepare_summary", "description": "",
                 "parameters": {"type": "object", "properties": {},
                                "required": ["session_id"]}},
            ],
        }
    fake.initialize_agent = _init  # type: ignore[method-assign]

    p = _provider_with_fake(monkeypatch, fake)
    try:
        names = {s["name"] for s in p.get_tool_schemas()}
        assert "styx_dialogue_search" in names
        assert "styx_dialogue_recent" in names
        assert "styx_dialogue_prepare_summary" in names
        # save и sessions сознательно НЕ обёрнуты:
        assert "styx_dialogue_save" not in names
        assert "styx_dialogue_sessions" not in names
    finally:
        p.shutdown()


# ── styx_dialogue_search ─────────────────────────────────────────────


def test_dialogue_search_without_initialize_returns_error() -> None:
    p = StyxMemoryProvider()
    out = json.loads(
        p.handle_tool_call("styx_dialogue_search", {"query": "x"}),
    )
    assert "error" in out


def test_dialogue_search_empty_query_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call("styx_dialogue_search", {"query": ""}),
        )
        assert "error" in out
        assert "query" in out["error"].lower()
    finally:
        p.shutdown()


def test_dialogue_search_proxies_default_hybrid(monkeypatch) -> None:
    fake = _FakeClient(
        search_response={
            "results": [{
                "memory_id": "m1", "role": "user", "content": "hello",
                "score": 0.9, "created_at": "2026-04-01T12:00:00Z",
                "session_id": "s1",
            }],
        }
    )
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_dialogue_search", {"query": "что обсуждали"},
        )
        out = json.loads(raw)
        assert len(out["results"]) == 1
        assert out["results"][0]["content"] == "hello"

        calls = [c for c in fake.calls if c[0] == "dialogue_search"]
        assert len(calls) == 1
        agent, query = calls[0][1]
        kwargs = calls[0][2]
        assert agent == "alpha"
        assert query == "что обсуждали"
        # Default semantic_only=False (hybrid).
        assert kwargs.get("semantic_only") is False
    finally:
        p.shutdown()


def test_dialogue_search_passes_semantic_only_flag(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_dialogue_search", {
            "query": "x", "semantic_only": True,
        })
        kwargs = [c for c in fake.calls if c[0] == "dialogue_search"][0][2]
        assert kwargs["semantic_only"] is True
    finally:
        p.shutdown()


def test_dialogue_search_passes_session_and_dates(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_dialogue_search", {
            "query": "x",
            "session_id": "sess-uuid",
            "after": "2026-01-01T00:00:00Z",
            "before": "2026-12-31T23:59:59Z",
            "limit": 5,
        })
        kwargs = [c for c in fake.calls if c[0] == "dialogue_search"][0][2]
        assert kwargs["session_id"] == "sess-uuid"
        assert kwargs["after"] == "2026-01-01T00:00:00Z"
        assert kwargs["before"] == "2026-12-31T23:59:59Z"
        assert kwargs["limit"] == 5
    finally:
        p.shutdown()


def test_dialogue_search_clamps_invalid_limit(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_dialogue_search", {"query": "x", "limit": 999})
        kwargs = [c for c in fake.calls if c[0] == "dialogue_search"][0][2]
        # limit > 50 — игнорируется, передаётся None.
        assert kwargs["limit"] is None
    finally:
        p.shutdown()


def test_dialogue_search_propagates_network_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def dialogue_search(self, agent_id, query, **kwargs):
            raise RuntimeError("network down")

    fake = _BoomClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call("styx_dialogue_search", {"query": "x"}),
        )
        assert "error" in out
        assert "dialogue_search failed" in out["error"]
    finally:
        p.shutdown()


# ── styx_dialogue_recent ─────────────────────────────────────────────


def test_dialogue_recent_proxies_default_limit(monkeypatch) -> None:
    fake = _FakeClient(
        recent_response={
            "rows": [{
                "memory_id": "m1", "role": "user", "content": "first",
                "created_at": "2026-04-01T12:00:00Z", "session_id": "s1",
            }, {
                "memory_id": "m2", "role": "assistant", "content": "second",
                "created_at": "2026-04-01T12:00:30Z", "session_id": "s1",
            }],
        }
    )
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call("styx_dialogue_recent", {})
        out = json.loads(raw)
        assert len(out["rows"]) == 2
        assert out["rows"][0]["content"] == "first"

        calls = [c for c in fake.calls if c[0] == "dialogue_recent"]
        assert len(calls) == 1
        assert calls[0][1] == ("alpha",)
    finally:
        p.shutdown()


def test_dialogue_recent_passes_session_and_before(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_dialogue_recent", {
            "session_id": "s1",
            "before": "2026-04-01T00:00:00Z",
            "limit": 50,
        })
        kwargs = [c for c in fake.calls if c[0] == "dialogue_recent"][0][2]
        assert kwargs["session_id"] == "s1"
        assert kwargs["before"] == "2026-04-01T00:00:00Z"
        assert kwargs["limit"] == 50
    finally:
        p.shutdown()


def test_dialogue_recent_propagates_network_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def dialogue_recent(self, agent_id, **kwargs):
            raise RuntimeError("boom")

    fake = _BoomClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call("styx_dialogue_recent", {}))
        assert "error" in out
        assert "dialogue_recent failed" in out["error"]
    finally:
        p.shutdown()


# ── styx_dialogue_prepare_summary ────────────────────────────────────


def test_prepare_summary_without_session_id_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call("styx_dialogue_prepare_summary", {}),
        )
        assert "error" in out
        assert "session_id" in out["error"].lower()
    finally:
        p.shutdown()


def test_prepare_summary_proxies_with_transcript(monkeypatch) -> None:
    fake = _FakeClient(
        prepare_response={
            "session_id": "sid",
            "message_count": 2,
            "first_message_at": "2026-04-01T12:00:00Z",
            "last_message_at": "2026-04-01T12:00:30Z",
            "transcript": (
                "[2026-04-01 12:00:00] Human: привет\n"
                "[2026-04-01 12:00:30] Agent: и тебе"
            ),
        }
    )
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_dialogue_prepare_summary",
            {"session_id": "sid"},
        )
        out = json.loads(raw)
        assert out["session_id"] == "sid"
        assert out["message_count"] == 2
        assert "Human:" in out["transcript"]
        assert "Agent:" in out["transcript"]

        calls = [
            c for c in fake.calls if c[0] == "dialogue_prepare_summary"
        ]
        assert len(calls) == 1
        agent, sid = calls[0][1]
        assert agent == "alpha"
        assert sid == "sid"
    finally:
        p.shutdown()


def test_prepare_summary_passes_limit(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_dialogue_prepare_summary", {
            "session_id": "sid", "limit": 500,
        })
        kwargs = [
            c for c in fake.calls if c[0] == "dialogue_prepare_summary"
        ][0][2]
        assert kwargs["limit"] == 500
    finally:
        p.shutdown()


def test_prepare_summary_propagates_network_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def dialogue_prepare_summary(self, agent_id, session_id, **kwargs):
            raise RuntimeError("boom")

    fake = _BoomClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call(
            "styx_dialogue_prepare_summary", {"session_id": "sid"},
        ))
        assert "error" in out
        assert "dialogue_prepare_summary failed" in out["error"]
    finally:
        p.shutdown()
