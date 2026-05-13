"""Тесты styx_reinterpret handle_tool_call в Hermes plugin (волна 22).

Pattern зеркалит test_search_archive_tool: mock StyxCoreClient,
проверяем что Hermes plugin корректно адаптирует tool call в HTTP
/reinterpret.
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
    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._response = response or {
            "status": "queued",
            "memory_id": "00000000-0000-0000-0000-000000000001",
            "task_id": "00000000-0000-0000-0000-000000000002",
            "application_id": 7,
            "message": "queued",
        }
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        self.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {
            "agent_id": agent_id,
            "tools": [
                {
                    "name": "styx_reinterpret",
                    "description": "explicit reinterpret",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "new_understanding_text": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": [
                            "memory_id", "new_understanding_text",
                        ],
                    },
                },
            ],
        }

    def shutdown_agent(self, agent_id):
        self.calls.append(("shutdown_agent", (agent_id,), {}))

    def reinterpret(
        self, agent_id, memory_id, text, **kwargs,
    ):
        self.calls.append(
            ("reinterpret", (agent_id, memory_id, text), kwargs)
        )
        return self._response

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


# ── schema ────────────────────────────────────────────────────────────


def test_get_tool_schemas_includes_reinterpret(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        names = {s["name"] for s in p.get_tool_schemas()}
        assert "styx_reinterpret" in names
    finally:
        p.shutdown()


# ── error paths ───────────────────────────────────────────────────────


def test_handle_tool_call_without_initialize_returns_error() -> None:
    p = StyxMemoryProvider()
    out = json.loads(
        p.handle_tool_call(
            "styx_reinterpret",
            {"memory_id": "x", "new_understanding_text": "y"},
        )
    )
    assert "error" in out


def test_handle_tool_call_missing_memory_id_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_reinterpret",
                {"new_understanding_text": "x"},
            )
        )
        assert "error" in out
        assert "memory_id" in out["error"]
    finally:
        p.shutdown()


def test_handle_tool_call_missing_text_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_reinterpret",
                {"memory_id": "00000000-0000-0000-0000-000000000001"},
            )
        )
        assert "error" in out
        assert "new_understanding_text" in out["error"]
    finally:
        p.shutdown()


def test_handle_tool_call_invalid_weight_type_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_reinterpret",
                {
                    "memory_id": "00000000-0000-0000-0000-000000000001",
                    "new_understanding_text": "x",
                    "weight": "not-a-number",
                },
            )
        )
        assert "error" in out
        assert "weight" in out["error"]
    finally:
        p.shutdown()


# ── happy path: passthrough ──────────────────────────────────────────


def test_handle_tool_call_proxies_to_client(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_reinterpret",
            {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "new_understanding_text": "новое понимание",
            },
        )
        out = json.loads(raw)
        assert out["status"] == "queued"
        assert out["application_id"] == 7

        calls = [c for c in fake.calls if c[0] == "reinterpret"]
        assert len(calls) == 1
        agent_arg, mid_arg, text_arg = calls[0][1]
        assert agent_arg == "alpha"
        assert mid_arg == "00000000-0000-0000-0000-000000000001"
        assert text_arg == "новое понимание"
        assert calls[0][2]["weight"] is None
    finally:
        p.shutdown()


def test_handle_tool_call_passes_weight(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call(
            "styx_reinterpret",
            {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "new_understanding_text": "x",
                "weight": 0.7,
            },
        )
        kwargs = [c for c in fake.calls if c[0] == "reinterpret"][0][2]
        assert kwargs["weight"] == 0.7
    finally:
        p.shutdown()


def test_handle_tool_call_returns_cooldown_passthrough(monkeypatch) -> None:
    """409 cooldown body должен пробросить status='cooldown' для LLM."""
    fake = _FakeClient(response={
        "status": "cooldown",
        "memory_id": "00000000-0000-0000-0000-000000000001",
        "next_available_at": "2026-05-06T12:00:00+00:00",
        "last_reinterpreted_at": "2026-05-05T12:00:00+00:00",
    })
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_reinterpret",
            {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "new_understanding_text": "новое",
            },
        )
        out = json.loads(raw)
        assert out["status"] == "cooldown"
        assert out["next_available_at"] is not None
    finally:
        p.shutdown()


def test_handle_tool_call_returns_memory_not_found_passthrough(monkeypatch) -> None:
    fake = _FakeClient(response={
        "status": "memory_not_found",
        "memory_id": "00000000-0000-0000-0000-000000000099",
    })
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_reinterpret",
            {
                "memory_id": "00000000-0000-0000-0000-000000000099",
                "new_understanding_text": "x",
            },
        )
        out = json.loads(raw)
        assert out["status"] == "memory_not_found"
    finally:
        p.shutdown()


def test_handle_tool_call_propagates_network_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def reinterpret(self, agent_id, memory_id, text, **kwargs):
            raise RuntimeError("network down")

    fake = _BoomClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_reinterpret",
                {
                    "memory_id": "00000000-0000-0000-0000-000000000001",
                    "new_understanding_text": "x",
                },
            )
        )
        assert "error" in out
        assert "reinterpret failed" in out["error"]
    finally:
        p.shutdown()
