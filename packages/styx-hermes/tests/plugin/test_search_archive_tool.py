"""Тесты styx_search_archive handle_tool_call в HTTP-wrapper (волна 20).

Pattern зеркалит test_recall_tool.py: mock StyxCoreClient, проверяем
что Hermes plugin корректно адаптирует tool call в HTTP /search_archive.
End-to-end (real БД) — в core integration tests + Docker batch suite.
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
    def __init__(self, search_response: dict | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._search_response = search_response or {
            "results": [], "total_matched": 0,
        }
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        self.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {
            "agent_id": agent_id,
            "tools": [
                {
                    "name": "styx_search_archive",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "scope": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                }
            ],
        }

    def shutdown_agent(self, agent_id):
        self.calls.append(("shutdown_agent", (agent_id,), {}))

    def search_archive(self, agent_id, query, **kwargs):
        self.calls.append(("search_archive", (agent_id, query), kwargs))
        return self._search_response

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


def test_get_tool_schemas_includes_search_archive(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        names = {s["name"] for s in p.get_tool_schemas()}
        assert "styx_search_archive" in names
    finally:
        p.shutdown()


# ── handler ───────────────────────────────────────────────────────────


def test_handle_tool_call_without_initialize_returns_error() -> None:
    p = StyxMemoryProvider()
    out = json.loads(p.handle_tool_call("styx_search_archive", {"query": "x"}))
    assert "error" in out


def test_handle_tool_call_empty_query_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call("styx_search_archive", {"query": ""}))
        assert "error" in out
        assert "query" in out["error"].lower()
    finally:
        p.shutdown()


def test_handle_tool_call_invalid_scope_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call(
            "styx_search_archive", {"query": "x", "scope": "weird"},
        ))
        assert "error" in out
        assert "scope" in out["error"].lower()
    finally:
        p.shutdown()


def test_handle_tool_call_proxies_with_default_scope(monkeypatch) -> None:
    fake = _FakeClient(
        search_response={
            "results": [
                {
                    "scope": "document",
                    "text": "stitched region",
                    "snippet": "stitched",
                    "score": 0.9,
                    "document_id": "doc-uuid",
                    "chunk_positions": [0, 1],
                }
            ],
            "total_matched": 1,
        }
    )
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_search_archive", {"query": "что мы решили"},
        )
        out = json.loads(raw)
        assert out["total_matched"] == 1
        assert out["results"][0]["text"] == "stitched region"

        search_calls = [c for c in fake.calls if c[0] == "search_archive"]
        assert len(search_calls) == 1
        agent_arg, query_arg = search_calls[0][1]
        kwargs = search_calls[0][2]
        assert agent_arg == "alpha"
        assert query_arg == "что мы решили"
        assert kwargs.get("scope") == "all"
    finally:
        p.shutdown()


def test_handle_tool_call_passes_scope_and_limit(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_search_archive", {
            "query": "test", "scope": "documents", "limit": 5,
        })
        kwargs = [c for c in fake.calls if c[0] == "search_archive"][0][2]
        assert kwargs["scope"] == "documents"
        assert kwargs["limit"] == 5
    finally:
        p.shutdown()


def test_handle_tool_call_passes_date_filters(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_search_archive", {
            "query": "x", "date_from": "2026-01-01T00:00:00Z",
            "date_to": "2026-12-31T23:59:59Z",
        })
        kwargs = [c for c in fake.calls if c[0] == "search_archive"][0][2]
        assert kwargs["date_from"] == "2026-01-01T00:00:00Z"
        assert kwargs["date_to"] == "2026-12-31T23:59:59Z"
    finally:
        p.shutdown()


def test_handle_tool_call_propagates_network_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def search_archive(self, agent_id, query, **kwargs):
            raise RuntimeError("network down")

    fake = _BoomClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call(
            "styx_search_archive", {"query": "x"},
        ))
        assert "error" in out
        assert "search_archive failed" in out["error"]
    finally:
        p.shutdown()


def test_handle_tool_call_clamps_invalid_limit(monkeypatch) -> None:
    """limit ≤ 0 → не передаётся (None)."""
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_search_archive", {"query": "x", "limit": 0})
        kwargs = [c for c in fake.calls if c[0] == "search_archive"][0][2]
        assert kwargs["limit"] is None
    finally:
        p.shutdown()
