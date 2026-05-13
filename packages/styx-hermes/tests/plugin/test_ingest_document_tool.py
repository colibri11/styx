"""Тесты styx_ingest_document handle_tool_call в Hermes plugin (волна 28).

Pattern зеркалит test_reinterpret_tool: mock StyxCoreClient, проверяем
что Hermes plugin корректно адаптирует tool call в HTTP /ingest_document.
End-to-end (real file + DB) — в core integration tests + Docker batch
suite.
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
            "document_id": "00000000-0000-0000-0000-000000000001",
            "deduplicated": False,
            "chunks_count": 3,
            "mime_type": "application/pdf",
            "original_name": "spec.pdf",
            "size_bytes": 12345,
            "char_count": 5000,
            "content_hash": "a" * 64,
        }
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        self.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {
            "agent_id": agent_id,
            "tools": [
                {
                    "name": "styx_ingest_document",
                    "description": "archive file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "source_ref": {"type": "string"},
                            "visibility": {"type": "string"},
                            "metadata": {"type": "object"},
                            "content_hash": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            ],
        }

    def shutdown_agent(self, agent_id):
        self.calls.append(("shutdown_agent", (agent_id,), {}))

    def ingest_document(self, agent_id, path, **kwargs):
        self.calls.append(("ingest_document", (agent_id, path), kwargs))
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


def test_get_tool_schemas_includes_ingest_document(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        names = {s["name"] for s in p.get_tool_schemas()}
        assert "styx_ingest_document" in names
    finally:
        p.shutdown()


# ── happy path ─────────────────────────────────────────────────────────


def test_handle_tool_call_passes_path_to_client(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_ingest_document",
                {"path": "/abs/path/spec.pdf"},
            )
        )
        assert out["document_id"] == "00000000-0000-0000-0000-000000000001"
        assert out["chunks_count"] == 3
        assert out["deduplicated"] is False
        # FakeClient зарегистрировал вызов.
        ingest_calls = [c for c in fake.calls if c[0] == "ingest_document"]
        assert len(ingest_calls) == 1
        _, args, kwargs = ingest_calls[0]
        assert args == ("alpha", "/abs/path/spec.pdf")
        assert kwargs["source_ref"] is None
        assert kwargs["visibility"] is None
        assert kwargs["metadata"] == {}
        assert kwargs["content_hash"] is None
    finally:
        p.shutdown()


def test_handle_tool_call_forwards_optional_params(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call(
            "styx_ingest_document",
            {
                "path": "/abs/x.docx",
                "source_ref": "ticket-7",
                "visibility": "private",
                "metadata": {"tag": "spec"},
                "content_hash": "f" * 64,
            },
        )
        _, _, kwargs = fake.calls[-1]
        assert kwargs["source_ref"] == "ticket-7"
        assert kwargs["visibility"] == "private"
        assert kwargs["metadata"] == {"tag": "spec"}
        assert kwargs["content_hash"] == "f" * 64
    finally:
        p.shutdown()


def test_handle_tool_call_deduplicated_response(monkeypatch) -> None:
    dedup = {
        "document_id": "00000000-0000-0000-0000-000000000099",
        "deduplicated": True,
        "chunks_count": 0,
        "mime_type": "application/pdf",
        "original_name": "spec.pdf",
        "size_bytes": 12345,
        "char_count": 0,
        "content_hash": "a" * 64,
    }
    fake = _FakeClient(response=dedup)
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_ingest_document",
                {"path": "/abs/spec.pdf"},
            )
        )
        assert out["deduplicated"] is True
        assert out["chunks_count"] == 0
    finally:
        p.shutdown()


# ── error paths ───────────────────────────────────────────────────────


def test_handle_tool_call_without_initialize_returns_error() -> None:
    p = StyxMemoryProvider()
    out = json.loads(
        p.handle_tool_call(
            "styx_ingest_document",
            {"path": "/abs/x.pdf"},
        )
    )
    assert "error" in out
    assert "before initialize" in out["error"]


def test_handle_tool_call_missing_path_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call("styx_ingest_document", {}))
        assert "error" in out
        assert "path" in out["error"]
    finally:
        p.shutdown()


def test_handle_tool_call_empty_path_returns_error(monkeypatch) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call("styx_ingest_document", {"path": "  "})
        )
        assert "error" in out
        assert "path" in out["error"]
    finally:
        p.shutdown()


def test_handle_tool_call_invalid_metadata_type_returns_error(
    monkeypatch,
) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_ingest_document",
                {"path": "/abs/x.pdf", "metadata": "not a dict"},
            )
        )
        assert "error" in out
        assert "metadata" in out["error"]
    finally:
        p.shutdown()


def test_handle_tool_call_http_exception_returns_error(monkeypatch) -> None:
    class _FailingClient(_FakeClient):
        def ingest_document(self, *a, **kw):
            raise RuntimeError("daemon down")

    fake = _FailingClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(
            p.handle_tool_call(
                "styx_ingest_document",
                {"path": "/abs/x.pdf"},
            )
        )
        assert "error" in out
        assert "ingest_document failed" in out["error"]
        assert "daemon down" in out["detail"]
    finally:
        p.shutdown()
