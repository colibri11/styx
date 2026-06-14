"""Тесты styx_recall handle_tool_call в HTTP-wrapper.

После split (v1.0.0) ``StyxMemoryProvider`` — тонкий HTTP клиент,
не имеет внутреннего embed/queries. Поведение handle_tool_call
проверяется через mock ``StyxCoreClient`` — проверяем логику
адаптера между Hermes ABC и HTTP API контрактом /recall.

Полный end-to-end recall (с реальной БД и Ollama) валидируется
ручным smoke в Docker compose стеке либо тестами в core
``tests/integration/`` (которые тестируют ``StyxMemoryCore`` без
HTTP-обёртки).
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
    """Фейк StyxCoreClient — фиксирует вызовы и возвращает заданный response."""

    def __init__(self, recall_response: dict | None = None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._recall_response = recall_response or {
            "memories": [],
            "queried_count": 0,
            "internal_duplicates_removed": 0,
            "elapsed_ms": 0,
        }
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        self.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {"agent_id": agent_id, "tools": [
            {
                "name": "styx_recall",
                "description": "...",
                "parameters": {"type": "object", "properties": {}, "required": ["query"]},
            }
        ]}

    def shutdown_agent(self, agent_id):
        self.calls.append(("shutdown_agent", (agent_id,), {}))

    def recall(self, agent_id, query, **kwargs):
        self.calls.append(("recall", (agent_id, query), kwargs))
        return self._recall_response

    def sync_turn(self, agent_id, **kwargs):
        self.calls.append(("sync_turn", (agent_id,), kwargs))
        return {"memory_ids": [], "recall_event_ids": []}

    def close(self):
        self.closed = True


def _provider_with_fake(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: _FakeClient,
) -> StyxMemoryProvider:
    """Заменяет StyxCoreClient на _FakeClient, инициализирует provider'а."""
    monkeypatch.setattr(
        "styx_hermes.providers.memory.StyxCoreClient",
        lambda *a, **kw: fake_client,
    )
    p = StyxMemoryProvider()
    p.initialize(session_id="sid-test", agent_identity="alpha")
    return p


# ── schema ────────────────────────────────────────────────────────────


def test_get_tool_schemas_static_catalog_before_initialize() -> None:
    """До /agent/initialize tool schemas — статический каталог ядра (не пуст).

    Hermes строит routing-индекс _tool_to_provider в add_provider() ДО
    initialize(); пустой ответ тут = каждый styx_* tool-call упадёт в
    "Unknown tool". До init отдаём чистый каталог StyxMemoryCore, после —
    авторитетные daemon-схемы (см. test_get_tool_schemas_populated_*)."""
    p = StyxMemoryProvider()
    names = {s["name"] for s in p.get_tool_schemas()}
    assert "styx_recall" in names


def test_get_tool_schemas_populated_after_initialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """После initialize tools заполняются из ответа core."""
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    schemas = p.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "styx_recall"
    p.shutdown()


# ── handler ───────────────────────────────────────────────────────────


def test_handle_tool_call_without_initialize_returns_error() -> None:
    p = StyxMemoryProvider()
    out = json.loads(p.handle_tool_call("styx_recall", {"query": "x"}))
    assert "error" in out


def test_handle_tool_call_unknown_tool_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermes ABC дефолт для неизвестного tool — NotImplementedError."""
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        with pytest.raises(NotImplementedError):
            p.handle_tool_call("unknown_tool", {})
    finally:
        p.shutdown()


def test_handle_tool_call_empty_query_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call("styx_recall", {"query": ""}))
        assert "error" in out
        assert "query" in out["error"].lower()
    finally:
        p.shutdown()


def test_handle_tool_call_proxies_to_client_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Корректный query → /recall с agent_id и формат JSON ответа."""
    fake = _FakeClient(
        recall_response={
            "memories": [
                {
                    "id": "uuid-1",
                    "content": "что мы решили по волнам Styx?",
                    "score": 1.0,
                    "role": "user",
                    "created_at": None,
                }
            ],
            "queried_count": 5,
            "internal_duplicates_removed": 1,
            "elapsed_ms": 42,
        }
    )
    p = _provider_with_fake(monkeypatch, fake)
    try:
        raw = p.handle_tool_call(
            "styx_recall", {"query": "что мы решили по волнам Styx?"}
        )
        out = json.loads(raw)
        assert "memories_text" in out
        assert "что мы решили по волнам Styx?" in out["memories_text"]
        assert out["count"] == 1
        assert out["queried_count"] == 5
        assert out["duplicates_removed"] == 1

        recall_calls = [c for c in fake.calls if c[0] == "recall"]
        assert len(recall_calls) == 1
        args = recall_calls[0][1]
        assert args == ("alpha", "что мы решили по волнам Styx?")
    finally:
        p.shutdown()


def test_handle_tool_call_respects_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """limit в допустимом диапазоне передаётся клиенту."""
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_recall", {"query": "x", "limit": 5})
        recall_calls = [c for c in fake.calls if c[0] == "recall"]
        kwargs = recall_calls[0][2]
        assert kwargs.get("limit") == 5
    finally:
        p.shutdown()


def test_handle_tool_call_clamps_invalid_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """limit вне диапазона [1, 20] — не передаётся (None)."""
    fake = _FakeClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        p.handle_tool_call("styx_recall", {"query": "x", "limit": 0})
        recall_calls = [c for c in fake.calls if c[0] == "recall"]
        kwargs = recall_calls[0][2]
        assert kwargs.get("limit") is None

        # limit=999 — тоже clamped.
        fake.calls.clear()
        p.handle_tool_call("styx_recall", {"query": "x", "limit": 999})
        kwargs2 = [c for c in fake.calls if c[0] == "recall"][0][2]
        assert kwargs2.get("limit") is None
    finally:
        p.shutdown()


def test_handle_tool_call_propagates_recall_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сетевая ошибка /recall → JSON с error, без exception."""

    class _BoomClient(_FakeClient):
        def recall(self, agent_id, query, **kwargs):
            raise RuntimeError("network down")

    fake = _BoomClient()
    p = _provider_with_fake(monkeypatch, fake)
    try:
        out = json.loads(p.handle_tool_call("styx_recall", {"query": "x"}))
        assert "error" in out
    finally:
        p.shutdown()
