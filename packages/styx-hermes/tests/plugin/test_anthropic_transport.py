"""Тесты ``StyxAnthropicTransport`` + cache stats push (волна 29 Phase E).

Override `extract_cache_stats` после обработки родительским
``AnthropicTransport`` шлёт результат в core daemon через
``StyxCoreClient.push_cache_stats``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from styx_hermes import _agent_session


@pytest.fixture(autouse=True)
def _reset_session():
    yield
    _agent_session.clear_session()


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.base_url = "http://fake"

    def push_cache_stats(self, agent_id: str, **kwargs: Any) -> None:
        self.calls.append(("push_cache_stats", (agent_id,), kwargs))


def _make_transport():
    """Lazy-import: AnthropicTransport требует Hermes на path'е."""
    from styx_hermes.engine.transport import StyxAnthropicTransport

    return StyxAnthropicTransport()


def _fake_response(*, cached: int, written: int) -> SimpleNamespace:
    """Имитация Anthropic response.usage shape."""
    usage = SimpleNamespace(
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=written,
    )
    return SimpleNamespace(usage=usage)


def test_extract_cache_stats_pushes_on_session(monkeypatch) -> None:
    """Когда session set'нут, stats push'аются с обоими полями."""
    fake = _FakeClient()
    _agent_session.set_session("alpha", fake)
    t = _make_transport()
    resp = _fake_response(cached=500, written=120)
    out = t.extract_cache_stats(resp)
    assert out == {"cached_tokens": 500, "creation_tokens": 120}
    assert len(fake.calls) == 1
    name, (agent_id,), kwargs = fake.calls[0]
    assert name == "push_cache_stats"
    assert agent_id == "alpha"
    assert kwargs == {"cache_read_tokens": 500, "cache_creation_tokens": 120}


def test_extract_cache_stats_pushes_zeros_too(monkeypatch) -> None:
    """Cache miss (0/0) — тоже push'ится: sample-count важен для ratio."""
    fake = _FakeClient()
    _agent_session.set_session("alpha", fake)
    t = _make_transport()
    resp = _fake_response(cached=0, written=0)
    out = t.extract_cache_stats(resp)
    # Default super возвращает None при 0/0
    assert out is None
    # Но push всё равно произошёл с 0/0 — observability invariant.
    assert len(fake.calls) == 1
    _, _, kwargs = fake.calls[0]
    assert kwargs == {"cache_read_tokens": 0, "cache_creation_tokens": 0}


def test_extract_cache_stats_no_session_skips_push() -> None:
    """Без active session — ничего не push'ится (silent)."""
    _agent_session.clear_session()
    t = _make_transport()
    resp = _fake_response(cached=10, written=0)
    out = t.extract_cache_stats(resp)
    # Default super возвращает stats как обычно.
    assert out == {"cached_tokens": 10, "creation_tokens": 0}


def test_extract_cache_stats_fail_open_on_push_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Если push падает — extract_cache_stats всё равно возвращает stats."""
    import logging

    class _BrokenClient(_FakeClient):
        def push_cache_stats(self, agent_id, **kwargs):
            super().push_cache_stats(agent_id, **kwargs)
            raise RuntimeError("simulated network failure")

    fake = _BrokenClient()
    _agent_session.set_session("alpha", fake)
    t = _make_transport()
    with caplog.at_level(logging.WARNING):
        out = t.extract_cache_stats(_fake_response(cached=42, written=0))
    assert out == {"cached_tokens": 42, "creation_tokens": 0}
    assert any("/agent/cache_stats" in rec.message for rec in caplog.records)


def test_register_with_hermes_includes_anthropic() -> None:
    """register_with_hermes вешает StyxAnthropicTransport на anthropic_messages."""
    from styx_hermes.engine import transport as _t

    registered: dict[str, Any] = {}
    _hermes_path = __import__("styx_hermes._hermes_path", fromlist=["ensure_on_path"])
    _hermes_path.ensure_on_path()
    from agent import transports as _agent_transports

    orig = _agent_transports.register_transport

    def _capture(name, cls):
        registered[name] = cls

    _agent_transports.register_transport = _capture
    try:
        _t.register_with_hermes()
    finally:
        _agent_transports.register_transport = orig

    assert "anthropic_messages" in registered
    assert registered["anthropic_messages"] is _t.StyxAnthropicTransport
