"""Тесты ``StyxMemoryProvider.prefetch()`` — Hermes recall-канал
(волна 29 Phase B).

Hermes зовёт ``prefetch(query)`` перед каждым LLM call'ом и аппендит
return text в input. Реализация — synchronous HTTP вызов к
``/context/assemble`` styx-core daemon с minimal messages list. Endpoint
возвращает ``system_prompt_addition`` — pre-formatted
``<styx-salient>...</styx-salient>`` строка (волны 26.7 + 30 family
taxonomy) либо ``None`` если памяти нет.
"""

from __future__ import annotations

import pytest

from styx_hermes import _agent_session
from styx_hermes.providers.memory import StyxMemoryProvider


@pytest.fixture(autouse=True)
def _reset_session():
    yield
    _agent_session.clear_session()


class _FakeClient:
    """Фейк StyxCoreClient — фиксирует assemble_context вызовы."""

    def __init__(self, assemble_response: dict | None = None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._assemble_response = assemble_response or {
            "messages": [],
            "estimated_tokens": 0,
            "system_prompt_addition": None,
            "prompt_authority": "assembled",
        }
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        self.calls.append(("initialize_agent", (agent_id,), kwargs))
        return {"agent_id": agent_id, "tools": []}

    def shutdown_agent(self, agent_id):
        self.calls.append(("shutdown_agent", (agent_id,), {}))

    def assemble_context(self, agent_id, messages, **kwargs):
        self.calls.append(("assemble_context", (agent_id, messages), kwargs))
        return self._assemble_response

    def close(self):
        self.closed = True


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: _FakeClient,
) -> StyxMemoryProvider:
    monkeypatch.setattr(
        "styx_hermes.providers.memory.StyxCoreClient",
        lambda *a, **kw: fake_client,
    )
    p = StyxMemoryProvider()
    p.initialize(session_id="sid-test", agent_identity="alpha")
    return p


def test_prefetch_before_initialize_returns_empty() -> None:
    """До initialize() prefetch() — no-op, возвращает пустую строку."""
    p = StyxMemoryProvider()
    assert p.prefetch("any query") == ""


def test_prefetch_empty_query_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой / whitespace query — без HTTP-вызова, возвращает ""."""
    fake = _FakeClient()
    p = _make_provider(monkeypatch, fake)
    assert p.prefetch("") == ""
    assert p.prefetch("   ") == ""
    # assemble_context не должен вызваться
    assemble_calls = [c for c in fake.calls if c[0] == "assemble_context"]
    assert assemble_calls == []


def test_prefetch_returns_salient_text_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Когда core возвращает system_prompt_addition — prefetch возвращает её."""
    salient = (
        "<styx-salient>\n"
        "[Styx — релевантные memories]\n"
        "- (assistant 2026-05-12) предыдущая важная мысль\n"
        "</styx-salient>"
    )
    fake = _FakeClient(
        assemble_response={
            "messages": [],
            "estimated_tokens": 0,
            "system_prompt_addition": salient,
            "prompt_authority": "assembled",
        }
    )
    p = _make_provider(monkeypatch, fake)
    out = p.prefetch("какой мой любимый цвет?")
    assert out == salient
    # Family-маркер должен быть на месте — это волна 30 invariant.
    assert out.startswith("<styx-salient>\n")
    assert out.endswith("</styx-salient>")


def test_prefetch_returns_empty_when_addition_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Когда нет памяти (system_prompt_addition=None) — возвращает ""."""
    fake = _FakeClient(
        assemble_response={
            "messages": [],
            "estimated_tokens": 0,
            "system_prompt_addition": None,
            "prompt_authority": "assembled",
        }
    )
    p = _make_provider(monkeypatch, fake)
    assert p.prefetch("любой запрос") == ""


def test_prefetch_passes_session_id_to_assemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session_id из Hermes передаётся в /context/assemble."""
    fake = _FakeClient()
    p = _make_provider(monkeypatch, fake)
    p.prefetch("вопрос пользователя", session_id="cli-session-42")
    assemble_calls = [c for c in fake.calls if c[0] == "assemble_context"]
    assert len(assemble_calls) == 1
    _, (agent_id, messages), kwargs = assemble_calls[0]
    assert agent_id == "alpha"
    assert messages == [{"role": "user", "content": "вопрос пользователя"}]
    assert kwargs.get("session_id") == "cli-session-42"


def test_prefetch_empty_session_id_passed_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустая строка session_id маппится в None — Hermes ABC default."""
    fake = _FakeClient()
    p = _make_provider(monkeypatch, fake)
    p.prefetch("вопрос")
    assemble_calls = [c for c in fake.calls if c[0] == "assemble_context"]
    assert assemble_calls[0][2].get("session_id") is None


def test_prefetch_fail_open_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """HTTP failure — возвращает "" + WARNING лог. Fail-open invariant."""
    import logging

    class _BrokenClient(_FakeClient):
        def assemble_context(self, agent_id, messages, **kwargs):
            self.calls.append(("assemble_context", (agent_id, messages), kwargs))
            raise RuntimeError("simulated network failure")

    fake = _BrokenClient()
    p = _make_provider(monkeypatch, fake)
    with caplog.at_level(logging.WARNING):
        out = p.prefetch("query")
    assert out == ""
    assert any("/context/assemble" in rec.message for rec in caplog.records)


def test_queue_prefetch_is_noop_for_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """queue_prefetch — TODO для Phase B+; сейчас no-op без HTTP-вызовов."""
    fake = _FakeClient()
    p = _make_provider(monkeypatch, fake)
    out = p.queue_prefetch("warm query")
    assert out is None
    assert [c for c in fake.calls if c[0] == "assemble_context"] == []


def test_prefetch_returns_str_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ABC contract — prefetch всегда возвращает str (не None, не int)."""
    fake = _FakeClient()
    p = _make_provider(monkeypatch, fake)
    assert isinstance(p.prefetch("q"), str)
    assert isinstance(p.prefetch(""), str)
