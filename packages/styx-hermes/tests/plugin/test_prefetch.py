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


def test_prefetch_never_duplicates_canonical_preturn_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider prefetch is inert; pre_llm_call owns the fenced envelope."""
    fake = _FakeClient(
        assemble_response={
            "messages": [],
            "estimated_tokens": 0,
            "system_prompt_addition": "<styx-continuity>must not duplicate</styx-continuity>",
            "prompt_authority": "assembled",
        }
    )
    p = _make_provider(monkeypatch, fake)
    assert p.prefetch("любой запрос", session_id="cli-session-42") == ""
    assert [c for c in fake.calls if c[0] == "assemble_context"] == []


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
