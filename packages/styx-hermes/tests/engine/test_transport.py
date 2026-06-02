"""Юнит-тесты StyxOpenAITransport (split-архитектура, host=Hermes).

Transport-класс живёт в ``styx_hermes.engine.transport``; agent_id
резолвится через ``styx_hermes._agent_session.set_session``. Чистые
helper'ы (``configure``, ``compute_prefix_digest``) — из core-модуля
``styx.engine.transport``.

Эталон стиля — ``tests/plugin/test_anthropic_transport.py``.
"""

from __future__ import annotations

import logging

import pytest

from styx.engine import transport as core_t
from styx_hermes import _agent_session
from styx_hermes.engine import transport as t


@pytest.fixture(autouse=True)
def _reset_session():
    """Сброс per-process session + core per-agent state после теста."""
    yield
    _agent_session.clear_session()
    core_t.reset_all()


class _FakeClient:
    """Минимальный stub StyxCoreClient — в build_kwargs используется
    только agent_id из session, клиент сам не дёргается."""

    def __init__(self) -> None:
        self.base_url = "http://fake"


def _set_agent(agent_id: str) -> None:
    _agent_session.set_session(agent_id, _FakeClient())


def _msgs(*pairs: tuple[str, str]) -> list[dict]:
    return [{"role": role, "content": content} for role, content in pairs]


# -- identity / inheritance ------------------------------------------------


def test_inherits_from_chat_completions() -> None:
    from agent.transports.base import ProviderTransport
    from agent.transports.chat_completions import ChatCompletionsTransport

    tr = t.StyxOpenAITransport()
    assert isinstance(tr, ChatCompletionsTransport)
    assert isinstance(tr, ProviderTransport)


def test_api_mode_is_chat_completions() -> None:
    assert t.StyxOpenAITransport().api_mode == "chat_completions"


def test_no_arg_constructor() -> None:
    """Hermes get_transport вызывает cls() без аргументов."""
    instance = t.StyxOpenAITransport()
    assert instance is not None


# -- prompt_cache_key resolution ------------------------------------------


def test_no_cache_key_when_unconfigured() -> None:
    """Без session и без session_id — Hermes default (нет ключа)."""
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert "prompt_cache_key" not in kwargs


def test_cache_key_from_session_agent_id() -> None:
    _set_agent("alpha")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert kwargs["prompt_cache_key"] == "alpha"


def test_explicit_prompt_cache_key_overrides_agent_id() -> None:
    _set_agent("alpha")
    core_t.configure("alpha", prompt_cache_key="custom-key")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert kwargs["prompt_cache_key"] == "custom-key"


def test_per_call_param_overrides_session() -> None:
    _set_agent("alpha")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs(
        "gpt-x", _msgs(("user", "hi")), prompt_cache_key="per-call"
    )
    assert kwargs["prompt_cache_key"] == "per-call"


def test_per_call_overrides_explicit_per_agent_override() -> None:
    """Верхнее звено цепочки: per-call бьёт ЯВНЫЙ per-agent override.

    session(alpha) + configure(alpha, prompt_cache_key=custom) задаёт
    per-agent override; per-call prompt_cache_key должен пересилить его.
    """
    _set_agent("alpha")
    core_t.configure("alpha", prompt_cache_key="custom-key")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs(
        "gpt-x", _msgs(("user", "hi")), prompt_cache_key="per-call"
    )
    assert kwargs["prompt_cache_key"] == "per-call"


def test_per_agent_override_used_when_no_per_call() -> None:
    """Компаньон: без per-call действует явный per-agent override."""
    _set_agent("alpha")
    core_t.configure("alpha", prompt_cache_key="custom-key")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert kwargs["prompt_cache_key"] == "custom-key"


def test_session_id_fallback_when_no_session() -> None:
    """Без active session — fallback на params.session_id."""
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs(
        "gpt-x", _msgs(("user", "hi")), session_id="sess-123"
    )
    assert kwargs["prompt_cache_key"] == "sess-123"


# -- super-class behaviour preserved --------------------------------------


def test_messages_passed_through() -> None:
    msgs = _msgs(("user", "first"), ("assistant", "second"))
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", msgs)
    assert kwargs["model"] == "gpt-x"
    assert kwargs["messages"] == msgs


def test_tools_passed_through() -> None:
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")), tools=tools)
    assert kwargs["tools"] == tools


def test_codex_sanitization_inherited(caplog) -> None:
    """convert_messages родителя дропает codex_* поля — наследуем без изменений."""
    msgs = [
        {
            "role": "user",
            "content": "x",
            "codex_reasoning_items": ["should-strip"],
        },
    ]
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", msgs)
    assert "codex_reasoning_items" not in kwargs["messages"][0]


# -- wire-log -------------------------------------------------------------


def test_wire_log_emits_digest(caplog: pytest.LogCaptureFixture) -> None:
    tr = t.StyxOpenAITransport()
    with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
        tr.build_kwargs("gpt-x", _msgs(("user", "hello")))
    assert any(
        "prefix_slice" in r.message and "digest=" in r.message
        for r in caplog.records
    )


def test_wire_log_digest_byte_stable_for_same_messages() -> None:
    same = _msgs(("user", "stable")) + _msgs(("assistant", "answer"))
    d1 = core_t.compute_prefix_digest(same)
    d2 = core_t.compute_prefix_digest(same)
    assert d1 == d2


def test_wire_log_digest_differs_for_different_prefix() -> None:
    a = _msgs(("user", "alpha"))
    b = _msgs(("user", "beta"))
    assert core_t.compute_prefix_digest(a) != core_t.compute_prefix_digest(b)


def test_wire_log_disabled_with_zero_head_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """head=0 (через core configure) выключает wire-log digest."""
    _set_agent("alpha")
    core_t.configure("alpha", wire_log_head_messages=0)
    tr = t.StyxOpenAITransport()
    with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
        tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert not any("prefix_slice" in r.message for r in caplog.records)


# -- registry registration -------------------------------------------------


def test_register_with_hermes_replaces_default() -> None:
    from agent.transports import get_transport, register_transport
    from agent.transports.chat_completions import ChatCompletionsTransport

    # Принудительно сбрасываем _REGISTRY к дефолту до проверки —
    # тест robustic к pre-state (в Docker-стеке Styx уже мог быть
    # зарегистрирован при bootstrap'е).
    register_transport("chat_completions", ChatCompletionsTransport)
    default_inst = get_transport("chat_completions")
    assert default_inst.__class__.__name__ == "ChatCompletionsTransport"

    t.register_with_hermes()
    try:
        inst = get_transport("chat_completions")
        assert isinstance(inst, t.StyxOpenAITransport)
    finally:
        register_transport("chat_completions", ChatCompletionsTransport)


# -- session state replacement (post-split, Q20) --------------------------


def test_set_session_replaces_state() -> None:
    """После split нет single-global conflict-guard: одна процесс-сессия,
    повторный set_session намеренно заменяет state (one-process-one-agent).
    """
    _set_agent("alpha")
    tr = t.StyxOpenAITransport()
    k1 = tr.build_kwargs("gpt-x", _msgs(("user", "x")))["prompt_cache_key"]
    assert k1 == "alpha"
    # Повторный set_session заменяет — это намеренное поведение.
    _set_agent("beta")
    k2 = tr.build_kwargs("gpt-x", _msgs(("user", "x")))["prompt_cache_key"]
    assert k2 == "beta"


# -- #10: wire-log не пишет slice_hex на INFO по умолчанию ---------------


def test_wire_log_does_not_emit_slice_hex_by_default(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """slice_hex не должен появляться в INFO-логах по умолчанию."""
    monkeypatch.delenv("STYX_WIRE_LOG_RAW", raising=False)
    tr = t.StyxOpenAITransport()
    with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
        tr.build_kwargs("gpt-x", _msgs(("user", "secret content")))
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, "должна быть хотя бы одна INFO запись wire-log"
    assert not any("slice_hex=" in r.message for r in info_records), (
        "slice_hex не должен быть в INFO записях по умолчанию"
    )


def test_wire_log_emits_slice_hex_with_env_flag(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При STYX_WIRE_LOG_RAW=1 slice_hex появляется в INFO."""
    monkeypatch.setenv("STYX_WIRE_LOG_RAW", "1")
    tr = t.StyxOpenAITransport()
    with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
        tr.build_kwargs("gpt-x", _msgs(("user", "content")))
    assert any("slice_hex=" in r.message for r in caplog.records), (
        "slice_hex должен появляться при STYX_WIRE_LOG_RAW=1"
    )


# -- cache_key стабильность между turn'ами -------------------------------


def test_cache_key_stable_across_turns_for_same_agent() -> None:
    _set_agent("alpha")
    tr = t.StyxOpenAITransport()
    k1 = tr.build_kwargs("gpt-x", _msgs(("user", "turn1")))["prompt_cache_key"]
    k2 = tr.build_kwargs("gpt-x", _msgs(("user", "turn2")))["prompt_cache_key"]
    assert k1 == k2 == "alpha"


def test_cache_key_unique_per_agent() -> None:
    """Два разных агента дают разные cache_key (сброс session между ними)."""
    tr = t.StyxOpenAITransport()
    _set_agent("alpha")
    k_a = tr.build_kwargs("gpt-x", _msgs(("user", "x")))["prompt_cache_key"]
    # Сбрасываем session перед вторым агентом (в реальности — отдельный процесс).
    _agent_session.clear_session()
    core_t.reset_all()
    _set_agent("beta")
    k_b = tr.build_kwargs("gpt-x", _msgs(("user", "x")))["prompt_cache_key"]
    assert k_a != k_b
