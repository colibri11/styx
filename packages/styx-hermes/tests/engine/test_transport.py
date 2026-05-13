"""Юнит-тесты StyxOpenAITransport."""

from __future__ import annotations

import logging

import pytest

from styx.engine import transport as t


@pytest.fixture(autouse=True)
def _reset_module_globals():
    t._reset_for_test()
    yield
    t._reset_for_test()


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
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert "prompt_cache_key" not in kwargs


def test_cache_key_from_configure_agent_id() -> None:
    t.configure(agent_id="alpha")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert kwargs["prompt_cache_key"] == "alpha"


def test_explicit_prompt_cache_key_overrides_agent_id() -> None:
    t.configure(agent_id="alpha", prompt_cache_key="custom-key")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs("gpt-x", _msgs(("user", "hi")))
    assert kwargs["prompt_cache_key"] == "custom-key"


def test_per_call_param_overrides_module_global() -> None:
    t.configure(agent_id="alpha")
    tr = t.StyxOpenAITransport()
    kwargs = tr.build_kwargs(
        "gpt-x", _msgs(("user", "hi")), prompt_cache_key="per-call"
    )
    assert kwargs["prompt_cache_key"] == "per-call"


def test_session_id_fallback_when_no_agent() -> None:
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
    assert any("prefix_slice digest=" in r.message for r in caplog.records)


def test_wire_log_digest_byte_stable_for_same_messages() -> None:
    same = _msgs(("user", "stable")) + _msgs(("assistant", "answer"))
    d1 = t.compute_prefix_digest(same)
    d2 = t.compute_prefix_digest(same)
    assert d1 == d2


def test_wire_log_digest_differs_for_different_prefix() -> None:
    a = _msgs(("user", "alpha"))
    b = _msgs(("user", "beta"))
    assert t.compute_prefix_digest(a) != t.compute_prefix_digest(b)


def test_wire_log_disabled_with_zero_head_messages(caplog: pytest.LogCaptureFixture) -> None:
    t.configure(wire_log_head_messages=0)
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


# -- #9: configure guard против конфликта agent_id ----------------------


def test_configure_rejects_conflicting_agent_id(caplog: pytest.LogCaptureFixture) -> None:
    """Если _AGENT_ID уже задан и пришло другое значение — log.error, не перетирать."""
    t.configure(agent_id="alpha")
    with caplog.at_level(logging.ERROR, logger="styx.engine.transport"):
        t.configure(agent_id="beta")
    # Значение не поменялось
    assert t._AGENT_ID == "alpha"
    # Ошибка залогирована
    assert any("conflicting agent_id" in r.message for r in caplog.records)


def test_configure_same_agent_id_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    """Повторный вызов с тем же agent_id — тихой ошибки нет."""
    t.configure(agent_id="alpha")
    with caplog.at_level(logging.ERROR, logger="styx.engine.transport"):
        t.configure(agent_id="alpha")
    assert t._AGENT_ID == "alpha"
    assert not any("conflicting agent_id" in r.message for r in caplog.records)


def test_configure_none_agent_id_is_noop() -> None:
    """configure(agent_id=None) не меняет уже установленное значение."""
    t.configure(agent_id="alpha")
    t.configure(agent_id=None)
    assert t._AGENT_ID == "alpha"


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
    t.configure(agent_id="alpha")
    tr = t.StyxOpenAITransport()
    k1 = tr.build_kwargs("gpt-x", _msgs(("user", "turn1")))["prompt_cache_key"]
    k2 = tr.build_kwargs("gpt-x", _msgs(("user", "turn2")))["prompt_cache_key"]
    assert k1 == k2 == "alpha"


def test_cache_key_unique_per_agent() -> None:
    """Два разных агента дают разные cache_key (сброс между ними)."""
    tr = t.StyxOpenAITransport()
    t.configure(agent_id="alpha")
    k_a = tr.build_kwargs("gpt-x", _msgs(("user", "x")))["prompt_cache_key"]
    # Сбрасываем module-global перед вторым агентом (в реальности — отдельный процесс)
    t._reset_for_test()
    t.configure(agent_id="beta")
    k_b = tr.build_kwargs("gpt-x", _msgs(("user", "x")))["prompt_cache_key"]
    assert k_a != k_b
