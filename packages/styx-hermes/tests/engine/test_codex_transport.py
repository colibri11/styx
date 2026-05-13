"""Юнит-тесты StyxCodexTransport — Responses API путь (Codex OAuth)."""

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
    """OpenAI-style messages (Responses transport сам конвертирует в input)."""
    return [{"role": role, "content": content} for role, content in pairs]


# -- identity / inheritance -----------------------------------------------


def test_inherits_from_responses_api_transport() -> None:
    from agent.transports.base import ProviderTransport
    from agent.transports.codex import ResponsesApiTransport

    tr = t.StyxCodexTransport()
    assert isinstance(tr, ResponsesApiTransport)
    assert isinstance(tr, ProviderTransport)


def test_api_mode_is_codex_responses() -> None:
    assert t.StyxCodexTransport().api_mode == "codex_responses"


def test_no_arg_constructor() -> None:
    """Hermes get_transport вызывает cls() без аргументов."""
    instance = t.StyxCodexTransport()
    assert instance is not None


# -- cache_key resolution -------------------------------------------------


def test_hermes_default_session_id_preserved_when_unconfigured() -> None:
    """Если Styx не сконфигурирован — поведение Hermes сохраняется.

    Hermes-default: ``prompt_cache_key = session_id``. Styx это уважает.
    """
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert kwargs["prompt_cache_key"] == "hermes-sess-1"


def test_agent_id_overrides_hermes_default() -> None:
    """С configure(agent_id=...) Styx переопределяет per-session на per-agent."""
    t.configure(agent_id="alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert kwargs["prompt_cache_key"] == "alpha"


def test_explicit_module_override_beats_agent_id() -> None:
    t.configure(agent_id="alpha", prompt_cache_key="custom")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert kwargs["prompt_cache_key"] == "custom"


def test_per_call_param_overrides_module() -> None:
    t.configure(agent_id="alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
        prompt_cache_key="per-call",
    )
    assert kwargs["prompt_cache_key"] == "per-call"


# -- codex_backend extra_headers -----------------------------------------


def test_codex_backend_headers_synced_to_override() -> None:
    """На is_codex_backend Hermes пишет session_id/x-client-request-id в
    extra_headers — Styx должен синхронизировать их с нашим cache_key.
    """
    t.configure(agent_id="alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
        is_codex_backend=True,
    )
    headers = kwargs["extra_headers"]
    assert headers["session_id"] == "alpha"
    assert headers["x-client-request-id"] == "alpha"


def test_codex_backend_headers_untouched_without_override() -> None:
    """Без agent_id мы не трогаем extra_headers — Hermes сам ставит session_id."""
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
        is_codex_backend=True,
    )
    headers = kwargs["extra_headers"]
    # Hermes-default подставил session_id из себя
    assert headers["session_id"] == "hermes-sess-1"


def test_non_codex_backend_no_extra_headers_added() -> None:
    """Когда не Codex backend — extra_headers не добавляем (Hermes тоже не).

    Уточнение: Hermes super не ставит extra_headers без is_codex_backend и
    is_xai_responses; Styx-override тоже не должен.
    """
    t.configure(agent_id="alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert "extra_headers" not in kwargs


# -- super-class behaviour preserved --------------------------------------


def test_input_built_from_messages() -> None:
    """super().build_kwargs конвертирует messages → Responses input items."""
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hello")), session_id="s"
    )
    assert "input" in kwargs
    assert isinstance(kwargs["input"], list)


def test_instructions_extracted_from_system_message() -> None:
    """Если первое сообщение system — оно идёт в instructions, не в input."""
    tr = t.StyxCodexTransport()
    msgs = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]
    kwargs = tr.build_kwargs("gpt-5.5", msgs, session_id="s")
    assert kwargs["instructions"] == "Be terse."


def test_store_false_inherited() -> None:
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs("gpt-5.5", _msgs(("user", "x")), session_id="s")
    assert kwargs["store"] is False


# -- wire-log on input payload --------------------------------------------


def test_wire_log_emits_digest_for_input(caplog: pytest.LogCaptureFixture) -> None:
    tr = t.StyxCodexTransport()
    with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
        tr.build_kwargs("gpt-5.5", _msgs(("user", "hello")), session_id="s")
    assert any("prefix_slice digest=" in r.message for r in caplog.records)


def test_wire_log_digest_stable_across_turns(caplog: pytest.LogCaptureFixture) -> None:
    """Если input items байт-в-байт одинаковые — digest идентичен."""
    t.configure(agent_id="alpha")
    tr = t.StyxCodexTransport()
    base = _msgs(("user", "stable preamble " + "filler " * 30))
    with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
        tr.build_kwargs("gpt-5.5", base, session_id="s1")
        tr.build_kwargs("gpt-5.5", base, session_id="s2")
    digests = [
        r.message.split("digest=")[1].split()[0]
        for r in caplog.records
        if "prefix_slice" in r.message
    ]
    assert len(digests) == 2
    assert digests[0] == digests[1]


# -- registry registration -------------------------------------------------


def test_register_with_hermes_replaces_codex_default() -> None:
    from agent.transports import get_transport, register_transport
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport

    # Принудительный сброс _REGISTRY к дефолтам — тест robustic к
    # pre-state (в Docker-стеке Styx уже зарегистрирован при bootstrap'е).
    register_transport("chat_completions", ChatCompletionsTransport)
    register_transport("codex_responses", ResponsesApiTransport)

    default_inst = get_transport("codex_responses")
    assert default_inst.__class__.__name__ == "ResponsesApiTransport"

    t.register_with_hermes()
    try:
        inst = get_transport("codex_responses")
        assert isinstance(inst, t.StyxCodexTransport)

        # И chat_completions тоже зарегистрирован тем же вызовом
        cc = get_transport("chat_completions")
        assert isinstance(cc, t.StyxOpenAITransport)
    finally:
        register_transport("codex_responses", ResponsesApiTransport)
        register_transport("chat_completions", ChatCompletionsTransport)


# -- cache_key стабильность между turn'ами ---------------------------------


def test_cache_key_stable_across_turns_for_same_agent() -> None:
    t.configure(agent_id="alpha")
    tr = t.StyxCodexTransport()
    k1 = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "t1")), session_id="s-1"
    )["prompt_cache_key"]
    k2 = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "t2")), session_id="s-2"
    )["prompt_cache_key"]
    assert k1 == k2 == "alpha"


def test_cache_key_unique_per_agent() -> None:
    """Два разных агента дают разные cache_key (сброс между ними)."""
    tr = t.StyxCodexTransport()
    t.configure(agent_id="alpha")
    k_a = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "x")), session_id="s"
    )["prompt_cache_key"]
    # Сбрасываем module-global перед вторым агентом (в реальности — отдельный процесс)
    t._reset_for_test()
    t.configure(agent_id="beta")
    k_b = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "x")), session_id="s"
    )["prompt_cache_key"]
    assert k_a != k_b
