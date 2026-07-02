"""Юнит-тесты StyxCodexTransport — Responses API путь (Codex OAuth).

Split-архитектура: transport-класс в ``styx_hermes.engine.transport``,
agent_id через ``styx_hermes._agent_session.set_session``. Эталон стиля
— ``tests/plugin/test_anthropic_transport.py``.
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
    """Минимальный stub StyxCoreClient — в build_kwargs дёргается только
    agent_id из session, сам клиент не используется."""

    def __init__(self) -> None:
        self.base_url = "http://fake"


def _set_agent(agent_id: str) -> None:
    _agent_session.set_session(agent_id, _FakeClient())


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


def test_hermes_default_preserved_when_unconfigured() -> None:
    """Если Styx session не set — Styx НЕ перетирает Hermes-default (passthrough).

    Контракт Styx здесь — «unconfigured → отдать то, что поставил super()»,
    а не конкретное значение cache_key. Сам Hermes-default менялся между
    версиями (v0.17.0: ``session_id``; v0.18.0: content-addressed
    ``_content_cache_key(...) or session_id`` = ``pck_<hash>``, фикс
    cron-cache-cold). Поэтому сверяемся с выводом самого
    ``ResponsesApiTransport``, а не хардкодим значение — тест устойчив к
    будущим сменам дефолта Hermes.
    """
    from agent.transports.codex import ResponsesApiTransport

    base_kwargs = ResponsesApiTransport().build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert kwargs["prompt_cache_key"] == base_kwargs["prompt_cache_key"]


def test_agent_id_overrides_hermes_default() -> None:
    """С active session Styx переопределяет per-session на per-agent."""
    _set_agent("alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert kwargs["prompt_cache_key"] == "alpha"


def test_explicit_module_override_beats_agent_id() -> None:
    _set_agent("alpha")
    core_t.configure("alpha", prompt_cache_key="custom")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "hi")), session_id="hermes-sess-1"
    )
    assert kwargs["prompt_cache_key"] == "custom"


def test_per_call_param_overrides_session() -> None:
    _set_agent("alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
        prompt_cache_key="per-call",
    )
    assert kwargs["prompt_cache_key"] == "per-call"


# -- github/xai cache_key opt-out (abi-5) ---------------------------------


def test_github_responses_omits_cache_key_with_active_session() -> None:
    """is_github_responses=True — Styx опускает prompt_cache_key, как Hermes.

    Hermes-default (agent/transports/codex.py:158) гейтит установку
    cache_key по ``not is_github_responses and not is_xai_responses``;
    GitHub Models opt-out из cache-key routing. Styx наследует это.
    """
    _set_agent("alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
        is_github_responses=True,
    )
    assert "prompt_cache_key" not in kwargs


def test_xai_responses_omits_cache_key_with_active_session() -> None:
    """is_xai_responses=True — Styx опускает prompt_cache_key, как Hermes.

    xAI Responses получает cache-key отдельно через extra_body, поэтому
    Hermes-default не ставит prompt_cache_key напрямую. Styx наследует.
    """
    _set_agent("alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
        is_xai_responses=True,
    )
    assert "prompt_cache_key" not in kwargs


def test_plain_codex_keeps_agent_id_cache_key_with_active_session() -> None:
    """Регрессия: без github/xai флагов prompt_cache_key == agent_id.

    Обычный codex/openai путь — поведение не меняется, gate пропускает.
    """
    _set_agent("alpha")
    tr = t.StyxCodexTransport()
    kwargs = tr.build_kwargs(
        "gpt-5.5",
        _msgs(("user", "hi")),
        session_id="hermes-sess-1",
    )
    assert kwargs["prompt_cache_key"] == "alpha"


# -- codex_backend extra_headers -----------------------------------------


def test_codex_backend_headers_synced_to_override() -> None:
    """На is_codex_backend Hermes пишет session_id/x-client-request-id в
    extra_headers — Styx должен синхронизировать их с нашим cache_key.
    """
    _set_agent("alpha")
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
    """Без active session мы не трогаем extra_headers — Hermes сам
    ставит session_id."""
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
    _set_agent("alpha")
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
    assert any(
        "prefix_slice" in r.message and "digest=" in r.message
        for r in caplog.records
    )


def test_wire_log_digest_stable_across_turns(caplog: pytest.LogCaptureFixture) -> None:
    """Если input items байт-в-байт одинаковые — digest идентичен."""
    _set_agent("alpha")
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
    _set_agent("alpha")
    tr = t.StyxCodexTransport()
    k1 = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "t1")), session_id="s-1"
    )["prompt_cache_key"]
    k2 = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "t2")), session_id="s-2"
    )["prompt_cache_key"]
    assert k1 == k2 == "alpha"


def test_cache_key_unique_per_agent() -> None:
    """Два разных агента дают разные cache_key (сброс session между ними)."""
    tr = t.StyxCodexTransport()
    _set_agent("alpha")
    k_a = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "x")), session_id="s"
    )["prompt_cache_key"]
    # Сбрасываем session перед вторым агентом (в реальности — отдельный процесс).
    _agent_session.clear_session()
    core_t.reset_all()
    _set_agent("beta")
    k_b = tr.build_kwargs(
        "gpt-5.5", _msgs(("user", "x")), session_id="s"
    )["prompt_cache_key"]
    assert k_a != k_b
