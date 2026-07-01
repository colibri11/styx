"""Юнит-тесты pre_llm_inject framework — configure / on_pre_llm_call / fail-open."""

from __future__ import annotations

import pytest

from styx.engine import pre_llm_inject
from styx.engine.pre_llm_inject import ChannelHandle


class _StubQueries:
    pass


def _handle(**overrides) -> ChannelHandle:
    base = dict(
        queries=_StubQueries(),
        self_state_enabled=True,
        self_state_min_norm=0.2,
        self_state_max_age_s=900.0,
    )
    base.update(overrides)
    return ChannelHandle(**base)


@pytest.fixture(autouse=True)
def _reset() -> None:
    pre_llm_inject.reset_all()
    yield
    pre_llm_inject.reset_all()


def test_get_handle_none_initially() -> None:
    assert pre_llm_inject.get_handle("test-agent") is None
    # До configure агент не зарегистрирован — is_enabled возвращает False.
    # Старая семантика глобального _ENABLED=True больше не применима:
    # фреймворк хранит per-agent state, нет state'а = нет канала.
    assert pre_llm_inject.is_enabled("test-agent") is False


def test_on_pre_llm_call_returns_none_when_not_configured() -> None:
    assert pre_llm_inject.on_pre_llm_call("test-agent", session_id="x") is None


def test_configure_then_on_pre_llm_call_with_no_channels() -> None:
    pre_llm_inject.configure("test-agent", handle=_handle(), channels=[])
    assert pre_llm_inject.get_handle("test-agent") is not None
    assert pre_llm_inject.on_pre_llm_call("test-agent", session_id="x") is None


def test_on_pre_llm_call_collects_non_null_channels() -> None:
    def ch_one(handle, kw): return "context line one"
    def ch_two(handle, kw): return None
    def ch_three(handle, kw): return "context line three"

    pre_llm_inject.configure("test-agent", 
        handle=_handle(),
        channels=[("one", ch_one), ("two", ch_two), ("three", ch_three)],
    )
    out = pre_llm_inject.on_pre_llm_call("test-agent", session_id="x")
    assert out == {"context": "context line one\n\ncontext line three"}


def test_on_pre_llm_call_returns_none_when_all_channels_null() -> None:
    def ch(handle, kw): return None

    pre_llm_inject.configure("test-agent", handle=_handle(), channels=[("only", ch)])
    assert pre_llm_inject.on_pre_llm_call("test-agent", session_id="x") is None


def test_on_pre_llm_call_skips_failing_channel(caplog) -> None:
    def ch_boom(handle, kw): raise RuntimeError("ollama down")
    def ch_ok(handle, kw): return "ok line"

    pre_llm_inject.configure("test-agent", 
        handle=_handle(),
        channels=[("boom", ch_boom), ("ok", ch_ok)],
    )
    with caplog.at_level("WARNING"):
        out = pre_llm_inject.on_pre_llm_call("test-agent", session_id="x")
    assert out == {"context": "ok line"}
    assert any("boom" in rec.getMessage() for rec in caplog.records)


def test_disabled_via_enabled_flag() -> None:
    def ch(handle, kw): return "would-be-context"

    pre_llm_inject.configure("test-agent", 
        handle=_handle(), channels=[("c", ch)], enabled=False,
    )
    assert pre_llm_inject.is_enabled("test-agent") is False
    assert pre_llm_inject.on_pre_llm_call("test-agent", session_id="x") is None


def test_double_configure_replaces_state() -> None:
    def ch_old(handle, kw): return "old"
    def ch_new(handle, kw): return "new"

    pre_llm_inject.configure("test-agent", handle=_handle(), channels=[("old", ch_old)])
    pre_llm_inject.configure("test-agent", handle=_handle(), channels=[("new", ch_new)])
    out = pre_llm_inject.on_pre_llm_call("test-agent", session_id="x")
    assert out == {"context": "new"}


def test_reset_clears_state() -> None:
    def ch(handle, kw): return "anything"
    pre_llm_inject.configure("test-agent", handle=_handle(), channels=[("c", ch)])
    assert pre_llm_inject.get_handle("test-agent") is not None
    pre_llm_inject.reset_all()
    assert pre_llm_inject.get_handle("test-agent") is None
    assert pre_llm_inject.on_pre_llm_call("test-agent", session_id="x") is None


def test_channel_receives_handle_and_kwargs() -> None:
    captured = {}

    def ch(handle, kw):
        captured["handle"] = handle
        captured["kw"] = kw
        return None

    h = _handle()
    pre_llm_inject.configure("test-agent", handle=h, channels=[("c", ch)])
    pre_llm_inject.on_pre_llm_call("test-agent", 
        session_id="sess-1", user_message="hello", model="gpt-x",
    )
    assert captured["handle"] is h
    assert captured["kw"]["session_id"] == "sess-1"
    assert captured["kw"]["user_message"] == "hello"
    assert captured["kw"]["model"] == "gpt-x"
