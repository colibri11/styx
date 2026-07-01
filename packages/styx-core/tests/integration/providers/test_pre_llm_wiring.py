"""Тесты wiring волны 15 — initialize() конфигурирует pre_llm_inject,
shutdown() сбрасывает; STYX_PRE_LLM_INJECT_ENABLED=0 полностью отключает.
"""

from __future__ import annotations

import uuid

import pytest

from styx.engine import focus_tracker, pre_llm_inject, salient_bridge
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    from styx.engine import transport as transport_mod
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()
    yield
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()


def test_initialize_configures_pre_llm_inject(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    assert pre_llm_inject.get_handle("alpha") is None
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = pre_llm_inject.get_handle("alpha")
        assert handle is not None
        assert handle.queries is p.queries
        assert handle.self_state_enabled is True
        assert handle.self_state_min_norm == 0.2
        assert handle.self_state_max_age_s == 900.0
        assert pre_llm_inject.is_enabled("alpha") is True
    finally:
        p.shutdown()


def test_shutdown_resets_pre_llm_inject(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    assert pre_llm_inject.get_handle("alpha") is not None
    p.shutdown()
    assert pre_llm_inject.get_handle("alpha") is None


def test_inject_disabled_via_env(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_PRE_LLM_INJECT_ENABLED=0 → framework не configure'ится."""
    monkeypatch.setenv("STYX_PRE_LLM_INJECT_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert pre_llm_inject.get_handle("alpha") is None
        # on_pre_llm_call возвращает None silently
        assert pre_llm_inject.on_pre_llm_call("alpha", session_id=sid) is None
    finally:
        p.shutdown()


def test_self_state_disabled_via_env(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_SELF_STATE_ENABLED=0 → handle.self_state_enabled=False
    → channel skip'нет, framework configured но channel молчит."""
    monkeypatch.setenv("STYX_SELF_STATE_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = pre_llm_inject.get_handle("alpha")
        assert handle is not None
        assert handle.self_state_enabled is False
    finally:
        p.shutdown()


def test_self_state_thresholds_propagate(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_SELF_STATE_MIN_NORM", "0.5")
    monkeypatch.setenv("STYX_SELF_STATE_MAX_AGE_S", "120.0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = pre_llm_inject.get_handle("alpha")
        assert handle is not None
        assert handle.self_state_min_norm == 0.5
        assert handle.self_state_max_age_s == 120.0
    finally:
        p.shutdown()
