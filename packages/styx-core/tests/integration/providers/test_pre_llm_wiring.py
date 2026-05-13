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
        assert handle.peer_vad_enabled is True
        assert handle.peer_vad_min_norm == 0.2
        assert handle.peer_vad_ttl_s == 60.0
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


def test_peer_vad_disabled_via_env(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_PRE_LLM_PEER_VAD_ENABLED=0 → handle.peer_vad_enabled=False
    → channel skip'нет, framework configured но channel молчит."""
    monkeypatch.setenv("STYX_PRE_LLM_PEER_VAD_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = pre_llm_inject.get_handle("alpha")
        assert handle is not None
        assert handle.peer_vad_enabled is False
    finally:
        p.shutdown()


def test_peer_vad_thresholds_propagate(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_PEER_VAD_MIN_NORM", "0.5")
    monkeypatch.setenv("STYX_PEER_VAD_TTL_S", "120.0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = pre_llm_inject.get_handle("alpha")
        assert handle is not None
        assert handle.peer_vad_min_norm == 0.5
        assert handle.peer_vad_ttl_s == 120.0
    finally:
        p.shutdown()
