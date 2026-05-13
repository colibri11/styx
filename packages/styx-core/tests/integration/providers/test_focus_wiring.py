"""Тесты wiring волны 10 — initialize() конфигурирует focus_tracker,
shutdown() сбрасывает; STYX_DRIFT_ENABLED=0 полностью отключает.
"""

from __future__ import annotations

import uuid

import pytest

from styx.engine import focus_tracker, salient_bridge
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    from styx.engine import transport as transport_mod
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    transport_mod._reset_for_test()
    yield
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    transport_mod._reset_for_test()


def test_initialize_configures_focus_tracker(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    assert focus_tracker.get_state("alpha") is None
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        state = focus_tracker.get_state("alpha")
        assert state is not None
        assert state.window == []
        assert state.cached_salient is None
    finally:
        p.shutdown()


def test_shutdown_resets_focus_tracker(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    assert focus_tracker.get_state("alpha") is not None
    p.shutdown()
    assert focus_tracker.get_state("alpha") is None


def test_drift_disabled_via_env(styx_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """STYX_DRIFT_ENABLED=0 → focus_tracker не configure'ится → fallback."""
    monkeypatch.setenv("STYX_DRIFT_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        # salient_bridge всё ещё на месте (волна 9), focus_tracker — нет
        assert salient_bridge.get_handle("alpha") is not None
        assert focus_tracker.get_state("alpha") is None
    finally:
        p.shutdown()


def test_drift_disabled_when_salient_disabled(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_SALIENT_ENABLED=0 → focus_tracker не configure'ится (нечего кэшировать)."""
    monkeypatch.setenv("STYX_SALIENT_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert salient_bridge.get_handle("alpha") is None
        assert focus_tracker.get_state("alpha") is None
    finally:
        p.shutdown()


def test_drift_threshold_and_window_propagate(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_DRIFT_THRESHOLD", "0.55")
    monkeypatch.setenv("STYX_FOCUS_WINDOW_SIZE", "5")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        # После Phase B параметры — поля FocusState (per-agent),
        # не module-global'ы. Проверяем через get_state(agent_id).
        state = focus_tracker.get_state("alpha")
        assert state is not None
        assert state.drift_threshold == pytest.approx(0.55)
        assert state.window_size == 5
    finally:
        p.shutdown()
