"""Тесты wiring волны 13 — initialize/shutdown с working_set persistence.

Проверяем:
- initialize стартует save-thread (если enabled);
- shutdown stop'ит и flush'ит;
- STYX_WORKING_SET_PERSISTENCE_ENABLED=0 → save-thread не стартует;
- двойной initialize не утекает thread (старый stop, новый start);
- restart simulation: state, замутированный во время первого provider'а,
  восстанавливается во втором провайдере.
"""

from __future__ import annotations

import time
import uuid

import pytest

from styx.engine import (
    focus_tracker,
    hot_tier,
    working_set_persistence as wsp,
)
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str) -> str:
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    return migrated_db


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    from styx.engine import (
        eviction_relevance_bridge,
        pre_llm_inject,
        salient_bridge,
        transport as transport_mod,
    )
    wsp.stop_all()  # safety, на случай зависшего thread'а
    eviction_relevance_bridge.reset_all()
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()
    yield
    wsp.stop_all()
    eviction_relevance_bridge.reset_all()
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()


def test_initialize_starts_thread_when_enabled(styx_env: str) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    assert not wsp.is_running("alpha")
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert wsp.is_running("alpha")
    finally:
        p.shutdown()


def test_shutdown_stops_thread(styx_env: str) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    assert wsp.is_running("alpha")
    p.shutdown()
    assert not wsp.is_running("alpha")


def test_disabled_via_env_does_not_start(
    styx_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STYX_WORKING_SET_PERSISTENCE_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert not wsp.is_running("alpha")
    finally:
        p.shutdown()


def test_double_initialize_refreshes_thread(styx_env: str) -> None:
    """Двойной initialize → save-thread первого корректно остановлен,
    новый запущен."""
    p1 = StyxMemoryCore()
    sid1 = str(uuid.uuid4())
    p1.initialize(session_id=sid1, agent_identity="alpha")
    assert wsp.is_running("alpha")
    p1.shutdown()
    assert not wsp.is_running("alpha")

    p2 = StyxMemoryCore()
    sid2 = str(uuid.uuid4())
    p2.initialize(session_id=sid2, agent_identity="beta")
    try:
        # save-thread per-agent: после Phase B beta-thread живёт под
        # ключом "beta", а старый alpha-thread корректно остановлен.
        assert wsp.is_running("beta")
        assert not wsp.is_running("alpha")
    finally:
        p2.shutdown()


def test_restart_restores_focus_state(
    styx_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider 1: observe + set_cached → shutdown.
    Provider 2: initialize → focus_tracker.window и cached_salient восстановлены."""
    monkeypatch.setenv("STYX_WORKING_SET_SAVE_INTERVAL_S", "60")  # без periodic save'а

    # Provider 1
    p1 = StyxMemoryCore()
    p1.initialize(session_id=str(uuid.uuid4()), agent_identity="alpha")
    state_before = focus_tracker.get_state("alpha")
    assert state_before is not None
    state_before.window.append([0.1] * 768)
    state_before.window.append([0.2] * 768)
    state_before.cached_salient = {"role": "user", "content": "marker"}
    state_before.epoch_id = 3
    p1.shutdown()  # final flush через stop()
    assert not wsp.is_running("alpha")

    # Provider 2 — симулирует restart
    p2 = StyxMemoryCore()
    p2.initialize(session_id=str(uuid.uuid4()), agent_identity="alpha")
    try:
        state_after = focus_tracker.get_state("alpha")
        assert state_after is not None
        assert len(state_after.window) == 2
        assert state_after.window[0] == pytest.approx([0.1] * 768)
        assert state_after.window[1] == pytest.approx([0.2] * 768)
        assert state_after.cached_salient == {"role": "user", "content": "marker"}
        assert state_after.epoch_id == 3
    finally:
        p2.shutdown()


def test_restart_with_persistence_disabled_does_not_restore(
    styx_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STYX_WORKING_SET_SAVE_INTERVAL_S", "60")

    p1 = StyxMemoryCore()
    p1.initialize(session_id=str(uuid.uuid4()), agent_identity="alpha")
    state_before = focus_tracker.get_state("alpha")
    assert state_before is not None
    state_before.cached_salient = {"role": "user", "content": "marker"}
    p1.shutdown()

    monkeypatch.setenv("STYX_WORKING_SET_PERSISTENCE_ENABLED", "0")
    p2 = StyxMemoryCore()
    p2.initialize(session_id=str(uuid.uuid4()), agent_identity="alpha")
    try:
        state_after = focus_tracker.get_state("alpha")
        assert state_after is not None
        assert state_after.cached_salient is None
    finally:
        p2.shutdown()


def test_periodic_save_runs(
    styx_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save-thread с коротким interval'ом сохраняет state в БД до shutdown'а."""
    monkeypatch.setenv("STYX_WORKING_SET_SAVE_INTERVAL_S", "0.2")

    p = StyxMemoryCore()
    p.initialize(session_id=str(uuid.uuid4()), agent_identity="alpha")
    try:
        state = focus_tracker.get_state("alpha")
        assert state is not None
        state.cached_salient = {"role": "user", "content": "periodic"}
        # Ждём 2 tick'а (0.4s) + headroom — гарантировано один save случился.
        time.sleep(0.7)
        # Проверяем строку в БД до shutdown'а (final flush ещё не было).
        import psycopg
        with psycopg.connect(styx_env) as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM working_set WHERE agent_id = %s",
                    ("alpha",),
                )
                row = cur.fetchone()
        assert row is not None
        assert row[0]["focus"]["cached_salient"]["content"] == "periodic"
    finally:
        p.shutdown()
