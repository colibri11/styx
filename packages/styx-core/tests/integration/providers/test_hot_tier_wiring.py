"""Тесты wiring волны 11 — initialize() configures hot_tier,
shutdown() сбрасывает; STYX_HOT_TIER_ENABLED=0 полностью отключает.
"""

from __future__ import annotations

import uuid

import pytest

from styx.engine import hot_tier
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    from styx.engine import (
        focus_tracker,
        pre_llm_inject,
        salient_bridge,
        transport as transport_mod,
    )
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()
    yield
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()


def test_initialize_configures_hot_tier(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    assert hot_tier.get_state("alpha") is None
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        s = hot_tier.get_state("alpha")
        assert s is not None
        assert s.entries == {}
        assert s.ttl_s == 300.0
        assert s.lru_bound == 100
    finally:
        p.shutdown()


def test_shutdown_resets_hot_tier(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    assert hot_tier.get_state("alpha") is not None
    p.shutdown()
    assert hot_tier.get_state("alpha") is None


def test_hot_tier_disabled_via_env(styx_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """STYX_HOT_TIER_ENABLED=0 → state остаётся None."""
    monkeypatch.setenv("STYX_HOT_TIER_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert hot_tier.get_state("alpha") is None
    finally:
        p.shutdown()


def test_double_initialize_does_not_leak_state(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Двойной initialize — state свежий, не загрязняется предыдущим.

    Persistence (волна 13) явно отключена — тест проверяет именно
    конфигурацию hot_tier при reinitialize, а не restore из БД.
    """
    monkeypatch.setenv("STYX_WORKING_SET_PERSISTENCE_ENABLED", "0")
    p1 = StyxMemoryCore()
    sid1 = str(uuid.uuid4())
    p1.initialize(session_id=sid1, agent_identity="alpha")
    s1 = hot_tier.get_state("alpha")
    assert s1 is not None
    # «Утечка» — добавим entry. Через configure (двойной initialize) state
    # должен пересоздаться.
    from styx.storage.queries import MemoryHit
    import datetime as _dt
    hot_tier.put_many("alpha", [MemoryHit(
        id=uuid.uuid4(), agent_id="alpha", kind="x", kind_src="subjective",
        role="user", content="x", metadata={},
        created_at=_dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc),
        score=0.5, match_score=0.5, embedding=[1.0] + [0.0] * 767,
    )])
    assert len(s1.entries) == 1

    p1.shutdown()
    assert hot_tier.get_state("alpha") is None

    p2 = StyxMemoryCore()
    sid2 = str(uuid.uuid4())
    p2.initialize(session_id=sid2, agent_identity="alpha")
    try:
        s2 = hot_tier.get_state("alpha")
        assert s2 is not None
        assert s2.entries == {}
    finally:
        p2.shutdown()


def test_custom_ttl_and_bound_via_env(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_HOT_TIER_TTL_S", "120")
    monkeypatch.setenv("STYX_HOT_TIER_LRU_BOUND", "42")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        s = hot_tier.get_state("alpha")
        assert s is not None
        assert s.ttl_s == 120.0
        assert s.lru_bound == 42
    finally:
        p.shutdown()
