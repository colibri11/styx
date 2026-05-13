"""Тесты wiring волны 12 — initialize/shutdown configure'ит/сбрасывает
eviction_relevance bridge; STYX_EVICTION_RELEVANCE_ENABLED=0 отключает.
"""

from __future__ import annotations

import uuid

import pytest

from styx.engine import eviction_relevance_bridge
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    from styx.engine import (
        focus_tracker,
        hot_tier,
        pre_llm_inject,
        salient_bridge,
        transport as transport_mod,
    )
    eviction_relevance_bridge.reset_all()
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()
    yield
    eviction_relevance_bridge.reset_all()
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()


def test_initialize_configures_eviction_relevance(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    assert eviction_relevance_bridge.get_handle("alpha") is None
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        h = eviction_relevance_bridge.get_handle("alpha")
        assert h is not None
        assert h.keep_k == 2
        assert h.threshold == 0.4
        assert h.agent_id == "alpha"
    finally:
        p.shutdown()


def test_shutdown_resets_eviction_relevance(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    assert eviction_relevance_bridge.get_handle("alpha") is not None
    p.shutdown()
    assert eviction_relevance_bridge.get_handle("alpha") is None


def test_disabled_via_env(styx_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYX_EVICTION_RELEVANCE_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert eviction_relevance_bridge.get_handle("alpha") is None
    finally:
        p.shutdown()


def test_custom_keep_k_and_threshold_via_env(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_EVICTION_RELEVANCE_KEEP_K", "5")
    monkeypatch.setenv("STYX_EVICTION_RELEVANCE_THRESHOLD", "0.6")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        h = eviction_relevance_bridge.get_handle("alpha")
        assert h is not None
        assert h.keep_k == 5
        assert h.threshold == 0.6
    finally:
        p.shutdown()


def test_double_initialize_refreshes_handle(styx_env) -> None:
    """После Phase B — handle per-agent: shutdown alpha убирает handle('alpha'),
    а initialize beta создаёт handle('beta')."""
    p1 = StyxMemoryCore()
    sid1 = str(uuid.uuid4())
    p1.initialize(session_id=sid1, agent_identity="alpha")
    h1 = eviction_relevance_bridge.get_handle("alpha")
    assert h1 is not None
    p1.shutdown()
    assert eviction_relevance_bridge.get_handle("alpha") is None

    p2 = StyxMemoryCore()
    sid2 = str(uuid.uuid4())
    p2.initialize(session_id=sid2, agent_identity="beta")
    try:
        h2 = eviction_relevance_bridge.get_handle("beta")
        assert h2 is not None
        assert h2.agent_id == "beta"
        # alpha при этом — отдельный namespace, всё ещё отсутствует.
        assert eviction_relevance_bridge.get_handle("alpha") is None
    finally:
        p2.shutdown()
