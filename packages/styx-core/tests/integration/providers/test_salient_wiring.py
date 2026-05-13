"""Тесты wiring волны 9 — initialize() конфигурирует salient_bridge,
shutdown() сбрасывает; STYX_SALIENT_ENABLED=0 полностью отключает.

Требует живую БД (initialize ходит в Postgres). Unit-тесты на config
loading — в tests/test_config.py.
"""

from __future__ import annotations

import uuid

import pytest

from styx.engine import salient_bridge
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


@pytest.fixture(autouse=True)
def _reset_handle() -> None:
    from styx.engine import transport as transport_mod
    salient_bridge.reset_all()
    transport_mod._reset_for_test()
    yield
    salient_bridge.reset_all()
    transport_mod._reset_for_test()


def test_initialize_configures_bridge(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    assert salient_bridge.get_handle("alpha") is None
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = salient_bridge.get_handle("alpha")
        assert handle is not None
        assert handle.queries is p.queries
        # Дефолты переданы из StyxConfig
        assert handle.timeout_s == 1.0
        assert handle.min_query_len == 20
    finally:
        p.shutdown()


def test_shutdown_resets_bridge(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    assert salient_bridge.get_handle("alpha") is not None
    p.shutdown()
    assert salient_bridge.get_handle("alpha") is None


def test_double_initialize_replaces_handle(styx_env) -> None:
    p = StyxMemoryCore()
    sid1 = str(uuid.uuid4())
    sid2 = str(uuid.uuid4())
    p.initialize(session_id=sid1, agent_identity="alpha")
    try:
        h1 = salient_bridge.get_handle("alpha")
        assert h1 is not None
        p.initialize(session_id=sid2, agent_identity="alpha")
        h2 = salient_bridge.get_handle("alpha")
        assert h2 is not None
        # Реинициализация заменила connection → queries — другой instance.
        assert h2.queries is not h1.queries
    finally:
        p.shutdown()


def test_salient_disabled_via_env(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_SALIENT_ENABLED", "0")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        # bridge не сконфигурирован → handle is None → compress() не инжектит
        assert salient_bridge.get_handle("alpha") is None
    finally:
        p.shutdown()


def test_salient_timeout_and_min_query_len_propagate(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_SALIENT_TIMEOUT_S", "2.5")
    monkeypatch.setenv("STYX_SALIENT_MIN_QUERY_LEN", "30")
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        handle = salient_bridge.get_handle("alpha")
        assert handle is not None
        assert handle.timeout_s == 2.5
        assert handle.min_query_len == 30
    finally:
        p.shutdown()
