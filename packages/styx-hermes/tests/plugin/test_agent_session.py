"""Unit-тесты styx_hermes._agent_session — module-global session state."""

from __future__ import annotations

import logging

import pytest

from styx_hermes import _agent_session
from styx_hermes.client import StyxCoreClient


@pytest.fixture(autouse=True)
def _clear_session():
    yield
    _agent_session.clear_session()


def test_get_session_none_initially() -> None:
    _agent_session.clear_session()
    assert _agent_session.get_session() is None


def test_set_then_get_session() -> None:
    client = StyxCoreClient(base_url="http://x", token=None)
    _agent_session.set_session("agent-a", client)
    out = _agent_session.get_session()
    assert out is not None
    agent_id, got_client = out
    assert agent_id == "agent-a"
    assert got_client is client


def test_set_replaces_previous() -> None:
    c1 = StyxCoreClient(base_url="http://a", token=None)
    c2 = StyxCoreClient(base_url="http://b", token=None)
    _agent_session.set_session("agent-a", c1)
    _agent_session.set_session("agent-b", c2)
    out = _agent_session.get_session()
    assert out is not None
    agent_id, client = out
    assert agent_id == "agent-b"
    assert client is c2


def test_replace_with_different_id_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Замена на ОТЛИЧНЫЙ agent_id шумит warning'ом (Q20), state заменяется."""
    c1 = StyxCoreClient(base_url="http://a", token=None)
    c2 = StyxCoreClient(base_url="http://b", token=None)
    _agent_session.set_session("agent-a", c1)
    with caplog.at_level(logging.WARNING, logger="styx_hermes._agent_session"):
        _agent_session.set_session("agent-b", c2)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "styx_hermes._agent_session"
    ]
    assert warnings, "замена на отличный agent_id должна логировать warning"
    assert any("agent-a" in r.getMessage() and "agent-b" in r.getMessage() for r in warnings)
    # State всё равно заменён — replace намеренный.
    out = _agent_session.get_session()
    assert out is not None
    assert out[0] == "agent-b"


def test_replace_with_same_id_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Idempotent повтор тем же agent_id — тихо, без warning."""
    c1 = StyxCoreClient(base_url="http://a", token=None)
    c2 = StyxCoreClient(base_url="http://b", token=None)
    _agent_session.set_session("agent-a", c1)
    with caplog.at_level(logging.WARNING, logger="styx_hermes._agent_session"):
        _agent_session.set_session("agent-a", c2)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "styx_hermes._agent_session"
    ]
    assert not warnings, "повтор тем же agent_id не должен логировать warning"
    # Клиент всё равно обновлён (state заменён).
    out = _agent_session.get_session()
    assert out is not None
    assert out[1] is c2


def test_clear_after_set() -> None:
    _agent_session.set_session("a", StyxCoreClient(base_url="http://x", token=None))
    _agent_session.clear_session()
    assert _agent_session.get_session() is None


def test_clear_idempotent() -> None:
    _agent_session.clear_session()
    _agent_session.clear_session()  # повторный — без ошибки
    assert _agent_session.get_session() is None
