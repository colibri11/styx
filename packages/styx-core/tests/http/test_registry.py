"""AgentRegistry — register / get / unregister / agent_id-isolation."""

from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException

from styx.http import registry
from styx.http.registry import AgentSession


def setup_function() -> None:
    registry.reset_all()


def teardown_function() -> None:
    registry.reset_all()


def test_get_unknown_raises_404():
    with pytest.raises(HTTPException) as ei:
        registry.get("ghost")
    assert ei.value.status_code == 404


def test_register_then_get():
    sess = registry.register("alyona", core=object())
    got = registry.get("alyona")
    assert got is sess


def test_double_register_replaces():
    s1 = registry.register("alyona", core="first")
    s2 = registry.register("alyona", core="second")
    assert registry.get("alyona") is s2
    assert s1 is not s2


def test_unregister_removes():
    registry.register("alyona", core=object())
    out = registry.unregister("alyona")
    assert out is not None
    with pytest.raises(HTTPException):
        registry.get("alyona")


def test_unregister_unknown_returns_none():
    assert registry.unregister("ghost") is None


def test_agent_isolation():
    """Два агента — независимые AgentSession."""
    a = registry.register("alyona", core="A")
    b = registry.register("vera", core="B")
    assert registry.get("alyona").core == "A"
    assert registry.get("vera").core == "B"
    assert "alyona" in registry.all_agent_ids()
    assert "vera" in registry.all_agent_ids()
    assert a is not b


def test_reset_all_clears():
    registry.register("alyona", core="X")
    registry.register("vera", core="Y")
    registry.reset_all()
    assert registry.all_agent_ids() == []


def test_session_has_lock():
    s = registry.register("alyona", core=object())
    assert isinstance(s.write_lock, type(threading.Lock()))


def test_session_carries_tools():
    schemas = [{"name": "tool1", "description": "x", "parameters": {}}]
    s = registry.register("alyona", core=object(), tool_schemas=schemas)
    assert s.tool_schemas == schemas
