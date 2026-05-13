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
    sess = registry.register("agent-a", core=object())
    got = registry.get("agent-a")
    assert got is sess


def test_double_register_replaces():
    s1 = registry.register("agent-a", core="first")
    s2 = registry.register("agent-a", core="second")
    assert registry.get("agent-a") is s2
    assert s1 is not s2


def test_unregister_removes():
    registry.register("agent-a", core=object())
    out = registry.unregister("agent-a")
    assert out is not None
    with pytest.raises(HTTPException):
        registry.get("agent-a")


def test_unregister_unknown_returns_none():
    assert registry.unregister("ghost") is None


def test_agent_isolation():
    """Два агента — независимые AgentSession."""
    a = registry.register("agent-a", core="A")
    b = registry.register("agent-b", core="B")
    assert registry.get("agent-a").core == "A"
    assert registry.get("agent-b").core == "B"
    assert "agent-a" in registry.all_agent_ids()
    assert "agent-b" in registry.all_agent_ids()
    assert a is not b


def test_reset_all_clears():
    registry.register("agent-a", core="X")
    registry.register("agent-b", core="Y")
    registry.reset_all()
    assert registry.all_agent_ids() == []


def test_session_has_lock():
    s = registry.register("agent-a", core=object())
    assert isinstance(s.write_lock, type(threading.Lock()))


def test_session_carries_tools():
    schemas = [{"name": "tool1", "description": "x", "parameters": {}}]
    s = registry.register("agent-a", core=object(), tool_schemas=schemas)
    assert s.tool_schemas == schemas
