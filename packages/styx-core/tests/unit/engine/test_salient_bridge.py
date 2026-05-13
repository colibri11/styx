"""Юнит-тесты salient_bridge — configure/get/reset module-global state."""

from __future__ import annotations

import pytest

from styx.engine import salient_bridge
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


class _FakeQueries:
    pass


class _FakeEmbed:
    @property
    def dim(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768


@pytest.fixture(autouse=True)
def _reset_handle() -> None:
    salient_bridge.reset_all()
    yield
    salient_bridge.reset_all()


def test_get_handle_returns_none_initially() -> None:
    assert salient_bridge.get_handle("test-agent") is None


def test_configure_then_get_handle() -> None:
    queries = _FakeQueries()
    embed = _FakeEmbed()

    salient_bridge.configure("test-agent", 
        queries=queries,
        embed_client=embed,
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=0.5,
        min_query_len=15,
    )

    handle = salient_bridge.get_handle("test-agent")
    assert handle is not None
    assert handle.queries is queries
    assert handle.embed_client is embed
    assert handle.recall_config is DEFAULT_RECALL_CONFIG
    assert handle.timeout_s == 0.5
    assert handle.min_query_len == 15


def test_configure_uses_defaults() -> None:
    salient_bridge.configure("test-agent", 
        queries=_FakeQueries(),
        embed_client=_FakeEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
    )
    handle = salient_bridge.get_handle("test-agent")
    assert handle is not None
    assert handle.timeout_s == 1.0
    assert handle.min_query_len == 20


def test_double_configure_replaces_handle() -> None:
    q1, q2 = _FakeQueries(), _FakeQueries()
    salient_bridge.configure("test-agent", 
        queries=q1, embed_client=_FakeEmbed(), recall_config=DEFAULT_RECALL_CONFIG,
    )
    salient_bridge.configure("test-agent", 
        queries=q2, embed_client=_FakeEmbed(), recall_config=DEFAULT_RECALL_CONFIG,
    )
    handle = salient_bridge.get_handle("test-agent")
    assert handle is not None
    assert handle.queries is q2


def test_reset_clears_handle() -> None:
    salient_bridge.configure("test-agent", 
        queries=_FakeQueries(),
        embed_client=_FakeEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
    )
    assert salient_bridge.get_handle("test-agent") is not None
    salient_bridge.reset_all()
    assert salient_bridge.get_handle("test-agent") is None
