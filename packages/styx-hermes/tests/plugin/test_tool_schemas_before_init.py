"""Контракт: get_tool_schemas() работает ДО initialize().

Hermes строит routing-индекс _tool_to_provider в MemoryManager.add_provider(),
который вызывается ДО initialize() (agent_init.py:1101 vs :1144). Если
StyxMemoryProvider.get_tool_schemas() до initialize вернёт [], индекс
построится пустым и любой styx_* tool-call упадёт в "Unknown tool" —
хотя схема при этом доходит до модели (get_tool_schemas зовётся повторно
ПОСЛЕ init). Этот тест фиксирует, что свежий провайдер (без initialize)
отдаёт непустой статический каталог ядра, а после init приоритет —
у авторитетных daemon-схем.

Найдено live-e2e на Hermes v0.16.0.
"""

from __future__ import annotations

from styx_hermes import _agent_session
from styx_hermes.providers.memory import StyxMemoryProvider

import pytest


@pytest.fixture(autouse=True)
def _reset_session():
    yield
    _agent_session.clear_session()


_EXPECTED_NAMES = {
    "styx_recall",
    "styx_search_archive",
    "styx_reinterpret",
    "styx_dialogue_search",
    "styx_dialogue_recent",
    "styx_dialogue_prepare_summary",
    "styx_ingest_document",
}


def test_get_tool_schemas_before_initialize_is_static_catalog() -> None:
    """Свежий провайдер (без initialize) отдаёт статический каталог ядра."""
    p = StyxMemoryProvider()
    schemas = p.get_tool_schemas()
    assert schemas, "get_tool_schemas до initialize не должен быть пустым"
    names = {s["name"] for s in schemas}
    missing = _EXPECTED_NAMES - names
    assert not missing, f"в каталоге до init отсутствуют tools: {missing}"


def test_authoritative_schemas_take_priority_after_init() -> None:
    """После init self._tool_schemas (daemon) имеют приоритет над каталогом."""
    p = StyxMemoryProvider()
    p._tool_schemas = [{"name": "x", "description": "...", "parameters": {}}]
    schemas = p.get_tool_schemas()
    assert [s["name"] for s in schemas] == ["x"]
