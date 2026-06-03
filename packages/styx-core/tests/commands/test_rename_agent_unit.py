"""Unit-тесты валидации ``run_rename_agent`` — без БД (волна 32).

Валидация (strip / непустота / ``old != new``) выполняется ДО любого
запроса к ``conn``, поэтому вызываем с ``conn=None`` — до обращения к
БД дело не доходит, ValueError кидается раньше. Integration-кейсы
(перенос / completeness / collision / dry_run / non-existent) — в
``tests/integration/test_rename_agent.py`` (нужен Postgres).
"""

from __future__ import annotations

import pytest

from styx.commands.rename_agent import run_rename_agent


def test_empty_old_raises() -> None:
    with pytest.raises(ValueError, match="old agent_id пуст"):
        run_rename_agent(conn=None, old="", new="b")


def test_whitespace_old_raises() -> None:
    with pytest.raises(ValueError, match="old agent_id пуст"):
        run_rename_agent(conn=None, old="   ", new="b")


def test_empty_new_raises() -> None:
    with pytest.raises(ValueError, match="new agent_id пуст"):
        run_rename_agent(conn=None, old="a", new="")


def test_whitespace_new_raises() -> None:
    with pytest.raises(ValueError, match="new agent_id пуст"):
        run_rename_agent(conn=None, old="a", new="  ")


def test_old_equals_new_raises() -> None:
    with pytest.raises(ValueError, match="old == new"):
        run_rename_agent(conn=None, old="same", new="same")


def test_old_equals_new_after_strip_raises() -> None:
    # strip приводит к равенству — тоже отказ (нечего переименовывать).
    with pytest.raises(ValueError, match="old == new"):
        run_rename_agent(conn=None, old=" same ", new="same")
