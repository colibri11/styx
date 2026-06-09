"""Host-side unit-тесты для StyxMemoryCore._guarded_write (волна 34).

Без Postgres — мокаем self._conn. Проверяем сам rollback-guard
context-manager в изоляции:

  (а) исключение внутри блока → вызывает self._conn.rollback() И
      пробрасывает исключение наружу (re-raise, D3);
  (б) чистый проход → rollback не зовётся.

Дополнительно: rollback внутри guard'а сам по себе может упасть
(соединение уже мёртвое) — guard это глотает (defensive), но исходное
исключение всё равно пробрасывает.

Транзакционные сценарии на реальном Postgres (FK→NULL, отравление→
восстановление, routed regress) — в integration-наборе (Phase C).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from styx.providers.memory import StyxMemoryCore


def _core_with_mock_conn() -> tuple[StyxMemoryCore, MagicMock]:
    """Инстанс провайдера с мок-соединением, без initialize/Postgres.

    __init__ не открывает соединение — только проставляет атрибуты,
    поэтому конструируем напрямую и подменяем self._conn.
    """
    core = StyxMemoryCore(agent_id="test-agent")
    conn = MagicMock(name="conn")
    core._conn = conn  # type: ignore[assignment]
    return core, conn


def test_guarded_write_rollback_and_reraise_on_exception() -> None:
    """(а) Исключение внутри блока → rollback + re-raise."""
    core, conn = _core_with_mock_conn()

    sentinel = RuntimeError("insert упал (например, ForeignKeyViolation)")
    with pytest.raises(RuntimeError) as excinfo:
        with core._guarded_write("unit_label"):
            raise sentinel

    # Исходное исключение проброшено без подмены (D3 — никакого swallowing).
    assert excinfo.value is sentinel
    # Соединение откатано ровно один раз (defensive rollback).
    conn.rollback.assert_called_once_with()


def test_guarded_write_no_rollback_on_clean_pass() -> None:
    """(б) Чистый проход → rollback НЕ зовётся."""
    core, conn = _core_with_mock_conn()

    marker: list[str] = []
    with core._guarded_write("unit_label"):
        marker.append("body-executed")

    assert marker == ["body-executed"]
    conn.rollback.assert_not_called()


def test_guarded_write_commit_inside_block_not_touched() -> None:
    """Commit'ы внутри блока — забота тела, guard их не трогает на
    успешном проходе (rollback не зовётся, commit проброшен телом)."""
    core, conn = _core_with_mock_conn()

    with core._guarded_write("unit_label"):
        core._conn.commit()  # type: ignore[union-attr]

    conn.commit.assert_called_once_with()
    conn.rollback.assert_not_called()


def test_guarded_write_swallows_rollback_failure_but_reraises_original() -> None:
    """Если сам rollback падает (мёртвое соединение) — guard глотает
    эту вторичную ошибку (defensive), но исходное исключение всё равно
    пробрасывает наружу."""
    core, conn = _core_with_mock_conn()
    conn.rollback.side_effect = RuntimeError("rollback тоже упал")

    original = ValueError("primary failure")
    with pytest.raises(ValueError) as excinfo:
        with core._guarded_write("unit_label"):
            raise original

    # Наружу выходит ИСХОДНОЕ исключение, не ошибка rollback'а.
    assert excinfo.value is original
    conn.rollback.assert_called_once_with()
