"""``run_rename_agent`` integration: реальный Postgres (волна 32).

Маркер ``STYX_TEST_DATABASE_URL`` (как у остальных integration). Каждый
тест работает на ``migrated_db`` (миграции применены, БД пуста).

Покрытие (см. wave-doc § «Приёмка»):
- rename переносит строки old → new по всем засеянным таблицам;
- completeness: после rename ``old`` не остаётся НИ в одной
  agent_id-таблице из ``information_schema`` (гард на будущие таблицы);
- collision refuse: ``new`` существует → ошибка, БД не изменена;
- dry_run: counts отчитаны, БД не изменена;
- non-existent old → ошибка.

Unit-кейсы валидации (без БД) — в ``tests/commands/test_rename_agent_unit.py``.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from styx.commands.rename_agent import (
    RenameResult,
    _agent_id_tables,
    run_rename_agent,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


# ── seed helpers ────────────────────────────────────────────────────
#
# Прямой INSERT с минимальными обязательными колонками (сверено по
# schema/*.sql, не угадано). Засеваем НЕСКОЛЬКО agent_id-таблиц:
# sessions, memories, working_set, emotional_state, emotional_baseline.
# (relations / chunks намеренно НЕ берём — у них НЕТ колонки agent_id,
# scope через FK на UUID; rename их не трогает by design.)


def _seed_agent(conn: "psycopg.Connection", agent: str, *, n_memories: int = 2) -> dict[str, int]:
    """Засеять агента строками в нескольких agent_id-таблицах.

    Возвращает {table: ожидаемое_число_строк} для последующих assert'ов.
    """
    expected: dict[str, int] = {}
    with conn.cursor() as cur:
        session_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO sessions (id, agent_id) VALUES (%s, %s)",
            (session_id, agent),
        )
        expected["sessions"] = 1

        for i in range(n_memories):
            cur.execute(
                "INSERT INTO memories (agent_id, role, content) "
                "VALUES (%s, 'summary', %s)",
                (agent, f"factum {agent} #{i}"),
            )
        expected["memories"] = n_memories

        cur.execute(
            "INSERT INTO working_set (agent_id, payload) VALUES (%s, %s::jsonb)",
            (agent, "{}"),
        )
        expected["working_set"] = 1

        cur.execute(
            "INSERT INTO emotional_state (agent_id, valence, arousal, dominance) "
            "VALUES (%s, 0.1, 0.2, 0.3)",
            (agent,),
        )
        expected["emotional_state"] = 1

        cur.execute(
            "INSERT INTO emotional_baseline (agent_id) VALUES (%s)",
            (agent,),
        )
        expected["emotional_baseline"] = 1
    conn.commit()
    return expected


def _count(conn: "psycopg.Connection", table: str, agent: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            psycopg.sql.SQL("SELECT count(*) FROM {} WHERE agent_id = %s").format(
                psycopg.sql.Identifier(table)
            ),
            (agent,),
        )
        return int(cur.fetchone()[0])


def _total_for_agent(conn: "psycopg.Connection", agent: str) -> int:
    return sum(_count(conn, t, agent) for t in _agent_id_tables(conn))


# ── rename переносит ────────────────────────────────────────────────


def test_rename_moves_rows(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        expected = _seed_agent(conn, "a", n_memories=3)
        total_seed = sum(expected.values())

        result = run_rename_agent(conn=conn, old="a", new="b")
        conn.commit()

        assert isinstance(result, RenameResult)
        assert result.old == "a"
        assert result.new == "b"
        assert result.dry_run is False
        assert result.total_rows == total_seed

        # Per-table: засеянные таблицы перенесены, 0 у 'a', N у 'b'.
        per_table = dict(result.tables)
        for table, n in expected.items():
            assert _count(conn, table, "a") == 0, table
            assert _count(conn, table, "b") == n, table
            assert per_table[table] == n, table


# ── completeness (гард на будущие agent_id-таблицы) ─────────────────


def test_rename_completeness_old_gone_everywhere(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        _seed_agent(conn, "a")
        run_rename_agent(conn=conn, old="a", new="b")
        conn.commit()

        # Ни одной agent_id-таблицы (из information_schema) не должно
        # остаться со строкой 'a' — гард на таблицы будущих волн.
        for table in _agent_id_tables(conn):
            assert _count(conn, table, "a") == 0, f"осталась строка 'a' в {table}"


# ── collision refuse ────────────────────────────────────────────────


def test_rename_collision_refuses_and_no_change(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        _seed_agent(conn, "a")
        _seed_agent(conn, "b")
        before_a = _total_for_agent(conn, "a")
        before_b = _total_for_agent(conn, "b")

        with pytest.raises(ValueError, match="уже существует"):
            run_rename_agent(conn=conn, old="a", new="b")
        # Откатываем транзакцию (как сделал бы `with conn:` на исключении).
        conn.rollback()

        assert _total_for_agent(conn, "a") == before_a
        assert _total_for_agent(conn, "b") == before_b


# ── dry_run ─────────────────────────────────────────────────────────


def test_rename_dry_run_counts_without_change(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        expected = _seed_agent(conn, "a")
        total_seed = sum(expected.values())

        result = run_rename_agent(conn=conn, old="a", new="b", dry_run=True)
        conn.commit()

        assert result.dry_run is True
        assert result.total_rows == total_seed
        per_table = dict(result.tables)
        for table, n in expected.items():
            assert per_table[table] == n, table

        # БД не изменена: 'a' на месте, 'b' нет нигде.
        assert _total_for_agent(conn, "a") == total_seed
        assert _total_for_agent(conn, "b") == 0


# ── non-existent old ────────────────────────────────────────────────


def test_rename_nonexistent_old_raises(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        with pytest.raises(ValueError, match="не найден"):
            run_rename_agent(conn=conn, old="ghost", new="b")
