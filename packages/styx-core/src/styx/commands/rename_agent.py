"""``styx rename-agent <old> <new>`` — переименование ``agent_id``.

Волна 32. Этап 1 плана миграции agent-a/agent-b (memorybox→styx):
приведение ``agent_id`` к именам Hermes-профилей.

``agent_id`` — денормализованная строка в agent-scoped таблицах
(на момент волны 32 — 9: memories, sessions, working_set,
emotional_state, emotional_baseline, documents,
memory_reinterpretations, reinterpret_applications,
memory_consolidation_applications; точный список — schema-driven,
см. ниже). **НЕ FK**, нет ``agents``-таблицы, нет ``ON UPDATE CASCADE``.
Вся структурная целостность — на UUID'ах (``memory_id``,
``document_id``, ``session_id``, граф ``source_id/target_id``).
UUID-scoped таблицы (``relations``, ``recall_events``, ``chunks``,
``llm_tasks`` — у них колонки ``agent_id`` НЕТ) следуют за памятью
автоматически: их FK на ``memories``/``documents`` (UUID) не меняются.
⇒ чистый rename = ``UPDATE ... SET agent_id`` по agent_id-таблицам, без
ремапа UUID; эмбеддинги, граф, переосмысления, эмоц-снапшоты остаются
байт-в-байт (см. wave-doc § «Ключевой факт схемы»).

Инвариант волны 27 («деградация линии `я` запрещена»): пропущенная
таблица = расщеплённое `я`. Поэтому список таблиц — **schema-driven**
из ``information_schema``, НЕ хардкод (хардкод протухнет на новой
волне со своей agent_id-таблицей).

``run_rename_agent`` НЕ коммитит — транзакция caller-owned. Атомарность
всех UPDATE'ов обеспечивает один ``with conn:`` блок в CLI (commit на
чистом выходе / rollback на исключении). Частичный rename =
расщеплённое `я`, хуже отказа.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg import sql

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenameResult:
    old: str
    new: str
    # (table_name, rowcount) — для dry_run rowcount = count(*) WHERE
    # agent_id=old (would-rename); для реального прогона — UPDATE rowcount.
    tables: list[tuple[str, int]]
    total_rows: int
    dry_run: bool


def _agent_id_tables(conn: "psycopg.Connection") -> list[str]:
    """Список public-таблиц с колонкой ``agent_id`` (schema-driven).

    Источник истины — ``information_schema.columns`` в рантайме, НЕ
    хардкод: новая волна со своей agent_id-таблицей не должна молча
    выпасть из rename'а (иначе расщеплённое `я`).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.columns "
            "WHERE column_name = 'agent_id' AND table_schema = 'public' "
            "ORDER BY table_name"
        )
        return [row[0] for row in cur.fetchall()]


def _count_for_agent(
    conn: "psycopg.Connection", table: str, agent_id: str
) -> int:
    """count(*) рядов таблицы с заданным agent_id. table — Identifier."""
    query = sql.SQL("SELECT count(*) FROM {} WHERE agent_id = %s").format(
        sql.Identifier(table)
    )
    with conn.cursor() as cur:
        cur.execute(query, (agent_id,))
        return int(cur.fetchone()[0])


def run_rename_agent(
    *,
    conn: "psycopg.Connection",
    old: str,
    new: str,
    dry_run: bool = False,
) -> RenameResult:
    """Переименовать ``agent_id`` ``old`` → ``new`` по всем agent-scoped
    таблицам. НЕ коммитит — транзакция caller-owned (атомарность одной
    tx даёт «всё-или-ничего»).

    Валидация / existence / collision выполняются до любого UPDATE и
    кидают ``ValueError`` с actionable-сообщением. ``dry_run`` считает
    per-table count'ы для ``old`` без записи.
    """
    old_s = (old or "").strip()
    new_s = (new or "").strip()
    if not old_s:
        raise ValueError("old agent_id пуст")
    if not new_s:
        raise ValueError("new agent_id пуст")
    if old_s == new_s:
        raise ValueError(f"old == new ({old_s!r}) — переименовывать нечего")

    tables = _agent_id_tables(conn)
    if not tables:
        raise ValueError(
            "не найдено ни одной таблицы с колонкой agent_id — "
            "схема не применена? (запусти styx migrate)"
        )

    # Existence: old обязан существовать хотя бы где-то.
    old_counts: dict[str, int] = {}
    total_old = 0
    for t in tables:
        n = _count_for_agent(conn, t, old_s)
        old_counts[t] = n
        total_old += n
    if total_old == 0:
        raise ValueError(f"agent '{old_s}' не найден (нет данных)")

    # Collision refuse: new не должен существовать ни в одной таблице.
    for t in tables:
        if _count_for_agent(conn, t, new_s) > 0:
            raise ValueError(
                f"agent '{new_s}' уже существует — откажусь сливать; "
                f"найдены строки в таблице '{t}'. Слияние агентов "
                "(--merge) не поддержано в этой волне; выбери свободное "
                "имя или удали данные нового агента вручную."
            )

    if dry_run:
        per_table = [(t, old_counts[t]) for t in tables]
        log.info(
            "rename-agent dry-run: %r → %r — would rename %d rows across %d tables",
            old_s, new_s, total_old, len(tables),
        )
        return RenameResult(
            old=old_s,
            new=new_s,
            tables=per_table,
            total_rows=total_old,
            dry_run=True,
        )

    log.info("rename-agent: %r → %r — старт по %d таблицам", old_s, new_s, len(tables))
    per_table_updated: list[tuple[str, int]] = []
    total_updated = 0
    for t in tables:
        query = sql.SQL("UPDATE {} SET agent_id = %s WHERE agent_id = %s").format(
            sql.Identifier(t)
        )
        with conn.cursor() as cur:
            cur.execute(query, (new_s, old_s))
            n = cur.rowcount
        per_table_updated.append((t, n))
        total_updated += n
        if n:
            log.info("rename-agent: %s — %d рядов %r → %r", t, n, old_s, new_s)

    log.info(
        "rename-agent: %r → %r — переименовано %d рядов в %d таблицах "
        "(транзакция не закоммичена — caller-owned)",
        old_s, new_s, total_updated, len(tables),
    )
    return RenameResult(
        old=old_s,
        new=new_s,
        tables=per_table_updated,
        total_rows=total_updated,
        dry_run=False,
    )
