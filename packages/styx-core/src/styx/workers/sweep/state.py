"""KV-доступ к ``consolidation_state`` (memorybox 009 port).

Таблица — один JSONB value на key. Используется sweep'ом для
- ``lifecycle_thresholds`` — EMA-сглаженные пороги fresh→settled и
  settled→dormant.
- ``last_sweep_heartbeat`` — отметка времени последнего runа (для
  диагностики hung sweep'а).

Все писатели должны commit'ить свои изменения сами; модуль не делает
commit.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def get_state(conn: psycopg.Connection, key: str) -> dict[str, Any] | None:
    """SELECT значения по ключу. ``None`` если ключа нет."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT value FROM consolidation_state WHERE key = %s", (key,)
        )
        row = cur.fetchone()
    if row is None:
        return None
    return row["value"]


def set_state(
    conn: psycopg.Connection, key: str, value: dict[str, Any]
) -> None:
    """UPSERT JSONB value по ключу. Не коммитит."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO consolidation_state (key, value, updated_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (key) DO UPDATE "
            "  SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
            (key, Jsonb(value)),
        )
