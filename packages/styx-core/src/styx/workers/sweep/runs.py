"""Учёт sweep-прогонов в таблице ``sweep_runs``.

INSERT'нуть строку при старте, UPDATE'нуть при завершении —
observability для операционной диагностики (был ли последний sweep,
сколько занял, какой status).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def start_run(conn: psycopg.Connection, started_at: _dt.datetime) -> uuid.UUID:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sweep_runs (started_at, status) "
            "VALUES (%s, 'running') RETURNING id",
            (started_at,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("INSERT в sweep_runs не вернул id")
    return row[0]


def finish_run(
    conn: psycopg.Connection,
    sweep_id: uuid.UUID,
    status: str,
    summary: dict[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sweep_runs "
            "   SET finished_at = now(), status = %s, summary = %s, errors = %s "
            " WHERE id = %s",
            (status, Jsonb(summary), Jsonb(errors), sweep_id),
        )
