"""Top-level sweep runner с advisory lock + per-task try/except.

Прямой port `consolidation/sweep.ts` (memorybox). Один runner крутит
список задач последовательно, каждая со своим transactional boundary.
Advisory lock — чтобы два конкурирующих ``styx-worker`` не запустили
sweep одновременно.

В волне 7b — единственная задача `lifecycle_refresh`. Дальнейшие волны
добавляют свои.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import psycopg

from styx.observability.logging import log_event
from styx.workers.sweep.lifecycle import lifecycle_refresh
from styx.workers.sweep.runs import finish_run, start_run
from styx.workers.sweep.state import set_state

log = logging.getLogger(__name__)


# Тот же ключ что в memorybox `consolidation/sweep.ts` — на случай
# если когда-нибудь Styx и memorybox будут в одной БД (маловероятно
# но без потерь).
SWEEP_ADVISORY_LOCK_KEY = 9876543210

DEFAULT_LOCK_TIMEOUT_S = 1800


@dataclass(frozen=True)
class SweepTask:
    name: str
    fn: Callable[[psycopg.Connection], dict[str, Any]]


@dataclass(frozen=True)
class SweepResult:
    status: str  # success | partial | failed
    sweep_id: Any | None  # uuid.UUID | None
    summary: dict[str, Any]
    errors: list[dict[str, Any]]
    skipped: bool


def _build_default_task_list() -> list[SweepTask]:
    return [
        SweepTask(name="lifecycle_refresh", fn=lifecycle_refresh),
    ]


def run_sweep(
    dsn: str,
    *,
    tasks: list[SweepTask] | None = None,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
) -> SweepResult:
    """Запустить полный sweep. Открывает свою connection, держит advisory
    lock на её session, после освобождает и закрывает."""
    task_list = tasks if tasks is not None else _build_default_task_list()
    started_monotonic = time.monotonic()

    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s::bigint) AS ok",
                (SWEEP_ADVISORY_LOCK_KEY,),
            )
            row = cur.fetchone()
        conn.commit()
        if not row or not row[0]:
            log.warning("sweep skipped: advisory lock held by another instance")
            return SweepResult(
                status="success",
                sweep_id=None,
                summary={},
                errors=[],
                skipped=True,
            )

        started_at = _dt.datetime.now(tz=_dt.timezone.utc)
        sweep_id = None
        try:
            sweep_id = start_run(conn, started_at)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.error("sweep: start_run failed: %s", exc)
            return SweepResult(
                status="failed",
                sweep_id=None,
                summary={},
                errors=[{"task": "sweep_start", "message": str(exc)}],
                skipped=False,
            )

        # Heartbeat — чтобы операторы могли увидеть hung sweep.
        try:
            set_state(
                conn,
                "last_sweep_heartbeat",
                {
                    "pid": os.getpid(),
                    "started_at": started_at.isoformat(),
                },
            )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("sweep: heartbeat write failed: %s", exc)

        lock_timeout_ms = max(1000, int(lock_timeout_s * 1000))
        summary: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []

        for task in task_list:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SET statement_timeout = {lock_timeout_ms}"
                    )
                conn.commit()
                result = task.fn(conn)
                summary[task.name] = result
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                log.warning("sweep task %s failed: %s", task.name, msg)
                errors.append({"task": task.name, "message": msg})
                summary[task.name] = {"error": msg}
                # Откатить любые незакоммиченные изменения провалившейся
                # задачи — чтобы следующая стартовала с чистой tx.
                try:
                    conn.rollback()
                except psycopg.Error:
                    pass
            finally:
                try:
                    with conn.cursor() as cur:
                        cur.execute("RESET statement_timeout")
                    conn.commit()
                except psycopg.Error:
                    pass

        any_succeeded = any(
            isinstance(s, dict) and "error" not in s for s in summary.values()
        )
        if not errors:
            status = "success"
        elif any_succeeded:
            status = "partial"
        else:
            status = "failed"

        try:
            finish_run(conn, sweep_id, status, summary, errors)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("sweep: finish_run failed: %s", exc)

        log_event(
            log,
            "sweep_cycle",
            status=status,
            tasks=len(task_list),
            errors=len(errors),
            elapsed_s=round(time.monotonic() - started_monotonic, 3),
        )
        return SweepResult(
            status=status,
            sweep_id=sweep_id,
            summary=summary,
            errors=errors,
            skipped=False,
        )
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock(%s::bigint)",
                    (SWEEP_ADVISORY_LOCK_KEY,),
                )
            conn.commit()
        except psycopg.Error as exc:
            log.warning("sweep: advisory_unlock failed: %s", exc)
        try:
            conn.close()
        except psycopg.Error:
            pass
