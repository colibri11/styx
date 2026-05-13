"""Reinterpret apply-sweeper — port memorybox `reinterpret-apply-sweeper.ts`
(волна 22).

Periodic-task в worker'е. Раз в `apply_tick_s` (default 30s):
- per-agent loop: `is_active(agent_id) == True` → fast-path skip;
- иначе — load pending_sleep applications + JOIN llm_tasks;
- для каждой row dispatch:
  - task_status='failed' → `mark_skipped`.
  - task done && skip-shape result → `mark_skipped`.
  - task done && merged-shape result → re-check `is_active`, если
    idle — apply (UPDATE memory + INSERT audit + UPDATE applications).
  - task pending/running → defer, попробуем на следующем tick'е.

Каждая row — отдельный psycopg3 atomic block (`conn.transaction()`):
COMMIT при успехе, ROLLBACK при exception. Один сломавшийся row не
ломает остальные.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg

from styx import turn_state
from styx.storage.queries import AgentScopedQueries, list_subject_agents

log = logging.getLogger(__name__)


@dataclass
class SweepSummary:
    applied: int = 0
    skipped: int = 0
    deferred: int = 0
    errors: int = 0


def _is_skip_shape(result: Any) -> bool:
    """LLM handler skip-shape: skip=True. Также покрывает legacy
    {"skipped": "memory_gone"|"no_memory_id"} формы."""
    if not isinstance(result, dict):
        return False
    if result.get("skip") is True:
        return True
    if isinstance(result.get("skipped"), str):
        return True
    return False


def _is_merged_shape(result: Any) -> bool:
    """skip=False с обязательными полями merged_text/embedding/previous*."""
    if not isinstance(result, dict):
        return False
    if result.get("skip") is not False:
        return False
    return (
        isinstance(result.get("merged_text"), str)
        and isinstance(result.get("merged_embedding"), list)
        and isinstance(result.get("previous_text"), str)
        and isinstance(result.get("previous_embedding"), list)
        and isinstance(result.get("weight_applied"), (int, float))
    )


def run_reinterpret_apply_sweep(
    conn: psycopg.Connection,
    *,
    now: _dt.datetime | None = None,
) -> SweepSummary:
    """Один проход sweeper'а. Возвращает summary. Не делает rollback —
    psycopg3 `conn.transaction()` per row делает atomic block.
    """
    summary = SweepSummary()
    agents = list_subject_agents(conn)
    for agent_id in agents:
        # Fast-path skip — owner в активном turn'е.
        if turn_state.is_active(agent_id, now=now):
            continue

        try:
            queries = AgentScopedQueries(conn, agent_id)
            pending = queries.load_pending_reinterpret_applications()
        except Exception as exc:  # noqa: BLE001
            summary.errors += 1
            log.warning(
                "reinterpret_apply: scope failure for %s: %s",
                agent_id, exc,
            )
            try:
                conn.rollback()
            except Exception:
                pass
            continue

        for row in pending:
            try:
                with conn.transaction():
                    _dispatch_application(
                        conn=conn, agent_id=agent_id, row=row,
                        summary=summary, now=now,
                    )
            except Exception as exc:  # noqa: BLE001
                summary.errors += 1
                log.warning(
                    "reinterpret_apply: row %s упал: %s",
                    row.get("application_id"), exc,
                )

    return summary


def _dispatch_application(
    *,
    conn: psycopg.Connection,
    agent_id: str,
    row: dict,
    summary: SweepSummary,
    now: _dt.datetime | None,
) -> None:
    """Per-row dispatch внутри `conn.transaction()` block'а."""
    queries = AgentScopedQueries(conn, agent_id)
    application_id = int(row["application_id"])
    task_status = row["task_status"]
    task_result = row["task_result"]

    if task_status == "failed":
        queries.mark_reinterpret_skipped(application_id)
        summary.skipped += 1
        return

    if task_status != "done":
        # pending/running — LLM ещё считает.
        summary.deferred += 1
        return

    # task done.
    if _is_skip_shape(task_result):
        queries.mark_reinterpret_skipped(application_id)
        summary.skipped += 1
        return

    if not _is_merged_shape(task_result):
        # Defensive — unexpected shape тоже skipped (ничего не применяем).
        queries.mark_reinterpret_skipped(application_id)
        summary.skipped += 1
        return

    # Re-check is_active перед apply — race window между agent-list step
    # и dispatch'ем (owner мог открыть turn).
    if turn_state.is_active(agent_id, now=now):
        summary.deferred += 1
        return

    memory_id = uuid.UUID(row["memory_id"])
    merged_text = task_result["merged_text"]
    merged_embedding = [float(x) for x in task_result["merged_embedding"]]
    previous_text = task_result["previous_text"]
    previous_embedding = [float(x) for x in task_result["previous_embedding"]]
    new_understanding_text = task_result.get("new_understanding_text", "")
    weight_applied = float(task_result["weight_applied"])

    rc = queries.apply_reinterpret_update(
        memory_id=memory_id,
        merged_text=merged_text,
        merged_embedding=merged_embedding,
    )
    if rc == 0:
        # Memory исчезла между handler'ом и apply'ем (CASCADE на DELETE
        # снёс бы и task — но защищаемся).
        log.warning(
            "reinterpret_apply: memory %s исчезла, mark_skipped",
            memory_id,
        )
        queries.mark_reinterpret_skipped(application_id)
        summary.skipped += 1
        return

    queries.insert_memory_reinterpretation(
        memory_id=memory_id,
        previous_text=previous_text,
        new_understanding_text=new_understanding_text,
        merged_text=merged_text,
        previous_embedding=previous_embedding,
        merged_embedding=merged_embedding,
        weight_applied=weight_applied,
    )
    queries.mark_reinterpret_applied(application_id)
    summary.applied += 1
