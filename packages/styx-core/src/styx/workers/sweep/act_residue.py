"""Bounded retry sweep for Wave 38 act-residue reductions.

The handler records dependency failures as ``retryable`` in the same
transaction in which the claimed queue task becomes done.  This sweeper is
the only component that turns that durable state into another coordinate-only
queue task, and terminalises rows whose bounded attempt budget is exhausted.

The caller owns the outer transaction.  Each ledger row is isolated by a
savepoint so a raced or malformed row cannot discard successful work for the
remaining rows.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from styx.storage.act_reduction import (
    ActReductionConflict,
    DEFAULT_REDUCER_VERSION,
    mark_act_reduction_retryable,
    mark_act_reduction_terminal_failure,
    schedule_act_reduction,
)
from styx.storage.cognition import lock_agent_line

log = logging.getLogger(__name__)


MAX_SWEEP_LIMIT = 128


@dataclass
class ActResidueSweepSummary:
    scanned: int = 0
    reconciled: int = 0
    requeued: int = 0
    terminalized: int = 0
    raced: int = 0
    errors: int = 0


def run_act_residue_retry_sweep(
    conn: psycopg.Connection,
    *,
    max_attempts: int = 3,
    limit: int = 32,
) -> ActResidueSweepSummary:
    """Requeue retryable reductions or close their exhausted lifecycle.

    ``attempt_count`` is incremented only when a handler starts, therefore
    ``max_attempts=3`` means one initial execution and at most two retries.
    Rows are intentionally discovered without ``FOR UPDATE``: the storage API
    acquires locks in its canonical agent-line/ledger order and revalidates the
    transition, making concurrent sweepers duplicate-safe.
    """
    if isinstance(max_attempts, bool) or not 1 <= max_attempts <= 20:
        raise ValueError("max_attempts должен быть int 1..20")
    if isinstance(limit, bool) or not 1 <= limit <= MAX_SWEEP_LIMIT:
        raise ValueError(f"limit должен быть int 1..{MAX_SWEEP_LIMIT}")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT r.agent_id,r.act_id,r.reducer_version,r.input_hash,"
            "       r.task_id,r.status,r.attempt_count,"
            "       t.status AS task_status,t.retry_count AS task_retry_count,"
            "       a.status AS act_status "
            "FROM cognitive_act_reductions r "
            "LEFT JOIN llm_tasks t ON t.id=r.task_id "
            "JOIN cognitive_acts a ON a.id=r.act_id AND a.agent_id=r.agent_id "
            "WHERE r.status='retryable' "
            "   OR (r.status IN ('pending','running') AND t.status='failed') "
            "ORDER BY r.updated_at,r.id LIMIT %s",
            (limit,),
        )
        rows = list(cur.fetchall())

    summary = ActResidueSweepSummary(scanned=len(rows))
    for row in rows:
        try:
            with conn.transaction():
                _dispatch_retryable(
                    conn,
                    row=row,
                    max_attempts=max_attempts,
                    summary=summary,
                )
        except ActReductionConflict:
            # Another sweeper or writer won after discovery.  Its durable
            # transition is authoritative; this pass has nothing to repair.
            summary.raced += 1
        except Exception as exc:  # noqa: BLE001 -- isolate one durable row
            summary.errors += 1
            log.warning(
                "act_residue retry sweep failed: agent=%s act=%s",
                row.get("agent_id"),
                row.get("act_id"),
            )
            log.debug(
                "act_residue retry sweep error_class=%s", type(exc).__name__
            )
    return summary


def _dispatch_retryable(
    conn: psycopg.Connection,
    *,
    row: dict,
    max_attempts: int,
    summary: ActResidueSweepSummary,
) -> None:
    agent_id = row.get("agent_id")
    act_id = row.get("act_id")
    reducer_version = row.get("reducer_version", DEFAULT_REDUCER_VERSION)
    input_hash = row.get("input_hash")
    task_id = row.get("task_id")
    status = row.get("status")
    task_status = row.get("task_status")
    attempt_count = row.get("attempt_count")
    task_retry_count = row.get("task_retry_count", 0)
    act_status = row.get("act_status")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("retryable row has invalid agent_id")
    try:
        act_id = act_id if isinstance(act_id, uuid.UUID) else uuid.UUID(str(act_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("retryable row has invalid act_id") from exc
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
        raise ValueError("retryable row has invalid attempt_count")
    if task_retry_count is None:
        task_retry_count = 0
    if isinstance(task_retry_count, bool) or not isinstance(task_retry_count, int):
        raise ValueError("retryable row has invalid task_retry_count")
    orphan = status in {"pending", "running"} and task_status == "failed"
    if status != "retryable" and not orphan:
        raise ActReductionConflict("reduction is no longer sweepable")

    # Reconciliation followed by scheduling must preserve the storage-wide
    # lock order: agent line first, reduction ledger second.
    lock_agent_line(conn, agent_id)

    # Failed acts are causal pass-through outcomes, not reducer prompts.  Their
    # no-task ledger waits until an immutable declared parent's terminal
    # frontier can be inherited.  Re-running the scheduler performs only that
    # bounded storage resolution and never creates an LLM task.
    if act_status == "failed" and task_id is None:
        outcome = schedule_act_reduction(
            conn,
            agent_id,
            act_id,
            reducer_version=reducer_version,
            retry=True,
        )
        if outcome.status == "no_residue":
            summary.reconciled += 1
        elif outcome.status == "terminal_failure":
            summary.reconciled += 1
            summary.terminalized += 1
        return

    # A failed queue task may have rolled back the ledger's running transition.
    # Count that durable queue failure so recovery stays bounded even when the
    # ledger still says pending with attempt_count=0.
    effective_attempts = attempt_count + (task_retry_count if orphan else 0)

    if orphan and task_retry_count:
        # The handler transaction (including its ledger attempt increment)
        # rolled back, while runtime durably counted the failed/crashed claim.
        # Fold that disjoint count into the ledger before replacing the task,
        # otherwise each new task would lose the crash budget.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE cognitive_act_reductions SET attempt_count=%s,"
                "updated_at=clock_timestamp() WHERE agent_id=%s AND act_id=%s "
                "AND reducer_version=%s AND task_id=%s AND input_hash=%s",
                (
                    effective_attempts,
                    agent_id,
                    act_id,
                    reducer_version,
                    task_id,
                    input_hash,
                ),
            )
            if cur.rowcount != 1:
                raise ActReductionConflict("attempt reconciliation raced")

    if effective_attempts >= max_attempts:
        if task_id is None:
            raise ValueError("exhausted retryable row has no task_id")
        mark_act_reduction_terminal_failure(
            conn,
            agent_id,
            act_id,
            reducer_version=reducer_version,
            task_id=task_id,
            input_hash=input_hash,
            error_code=("orphan_retry_exhausted" if orphan else "retry_exhausted"),
        )
        if orphan:
            summary.reconciled += 1
        summary.terminalized += 1
        return

    if orphan:
        if task_id is None:
            raise ValueError("orphan reduction row has no task_id")
        mark_act_reduction_retryable(
            conn,
            agent_id,
            act_id,
            reducer_version=reducer_version,
            task_id=task_id,
            input_hash=input_hash,
            error_code="orphan_failed_task",
        )
        summary.reconciled += 1

    outcome = schedule_act_reduction(
        conn,
        agent_id,
        act_id,
        reducer_version=reducer_version,
        retry=True,
    )
    if outcome.duplicate:
        summary.raced += 1
    else:
        summary.requeued += 1
