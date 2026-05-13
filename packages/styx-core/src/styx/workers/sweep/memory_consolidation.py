"""Memory-over-memory daily consolidation — periodic-task'и (волна 22).

Содержит две функции:

- ``run_memory_consolidation_scheduler_tick`` — раз в час: для каждого
  subject-агента проверяет cooldown (23h), при passed: SELECT
  candidate window (7d..24h), greedy-clustering (cosine ≥ 0.88,
  size 3..8), enqueue task'и + applications.

- ``run_memory_consolidation_apply_sweep`` — раз в 30s: per-agent loop
  с fast-path skip на `is_active`; для done+merged-shape — apply
  (INSERT new memory + UPDATE source.superseded_by + UPDATE
  applications status='applied').
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg

from styx import turn_state
from styx.engine.memory_consolidation import (
    ClusterCandidate,
    MemoryConsolidationConfig,
    build_clusters,
    cooldown_elapsed,
    pick_consolidated_kind,
    pick_consolidated_visibility,
)
from styx.storage.queries import (
    AgentScopedQueries,
    enqueue_llm_task,
    get_memory_daily_state,
    list_subject_agents,
    parse_vector,
    set_memory_daily_state,
)
from styx.workers.handlers.memory_daily_consolidation import (
    MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
)
from styx.workers.sweep.reinterpret_apply import SweepSummary

log = logging.getLogger(__name__)


# ── Scheduler ────────────────────────────────────────────────────────


def run_memory_consolidation_scheduler_tick(
    conn: psycopg.Connection,
    *,
    config: MemoryConsolidationConfig,
    now: _dt.datetime | None = None,
) -> int:
    """Один tick scheduler'а. Возвращает кол-во enqueued task'ов
    (sum across агентов). Caller (periodic-task wrapper в LlmWorker)
    управляет commit'ом."""
    if not config.enabled:
        return 0
    moment = now if now is not None else _dt.datetime.now(tz=_dt.timezone.utc)
    agents = list_subject_agents(conn)
    total_enqueued = 0
    for agent_id in agents:
        try:
            enq = _maybe_run_for_agent(conn, agent_id, at=moment, config=config)
            total_enqueued += enq
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory_consolidation scheduler: agent=%s упал: %s",
                agent_id, exc,
            )
            try:
                conn.rollback()
            except Exception:
                pass
    return total_enqueued


def _maybe_run_for_agent(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    at: _dt.datetime,
    config: MemoryConsolidationConfig,
) -> int:
    """Один проход scheduler'а на agent. Возвращает кол-во enqueued."""
    state = get_memory_daily_state(conn, agent_id)
    if not cooldown_elapsed(state, at, hours=config.cooldown_hours):
        return 0

    queries = AgentScopedQueries(conn, agent_id)
    window_from = at - _dt.timedelta(days=config.window_days)
    window_to = at - _dt.timedelta(hours=config.window_tail_hours)
    rows = queries.select_consolidation_window(
        window_from=window_from, window_to=window_to,
    )

    items: list[ClusterCandidate] = []
    for r in rows:
        emb = parse_vector(r.get("embedding"))
        if emb and len(emb) > 0:
            items.append(ClusterCandidate(id=r["id"], embedding=emb))

    clusters = build_clusters(
        items,
        cosine_threshold=config.cosine_threshold,
        min_size=config.min_cluster_size,
        max_size=config.max_cluster_size,
    )

    enqueued = 0
    for cluster in clusters:
        try:
            payload = {
                "agent_id": agent_id,
                "memory_ids": [str(mid) for mid in cluster.member_ids],
            }
            task_id = enqueue_llm_task(
                conn,
                task_type=MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
                payload=payload,
            )
            queries.insert_memory_consolidation_application(
                task_id=task_id, source_ids=cluster.member_ids,
            )
            enqueued += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory_consolidation enqueue failed agent=%s cluster=%r: %s",
                agent_id, cluster.member_ids, exc,
            )

    new_state = {
        "last_run_at": at.isoformat(),
        "last_window_to": at.isoformat(),
        "last_enqueued": enqueued,
    }
    set_memory_daily_state(conn, agent_id, new_state)
    return enqueued


# ── Apply-sweeper ────────────────────────────────────────────────────


def _is_skip_shape(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("skip") is True:
        return True
    if isinstance(result.get("skipped"), str):
        return True
    return False


def _is_merged_shape(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("skip") is not False:
        return False
    return (
        isinstance(result.get("consolidated_text"), str)
        and isinstance(result.get("consolidated_embedding"), list)
        and isinstance(result.get("source_ids"), list)
        and isinstance(result.get("source_kinds"), list)
        and isinstance(result.get("source_visibility"), list)
    )


def run_memory_consolidation_apply_sweep(
    conn: psycopg.Connection,
    *,
    now: _dt.datetime | None = None,
) -> SweepSummary:
    """Один проход apply-sweeper'а. Возвращает summary."""
    summary = SweepSummary()
    agents = list_subject_agents(conn)
    for agent_id in agents:
        if turn_state.is_active(agent_id, now=now):
            continue

        try:
            queries = AgentScopedQueries(conn, agent_id)
            pending = queries.load_pending_consolidation_applications()
        except Exception as exc:  # noqa: BLE001
            summary.errors += 1
            log.warning(
                "memory_consolidation_apply: scope failure agent=%s: %s",
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
                    _dispatch_consolidation_application(
                        conn=conn, agent_id=agent_id, row=row,
                        summary=summary, now=now,
                    )
            except Exception as exc:  # noqa: BLE001
                summary.errors += 1
                log.warning(
                    "memory_consolidation_apply: row %s упал: %s",
                    row.get("application_id"), exc,
                )

    return summary


def _dispatch_consolidation_application(
    *,
    conn: psycopg.Connection,
    agent_id: str,
    row: dict,
    summary: SweepSummary,
    now: _dt.datetime | None,
) -> None:
    queries = AgentScopedQueries(conn, agent_id)
    application_id = int(row["application_id"])
    task_status = row["task_status"]
    task_result = row["task_result"]

    if task_status == "failed":
        queries.mark_consolidation_skipped(application_id)
        summary.skipped += 1
        return

    if task_status != "done":
        summary.deferred += 1
        return

    if _is_skip_shape(task_result):
        queries.mark_consolidation_skipped(application_id)
        summary.skipped += 1
        return

    if not _is_merged_shape(task_result):
        queries.mark_consolidation_skipped(application_id)
        summary.skipped += 1
        return

    # Re-check is_active перед apply.
    if turn_state.is_active(agent_id, now=now):
        summary.deferred += 1
        return

    consolidated_text = task_result["consolidated_text"]
    embedding = [float(x) for x in task_result["consolidated_embedding"]]
    source_ids_raw = task_result["source_ids"]
    source_ids: list[uuid.UUID] = []
    for raw_id in source_ids_raw:
        try:
            source_ids.append(uuid.UUID(str(raw_id)))
        except ValueError:
            log.warning(
                "memory_consolidation_apply: invalid uuid in source_ids: %r",
                raw_id,
            )
            queries.mark_consolidation_skipped(application_id)
            summary.skipped += 1
            return
    source_kinds = list(task_result["source_kinds"])
    source_visibility = list(task_result["source_visibility"])

    kind = pick_consolidated_kind(source_kinds)
    visibility = pick_consolidated_visibility(source_visibility)

    new_memory_id = queries.insert_consolidated_memory(
        content=consolidated_text,
        embedding=embedding,
        kind=kind,
        visibility=visibility,
        source_ids=source_ids,
        application_id=application_id,
    )

    superseded_count = queries.mark_consolidation_sources_superseded(
        new_memory_id=new_memory_id, source_ids=source_ids,
    )
    if superseded_count == 0:
        # Все источники уже superseded'ы другим pipeline'ом. Новое
        # consolidated memory всё равно валидно (overarching meaning).
        log.info(
            "memory_consolidation_apply: 0 sources superseded for "
            "application %s — все источники уже под другим superseded'ом",
            application_id,
        )

    queries.mark_consolidation_applied(
        application_id=application_id, new_memory_id=new_memory_id,
    )
    summary.applied += 1
