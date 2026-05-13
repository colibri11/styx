"""Batch consolidation scheduler — periodic-task в styx-worker.

Каждые ``tick_interval_s`` секунд (default 60) проверяет всех
subject-агентов на условия триггера batch consolidation:
- ≥ TRIGGER_MESSAGES (20) новых user/assistant memories с предыдущего
  batch'а;
- И ≥ TRIGGER_INTERVAL_S (1200, 20 мин) с предыдущего batch'а
  (либо первый batch).

При выполнении обоих — INSERT в ``llm_tasks`` task'у с payload
{agent_id, window_from, window_to, with_overlap} и UPDATE
``consolidation_state[batch_consolidation:<agent_id>].last_batch_at``,
атомарно в одной транзакции.

``with_overlap=False`` если последняя реплика была > PERIOD_GAP_S
(6h) назад — новая «сессия», без overlap'а на старую.

Прямой port memorybox ``supervisor/batch-consolidation-scheduler.ts``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from styx.storage.queries import (
    AgentScopedQueries,
    enqueue_llm_task,
    get_batch_state,
    list_subject_agents,
    set_batch_state,
)
from styx.workers.handlers.dialogue_batch_consolidation import (
    DIALOGUE_BATCH_TASK_TYPE,
)

log = logging.getLogger(__name__)


# ── Constants (port memorybox batch-consolidation-scheduler.ts) ───────

TRIGGER_MESSAGES = 20
TRIGGER_INTERVAL_S = 1200  # 20 мин
PERIOD_GAP_S = 6 * 3600  # 6 ч


@dataclass(frozen=True)
class BatchSchedulerConfig:
    enabled: bool = True
    trigger_messages: int = TRIGGER_MESSAGES
    trigger_interval_s: int = TRIGGER_INTERVAL_S
    period_gap_s: int = PERIOD_GAP_S


@dataclass(frozen=True)
class TriggerCheck:
    new_msg_count: int
    latest_msg_at: _dt.datetime | None
    seconds_since_last_batch: int | None


# ── Pure functions ─────────────────────────────────────────────────────


def trigger_satisfied(
    check: TriggerCheck, *, config: BatchSchedulerConfig
) -> bool:
    """Оба условия выполнены: ≥ N новых сообщений И ≥ interval с
    последнего batch'а. Первый прогон (last_batch_at IS NULL) —
    достаточно одного condition'а по сообщениям."""
    if check.new_msg_count < config.trigger_messages:
        return False
    if check.seconds_since_last_batch is None:
        return True  # первый batch — нет проверки времени
    return check.seconds_since_last_batch >= config.trigger_interval_s


def period_closed(
    latest_msg_at: _dt.datetime | None,
    at: _dt.datetime,
    *,
    config: BatchSchedulerConfig,
) -> bool:
    """Прошло > PERIOD_GAP_S (6h) с последней реплики — это новая сессия."""
    if latest_msg_at is None:
        return True  # нет prior activity → fresh
    if latest_msg_at.tzinfo is None:
        latest_msg_at = latest_msg_at.replace(tzinfo=_dt.timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=_dt.timezone.utc)
    gap_s = (at - latest_msg_at).total_seconds()
    return gap_s > config.period_gap_s


# ── Per-agent core ─────────────────────────────────────────────────────


def run_trigger_check(
    conn: psycopg.Connection,
    agent_id: str,
    state: dict | None,
) -> TriggerCheck:
    """Спросить БД: сколько новых реплик, latest_msg_at, сколько прошло
    с последнего batch'а."""
    queries = AgentScopedQueries(conn, agent_id)
    prev_msg_at_iso = None if state is None else state.get(
        "last_message_created_at"
    )
    prev_msg_at: _dt.datetime | None = None
    if prev_msg_at_iso:
        try:
            prev_msg_at = _dt.datetime.fromisoformat(prev_msg_at_iso)
        except (ValueError, TypeError):
            prev_msg_at = None

    new_count = queries.count_dialogue_messages_since(since=prev_msg_at)
    latest_at = queries.latest_dialogue_at()

    seconds_since: int | None = None
    if state is not None:
        last_batch_iso = state.get("last_batch_at")
        if last_batch_iso:
            try:
                last_batch_at = _dt.datetime.fromisoformat(last_batch_iso)
                if last_batch_at.tzinfo is None:
                    last_batch_at = last_batch_at.replace(
                        tzinfo=_dt.timezone.utc
                    )
                now = _dt.datetime.now(tz=_dt.timezone.utc)
                seconds_since = int((now - last_batch_at).total_seconds())
            except (ValueError, TypeError):
                seconds_since = None

    return TriggerCheck(
        new_msg_count=new_count,
        latest_msg_at=latest_at,
        seconds_since_last_batch=seconds_since,
    )


def maybe_schedule(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    at: _dt.datetime,
    config: BatchSchedulerConfig,
) -> bool:
    """Атомарно проверить триггер и поставить task. True если поставлен.

    INSERT task + UPDATE state в одной транзакции — защита от двойной
    постановки при гонке. Caller (scheduler tick) делает commit; на
    rollback'е state не продвигается, следующий tick попробует снова.
    """
    state = get_batch_state(conn, agent_id)
    check = run_trigger_check(conn, agent_id, state)
    if not trigger_satisfied(check, config=config):
        return False

    window_from_iso = (
        state.get("last_message_created_at")
        if state is not None else None
    )
    window_to_iso = at.isoformat()
    with_overlap = not period_closed(
        check.latest_msg_at, at, config=config,
    )

    payload = {
        "agent_id": agent_id,
        "window_from": window_from_iso,
        "window_to": window_to_iso,
        "with_overlap": with_overlap,
    }

    enqueue_llm_task(
        conn, task_type=DIALOGUE_BATCH_TASK_TYPE, payload=payload,
    )
    new_state = {
        "last_batch_at": at.isoformat(),
        "last_message_created_at": (
            state.get("last_message_created_at") if state else None
        ),
        "last_message_id": (
            state.get("last_message_id") if state else None
        ),
        "last_window_end_at": (
            state.get("last_window_end_at") if state else None
        ),
    }
    set_batch_state(conn, agent_id, new_state)
    return True


def schedule_batch_tick(
    conn: psycopg.Connection,
    *,
    at: _dt.datetime | None = None,
    config: BatchSchedulerConfig,
) -> int:
    """Один tick scheduler'а. Возвращает кол-во поставленных task'ов.

    Не делает rollback — caller (periodic-task wrapper) контролирует
    транзакцию. На исключении в одном из агентов — log warning, идём
    к следующему. Безопасно: каждый maybe_schedule в своём
    savepoint'е если caller так настроил, иначе runtime откатит весь
    tick и повторит.
    """
    if not config.enabled:
        return 0

    moment = at if at is not None else _dt.datetime.now(tz=_dt.timezone.utc)
    agents = list_subject_agents(conn)
    scheduled = 0
    for agent_id in agents:
        try:
            if maybe_schedule(conn, agent_id, at=moment, config=config):
                scheduled += 1
                conn.commit()
            else:
                conn.commit()  # commit'им SELECT'ы (read-only ок)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "batch consolidation scheduler: agent=%s упал: %s",
                agent_id, exc,
            )
            try:
                conn.rollback()
            except Exception:
                pass
    return scheduled
