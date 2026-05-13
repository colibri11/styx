"""Temporal isolation of cognitive cycles — port memorybox 5d.

Tracks per-agent turn state so recall'ы в течение одного когнитивного
цикла видят стабильный snapshot памяти. Без этого фоновые workers
(batch consolidation, importance, etc.) могут изменить память между
шагами reasoning'а агента, что ломает когнитивную когерентность.

Контракт:
- `observe(agent_id)` — sticky внутри активного turn'а: первый вызов
  открывает turn (cycle_start = now), последующие в течение TTL
  возвращают тот же cycle_start. После TTL inactivity или явного
  close() — открывается новый turn.
- `close(agent_id)` — explicit marker конца turn'а. Зовётся из
  sync_turn после успешной записи user/assistant пары и embed'а.
- TTL 60s — safety net, если close() не вызвался (sync_turn упал).

Snapshot применяется в `recall_full`/`search_similar` как WHERE-фильтр:
`created_at <= cycle_start OR (kind_src IN ('subjective',
'subjective_tail') AND agent_id = current)`. Subjective-исключение
даёт «я положил, я помню» — sync_turn в этом же turn'е виден сразу.

Append всегда разрешён (INSERT не ломает rasoning). Update структурных
полей — только во сне (вне активного turn'а); write-gate в Styx
сейчас не реализован — нет потребителей (волна 14 пишет append-only
batch-memories, structural updates отложены до волны reinterpret).

Один Hermes-процесс == один агент == один TurnState (decisions § 5).
Module-global, по аналогии с salient_bridge / focus_tracker /
pre_llm_inject.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecallSnapshot:
    """Снимок состояния turn'а для прокидывания в recall path."""

    cycle_start: _dt.datetime
    agent_id: str


@dataclass
class TurnState:
    cycle_start: _dt.datetime
    last_activity_at: _dt.datetime


_STATES: dict[str, TurnState] = {}
_TTL_S: float = 60.0


def configure(*, ttl_s: float = 60.0) -> None:
    """Сконфигурировать TTL. Reset state'ов."""
    global _TTL_S, _STATES
    if ttl_s <= 0:
        raise ValueError(f"ttl_s должен быть > 0, получено {ttl_s}")
    _TTL_S = ttl_s
    _STATES = {}


def observe(agent_id: str, *, now: _dt.datetime | None = None) -> RecallSnapshot:
    """Зарегистрировать активность recall path'а. Возвращает snapshot.

    Открывает новый turn если предыдущий закрылся по TTL или его не
    было. Sticky: внутри активного turn'а возвращает тот же cycle_start.
    """
    t = now if now is not None else _now()
    state = _STATES.get(agent_id)
    if state is None:
        state = TurnState(cycle_start=t, last_activity_at=t)
        _STATES[agent_id] = state
        return RecallSnapshot(cycle_start=t, agent_id=agent_id)

    if (t - state.last_activity_at).total_seconds() > _TTL_S:
        # Previous turn timed out → открываем новый.
        state.cycle_start = t
    state.last_activity_at = t
    return RecallSnapshot(cycle_start=state.cycle_start, agent_id=agent_id)


def close(agent_id: str) -> None:
    """Явное закрытие turn'а. Вызывается из sync_turn после успешной
    записи user/assistant пары и embed-after-commit."""
    _STATES.pop(agent_id, None)


def is_active(
    agent_id: str, *, now: _dt.datetime | None = None
) -> bool:
    """True если у агента есть активный turn (внутри TTL)."""
    state = _STATES.get(agent_id)
    if state is None:
        return False
    t = now if now is not None else _now()
    return (t - state.last_activity_at).total_seconds() <= _TTL_S


def peek(agent_id: str) -> TurnState | None:
    """Read-only snapshot для тестов и /healthz."""
    state = _STATES.get(agent_id)
    if state is None:
        return None
    return TurnState(
        cycle_start=state.cycle_start,
        last_activity_at=state.last_activity_at,
    )


def reset(agent_id: str | None = None) -> None:
    """Сброс. С ``agent_id`` — только этот агент; без — все.

    Вызывается из ``shutdown()`` (per-agent) и тестов (per-agent или
    глобально).
    """
    global _STATES
    if agent_id is None:
        _STATES = {}
    else:
        _STATES.pop(agent_id, None)


def reset_all() -> None:
    """Алиас на ``reset(None)`` для daemon shutdown / tests."""
    reset(None)


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)
