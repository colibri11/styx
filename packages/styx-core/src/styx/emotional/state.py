"""Журнал ``emotional_state`` — append-only.

Прямой port из memorybox `emotional/state.ts`. Числа `INSTANT_DECAY_*`,
`EMOTIONAL_AXIS_*` оставлены буквально. RLS не используем — application-
level WHERE по agent_id (см. decisions § 5/§ 17.1).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# ── Constants (port memorybox state.ts:49-61) ─────────────────────────

EMOTIONAL_AXIS_MIN = -1.0
EMOTIONAL_AXIS_MAX = 1.0

INSTANT_DECAY_PER_MINUTE = 0.95
"""``v *= factor^minutes_elapsed``. Геометрическая прогрессия."""

INSTANT_DECAY_EPSILON = 0.005
"""Ниже этого порога (по любой оси) decay не пишется."""


# ── Types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmotionalVector:
    valence: float
    arousal: float
    dominance: float


NEUTRAL_VECTOR = EmotionalVector(0.0, 0.0, 0.0)


# ── Pure functions ────────────────────────────────────────────────────


def clamp_axis(value: float) -> float:
    if value < EMOTIONAL_AXIS_MIN:
        return EMOTIONAL_AXIS_MIN
    if value > EMOTIONAL_AXIS_MAX:
        return EMOTIONAL_AXIS_MAX
    return value


def clamp_vector(v: EmotionalVector) -> EmotionalVector:
    return EmotionalVector(
        valence=clamp_axis(v.valence),
        arousal=clamp_axis(v.arousal),
        dominance=clamp_axis(v.dominance),
    )


def max_abs(v: EmotionalVector) -> float:
    return max(abs(v.valence), abs(v.arousal), abs(v.dominance))


def decay_factor(minutes_elapsed: float) -> float:
    """``per_minute^minutes`` — геометрическая прогрессия."""
    return INSTANT_DECAY_PER_MINUTE ** minutes_elapsed


def apply_decay(vector: EmotionalVector, minutes_elapsed: float) -> EmotionalVector:
    f = decay_factor(minutes_elapsed)
    return EmotionalVector(
        valence=vector.valence * f,
        arousal=vector.arousal * f,
        dominance=vector.dominance * f,
    )


# ── DB-side ───────────────────────────────────────────────────────────


def read_last_state(
    conn: psycopg.Connection, agent_id: str
) -> tuple[EmotionalVector, _dt.datetime] | None:
    """Последняя точка журнала; ``None`` если истории нет."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT valence, arousal, dominance, at "
            "  FROM emotional_state "
            " WHERE agent_id = %s "
            " ORDER BY at DESC, id DESC LIMIT 1",
            (agent_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return (
        EmotionalVector(
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            dominance=float(row["dominance"]),
        ),
        row["at"],
    )


def append_emotional_state(
    conn: psycopg.Connection,
    agent_id: str,
    delta: EmotionalVector,
    *,
    source: str | None = None,
    metadata: dict | None = None,
) -> EmotionalVector:
    """Прибавить delta к последней точке (или к нейтрали при пустой истории),
    clamp в [-1, +1], INSERT новую точку. Возвращает получившееся состояние.

    Не делает commit.
    """
    last = read_last_state(conn, agent_id)
    base = last[0] if last is not None else NEUTRAL_VECTOR
    nxt = clamp_vector(
        EmotionalVector(
            valence=base.valence + delta.valence,
            arousal=base.arousal + delta.arousal,
            dominance=base.dominance + delta.dominance,
        )
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "  (agent_id, valence, arousal, dominance, source, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                agent_id,
                nxt.valence,
                nxt.arousal,
                nxt.dominance,
                source,
                Jsonb(metadata) if metadata is not None else None,
            ),
        )
    return nxt


@dataclass(frozen=True)
class ApplyDecayResult:
    decayed: bool
    point: EmotionalVector | None
    minutes_elapsed: float


def apply_instant_decay(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    now: _dt.datetime | None = None,
) -> ApplyDecayResult:
    """Один шаг decay для одного агента.

    Если последняя точка моложе минуты → no-op. Если в epsilon-окрестности
    нуля → no-op (журнал не раздуваем). Иначе INSERT decay-точки с
    ``source='decay'`` и ``metadata={"auto": true}``.

    Не делает commit.
    """
    if now is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)

    last = read_last_state(conn, agent_id)
    if last is None:
        return ApplyDecayResult(False, None, 0.0)
    vector, at = last

    # at может быть aware либо naive — приводим к aware UTC.
    if at.tzinfo is None:
        at = at.replace(tzinfo=_dt.timezone.utc)

    elapsed_seconds = (now - at).total_seconds()
    minutes = elapsed_seconds / 60.0
    if minutes < 1.0:
        return ApplyDecayResult(False, None, minutes)
    if max_abs(vector) < INSTANT_DECAY_EPSILON:
        return ApplyDecayResult(False, None, minutes)

    nxt = apply_decay(vector, minutes)
    # decay-точка сохраняется как абсолют, не как delta — поэтому
    # append'им через прямой INSERT, не через append_emotional_state.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "  (agent_id, at, valence, arousal, dominance, source, metadata) "
            "VALUES (%s, %s, %s, %s, %s, 'decay', %s)",
            (
                agent_id,
                now,
                nxt.valence,
                nxt.arousal,
                nxt.dominance,
                Jsonb({"auto": True}),
            ),
        )
    return ApplyDecayResult(True, nxt, minutes)


def list_active_agent_ids(conn: psycopg.Connection) -> list[str]:
    """``SELECT DISTINCT agent_id FROM memories``.

    Используется ``emotional_tick`` чтобы знать кому пересчитывать
    baseline. Worker'у не нужен agent-context, читает всех.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT agent_id FROM memories")
        return [r[0] for r in cur.fetchall()]
