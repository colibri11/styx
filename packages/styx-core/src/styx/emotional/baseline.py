"""Emotional baseline aggregator — медленный временной срез VAD.

Прямой port из memorybox `emotional/baseline.ts`. EMA α=0.98 над окном
60 мин. Запускается раз в минуту periodic-task'ом
``emotional_tick`` в worker-runtime.

Memorybox использует RLS — Styx нет (decisions § 5/§ 17.1), читаем/пишем
напрямую с явным WHERE по agent_id.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

log = logging.getLogger(__name__)


# ── Constants (port memorybox baseline.ts:21-24) ──────────────────────

BASELINE_EMA_ALPHA = 0.98
"""``next = α × current + (1 - α) × mean(instant)`` — медленный baseline."""

BASELINE_WINDOW_MINUTES = 60


# ── Types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmotionalBaseline:
    valence: float
    arousal: float
    dominance: float


@dataclass(frozen=True)
class RecomputeResult:
    skipped: bool
    baseline: EmotionalBaseline | None
    sample_size: int


# ── DB-side ───────────────────────────────────────────────────────────


def _read_mean_instant(
    conn: psycopg.Connection,
    agent_id: str,
    now: _dt.datetime,
    window_minutes: int,
) -> tuple[float | None, float | None, float | None, int]:
    window_start = now - _dt.timedelta(minutes=window_minutes)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT AVG(valence)::float8 AS mv, "
            "       AVG(arousal)::float8 AS ma, "
            "       AVG(dominance)::float8 AS md, "
            "       count(*)::int AS n "
            "  FROM emotional_state "
            " WHERE agent_id = %s "
            "   AND at >= %s AND at <= %s",
            (agent_id, window_start, now),
        )
        row = cur.fetchone()
    if row is None:
        return None, None, None, 0
    return row["mv"], row["ma"], row["md"], int(row["n"])


def _read_baseline(
    conn: psycopg.Connection, agent_id: str
) -> EmotionalBaseline | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT valence, arousal, dominance "
            "  FROM emotional_baseline WHERE agent_id = %s",
            (agent_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return EmotionalBaseline(
        valence=float(row["valence"]),
        arousal=float(row["arousal"]),
        dominance=float(row["dominance"]),
    )


def recompute_baseline(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    now: _dt.datetime | None = None,
    alpha: float = BASELINE_EMA_ALPHA,
    window_minutes: int = BASELINE_WINDOW_MINUTES,
) -> RecomputeResult:
    """Один EMA-шаг для одного агента.

    Пусто в окне → skip (не пишем нейтраль; см. memorybox baseline.ts:111).

    Не делает commit.
    """
    if now is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)

    mv, ma, md, n = _read_mean_instant(conn, agent_id, now, window_minutes)
    if n == 0 or mv is None or ma is None or md is None:
        return RecomputeResult(skipped=True, baseline=None, sample_size=0)

    current = _read_baseline(conn, agent_id)
    base_v = current.valence if current is not None else 0.0
    base_a = current.arousal if current is not None else 0.0
    base_d = current.dominance if current is not None else 0.0

    next_v = alpha * base_v + (1 - alpha) * mv
    next_a = alpha * base_a + (1 - alpha) * ma
    next_d = alpha * base_d + (1 - alpha) * md

    # mood_active намеренно не в INSERT-списке: владельцем колонки
    # будет волна mood-engine (memorybox 15). DEFAULT FALSE из миграции.
    # ON CONFLICT DO UPDATE SET тоже mood_active не включает —
    # существующее значение сохраняется.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_baseline "
            "  (agent_id, valence, arousal, dominance, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (agent_id) DO UPDATE "
            "  SET valence = EXCLUDED.valence, "
            "      arousal = EXCLUDED.arousal, "
            "      dominance = EXCLUDED.dominance, "
            "      updated_at = EXCLUDED.updated_at",
            (agent_id, next_v, next_a, next_d, now),
        )

    return RecomputeResult(
        skipped=False,
        baseline=EmotionalBaseline(valence=next_v, arousal=next_a, dominance=next_d),
        sample_size=n,
    )


def read_baseline_for_scoring(
    conn: psycopg.Connection, agent_id: str | None
) -> EmotionalBaseline | None:
    """Узкий SELECT для recall scoring'а. ``None`` на любой ошибке —
    резонанс не критичен, не валим recall.
    """
    if not agent_id:
        return None
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT valence, arousal, dominance "
                "  FROM emotional_baseline WHERE agent_id = %s",
                (agent_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        v = float(row["valence"])
        a = float(row["arousal"])
        d = float(row["dominance"])
        if not (math.isfinite(v) and math.isfinite(a) and math.isfinite(d)):
            return None
        return EmotionalBaseline(valence=v, arousal=a, dominance=d)
    except Exception as exc:  # noqa: BLE001 — fail-open
        log.warning("read_baseline_for_scoring failed: %s", exc)
        return None
