"""Emotional baseline aggregator — медленный временной срез VAD.

EMA α=0.98 в минуту над time-weighted окном 60 мин. Запускается раз в
минуту periodic-task'ом
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
class _BaselineRecord:
    value: EmotionalBaseline
    updated_at: _dt.datetime


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
) -> tuple[float | None, float | None, float | None, int, float | None, int | None]:
    window_start = now - _dt.timedelta(minutes=window_minutes)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "WITH selected AS ("
            "  SELECT id, at, valence, arousal, dominance, confidence "
            "    FROM emotional_state "
            "   WHERE agent_id = %s AND at >= %s AND at <= %s "
            "  UNION ALL "
            "  SELECT id, at, valence, arousal, dominance, confidence FROM ("
            "    SELECT id, at, valence, arousal, dominance, confidence "
            "      FROM emotional_state "
            "     WHERE agent_id = %s AND at < %s "
            "     ORDER BY at DESC, id DESC LIMIT 1"
            "  ) prior"
            "), ordered AS ("
            "  SELECT *, greatest(at, %s) AS span_start, "
            "         least(lead(at, 1, %s) OVER (ORDER BY at, id), %s) AS span_end "
            "    FROM selected"
            "), spans AS ("
            "  SELECT *, extract(epoch FROM (span_end - span_start))::float8 AS seconds "
            "    FROM ordered WHERE span_end > span_start"
            ") "
            "SELECT (sum(valence::float8 * seconds) / nullif(sum(seconds), 0)) AS mv, "
            "       (sum(arousal::float8 * seconds) / nullif(sum(seconds), 0)) AS ma, "
            "       (sum(dominance::float8 * seconds) / nullif(sum(seconds), 0)) AS md, "
            "       count(*)::int AS n, "
            "       CASE WHEN count(confidence) > 0 THEN "
            "         sum(coalesce(confidence::float8, 0.0) * seconds) "
            "         / nullif(sum(seconds), 0) "
            "       ELSE NULL END AS mc, "
            "       (array_agg(id ORDER BY at DESC, id DESC))[1] AS source_state_id "
            "  FROM spans",
            (
                agent_id, window_start, now,
                agent_id, window_start,
                window_start, now, now,
            ),
        )
        row = cur.fetchone()
    if row is None:
        return None, None, None, 0, None, None
    return (
        row["mv"], row["ma"], row["md"], int(row["n"]), row["mc"],
        int(row["source_state_id"]) if row["source_state_id"] is not None else None,
    )


def _read_baseline(
    conn: psycopg.Connection, agent_id: str
) -> _BaselineRecord | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT valence, arousal, dominance, updated_at "
            "  FROM emotional_baseline WHERE agent_id = %s",
            (agent_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return _BaselineRecord(
        value=EmotionalBaseline(
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            dominance=float(row["dominance"]),
        ),
        updated_at=row["updated_at"],
    )


def _lock_agent_timeline(conn: psycopg.Connection, agent_id: str) -> None:
    """Serialize the baseline snapshot with state transitions for this agent."""
    if not agent_id.strip():
        raise ValueError("agent_id не должен быть пустым")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"styx:emotional_state:{agent_id}",),
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

    _lock_agent_timeline(conn, agent_id)
    current = _read_baseline(conn, agent_id)
    if current is not None:
        updated_at = current.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=_dt.timezone.utc)
        if now <= updated_at:
            return RecomputeResult(
                skipped=True,
                baseline=current.value,
                sample_size=0,
            )

    mv, ma, md, n, mean_confidence, source_state_id = _read_mean_instant(
        conn, agent_id, now, window_minutes
    )
    if n == 0 or mv is None or ma is None or md is None:
        return RecomputeResult(skipped=True, baseline=None, sample_size=0)

    base_v = current.value.valence if current is not None else 0.0
    base_a = current.value.arousal if current is not None else 0.0
    base_d = current.value.dominance if current is not None else 0.0

    # α задан на минуту. Повторный технический tick в тот же момент не
    # должен второй раз двигать baseline; пропущенные минуты, наоборот,
    # учитываются степенью α.
    if current is None:
        effective_alpha = alpha
    else:
        updated_at = current.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=_dt.timezone.utc)
        elapsed_minutes = max(0.0, (now - updated_at).total_seconds() / 60.0)
        effective_alpha = alpha ** elapsed_minutes

    next_v = effective_alpha * base_v + (1 - effective_alpha) * mv
    next_a = effective_alpha * base_a + (1 - effective_alpha) * ma
    next_d = effective_alpha * base_d + (1 - effective_alpha) * md

    # Legacy mood_active намеренно не меняется: причинная проекция не
    # создаёт отдельную mood-сущность. Сохраняем ABI и текущее значение.
    # ON CONFLICT DO UPDATE SET тоже mood_active не включает —
    # существующее значение сохраняется.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_baseline "
            "  (agent_id, valence, arousal, dominance, updated_at, "
            "   source_window_from, source_window_to, sample_size, confidence, "
            "   source_state_id, computation_version) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (agent_id) DO UPDATE "
            "  SET valence = EXCLUDED.valence, "
            "      arousal = EXCLUDED.arousal, "
            "      dominance = EXCLUDED.dominance, "
            "      updated_at = EXCLUDED.updated_at, "
            "      source_window_from = EXCLUDED.source_window_from, "
            "      source_window_to = EXCLUDED.source_window_to, "
            "      sample_size = EXCLUDED.sample_size, "
            "      confidence = EXCLUDED.confidence, "
            "      source_state_id = EXCLUDED.source_state_id, "
            "      computation_version = EXCLUDED.computation_version",
            (
                agent_id, next_v, next_a, next_d, now,
                now - _dt.timedelta(minutes=window_minutes), now, n,
                mean_confidence, source_state_id, "time-weighted-v1",
            ),
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
