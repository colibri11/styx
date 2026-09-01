"""Журнал ``emotional_state`` — append-only.

Прямой port из memorybox `emotional/state.ts`. Числа `INSTANT_DECAY_*`,
`EMOTIONAL_AXIS_*` оставлены буквально. RLS не используем — application-
level WHERE по agent_id (см. decisions § 5/§ 17.1).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any

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

ACTIVE_CAUSE_LEASE_MINUTES = 15
"""Без нового причинного свидетельства support не живёт бесконечно."""


# ── Types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmotionalVector:
    valence: float
    arousal: float
    dominance: float


NEUTRAL_VECTOR = EmotionalVector(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class EmotionalEventRecord:
    """Наблюдаемое событие, которое может повлиять на состояние.

    Это свидетельство с координатами и неопределённостью, не назначение
    агенту готовой эмоции. ``signal`` описывает распознанный входной сигнал;
    вычисленный state transition хранится отдельно в ``emotional_state``.
    """

    id: int
    agent_id: str
    occurred_at: _dt.datetime
    observed_at: _dt.datetime
    source_kind: str
    source_ref: str | None
    idempotency_key: str | None
    signal: EmotionalVector | None
    intensity: float | None
    confidence: float | None
    cause_summary: str | None
    cause_status: str
    cause_status_at: _dt.datetime | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EmotionalEventWriteResult:
    event: EmotionalEventRecord
    duplicate: bool


@dataclass(frozen=True)
class EmotionalStateRecord:
    """Полная append-only точка вычисленного состояния агента."""

    id: int
    agent_id: str
    at: _dt.datetime
    vector: EmotionalVector
    source: str | None
    metadata: dict[str, Any]
    parent_state_id: int | None
    event_id: int | None
    delta: EmotionalVector | None
    intensity: float | None
    confidence: float | None
    causal_context: tuple[dict[str, Any], ...]
    computation_version: str | None
    transition_confidence: float | None = None


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


def active_cause_support(
    causal_context: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> EmotionalVector:
    """Суммарный неизгасающий вклад причин, которые всё ещё действуют.

    Новые causal-turn records сохраняют уже взвешенную delta каждой причины.
    Legacy/неполные компоненты честно не дают support: из сырого label или
    reaction нельзя восстанавливать неизвестный reducer задним числом.
    """
    v = a = d = 0.0
    for item in causal_context:
        if item.get("cause_active") is not True:
            continue
        raw = item.get("weighted_delta")
        if not isinstance(raw, list) or len(raw) != 3:
            continue
        if any(
            isinstance(axis, bool)
            or not isinstance(axis, (int, float))
            or not math.isfinite(float(axis))
            for axis in raw
        ):
            continue
        v += float(raw[0])
        a += float(raw[1])
        d += float(raw[2])
    return clamp_vector(EmotionalVector(v, a, d))


def aggregate_state_confidence(
    previous_vector: EmotionalVector,
    previous_confidence: float | None,
    delta: EmotionalVector,
    transition_confidence: float | None,
) -> float | None:
    """Confidence всей проекции, отдельно от confidence одного перехода.

    Весом служит фактическая амплитуда уже удерживаемого состояния и нового
    изменения. Неизвестная уверенность всё равно занимает вес в знаменателе:
    иначе большой неопределённый компонент ложно сохранил бы старую высокую confidence.
    Если вся существенная масса неизвестна, результат остаётся ``None``.
    """
    previous_weight = max_abs(previous_vector)
    transition_weight = max_abs(delta)
    total_weight = previous_weight + transition_weight
    known_weight = 0.0
    numerator = 0.0
    if previous_confidence is not None and previous_weight > 0.0:
        numerator += previous_confidence * previous_weight
        known_weight += previous_weight
    if transition_confidence is not None and transition_weight > 0.0:
        numerator += transition_confidence * transition_weight
        known_weight += transition_weight
    if total_weight > 0.0:
        return numerator / total_weight if known_weight > 0.0 else None
    return previous_confidence


# ── DB-side ───────────────────────────────────────────────────────────


def _validate_unit_interval(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} должен быть finite number в [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} должен быть finite number в [0, 1]")
    return parsed


def _validate_vector(vector: EmotionalVector, name: str) -> EmotionalVector:
    values = (vector.valence, vector.arousal, vector.dominance)
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} должен содержать только finite values")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} должен содержать только finite values")
    if not all(EMOTIONAL_AXIS_MIN <= v <= EMOTIONAL_AXIS_MAX for v in values):
        raise ValueError(f"{name} должен быть в [-1, 1] по каждой оси")
    return vector


def _lock_agent_state(conn: psycopg.Connection, agent_id: str) -> None:
    """Межпроцессный transaction-scoped lock эмоциональной линии агента.

    Daemon hot-path и worker decay/batch живут в разных процессах, поэтому
    Python lock недостаточен. Advisory lock берётся до read-last и держится
    PostgreSQL до commit/rollback вызывающего кода.
    """
    if not agent_id.strip():
        raise ValueError("agent_id не должен быть пустым")
    lock_name = f"styx:emotional_state:{agent_id}"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (lock_name,),
        )


def _event_from_row(row: dict[str, Any]) -> EmotionalEventRecord:
    signal = None
    if row["valence"] is not None:
        signal = EmotionalVector(
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            dominance=float(row["dominance"]),
        )
    return EmotionalEventRecord(
        id=int(row["id"]),
        agent_id=str(row["agent_id"]),
        occurred_at=row["occurred_at"],
        observed_at=row["observed_at"],
        source_kind=str(row["source_kind"]),
        source_ref=row["source_ref"],
        idempotency_key=row["idempotency_key"],
        signal=signal,
        intensity=(
            float(row["intensity"]) if row["intensity"] is not None else None
        ),
        confidence=(
            float(row["confidence"]) if row["confidence"] is not None else None
        ),
        cause_summary=row["cause_summary"],
        cause_status=str(row["cause_status"]),
        cause_status_at=row["cause_status_at"],
        metadata=dict(row["metadata"] or {}),
    )


def _state_from_row(row: dict[str, Any]) -> EmotionalStateRecord:
    delta = None
    if row["delta_valence"] is not None:
        delta = EmotionalVector(
            valence=float(row["delta_valence"]),
            arousal=float(row["delta_arousal"]),
            dominance=float(row["delta_dominance"]),
        )
    raw_context = row["causal_context"] or []
    return EmotionalStateRecord(
        id=int(row["id"]),
        agent_id=str(row["agent_id"]),
        at=row["at"],
        vector=EmotionalVector(
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            dominance=float(row["dominance"]),
        ),
        source=row["source"],
        metadata=dict(row["metadata"] or {}),
        parent_state_id=(
            int(row["parent_state_id"])
            if row["parent_state_id"] is not None
            else None
        ),
        event_id=int(row["event_id"]) if row["event_id"] is not None else None,
        delta=delta,
        intensity=(
            float(row["intensity"]) if row["intensity"] is not None else None
        ),
        transition_confidence=(
            float(row["transition_confidence"])
            if row["transition_confidence"] is not None
            else None
        ),
        confidence=(
            float(row["confidence"]) if row["confidence"] is not None else None
        ),
        causal_context=tuple(dict(item) for item in raw_context),
        computation_version=row["computation_version"],
    )


def _select_event_by_idempotency_key(
    conn: psycopg.Connection, agent_id: str, idempotency_key: str
) -> EmotionalEventRecord | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, agent_id, occurred_at, observed_at, source_kind, "
            "       source_ref, idempotency_key, valence, arousal, dominance, "
            "       intensity, confidence, cause_summary, cause_status, "
            "       cause_status_at, metadata "
            "  FROM emotional_events "
            " WHERE agent_id = %s AND idempotency_key = %s",
            (agent_id, idempotency_key),
        )
        row = cur.fetchone()
    return None if row is None else _event_from_row(row)


def read_emotional_event(
    conn: psycopg.Connection, agent_id: str, event_id: int
) -> EmotionalEventRecord | None:
    """Read one evidence event under the explicit agent scope."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, agent_id, occurred_at, observed_at, source_kind, "
            "       source_ref, idempotency_key, valence, arousal, dominance, "
            "       intensity, confidence, cause_summary, cause_status, "
            "       cause_status_at, metadata "
            "  FROM emotional_events "
            " WHERE id = %s AND agent_id = %s",
            (event_id, agent_id),
        )
        row = cur.fetchone()
    return None if row is None else _event_from_row(row)


def read_emotional_event_by_idempotency_key(
    conn: psycopg.Connection, agent_id: str, idempotency_key: str
) -> EmotionalEventRecord | None:
    """Read a previously accepted event without invoking its evaluator again."""
    if not idempotency_key.strip():
        return None
    return _select_event_by_idempotency_key(conn, agent_id, idempotency_key)


def _append_cause_status(
    conn: psycopg.Connection,
    agent_id: str,
    cause_event_id: int,
    status: str,
    *,
    at: _dt.datetime | None = None,
    support: EmotionalVector | None = None,
    confidence: float | None = None,
    intensity: float | None = None,
    status_source_event_id: int | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    lease_expires_at = None
    if status == "active":
        lease_base = at or _dt.datetime.now(tz=_dt.timezone.utc)
        lease_expires_at = lease_base + _dt.timedelta(
            minutes=ACTIVE_CAUSE_LEASE_MINUTES
        )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_cause_status "
            "  (agent_id, cause_event_id, at, status, lease_expires_at, "
            "   support_valence, support_arousal, support_dominance, "
            "   confidence, intensity, status_source_event_id, context) "
            "VALUES (%s, %s, coalesce(%s, clock_timestamp()), %s, %s, %s, "
            "        %s, %s, %s, %s, %s, %s)",
            (
                agent_id,
                cause_event_id,
                at,
                status,
                lease_expires_at,
                support.valence if support is not None else None,
                support.arousal if support is not None else None,
                support.dominance if support is not None else None,
                confidence,
                intensity,
                status_source_event_id,
                Jsonb(context or {}),
            ),
        )


def _expire_cause_leases(
    conn: psycopg.Connection,
    agent_id: str,
    now: _dt.datetime,
) -> None:
    """Materialize lease expiry once; current lifecycle remains append-only."""
    with conn.cursor() as cur:
        cur.execute(
            "WITH latest AS ("
            "  SELECT DISTINCT ON (cause_event_id) cause_event_id, status, "
            "         lease_expires_at "
            "    FROM emotional_cause_status "
            "   WHERE agent_id = %s "
            "   ORDER BY cause_event_id, at DESC, id DESC"
            ") "
            "INSERT INTO emotional_cause_status "
            "  (agent_id, cause_event_id, at, status, context) "
            "SELECT %s, cause_event_id, %s, 'expired', "
            "       jsonb_build_object('reason', 'lease_expired') "
            "  FROM latest "
            " WHERE status = 'active' AND lease_expires_at <= %s",
            (agent_id, agent_id, now, now),
        )


def _latest_cause_statuses(
    conn: psycopg.Connection,
    agent_id: str,
) -> dict[int, dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT DISTINCT ON (cause_event_id) cause_event_id, status, "
            "       lease_expires_at, support_valence, support_arousal, "
            "       support_dominance, confidence, intensity, context "
            "  FROM emotional_cause_status "
            " WHERE agent_id = %s "
            " ORDER BY cause_event_id, at DESC, id DESC",
            (agent_id,),
        )
        return {int(row["cause_event_id"]): dict(row) for row in cur.fetchall()}


def _context_event_id(item: dict[str, Any]) -> int | None:
    raw = item.get("evidence_id", item.get("event_id"))
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return None
    return raw


def _context_status(item: dict[str, Any]) -> str | None:
    raw = item.get("status", item.get("cause_status"))
    if raw in {"active", "resolved", "superseded", "expired"}:
        return str(raw)
    if item.get("cause_active") is False:
        return "resolved"
    return "active" if item.get("cause_active") is True else None


def _context_support(item: dict[str, Any]) -> EmotionalVector | None:
    raw = item.get("weighted_delta")
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    if any(isinstance(value, bool) for value in raw):
        return None
    try:
        return _validate_vector(
            EmotionalVector(*(float(value) for value in raw)),
            "causal_context.weighted_delta",
        )
    except (TypeError, ValueError):
        return None


def _sync_cause_statuses(
    conn: psycopg.Connection,
    agent_id: str,
    causal_context: list[dict[str, Any]],
    *,
    status_source_event_id: int | None,
    now: _dt.datetime,
) -> None:
    _expire_cause_leases(conn, agent_id, now)
    latest = _latest_cause_statuses(conn, agent_id)
    for item in causal_context:
        cause_event_id = _context_event_id(item)
        status = _context_status(item)
        if cause_event_id is None or status is None:
            continue
        support = _context_support(item) if status == "active" else None
        current = latest.get(cause_event_id)
        current_support = None
        if current is not None and current["support_valence"] is not None:
            current_support = EmotionalVector(
                float(current["support_valence"]),
                float(current["support_arousal"]),
                float(current["support_dominance"]),
            )
        normalized_item = dict(item)
        # Projection-only lease coordinates must not look like a fresh
        # reaffirmation and silently extend the cause on every host turn.
        normalized_item.pop("lease_expires_at", None)
        current_context = dict(current.get("context") or {}) if current else {}
        current_context.pop("lease_expires_at", None)
        if current is not None and current["status"] == status:
            same_context = current_context == normalized_item
            if status != "active" or (
                current_support == support and same_context
            ):
                continue
        confidence = _validate_unit_interval(item.get("confidence"), "confidence")
        intensity = _validate_unit_interval(item.get("intensity"), "intensity")
        _append_cause_status(
            conn,
            agent_id,
            cause_event_id,
            status,
            at=now,
            support=support,
            confidence=confidence,
            intensity=intensity,
            status_source_event_id=status_source_event_id,
            context=normalized_item,
        )


def _active_cause_rows(
    conn: psycopg.Connection,
    agent_id: str,
    now: _dt.datetime,
) -> list[dict[str, Any]]:
    _expire_cause_leases(conn, agent_id, now)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "WITH latest AS ("
            "  SELECT DISTINCT ON (cause_event_id) * "
            "    FROM emotional_cause_status "
            "   WHERE agent_id = %s "
            "   ORDER BY cause_event_id, at DESC, id DESC"
            ") "
            "SELECT cause_event_id, lease_expires_at, support_valence, "
            "       support_arousal, support_dominance, confidence, intensity, "
            "       context "
            "  FROM latest "
            " WHERE status = 'active' AND lease_expires_at > %s "
            " ORDER BY cause_event_id",
            (agent_id, now),
        )
        return [dict(row) for row in cur.fetchall()]


def read_active_cause_lifecycle(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    now: _dt.datetime | None = None,
) -> dict[int, dict[str, Any]]:
    """Return the authoritative active, unexpired causes for one agent.

    JSON state projections are deliberately not authoritative for lifecycle
    mutations: they can be bounded, stale, or imported from a legacy row.
    Missing/invalid leases are never treated as active by the SQL predicate.
    """
    instant = now or _dt.datetime.now(tz=_dt.timezone.utc)
    _lock_agent_state(conn, agent_id)
    return {
        int(row["cause_event_id"]): row
        for row in _active_cause_rows(conn, agent_id, instant)
    }


def read_cause_lifecycle_statuses(
    conn: psycopg.Connection,
    agent_id: str,
    event_ids: set[int],
    *,
    now: _dt.datetime | None = None,
) -> dict[int, dict[str, Any]]:
    """Read a bounded current-status projection for recall provenance."""
    if not event_ids:
        return {}
    instant = now or _dt.datetime.now(tz=_dt.timezone.utc)
    _lock_agent_state(conn, agent_id)
    _expire_cause_leases(conn, agent_id, instant)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "WITH latest AS ("
            " SELECT DISTINCT ON (cause_event_id) cause_event_id,status,lease_expires_at "
            " FROM emotional_cause_status WHERE agent_id=%s AND cause_event_id = ANY(%s) "
            " ORDER BY cause_event_id,at DESC,id DESC) "
            "SELECT cause_event_id,status,lease_expires_at FROM latest",
            (agent_id, list(event_ids)),
        )
        return {
            int(row["cause_event_id"]): {
                "status": str(row["status"]),
                "lease_expires_at": row["lease_expires_at"],
                "active": row["status"] == "active"
                and row["lease_expires_at"] is not None
                and row["lease_expires_at"] > instant,
            }
            for row in cur.fetchall()
        }


def update_active_cause_lifecycle(
    conn: psycopg.Connection,
    agent_id: str,
    event_ids: tuple[int, ...] | list[int],
    *,
    status: str,
    status_source_event_id: int,
    at: _dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Resolve/supersede selected active causes or explicitly reaffirm them.

    Every id must name an active, unexpired cause of ``agent_id`` at ``at``.
    Reaffirmation copies the original support coordinates and only renews its
    lease; callers must not apply the support as a fresh affect delta.
    """
    if status not in {"active", "resolved", "superseded"}:
        raise ValueError(f"unsupported lifecycle status: {status!r}")
    ids = tuple(event_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in ids):
        raise ValueError("event_ids must contain positive integers")
    if len(set(ids)) != len(ids):
        raise ValueError("event_ids must be unique")
    instant = at or _dt.datetime.now(tz=_dt.timezone.utc)
    _lock_agent_state(conn, agent_id)
    active = read_active_cause_lifecycle(conn, agent_id, now=instant)
    if not set(ids) <= set(active):
        raise ValueError("event_ids do not identify active unexpired causes")
    selected: list[dict[str, Any]] = []
    for event_id in ids:
        row = active[event_id]
        support = None
        if status == "active" and row["support_valence"] is not None:
            support = EmotionalVector(
                float(row["support_valence"]),
                float(row["support_arousal"]),
                float(row["support_dominance"]),
            )
        context = dict(row.get("context") or {})
        context.pop("lease_expires_at", None)
        context["evidence_id"] = event_id
        context["status"] = status
        context["cause_active"] = status == "active"
        context["status_source_event_id"] = status_source_event_id
        _append_cause_status(
            conn,
            agent_id,
            event_id,
            status,
            at=instant,
            support=support,
            confidence=(
                float(row["confidence"])
                if row.get("confidence") is not None else None
            ),
            intensity=(
                float(row["intensity"])
                if row.get("intensity") is not None else None
            ),
            status_source_event_id=status_source_event_id,
            context=context,
        )
        selected.append(context)
    return selected


def revise_active_cause_lifecycle(
    conn: psycopg.Connection,
    agent_id: str,
    event_id: int,
    *,
    support: EmotionalVector,
    confidence: float,
    intensity: float,
    status_source_event_id: int,
    context_updates: dict[str, Any],
    at: _dt.datetime | None = None,
) -> tuple[dict[str, Any], EmotionalVector]:
    """Replace one active cause projection and return its previous support."""
    instant = at or _dt.datetime.now(tz=_dt.timezone.utc)
    _validate_vector(support, "support")
    _validate_unit_interval(confidence, "confidence")
    _validate_unit_interval(intensity, "intensity")
    _lock_agent_state(conn, agent_id)
    active = read_active_cause_lifecycle(conn, agent_id, now=instant)
    if event_id not in active:
        raise ValueError("event_id does not identify an active unexpired cause")
    row = active[event_id]
    old = EmotionalVector(
        float(row.get("support_valence") or 0.0),
        float(row.get("support_arousal") or 0.0),
        float(row.get("support_dominance") or 0.0),
    )
    context = dict(row.get("context") or {})
    context.update(context_updates)
    context.update({
        "evidence_id": event_id,
        "status": "active",
        "cause_active": True,
        "status_source_event_id": status_source_event_id,
        "weighted_delta": [support.valence, support.arousal, support.dominance],
        "confidence": confidence,
        "intensity": intensity,
    })
    context.pop("lease_expires_at", None)
    _append_cause_status(
        conn, agent_id, event_id, "active", at=instant, support=support,
        confidence=confidence, intensity=intensity,
        status_source_event_id=status_source_event_id, context=context,
    )
    return context, old


def _active_db_support(
    conn: psycopg.Connection,
    agent_id: str,
    now: _dt.datetime,
) -> tuple[EmotionalVector, list[dict[str, Any]], bool]:
    rows = _active_cause_rows(conn, agent_id, now)
    contexts: list[dict[str, Any]] = []
    support = NEUTRAL_VECTOR
    for row in rows:
        item = dict(row["context"] or {})
        item["evidence_id"] = int(row["cause_event_id"])
        item["status"] = "active"
        item["cause_active"] = True
        item["lease_expires_at"] = row["lease_expires_at"].isoformat()
        contexts.append(item)
        if row["support_valence"] is not None:
            support = EmotionalVector(
                support.valence + float(row["support_valence"]),
                support.arousal + float(row["support_arousal"]),
                support.dominance + float(row["support_dominance"]),
            )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM emotional_cause_status WHERE agent_id=%s)",
            (agent_id,),
        )
        has_lifecycle = bool(cur.fetchone()[0])
    return clamp_vector(support), contexts, has_lifecycle


def append_emotional_event(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    source_kind: str,
    occurred_at: _dt.datetime | None = None,
    source_ref: str | None = None,
    idempotency_key: str | None = None,
    signal: EmotionalVector | None = None,
    intensity: float | None = None,
    confidence: float | None = None,
    cause_summary: str | None = None,
    cause_status: str = "unknown",
    cause_status_at: _dt.datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> EmotionalEventWriteResult:
    """Append causal evidence, idempotently when a key is provided.

    Does not commit.  A duplicate key returns the original immutable event;
    the second payload never overwrites its evidence.
    """
    if not source_kind.strip():
        raise ValueError("source_kind не должен быть пустым")
    if len(source_kind) > 64:
        raise ValueError("source_kind длиннее 64 символов")
    if source_ref is not None and len(source_ref) > 512:
        raise ValueError("source_ref длиннее 512 символов")
    if idempotency_key is not None and not idempotency_key.strip():
        raise ValueError("idempotency_key не должен быть пустым")
    if idempotency_key is not None and len(idempotency_key) > 512:
        raise ValueError("idempotency_key длиннее 512 символов")
    if cause_summary is not None and len(cause_summary) > 1000:
        raise ValueError("cause_summary длиннее 1000 символов")
    if cause_status not in {"unknown", "active", "resolved", "superseded"}:
        raise ValueError(f"неизвестный cause_status: {cause_status!r}")
    if signal is not None:
        _validate_vector(signal, "signal")
    intensity = _validate_unit_interval(intensity, "intensity")
    confidence = _validate_unit_interval(confidence, "confidence")
    if occurred_at is None:
        occurred_at = _dt.datetime.now(tz=_dt.timezone.utc)
    if cause_status != "unknown" and cause_status_at is None:
        cause_status_at = _dt.datetime.now(tz=_dt.timezone.utc)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata должен быть object")
    meta = metadata or {}

    _lock_agent_state(conn, agent_id)
    if idempotency_key is not None:
        existing = _select_event_by_idempotency_key(
            conn, agent_id, idempotency_key
        )
        if existing is not None:
            return EmotionalEventWriteResult(event=existing, duplicate=True)

    params = (
        agent_id,
        occurred_at,
        source_kind,
        source_ref,
        idempotency_key,
        signal.valence if signal is not None else None,
        signal.arousal if signal is not None else None,
        signal.dominance if signal is not None else None,
        intensity,
        confidence,
        cause_summary,
        cause_status,
        cause_status_at,
        Jsonb(meta),
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO emotional_events "
            "  (agent_id, occurred_at, source_kind, source_ref, "
            "   idempotency_key, valence, arousal, dominance, intensity, "
            "   confidence, cause_summary, cause_status, cause_status_at, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (agent_id, idempotency_key) "
            "  WHERE idempotency_key IS NOT NULL DO NOTHING "
            "RETURNING id, agent_id, occurred_at, observed_at, source_kind, "
            "          source_ref, idempotency_key, valence, arousal, dominance, "
            "          intensity, confidence, cause_summary, cause_status, "
            "          cause_status_at, metadata",
            params,
        )
        row = cur.fetchone()
    if row is not None:
        event = _event_from_row(row)
        if cause_status != "unknown":
            normalized_status = (
                cause_status if cause_status != "resolved" else "resolved"
            )
            _append_cause_status(
                conn,
                agent_id,
                event.id,
                normalized_status,
                at=event.observed_at,
                confidence=confidence,
                intensity=intensity,
                context={
                    "evidence_id": event.id,
                    "source_ref": event.source_ref,
                    "cause": event.cause_summary,
                    "status": cause_status,
                    "cause_active": cause_status == "active",
                },
            )
        return EmotionalEventWriteResult(event=event, duplicate=False)
    if idempotency_key is None:
        raise RuntimeError("emotional_events INSERT не вернул строку")
    existing = _select_event_by_idempotency_key(conn, agent_id, idempotency_key)
    if existing is None:
        raise RuntimeError("idempotency conflict без видимой emotional_event")
    return EmotionalEventWriteResult(event=existing, duplicate=True)


def read_last_state_record(
    conn: psycopg.Connection, agent_id: str
) -> EmotionalStateRecord | None:
    """Последняя полная state-точка, включая transition и lineage."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, agent_id, valence, arousal, dominance, at, source, "
            "       metadata, parent_state_id, event_id, delta_valence, "
            "       delta_arousal, delta_dominance, intensity, "
            "       transition_confidence, confidence, "
            "       causal_context, computation_version "
            "  FROM emotional_state "
            " WHERE agent_id = %s "
            " ORDER BY at DESC, id DESC LIMIT 1",
            (agent_id,),
        )
        row = cur.fetchone()
    return None if row is None else _state_from_row(row)


def read_last_state(
    conn: psycopg.Connection, agent_id: str
) -> tuple[EmotionalVector, _dt.datetime] | None:
    """Последняя точка журнала; ``None`` если истории нет."""
    record = read_last_state_record(conn, agent_id)
    if record is None:
        return None
    return record.vector, record.at


def append_emotional_transition(
    conn: psycopg.Connection,
    agent_id: str,
    delta: EmotionalVector,
    *,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
    event_id: int | None = None,
    intensity: float | None = None,
    confidence: float | None = None,
    causal_context: (
        list[dict[str, Any]] | tuple[dict[str, Any], ...] | None
    ) = None,
    computation_version: str | None = None,
) -> EmotionalStateRecord:
    """Apply one causal transition and return its rich append-only record.

    The per-agent PostgreSQL lock closes the daemon/worker lost-update race.
    ``causal_context=None`` inherits the previous snapshot; an explicit empty
    list deliberately clears it.
    """
    _validate_vector(delta, "delta")
    intensity = _validate_unit_interval(intensity, "intensity")
    transition_confidence = _validate_unit_interval(confidence, "confidence")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata должен быть object")
    if causal_context is not None and not all(
        isinstance(item, dict) for item in causal_context
    ):
        raise ValueError("causal_context должен быть списком объектов")

    _lock_agent_state(conn, agent_id)
    last = read_last_state_record(conn, agent_id)
    base = last.vector if last is not None else NEUTRAL_VECTOR
    nxt = clamp_vector(
        EmotionalVector(
            valence=base.valence + delta.valence,
            arousal=base.arousal + delta.arousal,
            dominance=base.dominance + delta.dominance,
        )
    )
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    supplied_context = (
        [dict(item) for item in causal_context]
        if causal_context is not None
        else list(last.causal_context) if last is not None else []
    )
    if causal_context is not None:
        _sync_cause_statuses(
            conn,
            agent_id,
            supplied_context,
            status_source_event_id=event_id,
            now=now,
        )
    _support, active_context, has_lifecycle = _active_db_support(
        conn, agent_id, now
    )
    if has_lifecycle:
        # Active causes are never subject to the host's bounded context tail.
        # Only inactive explanatory history is bounded; support comes from the
        # normalized lifecycle journal, not from this JSON projection.
        active_ids = {
            _context_event_id(item) for item in active_context
        }
        untracked_active = [
            item for item in supplied_context
            if _context_status(item) == "active"
            and _context_event_id(item) is None
        ]
        inactive_history = [
            item for item in supplied_context
            if _context_status(item) != "active"
            and _context_event_id(item) not in active_ids
        ]
        context = [*active_context, *untracked_active, *inactive_history[-8:]]
    else:
        context = supplied_context
    state_confidence = aggregate_state_confidence(
        base,
        last.confidence if last is not None else None,
        EmotionalVector(
            nxt.valence - base.valence,
            nxt.arousal - base.arousal,
            nxt.dominance - base.dominance,
        ),
        transition_confidence,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "  (agent_id, valence, arousal, dominance, source, metadata, "
            "   parent_state_id, event_id, delta_valence, delta_arousal, "
            "   delta_dominance, intensity, transition_confidence, confidence, "
            "   causal_context, "
            "   computation_version) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "        %s, %s, %s, %s, %s) "
            "RETURNING id, agent_id, valence, arousal, dominance, at, source, "
            "          metadata, parent_state_id, event_id, delta_valence, "
            "          delta_arousal, delta_dominance, intensity, "
            "          transition_confidence, confidence, "
            "          causal_context, computation_version",
            (
                agent_id,
                nxt.valence,
                nxt.arousal,
                nxt.dominance,
                source,
                Jsonb(metadata or {}),
                last.id if last is not None else None,
                event_id,
                delta.valence,
                delta.arousal,
                delta.dominance,
                intensity,
                transition_confidence,
                state_confidence,
                Jsonb(context),
                computation_version,
            ),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("emotional_state INSERT не вернул строку")
    return _state_from_row(row)


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
    return append_emotional_transition(
        conn,
        agent_id,
        delta,
        source=source,
        metadata=metadata,
    ).vector


@dataclass(frozen=True)
class ApplyDecayResult:
    decayed: bool
    point: EmotionalVector | None
    minutes_elapsed: float


def _protected_support_components(
    conn: psycopg.Connection,
    agent_id: str,
    start: _dt.datetime,
    end: _dt.datetime,
) -> list[tuple[EmotionalVector, _dt.datetime]]:
    """Support present at ``start`` and the instant its protection ends.

    A missed worker interval can straddle a cause lease expiry. Treating the
    cause as either protected or residual for the whole interval biases the
    result. This helper reconstructs the boundary from the append-only
    lifecycle; an explicit reaffirmation may extend it, while resolution or
    supersession shortens it.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "WITH initial AS ("
            " SELECT DISTINCT ON (cause_event_id) cause_event_id, status, "
            "        lease_expires_at, support_valence, support_arousal, "
            "        support_dominance "
            "   FROM emotional_cause_status "
            "  WHERE agent_id=%s AND at <= %s "
            "  ORDER BY cause_event_id, at DESC, id DESC"
            ") "
            "SELECT * FROM initial WHERE status='active' "
            " AND lease_expires_at IS NOT NULL AND lease_expires_at > %s "
            " AND support_valence IS NOT NULL",
            (agent_id, start, start),
        )
        initial = [dict(row) for row in cur.fetchall()]
        if not initial:
            return []
        ids = [int(row["cause_event_id"]) for row in initial]
        cur.execute(
            "SELECT cause_event_id, at, status, lease_expires_at "
            "  FROM emotional_cause_status "
            " WHERE agent_id=%s AND cause_event_id = ANY(%s::bigint[]) "
            "   AND at > %s AND at <= %s "
            " ORDER BY at, id",
            (agent_id, ids, start, end),
        )
        later: dict[int, list[dict[str, Any]]] = {event_id: [] for event_id in ids}
        for row in cur.fetchall():
            later[int(row["cause_event_id"])].append(dict(row))

    components: list[tuple[EmotionalVector, _dt.datetime]] = []
    for row in initial:
        event_id = int(row["cause_event_id"])
        protected_until = row["lease_expires_at"]
        for change in later[event_id]:
            if change["at"] > protected_until:
                break
            if change["status"] == "active":
                renewed_until = change["lease_expires_at"]
                if renewed_until is not None:
                    protected_until = max(protected_until, renewed_until)
            else:
                protected_until = min(protected_until, change["at"])
                break
        components.append((
            EmotionalVector(
                float(row["support_valence"]),
                float(row["support_arousal"]),
                float(row["support_dominance"]),
            ),
            min(protected_until, end),
        ))
    return components


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

    _lock_agent_state(conn, agent_id)
    last = read_last_state_record(conn, agent_id)
    if last is None:
        return ApplyDecayResult(False, None, 0.0)
    vector, at = last.vector, last.at

    # at может быть aware либо naive — приводим к aware UTC.
    if at.tzinfo is None:
        at = at.replace(tzinfo=_dt.timezone.utc)

    elapsed_seconds = (now - at).total_seconds()
    minutes = elapsed_seconds / 60.0
    if minutes < 1.0:
        return ApplyDecayResult(False, None, minutes)
    if max_abs(vector) < INSTANT_DECAY_EPSILON:
        return ApplyDecayResult(False, None, minutes)

    protected_components = _protected_support_components(
        conn, agent_id, at, now
    )
    support, active_context, has_lifecycle = _active_db_support(
        conn, agent_id, now
    )
    if not has_lifecycle:
        support = active_cause_support(last.causal_context)
        decay_context = list(last.causal_context)
    else:
        active_ids = {_context_event_id(item) for item in active_context}
        inactive_history = [
            dict(item) for item in last.causal_context
            if _context_status(item) != "active"
            and _context_event_id(item) not in active_ids
        ]
        decay_context = [*active_context, *inactive_history]
    if has_lifecycle:
        start_support = EmotionalVector(
            sum(item.valence for item, _ in protected_components),
            sum(item.arousal for item, _ in protected_components),
            sum(item.dominance for item, _ in protected_components),
        )
        residual = EmotionalVector(
            vector.valence - start_support.valence,
            vector.arousal - start_support.arousal,
            vector.dominance - start_support.dominance,
        )
        decayed_residual = apply_decay(residual, minutes)
        retained = NEUTRAL_VECTOR
        for component, protected_until in protected_components:
            unprotected_minutes = max(
                0.0, (now - protected_until).total_seconds() / 60.0
            )
            decayed_component = apply_decay(component, unprotected_minutes)
            retained = EmotionalVector(
                retained.valence + decayed_component.valence,
                retained.arousal + decayed_component.arousal,
                retained.dominance + decayed_component.dominance,
            )
        nxt = clamp_vector(EmotionalVector(
            decayed_residual.valence + retained.valence,
            decayed_residual.arousal + retained.arousal,
            decayed_residual.dominance + retained.dominance,
        ))
    else:
        residual = EmotionalVector(
            vector.valence - support.valence,
            vector.arousal - support.arousal,
            vector.dominance - support.dominance,
        )
        decayed_residual = apply_decay(residual, minutes)
        nxt = clamp_vector(EmotionalVector(
            support.valence + decayed_residual.valence,
            support.arousal + decayed_residual.arousal,
            support.dominance + decayed_residual.dominance,
        ))
    applied_delta = EmotionalVector(
        nxt.valence - vector.valence,
        nxt.arousal - vector.arousal,
        nxt.dominance - vector.dominance,
    )
    if max_abs(applied_delta) < INSTANT_DECAY_EPSILON:
        return ApplyDecayResult(False, None, minutes)

    active_intensities = [
        float(item["intensity"])
        for item in decay_context
        if item.get("cause_active") is True
        and isinstance(item.get("intensity"), (int, float))
        and not isinstance(item.get("intensity"), bool)
    ]
    decayed_intensity = (
        last.intensity * decay_factor(minutes)
        if last.intensity is not None else None
    )
    next_intensity = max(
        [value for value in [decayed_intensity, *active_intensities] if value is not None],
        default=None,
    )
    # decay-точка сохраняется как абсолют, не как delta — поэтому
    # append'им через прямой INSERT, не через append_emotional_state.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "  (agent_id, at, valence, arousal, dominance, source, metadata, "
            "   parent_state_id, delta_valence, delta_arousal, delta_dominance, "
            "   intensity, transition_confidence, confidence, causal_context, "
            "   computation_version) "
            "VALUES (%s, %s, %s, %s, %s, 'decay', %s, %s, %s, %s, %s, "
            "        %s, NULL, %s, %s, %s)",
            (
                agent_id,
                now,
                nxt.valence,
                nxt.arousal,
                nxt.dominance,
                Jsonb({"auto": True}),
                last.id,
                applied_delta.valence,
                applied_delta.arousal,
                applied_delta.dominance,
                next_intensity,
                last.confidence,
                Jsonb(decay_context),
                last.computation_version,
            ),
        )
    return ApplyDecayResult(True, nxt, minutes)


def list_active_agent_ids(conn: psycopg.Connection) -> list[str]:
    """Agents with dialogue or affect state requiring periodic upkeep.

    Affect observation may commit before dialogue ingestion fails.  Such an
    agent still needs lease expiry/decay, so enumeration cannot rely on
    ``memories`` alone.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT agent_id FROM memories "
            "UNION SELECT agent_id FROM emotional_state "
            "UNION SELECT agent_id FROM emotional_cause_status"
        )
        return [r[0] for r in cur.fetchall()]
