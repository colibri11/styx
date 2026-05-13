"""Lifecycle sweep — fresh → settled → dormant.

Прямой port из memorybox `consolidation/tasks/lifecycle-refresh.ts`.
Числа `DEFAULT_AUTOTUNE`, `DEFAULT_TRANSITION_BATCH_SIZE` оставлены
буквально.

Алгоритм одной итерации:

1. Compute targets — сколько памятей должно быть в каждом bucket'е
   (fresh / settled / dormant).
2. Compute thresholds — PERCENTILE_CONT по age и idle.
3. Clamp + EMA smoothing — против previous из `consolidation_state`.
4. Apply transitions — batch UPDATE с FOR UPDATE SKIP LOCKED.
5. Persist smoothed thresholds в `consolidation_state`.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from styx.workers.sweep.state import get_state, set_state


# ── Constants (port memorybox lifecycle-refresh.ts:14-32) ──────────────

DEFAULT_AUTOTUNE: dict[str, Any] = {
    "mode": "budget",
    "context_budget_tokens": 20000,
    "recall_budget_items": 10,
    "fresh_multiplier": 10,
    "settled_multiplier": 50,
    "max_fresh_share": 0.5,
    "min_fresh_size": 20,
    "min_population_for_tuning": 100,
    "fresh_share": 0.20,
    "dormant_share": 0.25,
    "smoothing": 0.3,
    "bounds": {
        "fresh_to_settled_age_days": {"min": 1, "max": 60},
        "settled_to_dormant_idle_days": {"min": 7, "max": 730},
    },
}

DEFAULT_TRANSITION_BATCH_SIZE = 500
TRANSITION_SAFETY_CAP = 100_000  # макс. итераций цикла, как в memorybox

LIFECYCLE_STATE_KEY = "lifecycle_thresholds"


# ── Pure functions ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LifecycleTargets:
    fresh: int
    settled: int
    dormant: int


def resolve_autotune_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Заполнить переданный конфиг дефолтами memorybox'а."""
    d = DEFAULT_AUTOTUNE
    if cfg is None:
        return _deep_copy_dict(d)
    bounds_cfg = cfg.get("bounds") or {}
    bf = bounds_cfg.get("fresh_to_settled_age_days") or {}
    bi = bounds_cfg.get("settled_to_dormant_idle_days") or {}
    return {
        "mode": cfg.get("mode", d["mode"]),
        "context_budget_tokens": cfg.get(
            "context_budget_tokens", d["context_budget_tokens"]
        ),
        "recall_budget_items": cfg.get(
            "recall_budget_items", d["recall_budget_items"]
        ),
        "fresh_multiplier": cfg.get("fresh_multiplier", d["fresh_multiplier"]),
        "settled_multiplier": cfg.get(
            "settled_multiplier", d["settled_multiplier"]
        ),
        "max_fresh_share": cfg.get("max_fresh_share", d["max_fresh_share"]),
        "min_fresh_size": cfg.get("min_fresh_size", d["min_fresh_size"]),
        "min_population_for_tuning": cfg.get(
            "min_population_for_tuning", d["min_population_for_tuning"]
        ),
        "fresh_share": cfg.get("fresh_share", d["fresh_share"]),
        "dormant_share": cfg.get("dormant_share", d["dormant_share"]),
        "smoothing": cfg.get("smoothing", d["smoothing"]),
        "bounds": {
            "fresh_to_settled_age_days": {
                "min": bf.get(
                    "min", d["bounds"]["fresh_to_settled_age_days"]["min"]
                ),
                "max": bf.get(
                    "max", d["bounds"]["fresh_to_settled_age_days"]["max"]
                ),
            },
            "settled_to_dormant_idle_days": {
                "min": bi.get(
                    "min",
                    d["bounds"]["settled_to_dormant_idle_days"]["min"],
                ),
                "max": bi.get(
                    "max",
                    d["bounds"]["settled_to_dormant_idle_days"]["max"],
                ),
            },
        },
    }


def _deep_copy_dict(src: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in src.items():
        out[k] = _deep_copy_dict(v) if isinstance(v, dict) else v
    return out


def clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        return lo
    return max(lo, min(hi, value))


def compute_targets(total: int, cfg: dict[str, Any]) -> LifecycleTargets:
    """Вернуть целевые размеры fresh/settled/dormant.

    Прямой port `computeTargets` (memorybox lifecycle-refresh.ts:108).
    """
    if total <= 0:
        return LifecycleTargets(0, 0, 0)
    if cfg["mode"] == "budget":
        fresh_target = min(
            cfg["recall_budget_items"] * cfg["fresh_multiplier"],
            int(total * cfg["max_fresh_share"]),
        )
        fresh_target = max(fresh_target, cfg["min_fresh_size"])
        fresh_target = min(fresh_target, total)
        settled_target = min(
            fresh_target * cfg["settled_multiplier"], total - fresh_target
        )
        dormant_target = total - fresh_target - settled_target
    else:  # fixed_share
        fresh_target = int(total * cfg["fresh_share"])
        dormant_target = int(total * cfg["dormant_share"])
        settled_target = total - fresh_target - dormant_target

    settled_target = max(settled_target, 0)
    dormant_target = max(dormant_target, 0)
    return LifecycleTargets(
        fresh=fresh_target, settled=settled_target, dormant=dormant_target
    )


def apply_smoothing(
    current: float, previous: float | None, alpha: float
) -> float:
    """``next = previous + α × (current - previous)`` (или current если
    previous=None)."""
    if previous is None:
        return current
    return previous + alpha * (current - previous)


# ── DB-side functions ─────────────────────────────────────────────────


def compute_distribution(
    conn: psycopg.Connection, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Один SELECT: total + age/idle quantile'ы.

    Quantile'ы вычисляются если total ≥ min_population_for_tuning,
    иначе возвращаем None (cold-start пойдёт по bounds.min).
    """
    targets_total_query = (
        "SELECT count(*)::int AS total FROM memories WHERE superseded_by IS NULL"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(targets_total_query)
        row = cur.fetchone()
    total = int(row["total"]) if row else 0

    if total < cfg["min_population_for_tuning"]:
        return {"total": total, "fresh_quantile": None, "idle_quantile": None}

    targets = compute_targets(total, cfg)
    if total <= 0:
        return {"total": total, "fresh_quantile": None, "idle_quantile": None}
    # PERCENTILE_CONT для age (fresh→settled) и idle (settled→dormant)
    fresh_p = max(0.0, min(1.0, 1.0 - targets.fresh / total))
    dormant_p = max(0.0, min(1.0, targets.dormant / total))
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT "
            "  percentile_cont(%(fp)s) WITHIN GROUP ("
            "    ORDER BY EXTRACT(epoch FROM now() - created_at) / 86400.0"
            "  )::float8 AS fresh_q, "
            "  percentile_cont(%(dp)s) WITHIN GROUP ("
            "    ORDER BY EXTRACT(epoch FROM now() - "
            "      COALESCE(last_accessed_at, created_at)) / 86400.0"
            "  )::float8 AS idle_q "
            "FROM memories WHERE superseded_by IS NULL",
            {"fp": fresh_p, "dp": dormant_p},
        )
        qrow = cur.fetchone()
    return {
        "total": total,
        "fresh_quantile": float(qrow["fresh_q"]) if qrow["fresh_q"] is not None else None,
        "idle_quantile": float(qrow["idle_q"]) if qrow["idle_q"] is not None else None,
    }


def apply_fresh_to_settled(
    conn: psycopg.Connection, age_days: float, batch_size: int
) -> int:
    """UPDATE batch'ами фрешные старее age_days → settled."""
    return _apply_lifecycle_transition(
        conn,
        from_lifecycle="fresh",
        to_lifecycle="settled",
        where_clause="now() - created_at > make_interval(secs => %s)",
        threshold_value=age_days * 86400.0,
        batch_size=batch_size,
    )


def apply_settled_to_dormant(
    conn: psycopg.Connection, idle_days: float, batch_size: int
) -> int:
    """UPDATE batch'ами settled с idle ≥ idle_days → dormant."""
    return _apply_lifecycle_transition(
        conn,
        from_lifecycle="settled",
        to_lifecycle="dormant",
        where_clause=(
            "now() - COALESCE(last_accessed_at, created_at) > "
            "make_interval(secs => %s)"
        ),
        threshold_value=idle_days * 86400.0,
        batch_size=batch_size,
    )


def _apply_lifecycle_transition(
    conn: psycopg.Connection,
    *,
    from_lifecycle: str,
    to_lifecycle: str,
    where_clause: str,
    threshold_value: float,
    batch_size: int,
) -> int:
    """Цикл UPDATE'ов с FOR UPDATE SKIP LOCKED, safety-cap иначе бесконечный."""
    total = 0
    iterations = 0
    while iterations < TRANSITION_SAFETY_CAP:
        iterations += 1
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE memories "
                f"   SET lifecycle = %s "
                f" WHERE id IN ( "
                f"   SELECT id FROM memories "
                f"    WHERE lifecycle = %s "
                f"      AND superseded_by IS NULL "
                f"      AND {where_clause} "
                f"    ORDER BY id "
                f"    FOR UPDATE SKIP LOCKED "
                f"    LIMIT %s "
                f" )",
                (to_lifecycle, from_lifecycle, threshold_value, batch_size),
            )
            n = cur.rowcount or 0
        conn.commit()
        total += n
        if n < batch_size:
            return total
    raise RuntimeError(
        f"lifecycle transition {from_lifecycle}→{to_lifecycle} превысил "
        f"safety cap {TRANSITION_SAFETY_CAP}"
    )


# ── Top-level lifecycle_refresh ────────────────────────────────────────


def lifecycle_refresh(
    conn: psycopg.Connection,
    cfg: dict[str, Any] | None = None,
    *,
    batch_size: int = DEFAULT_TRANSITION_BATCH_SIZE,
) -> dict[str, Any]:
    """Один прогон lifecycle sweep'а. Возвращает summary."""
    resolved = resolve_autotune_config(cfg)
    dist = compute_distribution(conn, resolved)
    total = dist["total"]
    fresh_q = dist["fresh_quantile"]
    idle_q = dist["idle_quantile"]

    bounds = resolved["bounds"]
    fresh_min = float(bounds["fresh_to_settled_age_days"]["min"])
    fresh_max = float(bounds["fresh_to_settled_age_days"]["max"])
    idle_min = float(bounds["settled_to_dormant_idle_days"]["min"])
    idle_max = float(bounds["settled_to_dormant_idle_days"]["max"])

    if fresh_q is None:
        # cold-start: данных мало или их нет — берём bounds.min
        fresh_age_clamped = fresh_min
        idle_clamped = idle_min
    else:
        fresh_age_clamped = clamp(fresh_q, fresh_min, fresh_max)
        idle_clamped = clamp(
            idle_q if idle_q is not None else idle_min, idle_min, idle_max
        )

    prev = get_state(conn, LIFECYCLE_STATE_KEY)
    prev_fresh = (
        float(prev["fresh_to_settled_age_days"])
        if prev and "fresh_to_settled_age_days" in prev
        else None
    )
    prev_idle = (
        float(prev["settled_to_dormant_idle_days"])
        if prev and "settled_to_dormant_idle_days" in prev
        else None
    )
    alpha = float(resolved["smoothing"])
    fresh_smoothed = apply_smoothing(fresh_age_clamped, prev_fresh, alpha)
    idle_smoothed = apply_smoothing(idle_clamped, prev_idle, alpha)

    # Persist пороги перед apply (чтобы после краша они сохранились
    # даже если apply упал).
    set_state(
        conn,
        LIFECYCLE_STATE_KEY,
        {
            "fresh_to_settled_age_days": fresh_smoothed,
            "settled_to_dormant_idle_days": idle_smoothed,
            "computed_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "total_population": total,
        },
    )
    conn.commit()

    fresh_to_settled_count = apply_fresh_to_settled(
        conn, fresh_smoothed, batch_size
    )
    settled_to_dormant_count = apply_settled_to_dormant(
        conn, idle_smoothed, batch_size
    )
    transitions_total = fresh_to_settled_count + settled_to_dormant_count

    return {
        "total": total,
        "fresh_to_settled_age_days": fresh_smoothed,
        "settled_to_dormant_idle_days": idle_smoothed,
        "fresh_to_settled_count": fresh_to_settled_count,
        "settled_to_dormant_count": settled_to_dormant_count,
        "transitions_total": transitions_total,
    }
