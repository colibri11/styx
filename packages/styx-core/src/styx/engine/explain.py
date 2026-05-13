"""Pure-Python helpers для explain endpoints (волна 25).

Port из ``openclaw-memorybox/src/tools/explain.ts`` — три функции
(``buildFactorsBlock``, decay projection math, preview/hash truncation).

Без зависимостей на БД — всё на готовых row'ах из ``AgentScopedQueries``.
SQL-выражения для скоринга строит ``storage/scoring.build_factor_exprs``;
здесь только ассемблируем JSON-блоки из row + factor metadata.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..storage.importance import DEFAULT_LAMBDA_BY_KIND, MemoryKind
from ..storage.scoring import DecayConfig, FactorExprs


def truncate_preview(s: str | None, max_len: int = 200) -> str:
    """Усекает строку до ``max_len`` chars + ``…`` хвостик.

    Port ``truncate`` (memorybox explain.ts:40). ``None``/пустая → "".
    """
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…"


def hex_query_hash_short(q: bytes | memoryview | None) -> str:
    """``0x`` + первые 16 hex chars sha256-хеша.

    Port ``"0x" + (...).toString("hex").slice(0, 16)`` (explain.ts:466).
    None → пустая строка.
    """
    if q is None:
        return ""
    if isinstance(q, memoryview):
        q = bytes(q)
    return "0x" + q.hex()[:16]


def _recency_rule_for_age(age_days: float) -> str:
    """Описание правила recency_boost для конкретного age_days.

    Port ``recencyRuleForAge`` (explain.ts:84). Совпадает с CASE
    в build_factor_exprs.
    """
    if age_days < 1:
        return "< 1 day"
    if age_days < 7:
        return "< 7 days"
    return ">= 7 days"


def _grace_multiplier(
    *, decay_enabled: bool, importance_final: float | None
) -> tuple[float, str]:
    """grace_multiplier + reason для decay-блока.

    Воспроизводит правило, которое build_factor_exprs кодирует в SQL
    ``CASE WHEN importance_final IS NULL THEN 0.3 ELSE GREATEST(0.3,
    1 - 0.7 * importance_final) END`` (scoring.py:165-170).
    """
    if not decay_enabled:
        return 0.0, "decay disabled"
    if importance_final is None:
        return 0.3, "importance_final IS NULL"
    raw = max(0.3, 1.0 - 0.7 * float(importance_final))
    if raw == 0.3:
        return 0.3, "floor (0.3)"
    return raw, f"importance_final={float(importance_final)}"


def _f(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def build_factors_block(
    row: Mapping[str, Any],
    factor_meta: FactorExprs,
    decay_config: DecayConfig | None,
) -> dict[str, Any]:
    """Собирает FactorsBlock из row + factor metadata.

    Port ``buildFactorsBlock`` (memorybox explain.ts:98-201). row должен
    содержать все factor-колонки которые ``factor_select_columns``
    запросил в SELECT.

    base_match.hybrid_weights — реальные веса из FactorExprs (либо
    {vector=1, bm25=0} для pure vector). usage_factor.formula зависит
    от того есть ли personalisation (p75 > 0).

    decay.formula зависит от decay_enabled.
    """
    kind: MemoryKind = row.get("kind", "episode")  # type: ignore[assignment]
    age_days = _f(row.get("age_days_sql"))
    importance_final_raw = row.get("importance_final")
    importance_final = (
        None if importance_final_raw is None else float(importance_final_raw)
    )
    importance_provisional = _f(row.get("importance_provisional"), 0.5)
    importance_effective = _f(
        row.get("importance_effective"), importance_provisional
    )
    effective_lambda = _f(row.get("effective_lambda"))
    lambda_base = _f(row.get("lambda_base"))

    grace_multiplier, grace_reason = _grace_multiplier(
        decay_enabled=factor_meta.decay_enabled,
        importance_final=importance_final,
    )

    base_match_mode = factor_meta.base_match_mode
    bm25_value = row.get("bm25_rank")

    if factor_meta.base_match_mode == "hybrid":
        hybrid_weights = {
            "vector": float(factor_meta.vector_weight),
            "bm25": float(factor_meta.bm25_weight),
        }
    else:
        hybrid_weights = {"vector": 1.0, "bm25": 0.0}

    if factor_meta.usage_norm_p75 > 0:
        usage_formula = "1 + 0.12 * LEAST(1, usage_count_30d / p75_agent)"
    else:
        usage_formula = (
            "1.0 (no personalisation — no used_in_output signal "
            "for this agent yet)"
        )

    if factor_meta.decay_enabled:
        decay_formula = "exp(-effective_lambda * age_days)"
    else:
        decay_formula = "1.0 (decay disabled)"

    overrides = (decay_config.lambdas if decay_config is not None else None) or {}
    if kind in overrides:
        lambda_source = f"kind={kind} (config override)"
    else:
        lambda_source = f"kind={kind}"

    return {
        "base_match": {
            "value": _f(row.get("base_match")),
            "mode": base_match_mode,
            "vector_sim": _f(row.get("vector_sim")),
            "bm25": None if bm25_value is None else float(bm25_value),
            "hybrid_weights": hybrid_weights,
        },
        "relevance": {
            "value": _f(row.get("relevance_factor"), 1.0),
            "description": "growing from default via Hebbian reinforcement on access",
        },
        "recency_boost": {
            "value": _f(row.get("recency_factor"), 1.0),
            "rule": _recency_rule_for_age(age_days),
            "age_days": age_days,
        },
        "frequency_boost": {
            "value": _f(row.get("frequency_factor"), 1.0),
            "formula": "1 + 0.3 * ln(access_count + 1)",
            "access_count": int(_f(row.get("access_count"))),
        },
        "lifecycle_factor": {
            "value": _f(row.get("lifecycle_factor"), 1.0),
            "state": row.get("lifecycle") or "fresh",
        },
        "feedback_factor": {
            "value": _f(row.get("feedback_factor"), 1.0),
            "formula": "1 + 0.05 * usefulness",
            "usefulness": _f(row.get("usefulness")),
        },
        "importance_factor": {
            "value": _f(row.get("importance_factor"), 0.7),
            "formula": "0.4 + 0.6 * effective",
            "provisional": importance_provisional,
            "final": importance_final,
            "effective": importance_effective,
            "source": "final" if importance_final is not None else "provisional",
            "llm_task_status": row.get("llm_task_status"),
            "llm_task_created_at": row.get("llm_task_created_at"),
        },
        "diversity_bonus": {
            "value": _f(row.get("diversity_factor"), 1.0),
            "formula": "1 + 0.2 * ln(1 + unique_query_count)",
            "unique_query_count": int(_f(row.get("unique_query_count"))),
        },
        "usage_factor": {
            "value": _f(row.get("usage_factor"), 1.0),
            "formula": usage_formula,
            "usage_count_30d": int(_f(row.get("usage_count_30d"))),
            "p75_agent": float(factor_meta.usage_norm_p75),
            "weight": 0.12,
        },
        "decay_factor": {
            "value": _f(row.get("decay_factor"), 1.0),
            "formula": decay_formula,
            "age_days": age_days,
            "lambda_base": lambda_base,
            "lambda_source": lambda_source,
            "grace_multiplier": grace_multiplier,
            "grace_reason": grace_reason,
            "effective_lambda": effective_lambda,
        },
    }


def lifetime_decay_projections(
    *,
    kind: str,
    age_days: float,
    importance_final: float | None,
    relevance: float,
    decay_config: DecayConfig | None,
    prune_min_relevance: float | None,
) -> dict[str, Any]:
    """Pure-Python decay-блок для lifetime-режима.

    Port `explainLifetime` decay-секции (explain.ts:413-451). Считает
    эффективную lambda, current/projected decay, и
    ``estimated_days_to_prune_threshold`` через закрытое выражение
    ``d = ln(relevance/threshold) / lambda - age_days``.

    Возвращает dict, готовый положить в response.decay.
    """
    decay_enabled = decay_config is None or decay_config.enabled

    overrides = (decay_config.lambdas if decay_config is not None else None) or {}
    lambda_base_raw = overrides.get(kind)  # type: ignore[arg-type]
    if lambda_base_raw is None:
        lambda_base_raw = DEFAULT_LAMBDA_BY_KIND.get(kind, 0.01)  # type: ignore[arg-type]
    lambda_base = float(lambda_base_raw)

    if not decay_enabled:
        grace_multiplier = 0.0
        grace_period_active = False
    elif importance_final is None:
        grace_multiplier = 0.3
        grace_period_active = True
    else:
        grace_multiplier = max(0.3, 1.0 - 0.7 * float(importance_final))
        grace_period_active = False

    effective_lambda = lambda_base * grace_multiplier if decay_enabled else 0.0
    if decay_enabled:
        current_decay = math.exp(-effective_lambda * age_days)
        projected_30 = math.exp(-effective_lambda * (age_days + 30))
        projected_365 = math.exp(-effective_lambda * (age_days + 365))
    else:
        current_decay = 1.0
        projected_30 = 1.0
        projected_365 = 1.0

    days_to_prune: float | None = None
    if (
        decay_enabled
        and prune_min_relevance is not None
        and effective_lambda > 0
    ):
        threshold = float(prune_min_relevance)
        if relevance * current_decay > threshold and threshold > 0:
            d = math.log(relevance / threshold) / effective_lambda - age_days
            days_to_prune = d if d > 0 else 0.0

    return {
        "lambda_base": lambda_base,
        "effective_lambda": effective_lambda,
        "grace_period_active": grace_period_active,
        "current_decay_factor": current_decay,
        "projected_decay_in_30d": projected_30,
        "projected_decay_in_365d": projected_365,
        "estimated_days_to_prune_threshold": days_to_prune,
    }


_LIFECYCLE_MULTIPLIER: dict[str, float] = {
    "fresh": 1.0,
    "settled": 0.85,
    "dormant": 0.3,
}


def lifecycle_multiplier(state: str | None) -> float:
    """Multiplier для lifecycle-блока в lifetime-mode.

    Port ``lifecycleMultiplier`` (explain.ts:498-500). Совпадает с
    CASE в build_factor_exprs.
    """
    if state is None:
        return 1.0
    return _LIFECYCLE_MULTIPLIER.get(state, 1.0)


__all__ = [
    "build_factors_block",
    "hex_query_hash_short",
    "lifecycle_multiplier",
    "lifetime_decay_projections",
    "truncate_preview",
]
