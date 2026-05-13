"""Composite scoring SQL fragments — port из memorybox buildFactorExprs.

Прямой port из ``openclaw-memorybox/src/tools/memory.ts:454+``.
Числа, формулы и эмиттируемый SQL буквальны (decisions.md § 17.5).

Композитный score для memories:

    base_match × relevance × recency × frequency × lifecycle ×
    feedback × importance × diversity × decay × usage × emotional_resonance

В волне 7 Styx нейтральные multiplier'ы дают:

    score ≈ vec_sim × recency × (0.4 + 0.6 × importance_provisional) × decay

— остальные факторы становятся нетривиальными по мере подключения
workers (волны 7a-d).

EmotionalResonanceWeight = 0.1 (constant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .importance import (
    DEFAULT_LAMBDA_BY_KIND,
    MemoryKind,
    build_lambda_case_expr,
)
from .search_weights import SearchConfig, WeightPair, compute_weights

EMOTIONAL_RESONANCE_WEIGHT = 0.1

BaseMatchMode = Literal["vector", "hybrid"]


@dataclass(frozen=True)
class EmotionalBaseline:
    valence: float
    arousal: float
    dominance: float


@dataclass(frozen=True)
class DecayConfig:
    enabled: bool = True
    lambdas: Mapping[MemoryKind, float] | None = None


@dataclass(frozen=True)
class FactorExprs:
    vector_sim_expr: str
    bm25_expr: str | None
    base_match_expr: str
    base_match_mode: BaseMatchMode
    vector_weight: float
    bm25_weight: float
    relevance_ref_expr: str
    age_days_expr: str
    recency_expr: str
    frequency_expr: str
    lifecycle_expr: str
    feedback_expr: str
    importance_expr: str
    importance_effective_expr: str
    diversity_expr: str
    decay_expr: str
    decay_enabled: bool
    lambda_base_expr: str
    effective_lambda_expr: str
    usage_count_expr: str
    usage_factor_expr: str
    usage_norm_p75: float
    usage_lateral_from: str
    emotional_resonance_expr: str
    score_expr: str
    table_alias: str | None


@dataclass(frozen=True)
class BuildFactorExprsOptions:
    """Опции, эквивалентные ``BuildFactorExprsOptions`` из TS."""

    text_query_param_index: int | None
    search_config: SearchConfig | None = None
    decay_config: DecayConfig | None = None
    table_alias: str | None = None
    omit_base_match: bool = False
    usage_norm_p75: float | None = None
    emotional_baseline: EmotionalBaseline | None = None


def build_factor_exprs(
    inp: Mapping[str, object],
    options: BuildFactorExprsOptions,
) -> FactorExprs:
    """Собрать SQL-фрагменты для composite scoring.

    Прямой port ``buildFactorExprs`` (memory.ts:454).

    ``inp`` отражает TS-сигнатуру ``{ text_query?: string; query?: string }``
    — фактически интересует только наличие ``text_query`` для активации
    hybrid-ветки. ``options.text_query_param_index`` — индекс bind-param'а
    в финальной SQL-strings'е (например 2 если query embedding уже на $1).
    """
    weights: WeightPair
    if inp.get("text_query"):
        weights = compute_weights(str(inp["text_query"]), options.search_config)
    else:
        weights = WeightPair(0.0, 0.0)

    p = f"{options.table_alias}." if options.table_alias else ""

    vector_sim_expr = f"(1 - ({p}embedding <=> $1))"
    bm25_expr: str | None = None
    base_match_expr: str
    base_match_mode: BaseMatchMode
    if options.text_query_param_index is not None:
        bm25_expr = (
            f"ts_rank({p}content_tsv, "
            f"plainto_tsquery('simple', ${options.text_query_param_index}), 32)"
        )
        base_match_expr = (
            f"({weights.vector_weight} * {vector_sim_expr} + "
            f"{weights.bm25_weight} * {bm25_expr})"
        )
        base_match_mode = "hybrid"
    else:
        base_match_expr = vector_sim_expr
        base_match_mode = "vector"

    recency_expr = (
        f"CASE WHEN now() - {p}created_at < interval '1 day' THEN 1.3\n"
        f"       WHEN now() - {p}created_at < interval '7 days' THEN 1.1\n"
        f"       ELSE 1.0 END"
    )
    frequency_expr = f"(1 + 0.3 * ln({p}access_count + 1))"
    lifecycle_expr = (
        f"CASE {p}lifecycle\n"
        f"    WHEN 'fresh' THEN 1.0\n"
        f"    WHEN 'settled' THEN 0.85\n"
        f"    WHEN 'dormant' THEN 0.3\n"
        f"    ELSE 1.0 END"
    )
    feedback_expr = f"(1 + 0.05 * {p}usefulness)"

    importance_effective_expr = (
        f"COALESCE({p}importance_final, {p}importance_provisional, 0.5)"
    )
    importance_expr = f"(0.4 + 0.6 * {importance_effective_expr})"

    diversity_expr = f"(1 + 0.2 * ln(1 + {p}unique_query_count))"

    decay_enabled = options.decay_config is None or options.decay_config.enabled
    raw_lambda_case = build_lambda_case_expr(
        options.decay_config.lambdas if options.decay_config else None,
        table_alias=options.table_alias,
    )
    lambda_base_expr = raw_lambda_case
    effective_lambda_expr: str
    decay_expr: str
    if decay_enabled:
        effective_lambda_expr = (
            f"(\n"
            f"      ({lambda_base_expr}) *\n"
            f"      CASE WHEN {p}importance_final IS NULL\n"
            f"           THEN 0.3\n"
            f"           ELSE GREATEST(0.3, 1 - 0.7 * {p}importance_final)\n"
            f"      END\n"
            f"    )"
        )
        decay_expr = (
            f"exp(-({effective_lambda_expr}) * "
            f"EXTRACT(EPOCH FROM (now() - {p}created_at)) / 86400.0)"
        )
    else:
        effective_lambda_expr = "0"
        decay_expr = "1.0"

    outer_ref = f"{options.table_alias}.id" if options.table_alias else "memories.id"
    usage_lateral_from = (
        f"LEFT JOIN LATERAL (\n"
        f"       SELECT count(*)::double precision AS uc\n"
        f"         FROM recall_events re\n"
        f"        WHERE re.memory_id = {outer_ref}\n"
        f"          AND re.used_in_output = true\n"
        f"          AND re.matched_at > now() - interval '30 days'\n"
        f"     ) _mb_usage ON true"
    )
    usage_count_expr = "_mb_usage.uc"
    usage_norm_p75 = (
        options.usage_norm_p75
        if options.usage_norm_p75 is not None and options.usage_norm_p75 > 0
        else 0.0
    )
    if usage_norm_p75 > 0:
        usage_factor_expr = (
            f"(1 + 0.12 * LEAST(1.0::double precision, "
            f"{usage_count_expr} / {usage_norm_p75}::double precision))"
        )
    else:
        usage_factor_expr = "1.0::double precision"

    emotional_resonance_expr = _build_emotional_resonance_expr(
        options.emotional_baseline, p
    )

    composed_rest_expr = (
        f"{p}relevance\n"
        f"* {recency_expr}\n"
        f"* {frequency_expr}\n"
        f"* {lifecycle_expr}\n"
        f"* {feedback_expr}\n"
        f"* {importance_expr}\n"
        f"* {diversity_expr}\n"
        f"* {decay_expr}\n"
        f"* {usage_factor_expr}\n"
        f"* {emotional_resonance_expr}"
    )
    if options.omit_base_match:
        score_expr = composed_rest_expr
    else:
        score_expr = f"({base_match_expr})\n* {composed_rest_expr}"

    relevance_ref_expr = f"{p}relevance"
    age_days_expr = f"EXTRACT(EPOCH FROM (now() - {p}created_at)) / 86400.0"

    return FactorExprs(
        vector_sim_expr=vector_sim_expr,
        bm25_expr=bm25_expr,
        base_match_expr=base_match_expr,
        base_match_mode=base_match_mode,
        vector_weight=weights.vector_weight,
        bm25_weight=weights.bm25_weight,
        relevance_ref_expr=relevance_ref_expr,
        age_days_expr=age_days_expr,
        recency_expr=recency_expr,
        frequency_expr=frequency_expr,
        lifecycle_expr=lifecycle_expr,
        feedback_expr=feedback_expr,
        importance_expr=importance_expr,
        importance_effective_expr=importance_effective_expr,
        diversity_expr=diversity_expr,
        decay_expr=decay_expr,
        decay_enabled=decay_enabled,
        lambda_base_expr=lambda_base_expr,
        effective_lambda_expr=effective_lambda_expr,
        usage_count_expr=usage_count_expr,
        usage_factor_expr=usage_factor_expr,
        usage_norm_p75=usage_norm_p75,
        usage_lateral_from=usage_lateral_from,
        emotional_resonance_expr=emotional_resonance_expr,
        score_expr=score_expr,
        table_alias=options.table_alias,
    )


def _build_emotional_resonance_expr(
    baseline: EmotionalBaseline | None, column_prefix: str
) -> str:
    """SQL для emotional_resonance факторa.

    Port ``buildEmotionalResonanceExpr`` (memory.ts:629). Без baseline
    или с не-finite значениями — нейтральный ``1.0``.

    Формула:
        factor = 1 + W * (1 - clamp(dist / sqrt(12), 0, 1))
    где dist = Euclidean между memory.emotional_context_* и baseline,
    sqrt(12) — диаметр куба [-1, 1]^3.
    """
    if baseline is None:
        return "1.0::double precision"
    if not all(_is_finite(v) for v in (baseline.valence, baseline.arousal, baseline.dominance)):
        return "1.0::double precision"

    v_lit = _format_sql_number(baseline.valence)
    a_lit = _format_sql_number(baseline.arousal)
    d_lit = _format_sql_number(baseline.dominance)
    return (
        f"CASE\n"
        f"    WHEN {column_prefix}emotional_context_valence IS NULL\n"
        f"      OR {column_prefix}emotional_context_arousal IS NULL\n"
        f"      OR {column_prefix}emotional_context_dominance IS NULL\n"
        f"    THEN 1.0::double precision\n"
        f"    ELSE (1 + {EMOTIONAL_RESONANCE_WEIGHT} * (1 - LEAST(1.0::double precision, sqrt(\n"
        f"        power({column_prefix}emotional_context_valence::double precision - ({v_lit}), 2)\n"
        f"      + power({column_prefix}emotional_context_arousal::double precision - ({a_lit}), 2)\n"
        f"      + power({column_prefix}emotional_context_dominance::double precision - ({d_lit}), 2)\n"
        f"    ) / sqrt(12::double precision))))\n"
        f"  END"
    )


def _format_sql_number(n: float) -> str:
    """Преобразование float → SQL-литерал (без потери точности).

    Port ``formatSqlNumber`` (memory.ts:664).
    """
    return repr(n)


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


__all__ = [
    "BuildFactorExprsOptions",
    "DecayConfig",
    "EMOTIONAL_RESONANCE_WEIGHT",
    "EmotionalBaseline",
    "FactorExprs",
    "build_factor_exprs",
]
