"""Unit-тесты для scoring.py — port buildFactorExprs из memorybox."""

from __future__ import annotations

from styx.storage.scoring import (
    EMOTIONAL_RESONANCE_WEIGHT,
    BuildFactorExprsOptions,
    DecayConfig,
    EmotionalBaseline,
    build_factor_exprs,
)


# ── base_match: vector vs hybrid ──────────────────────────────────────


def test_vector_only_when_no_text_query() -> None:
    out = build_factor_exprs({}, BuildFactorExprsOptions(text_query_param_index=None))
    assert out.base_match_mode == "vector"
    assert out.base_match_expr == "(1 - (embedding <=> $1))"
    assert out.bm25_expr is None
    assert out.vector_weight == 0.0
    assert out.bm25_weight == 0.0


def test_hybrid_when_text_query_present() -> None:
    """Adaptive weights активируются только при наличии text_query."""
    out = build_factor_exprs(
        {"text_query": "find me something"},  # 3 слова → 0.7/0.3
        BuildFactorExprsOptions(text_query_param_index=2),
    )
    assert out.base_match_mode == "hybrid"
    assert out.bm25_expr == "ts_rank(content_tsv, plainto_tsquery('simple', $2), 32)"
    assert out.vector_weight == 0.7
    assert out.bm25_weight == 0.3
    assert "0.7 * (1 - (embedding <=> $1))" in out.base_match_expr
    assert "0.3 * ts_rank" in out.base_match_expr


def test_short_query_vector_heavy() -> None:
    out = build_factor_exprs(
        {"text_query": "X"},
        BuildFactorExprsOptions(text_query_param_index=2),
    )
    assert out.vector_weight == 0.8
    assert out.bm25_weight == 0.2


def test_long_query_bm25_balanced() -> None:
    out = build_factor_exprs(
        {"text_query": "one two three four five six"},
        BuildFactorExprsOptions(text_query_param_index=2),
    )
    assert out.vector_weight == 0.6
    assert out.bm25_weight == 0.4


# ── table alias ───────────────────────────────────────────────────────


def test_table_alias_qualifies_columns() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(text_query_param_index=None, table_alias="m"),
    )
    assert out.vector_sim_expr == "(1 - (m.embedding <=> $1))"
    assert "m.created_at" in out.recency_expr
    assert "m.access_count" in out.frequency_expr
    assert "m.lifecycle" in out.lifecycle_expr
    assert "m.usefulness" in out.feedback_expr
    assert "m.importance_final" in out.importance_effective_expr
    assert "m.importance_provisional" in out.importance_effective_expr
    assert "m.unique_query_count" in out.diversity_expr
    assert "m.relevance" in out.relevance_ref_expr
    assert "re.memory_id = m.id" in out.usage_lateral_from


def test_no_table_alias_uses_bare_columns() -> None:
    out = build_factor_exprs({}, BuildFactorExprsOptions(text_query_param_index=None))
    assert out.vector_sim_expr == "(1 - (embedding <=> $1))"
    assert "re.memory_id = memories.id" in out.usage_lateral_from


# ── decay ─────────────────────────────────────────────────────────────


def test_decay_enabled_by_default() -> None:
    out = build_factor_exprs({}, BuildFactorExprsOptions(text_query_param_index=None))
    assert out.decay_enabled is True
    assert "exp(" in out.decay_expr
    assert "EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0" in out.decay_expr
    # importance-aware масштабирование лямбды.
    assert "importance_final IS NULL" in out.effective_lambda_expr
    assert "GREATEST(0.3" in out.effective_lambda_expr


def test_decay_disabled_collapses_to_one() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(
            text_query_param_index=None,
            decay_config=DecayConfig(enabled=False),
        ),
    )
    assert out.decay_enabled is False
    assert out.decay_expr == "1.0"
    assert out.effective_lambda_expr == "0"


def test_decay_lambda_per_kind_default() -> None:
    out = build_factor_exprs({}, BuildFactorExprsOptions(text_query_param_index=None))
    # Default lambdas из DEFAULT_LAMBDA_BY_KIND зашиты в lambda_base_expr.
    assert "WHEN 'decision' THEN 0.003" in out.lambda_base_expr
    assert "WHEN 'fact' THEN 0.005" in out.lambda_base_expr
    assert "WHEN 'episode' THEN 0.02" in out.lambda_base_expr


def test_decay_lambda_override() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(
            text_query_param_index=None,
            decay_config=DecayConfig(lambdas={"fact": 0.999}),
        ),
    )
    assert "WHEN 'fact' THEN 0.999" in out.lambda_base_expr
    # Остальные дефолты сохранены.
    assert "WHEN 'decision' THEN 0.003" in out.lambda_base_expr


# ── usage factor ──────────────────────────────────────────────────────


def test_usage_factor_neutral_when_p75_zero() -> None:
    out = build_factor_exprs({}, BuildFactorExprsOptions(text_query_param_index=None))
    assert out.usage_norm_p75 == 0.0
    assert out.usage_factor_expr == "1.0::double precision"


def test_usage_factor_active_when_p75_positive() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(text_query_param_index=None, usage_norm_p75=5.0),
    )
    assert out.usage_norm_p75 == 5.0
    assert "0.12 * LEAST" in out.usage_factor_expr
    assert "_mb_usage.uc / 5.0::double precision" in out.usage_factor_expr


def test_usage_factor_negative_p75_treated_as_zero() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(text_query_param_index=None, usage_norm_p75=-1.0),
    )
    assert out.usage_norm_p75 == 0.0
    assert out.usage_factor_expr == "1.0::double precision"


# ── emotional resonance ──────────────────────────────────────────────


def test_emotional_resonance_neutral_without_baseline() -> None:
    out = build_factor_exprs({}, BuildFactorExprsOptions(text_query_param_index=None))
    assert out.emotional_resonance_expr == "1.0::double precision"


def test_emotional_resonance_active_with_baseline() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(
            text_query_param_index=None,
            emotional_baseline=EmotionalBaseline(
                valence=0.3, arousal=-0.1, dominance=0.5
            ),
        ),
    )
    assert "emotional_context_valence IS NULL" in out.emotional_resonance_expr
    assert "0.3" in out.emotional_resonance_expr
    assert "-0.1" in out.emotional_resonance_expr
    assert "0.5" in out.emotional_resonance_expr
    assert "sqrt(12::double precision)" in out.emotional_resonance_expr
    assert f"1 + {EMOTIONAL_RESONANCE_WEIGHT}" in out.emotional_resonance_expr


def test_emotional_resonance_neutral_when_baseline_not_finite() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(
            text_query_param_index=None,
            emotional_baseline=EmotionalBaseline(
                valence=float("nan"), arousal=0.0, dominance=0.0
            ),
        ),
    )
    assert out.emotional_resonance_expr == "1.0::double precision"


# ── omit_base_match (causal anchors path) ─────────────────────────────


def test_omit_base_match_drops_leading_factor() -> None:
    out = build_factor_exprs(
        {},
        BuildFactorExprsOptions(text_query_param_index=None, omit_base_match=True),
    )
    # score_expr не начинается с base_match скобки.
    assert not out.score_expr.startswith("(1 - (embedding")
    # И base_match выражение остаётся доступно для diagnostics.
    assert out.base_match_expr == "(1 - (embedding <=> $1))"


def test_score_expr_contains_all_factors() -> None:
    """Прямой тест: финальный SQL содержит все 11 факторов как multiplier'ов."""
    out = build_factor_exprs(
        {"text_query": "abc def"},
        BuildFactorExprsOptions(text_query_param_index=2),
    )
    # base_match
    assert "(1 - (embedding <=> $1))" in out.score_expr
    # relevance
    assert "relevance" in out.score_expr
    # recency
    assert "interval '1 day'" in out.score_expr
    assert "interval '7 days'" in out.score_expr
    # frequency
    assert "0.3 * ln(access_count + 1)" in out.score_expr
    # lifecycle
    assert "WHEN 'fresh' THEN 1.0" in out.score_expr
    assert "WHEN 'settled' THEN 0.85" in out.score_expr
    assert "WHEN 'dormant' THEN 0.3" in out.score_expr
    # feedback
    assert "0.05 * usefulness" in out.score_expr
    # importance
    assert "0.4 + 0.6" in out.score_expr
    assert "COALESCE(importance_final, importance_provisional, 0.5)" in out.score_expr
    # diversity
    assert "0.2 * ln(1 + unique_query_count)" in out.score_expr
    # decay
    assert "exp(" in out.score_expr
    assert "86400.0" in out.score_expr
    # usage_factor — neutral 1.0 здесь
    assert "1.0::double precision" in out.score_expr
    # emotional_resonance — neutral 1.0 здесь
