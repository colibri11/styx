"""Unit-тесты для ``engine/explain.py`` (волна 25).

Pure helpers: build_factors_block, lifetime_decay_projections,
truncate_preview, hex_query_hash_short, lifecycle_multiplier.
"""

from __future__ import annotations

import math

import pytest

from styx.engine.explain import (
    build_factors_block,
    hex_query_hash_short,
    lifecycle_multiplier,
    lifetime_decay_projections,
    truncate_preview,
)
from styx.storage.scoring import (
    BuildFactorExprsOptions,
    DecayConfig,
    build_factor_exprs,
)


# ── truncate_preview ────────────────────────────────────────────────


def test_truncate_preview_empty() -> None:
    assert truncate_preview("") == ""
    assert truncate_preview(None) == ""


def test_truncate_preview_short_unchanged() -> None:
    assert truncate_preview("hello") == "hello"


def test_truncate_preview_long_truncated() -> None:
    s = "x" * 250
    out = truncate_preview(s)
    assert len(out) == 201  # 200 + ellipsis
    assert out.endswith("…")
    assert out[:200] == "x" * 200


def test_truncate_preview_custom_max_len() -> None:
    out = truncate_preview("abcdefgh", max_len=3)
    assert out == "abc…"


def test_truncate_preview_exact_boundary() -> None:
    s = "x" * 200
    assert truncate_preview(s) == s  # ровно 200 — без хвостика


# ── hex_query_hash_short ────────────────────────────────────────────


def test_hex_query_hash_short_none() -> None:
    assert hex_query_hash_short(None) == ""


def test_hex_query_hash_short_basic() -> None:
    payload = bytes.fromhex("0123456789abcdef" + "ff" * 24)
    assert hex_query_hash_short(payload) == "0x0123456789abcdef"


def test_hex_query_hash_short_memoryview() -> None:
    payload = memoryview(bytes.fromhex("aabbccdd" + "00" * 28))
    assert hex_query_hash_short(payload) == "0xaabbccdd00000000"


# ── lifecycle_multiplier ────────────────────────────────────────────


def test_lifecycle_multiplier_known_states() -> None:
    assert lifecycle_multiplier("fresh") == 1.0
    assert lifecycle_multiplier("settled") == 0.85
    assert lifecycle_multiplier("dormant") == 0.3


def test_lifecycle_multiplier_unknown_or_none() -> None:
    assert lifecycle_multiplier(None) == 1.0
    assert lifecycle_multiplier("zombie") == 1.0


# ── lifetime_decay_projections ──────────────────────────────────────


def test_lifetime_decay_disabled() -> None:
    out = lifetime_decay_projections(
        kind="fact",
        age_days=10.0,
        importance_final=0.5,
        relevance=1.0,
        decay_config=DecayConfig(enabled=False),
        prune_min_relevance=0.1,
    )
    assert out["effective_lambda"] == 0.0
    assert out["current_decay_factor"] == 1.0
    assert out["projected_decay_in_30d"] == 1.0
    assert out["projected_decay_in_365d"] == 1.0
    assert out["estimated_days_to_prune_threshold"] is None
    assert out["grace_period_active"] is False


def test_lifetime_decay_enabled_importance_final_null() -> None:
    out = lifetime_decay_projections(
        kind="episode",
        age_days=0.0,
        importance_final=None,
        relevance=1.0,
        decay_config=None,
        prune_min_relevance=None,
    )
    # episode default lambda = 0.020 × grace 0.3 = 0.006
    assert out["lambda_base"] == pytest.approx(0.020)
    assert out["effective_lambda"] == pytest.approx(0.020 * 0.3)
    assert out["grace_period_active"] is True
    assert out["current_decay_factor"] == pytest.approx(1.0)
    assert out["projected_decay_in_30d"] == pytest.approx(
        math.exp(-0.020 * 0.3 * 30)
    )


def test_lifetime_decay_grace_max_floor() -> None:
    # importance_final=1.0 → 1 - 0.7*1 = 0.3 (всегда floor)
    out = lifetime_decay_projections(
        kind="fact",
        age_days=5.0,
        importance_final=1.0,
        relevance=1.0,
        decay_config=None,
        prune_min_relevance=None,
    )
    # 0.005 lambda × 0.3 grace = 0.0015 effective
    assert out["effective_lambda"] == pytest.approx(0.005 * 0.3)
    assert out["grace_period_active"] is False


def test_lifetime_decay_grace_above_floor() -> None:
    # importance_final=0.5 → max(0.3, 1 - 0.7*0.5) = max(0.3, 0.65) = 0.65
    out = lifetime_decay_projections(
        kind="fact",
        age_days=0.0,
        importance_final=0.5,
        relevance=1.0,
        decay_config=None,
        prune_min_relevance=None,
    )
    assert out["effective_lambda"] == pytest.approx(0.005 * 0.65)


def test_lifetime_decay_lambda_override() -> None:
    cfg = DecayConfig(enabled=True, lambdas={"fact": 0.1})  # type: ignore[arg-type]
    out = lifetime_decay_projections(
        kind="fact",
        age_days=0.0,
        importance_final=None,
        relevance=1.0,
        decay_config=cfg,
        prune_min_relevance=None,
    )
    assert out["lambda_base"] == pytest.approx(0.1)


def test_lifetime_decay_prune_solver_converges() -> None:
    # relevance=1.0, lambda*age=0 → current_decay=1.0
    # threshold=0.5, effective_lambda=0.005*0.65=0.00325
    # d = ln(1/0.5)/0.00325 - 0 ≈ 213.247
    out = lifetime_decay_projections(
        kind="fact",
        age_days=0.0,
        importance_final=0.5,
        relevance=1.0,
        decay_config=None,
        prune_min_relevance=0.5,
    )
    expected = math.log(1.0 / 0.5) / (0.005 * 0.65)
    assert out["estimated_days_to_prune_threshold"] == pytest.approx(expected)


def test_lifetime_decay_prune_already_below() -> None:
    # relevance * current_decay уже под threshold → None
    out = lifetime_decay_projections(
        kind="fact",
        age_days=10000.0,  # сильно остарело
        importance_final=None,
        relevance=1.0,
        decay_config=None,
        prune_min_relevance=0.99,
    )
    assert out["estimated_days_to_prune_threshold"] is None


def test_lifetime_decay_prune_threshold_zero_safe() -> None:
    # threshold=0 → log(1/0) — избегаем divide-by-zero, возвращаем None
    out = lifetime_decay_projections(
        kind="fact",
        age_days=1.0,
        importance_final=0.5,
        relevance=1.0,
        decay_config=None,
        prune_min_relevance=0.0,
    )
    assert out["estimated_days_to_prune_threshold"] is None


# ── build_factors_block ─────────────────────────────────────────────


def _factors_for_test(
    *,
    text_query: str | None,
    table_alias: str = "m",
    decay_config: DecayConfig | None = None,
):
    """Helper: build_factor_exprs с реалистичными опциями."""
    text_query_param_index = 2 if text_query else None
    inp: dict[str, object] = {"text_query": text_query} if text_query else {}
    return build_factor_exprs(
        inp,
        BuildFactorExprsOptions(
            text_query_param_index=text_query_param_index,
            decay_config=decay_config,
            table_alias=table_alias,
            usage_norm_p75=0.0,
            emotional_baseline=None,
        ),
    )


def _row_for_test(**overrides: object) -> dict[str, object]:
    base = {
        "kind": "fact",
        "age_days_sql": 0.5,
        "importance_provisional": 0.5,
        "importance_final": None,
        "importance_effective": 0.5,
        "lambda_base": 0.005,
        "effective_lambda": 0.0015,
        "vector_sim": 0.7,
        "bm25_rank": None,
        "base_match": 0.7,
        "relevance_factor": 1.0,
        "recency_factor": 1.3,
        "frequency_factor": 1.0,
        "lifecycle": "fresh",
        "lifecycle_factor": 1.0,
        "feedback_factor": 1.0,
        "importance_factor": 0.7,
        "diversity_factor": 1.0,
        "decay_factor": 0.99925,
        "usage_count_30d": 0,
        "usage_factor": 1.0,
        "access_count": 0,
        "usefulness": 0.0,
        "unique_query_count": 0,
        "llm_task_status": None,
        "llm_task_created_at": None,
    }
    base.update(overrides)
    return base


def test_build_factors_block_pure_vector() -> None:
    factors = _factors_for_test(text_query=None)
    row = _row_for_test()
    block = build_factors_block(row, factors, decay_config=None)
    assert block["base_match"]["mode"] == "vector"
    assert block["base_match"]["bm25"] is None
    assert block["base_match"]["hybrid_weights"] == {"vector": 1.0, "bm25": 0.0}
    assert block["recency_boost"]["rule"] == "< 1 day"
    assert block["importance_factor"]["source"] == "provisional"
    assert block["importance_factor"]["final"] is None
    assert block["decay_factor"]["grace_reason"] == "importance_final IS NULL"
    assert block["decay_factor"]["grace_multiplier"] == 0.3
    assert block["decay_factor"]["formula"] == "exp(-effective_lambda * age_days)"


def test_build_factors_block_hybrid_mode() -> None:
    factors = _factors_for_test(text_query="hello world")
    row = _row_for_test(bm25_rank=0.42, base_match=0.55)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["base_match"]["mode"] == "hybrid"
    assert block["base_match"]["bm25"] == pytest.approx(0.42)
    assert block["base_match"]["hybrid_weights"]["vector"] > 0
    assert block["base_match"]["hybrid_weights"]["bm25"] > 0


def test_build_factors_block_recency_rule_settled_age() -> None:
    factors = _factors_for_test(text_query=None)
    row = _row_for_test(age_days_sql=3.0, recency_factor=1.1)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["recency_boost"]["rule"] == "< 7 days"
    assert block["recency_boost"]["age_days"] == 3.0


def test_build_factors_block_recency_rule_old_age() -> None:
    factors = _factors_for_test(text_query=None)
    row = _row_for_test(age_days_sql=10.0, recency_factor=1.0)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["recency_boost"]["rule"] == ">= 7 days"


def test_build_factors_block_grace_floor_with_high_importance() -> None:
    factors = _factors_for_test(text_query=None)
    # Чтобы строго попасть в floor (0.3), importance_final должен быть
    # > 1 (формула 1-0.7x даёт <0.3 → max клампит к 0.3). Возьмём 1.5,
    # хотя в реальности importance_final ∈ [0,1]. В этом тесте мы
    # покрываем сам floor-branch helper'а.
    row = _row_for_test(importance_final=1.5, importance_effective=1.0)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["decay_factor"]["grace_multiplier"] == 0.3
    assert block["decay_factor"]["grace_reason"] == "floor (0.3)"


def test_build_factors_block_grace_at_one_importance_above_floor() -> None:
    """importance_final=1.0 → 1 - 0.7*1 = 0.30000000000000004 (FP-дрейф),
    max(0.3, 0.300...4) = 0.300...4 — это `importance_final=1.0` reason
    (НЕ floor). Memorybox имеет ту же семантику: `===` 0.3 false."""
    factors = _factors_for_test(text_query=None)
    row = _row_for_test(importance_final=1.0, importance_effective=1.0)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["decay_factor"]["grace_multiplier"] == pytest.approx(
        0.3, abs=1e-10
    )
    assert "importance_final=1.0" in block["decay_factor"]["grace_reason"]


def test_build_factors_block_grace_above_floor() -> None:
    factors = _factors_for_test(text_query=None)
    row = _row_for_test(importance_final=0.5, importance_effective=0.5)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["decay_factor"]["grace_multiplier"] == pytest.approx(0.65)
    assert "importance_final=0.5" in block["decay_factor"]["grace_reason"]


def test_build_factors_block_decay_disabled() -> None:
    cfg = DecayConfig(enabled=False)
    factors = _factors_for_test(text_query=None, decay_config=cfg)
    row = _row_for_test(decay_factor=1.0)
    block = build_factors_block(row, factors, decay_config=cfg)
    assert block["decay_factor"]["formula"] == "1.0 (decay disabled)"
    assert block["decay_factor"]["grace_multiplier"] == 0.0
    assert block["decay_factor"]["grace_reason"] == "decay disabled"


def test_build_factors_block_lambda_source_override() -> None:
    cfg = DecayConfig(enabled=True, lambdas={"fact": 0.1})  # type: ignore[arg-type]
    factors = _factors_for_test(text_query=None, decay_config=cfg)
    row = _row_for_test(lambda_base=0.1, effective_lambda=0.03)
    block = build_factors_block(row, factors, decay_config=cfg)
    assert "config override" in block["decay_factor"]["lambda_source"]


def test_build_factors_block_usage_no_personalization() -> None:
    factors = _factors_for_test(text_query=None)
    row = _row_for_test()
    block = build_factors_block(row, factors, decay_config=None)
    # usage_norm_p75=0 → формула «no personalisation»
    assert "no personalisation" in block["usage_factor"]["formula"]
    assert block["usage_factor"]["p75_agent"] == 0.0


def test_build_factors_block_importance_final_present() -> None:
    factors = _factors_for_test(text_query=None)
    row = _row_for_test(importance_final=0.85, importance_effective=0.85)
    block = build_factors_block(row, factors, decay_config=None)
    assert block["importance_factor"]["source"] == "final"
    assert block["importance_factor"]["final"] == 0.85
