"""Pure-function unit-тесты для autotune lifecycle sweep'а."""

from __future__ import annotations

from styx.workers.sweep.lifecycle import (
    DEFAULT_AUTOTUNE,
    apply_smoothing,
    clamp,
    compute_targets,
    resolve_autotune_config,
)


# ── resolve_autotune_config ─────────────────────────────────────────


def test_resolve_default() -> None:
    cfg = resolve_autotune_config(None)
    assert cfg["mode"] == "budget"
    assert cfg["context_budget_tokens"] == 20000
    assert cfg["bounds"]["fresh_to_settled_age_days"]["min"] == 1
    assert cfg["bounds"]["fresh_to_settled_age_days"]["max"] == 60
    assert cfg["bounds"]["settled_to_dormant_idle_days"]["min"] == 7
    assert cfg["bounds"]["settled_to_dormant_idle_days"]["max"] == 730


def test_resolve_partial_override() -> None:
    cfg = resolve_autotune_config(
        {"mode": "fixed_share", "fresh_share": 0.3}
    )
    assert cfg["mode"] == "fixed_share"
    assert cfg["fresh_share"] == 0.3
    # Не указанные — defaults.
    assert cfg["smoothing"] == DEFAULT_AUTOTUNE["smoothing"]


def test_resolve_bounds_partial_override() -> None:
    cfg = resolve_autotune_config(
        {"bounds": {"fresh_to_settled_age_days": {"min": 5}}}
    )
    assert cfg["bounds"]["fresh_to_settled_age_days"]["min"] == 5
    assert cfg["bounds"]["fresh_to_settled_age_days"]["max"] == 60
    assert cfg["bounds"]["settled_to_dormant_idle_days"]["min"] == 7


def test_resolve_does_not_mutate_default() -> None:
    cfg = resolve_autotune_config(None)
    cfg["mode"] = "fixed_share"
    cfg["bounds"]["fresh_to_settled_age_days"]["min"] = 999
    # Дефолт не изменился.
    assert DEFAULT_AUTOTUNE["mode"] == "budget"
    assert DEFAULT_AUTOTUNE["bounds"]["fresh_to_settled_age_days"]["min"] == 1


# ── compute_targets ────────────────────────────────────────────────


def test_compute_targets_budget_small() -> None:
    """total меньше fresh_multiplier × recall_budget_items — берём min_fresh_size,
    settled max возможный, dormant остаток."""
    cfg = resolve_autotune_config(None)
    t = compute_targets(50, cfg)
    assert t.fresh + t.settled + t.dormant == 50
    assert t.fresh >= 0
    # min_fresh_size=20, max_fresh_share=0.5, fresh_multiplier=10,
    # recall_budget_items=10 → fresh_target = min(100, 25)=25, max(25, 20)=25
    assert t.fresh == 25


def test_compute_targets_budget_large() -> None:
    """total большой — settled съедает больше всего."""
    cfg = resolve_autotune_config(None)
    t = compute_targets(10000, cfg)
    assert t.fresh + t.settled + t.dormant == 10000
    # fresh_target = min(100, 5000)=100, max(100, 20)=100
    assert t.fresh == 100
    # settled_target = min(100*50=5000, 9900)=5000
    assert t.settled == 5000
    # dormant = 10000 - 100 - 5000 = 4900
    assert t.dormant == 4900


def test_compute_targets_fixed_share() -> None:
    cfg = resolve_autotune_config({"mode": "fixed_share"})
    t = compute_targets(1000, cfg)
    assert t.fresh + t.settled + t.dormant == 1000
    assert t.fresh == 200  # 0.20 * 1000
    assert t.dormant == 250  # 0.25 * 1000
    assert t.settled == 550


def test_compute_targets_zero_total() -> None:
    cfg = resolve_autotune_config(None)
    t = compute_targets(0, cfg)
    assert t == compute_targets(-1, cfg)
    assert t.fresh == 0 and t.settled == 0 and t.dormant == 0


# ── apply_smoothing ────────────────────────────────────────────────


def test_apply_smoothing_no_previous() -> None:
    """Без previous → берём current."""
    assert apply_smoothing(5.0, None, 0.3) == 5.0


def test_apply_smoothing_with_previous() -> None:
    """next = prev + α(current - prev)."""
    out = apply_smoothing(current=10.0, previous=5.0, alpha=0.3)
    assert abs(out - (5.0 + 0.3 * 5.0)) < 1e-9


def test_apply_smoothing_alpha_one() -> None:
    """α=1 → next = current."""
    assert apply_smoothing(10.0, 5.0, 1.0) == 10.0


def test_apply_smoothing_alpha_zero() -> None:
    """α=0 → next = previous."""
    assert apply_smoothing(10.0, 5.0, 0.0) == 5.0


# ── clamp ──────────────────────────────────────────────────────────


def test_clamp_basic() -> None:
    assert clamp(5.0, 1.0, 10.0) == 5.0
    assert clamp(0.5, 1.0, 10.0) == 1.0
    assert clamp(15.0, 1.0, 10.0) == 10.0


def test_clamp_inverted_bounds() -> None:
    """Если lo > hi — fallback на lo."""
    assert clamp(5.0, 10.0, 1.0) == 10.0
