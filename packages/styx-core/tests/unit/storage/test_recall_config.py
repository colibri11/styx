"""Unit-тесты для recall_config.py — port из memorybox."""

from __future__ import annotations

from styx.storage.recall_config import (
    DEFAULT_RECALL_CONFIG,
    RecallConfig,
    resolve_recall_config,
)


def test_defaults_match_memorybox() -> None:
    cfg = DEFAULT_RECALL_CONFIG
    # full
    assert cfg.full.memory_limit == 6
    assert cfg.full.dialogue_limit == 3
    assert cfg.full.chunk_limit == 3
    # Волна 8: откалибровано под embeddinggemma (decisions.md § 22).
    assert cfg.full.min_score == 0.32
    assert cfg.full.internal_dedup_similarity == 0.92
    # companion.dialogue
    assert cfg.companion.dialogue.enabled is True
    assert cfg.companion.dialogue.limit == 5
    assert cfg.companion.dialogue.min_score == 0.6
    assert cfg.companion.dialogue.session_scope == "all"
    # companion.structured
    assert cfg.companion.structured.enabled is False
    assert cfg.companion.structured.limit == 4
    assert cfg.companion.structured.min_score == 0.7
    assert cfg.companion.structured.internal_dedup_similarity == 0.92
    # companion.hot_tier
    assert cfg.companion.hot_tier.enabled is True
    assert cfg.companion.hot_tier.limit == 5
    # token budget
    assert cfg.token_budget_fraction == 0.1


def test_resolve_no_partial_returns_defaults() -> None:
    cfg = resolve_recall_config(None)
    assert cfg == DEFAULT_RECALL_CONFIG


def test_resolve_partial_full_override() -> None:
    cfg = resolve_recall_config({"full": {"min_score": 0.5, "memory_limit": 10}})
    assert cfg.full.min_score == 0.5
    assert cfg.full.memory_limit == 10
    # Остальные дефолты на месте.
    assert cfg.full.dialogue_limit == 3
    assert cfg.full.internal_dedup_similarity == 0.92
    assert cfg.companion == DEFAULT_RECALL_CONFIG.companion


def test_resolve_partial_companion_dialogue_override() -> None:
    cfg = resolve_recall_config(
        {"companion": {"dialogue": {"min_score": 0.4, "session_scope": "current"}}}
    )
    assert cfg.companion.dialogue.min_score == 0.4
    assert cfg.companion.dialogue.session_scope == "current"
    assert cfg.companion.dialogue.limit == 5  # default
    assert cfg.full == DEFAULT_RECALL_CONFIG.full


def test_resolve_unknown_keys_silently_ignored() -> None:
    cfg = resolve_recall_config({"full": {"unknown_field": "garbage"}})
    assert cfg == DEFAULT_RECALL_CONFIG


def test_resolve_token_budget_fraction_override() -> None:
    cfg = resolve_recall_config({"token_budget_fraction": 0.2})
    assert cfg.token_budget_fraction == 0.2


def test_resolve_token_budget_fraction_invalid_falls_back() -> None:
    cfg = resolve_recall_config({"token_budget_fraction": "bad"})
    assert cfg.token_budget_fraction == 0.1


def test_resolve_companion_structured_override() -> None:
    cfg = resolve_recall_config(
        {"companion": {"structured": {"enabled": True, "limit": 8}}}
    )
    assert cfg.companion.structured.enabled is True
    assert cfg.companion.structured.limit == 8
    assert cfg.companion.structured.min_score == 0.7  # default


def test_default_recall_config_immutable() -> None:
    """frozen=True — попытка перезаписать поле падает."""
    import dataclasses

    try:
        DEFAULT_RECALL_CONFIG.full.min_score = 0.99  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("DEFAULT_RECALL_CONFIG не immutable")
