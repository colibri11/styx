"""Unit-тесты для search_weights.py — port из memorybox."""

from __future__ import annotations

import pytest

from styx.storage.search_weights import SearchConfig, compute_weights


def test_explicit_vector_mode_overrides_everything() -> None:
    out = compute_weights("a b c d e f g", SearchConfig(mode="vector"))
    assert out.vector_weight == 1.0
    assert out.bm25_weight == 0.0


def test_explicit_bm25_mode_overrides_everything() -> None:
    out = compute_weights("short", SearchConfig(mode="bm25"))
    assert out.vector_weight == 0.0
    assert out.bm25_weight == 1.0


@pytest.mark.parametrize(
    "query,expected_vec,expected_bm25",
    [
        ("", 0.8, 0.2),                 # 0 слов → vector-heavy
        ("one", 0.8, 0.2),               # 1 слово
        ("two words", 0.8, 0.2),         # 2 слова — boundary inclusive
        ("three good words", 0.7, 0.3),  # 3 слова
        ("four good little words", 0.7, 0.3),  # 4 слова — boundary inclusive
        ("five short medium long words", 0.6, 0.4),  # 5+ слов
        ("a b c d e f g h i j k l m", 0.6, 0.4),
    ],
)
def test_adaptive_weights_table(
    query: str, expected_vec: float, expected_bm25: float
) -> None:
    out = compute_weights(query)
    assert out.vector_weight == expected_vec
    assert out.bm25_weight == expected_bm25


def test_adaptive_disabled_uses_defaults() -> None:
    out = compute_weights(
        "ignore this query length",
        SearchConfig(adaptive_weights=False),
    )
    assert out.vector_weight == 0.7
    assert out.bm25_weight == 0.3


def test_adaptive_disabled_with_overrides() -> None:
    out = compute_weights(
        "any",
        SearchConfig(
            adaptive_weights=False,
            default_vector_weight=0.5,
            default_bm25_weight=0.5,
        ),
    )
    assert out.vector_weight == 0.5
    assert out.bm25_weight == 0.5


def test_whitespace_collapse_in_word_count() -> None:
    """Множественные пробелы / табы / переводы строк не считаются как слова."""
    out = compute_weights("  hello\t\nworld  ")
    # 2 слова (не 4) → 0.8/0.2
    assert out.vector_weight == 0.8
