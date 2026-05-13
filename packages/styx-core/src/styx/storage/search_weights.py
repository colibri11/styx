"""Adaptive vector/BM25 weights для hybrid search.

Прямой port из ``openclaw-memorybox/src/search-weights.ts``. Числа
буквальны (decisions.md § 17.5).

Strategy: короткие запросы лежат на векторе (semantic), длинные — на
BM25 (keyword match), потому что у длинного запроса слов-якорей больше
и BM25 их лучше ловит.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SearchMode = Literal["hybrid", "vector", "bm25"]


@dataclass(frozen=True)
class SearchConfig:
    mode: SearchMode | None = None
    adaptive_weights: bool | None = None
    default_vector_weight: float | None = None
    default_bm25_weight: float | None = None


@dataclass(frozen=True)
class WeightPair:
    vector_weight: float
    bm25_weight: float


_WORD_SPLIT_RE = re.compile(r"\s+")


def compute_weights(query: str, config: SearchConfig | None = None) -> WeightPair:
    """Adaptive vector/BM25 weights в зависимости от длины запроса.

    Port ``computeWeights`` (search-weights.ts:9).
    """
    if config is None:
        config = SearchConfig()

    if config.mode == "vector":
        return WeightPair(1.0, 0.0)
    if config.mode == "bm25":
        return WeightPair(0.0, 1.0)

    if config.adaptive_weights is False:
        return WeightPair(
            config.default_vector_weight if config.default_vector_weight is not None else 0.7,
            config.default_bm25_weight if config.default_bm25_weight is not None else 0.3,
        )

    word_count = sum(1 for w in _WORD_SPLIT_RE.split(query) if w)

    if word_count <= 2:
        return WeightPair(0.8, 0.2)
    if word_count <= 4:
        return WeightPair(0.7, 0.3)
    return WeightPair(0.6, 0.4)
