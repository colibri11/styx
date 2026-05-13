"""Greedy single-link dedup по cosine similarity.

Прямой port из ``openclaw-memorybox/src/recall/internal-dedup.ts``.
Каждый кластер представлен одним winner'ом — элементом с наивысшим
``score``. Элементы без ``embedding`` идут как есть, в кластеризацию
не попадают.

Используется в recall-pipeline для отсечения почти-дублей в top-N
(после `min_score` фильтра, до slice'а до `memoryLimit`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Iterable, Protocol, Sequence, TypeVar


class _DedupItem(Protocol):
    score: float
    embedding: Sequence[float] | None


T = TypeVar("T", bound=_DedupItem)


@dataclass(frozen=True)
class DedupResult(Generic[T]):
    kept: list[T]
    removed: int


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Возвращает 0 при разных длинах или нулевом векторе.

    Port ``cosineSimilarity`` (internal-dedup.ts:11).
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag_a += x * x
        mag_b += y * y
    denom = math.sqrt(mag_a) * math.sqrt(mag_b)
    if denom == 0:
        return 0.0
    return dot / denom


def _has_embedding(item: _DedupItem) -> bool:
    return item.embedding is not None and len(item.embedding) > 0


def internal_dedup(items: Iterable[T], similarity_threshold: float) -> DedupResult[T]:
    """Greedy single-link clustering по cosine similarity.

    Каждый кластер сохраняет элемент с наивысшим ``score`` (winner).
    Элементы без ``embedding`` возвращаются passthrough — без участия
    в кластеризации.

    Port ``internalDedup`` (internal-dedup.ts:36).
    """
    items_list = list(items)
    if not items_list:
        return DedupResult(kept=[], removed=0)

    with_emb = [it for it in items_list if _has_embedding(it)]
    without_emb = [it for it in items_list if not _has_embedding(it)]

    if not with_emb:
        return DedupResult(kept=items_list, removed=0)

    sorted_items = sorted(with_emb, key=lambda x: x.score, reverse=True)
    clusters: list[list[T]] = []

    for item in sorted_items:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            assert item.embedding is not None and rep.embedding is not None
            sim = cosine_similarity(item.embedding, rep.embedding)
            if sim >= similarity_threshold:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    winners = [c[0] for c in clusters]
    removed = len(with_emb) - len(winners)

    return DedupResult(kept=[*winners, *without_emb], removed=removed)
