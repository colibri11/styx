"""Unit-тесты для internal_dedup.py — port из memorybox."""

from __future__ import annotations

from dataclasses import dataclass, field

from styx.storage.internal_dedup import (
    cosine_similarity,
    internal_dedup,
)


@dataclass
class _Item:
    score: float
    embedding: list[float] | None = field(default=None)
    label: str = ""


def test_cosine_similarity_identical() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_opposite() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_different_lengths_returns_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_dedup_empty() -> None:
    out = internal_dedup([], 0.9)
    assert out.kept == []
    assert out.removed == 0


def test_dedup_passthrough_when_no_embeddings() -> None:
    items = [_Item(score=0.8), _Item(score=0.5)]
    out = internal_dedup(items, 0.9)
    assert out.kept == items
    assert out.removed == 0


def test_dedup_keeps_winner_drops_close_neighbour() -> None:
    """Два почти-параллельных вектора — один winner с высшим score."""
    a = _Item(score=0.9, embedding=[1.0, 0.0, 0.0], label="winner")
    b = _Item(score=0.7, embedding=[0.999, 0.0447, 0.0], label="close-low-score")
    c = _Item(score=0.8, embedding=[0.0, 1.0, 0.0], label="orthogonal")

    out = internal_dedup([a, b, c], similarity_threshold=0.9)
    labels = sorted(it.label for it in out.kept)
    assert labels == ["orthogonal", "winner"]
    assert out.removed == 1


def test_dedup_threshold_below_keeps_all() -> None:
    """Очень мягкий порог 0.0 — все близкие в кластерах, выживет один на кластер.

    На threshold=0.0 даже ортогональные склеиваются. Ожидаем 1 winner.
    """
    a = _Item(score=0.9, embedding=[1.0, 0.0])
    b = _Item(score=0.7, embedding=[0.5, 0.5])
    out = internal_dedup([a, b], similarity_threshold=0.0)
    assert len(out.kept) == 1
    assert out.kept[0].score == 0.9


def test_dedup_threshold_high_keeps_all() -> None:
    """Очень строгий порог 1.001 — никто не близок, все остаются."""
    items = [
        _Item(score=0.9, embedding=[1.0, 0.0]),
        _Item(score=0.85, embedding=[0.0, 1.0]),
        _Item(score=0.8, embedding=[1.0, 0.0]),  # дубль первого
    ]
    out = internal_dedup(items, similarity_threshold=1.001)
    assert len(out.kept) == 3
    assert out.removed == 0


def test_dedup_mixed_with_and_without_embedding() -> None:
    """Без embedding — passthrough; с embedding — кластеризуются отдельно."""
    a = _Item(score=0.9, embedding=[1.0, 0.0], label="a")
    b = _Item(score=0.7, embedding=[0.999, 0.0447], label="b-close")
    c = _Item(score=0.8, embedding=None, label="no-emb")

    out = internal_dedup([a, b, c], similarity_threshold=0.9)
    labels = sorted(it.label for it in out.kept)
    assert labels == ["a", "no-emb"]
    assert out.removed == 1


def test_dedup_winner_is_highest_score_in_cluster() -> None:
    """Внутри кластера остаётся максимум по score, не по порядку прихода."""
    early = _Item(score=0.5, embedding=[1.0, 0.0])
    late = _Item(score=0.95, embedding=[1.0, 0.0])
    out = internal_dedup([early, late], similarity_threshold=0.9)
    assert len(out.kept) == 1
    assert out.kept[0].score == 0.95
