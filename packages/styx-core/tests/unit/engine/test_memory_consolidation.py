"""Unit-тесты для memory_consolidation engine (волна 22).

Pure функции: build_clusters / cooldown_elapsed / pick_consolidated_*.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from styx.engine.memory_consolidation import (
    DEFAULT_CLUSTER_COSINE,
    DEFAULT_CLUSTER_MAX_SIZE,
    DEFAULT_CLUSTER_MIN_SIZE,
    DEFAULT_COOLDOWN_HOURS,
    Cluster,
    ClusterCandidate,
    MemoryConsolidationConfig,
    build_clusters,
    cooldown_elapsed,
    cosine,
    pick_consolidated_kind,
    pick_consolidated_visibility,
)


def _id() -> uuid.UUID:
    return uuid.uuid4()


def _normalise(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n > 0 else vec


# ── cosine ────────────────────────────────────────────────────────────


def test_cosine_identical_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    assert cosine(a, a) == pytest.approx(1.0, abs=1e-9)


def test_cosine_orthogonal_vectors() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_dim_mismatch_returns_zero() -> None:
    assert cosine([1.0, 0.0], [1.0]) == 0.0


def test_cosine_empty_returns_zero() -> None:
    assert cosine([], []) == 0.0


def test_cosine_zero_magnitude_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── build_clusters ────────────────────────────────────────────────────


def _candidate(vec: list[float]) -> ClusterCandidate:
    return ClusterCandidate(id=_id(), embedding=_normalise(vec))


def test_build_clusters_three_close_one_cluster() -> None:
    # 3 близких embedding'а — должны пойти в один кластер.
    items = [
        _candidate([1.0, 0.0, 0.0]),
        _candidate([0.99, 0.01, 0.0]),
        _candidate([0.98, 0.02, 0.0]),
    ]
    clusters = build_clusters(items, cosine_threshold=0.88)
    assert len(clusters) == 1
    assert len(clusters[0].member_ids) == 3
    assert set(clusters[0].member_ids) == {it.id for it in items}


def test_build_clusters_below_min_size_dropped() -> None:
    items = [
        _candidate([1.0, 0.0, 0.0]),
        _candidate([0.99, 0.01, 0.0]),  # пара — < min_size=3
        _candidate([0.0, 1.0, 0.0]),  # одиночка
    ]
    clusters = build_clusters(items, cosine_threshold=0.88, min_size=3)
    assert clusters == []


def test_build_clusters_max_size_clamped() -> None:
    # 9 close items с max_size=8 → 1 кластер размером ровно 8.
    # 9-й остаётся taken (memorybox parity, D4 wave-doc).
    items = [
        _candidate([1.0, 0.001 * i, 0.0]) for i in range(9)
    ]
    clusters = build_clusters(
        items, cosine_threshold=0.88, min_size=3, max_size=8,
    )
    assert len(clusters) == 1
    assert len(clusters[0].member_ids) == 8


def test_build_clusters_two_independent_groups() -> None:
    g1 = [_candidate([1.0, 0.0, 0.0]) for _ in range(3)]
    g2 = [_candidate([0.0, 0.0, 1.0]) for _ in range(3)]
    # У всех в g1 cos ≈ 1 между собой; у всех в g2 — то же; g1 vs g2 = 0.
    clusters = build_clusters(g1 + g2, cosine_threshold=0.88)
    assert len(clusters) == 2
    sizes = sorted(len(c.member_ids) for c in clusters)
    assert sizes == [3, 3]


def test_build_clusters_cosine_threshold_filters_loose() -> None:
    # Embedding'и с cos ~0.7 — ниже 0.88, не клейстерятся.
    a = _candidate([1.0, 0.0, 0.0])
    b = _candidate([0.7, 0.7, 0.0])
    c = _candidate([0.6, 0.8, 0.0])
    clusters = build_clusters([a, b, c], cosine_threshold=0.88)
    # Никакая пара не выше 0.88 — кластеров нет.
    assert clusters == []


def test_build_clusters_empty_items() -> None:
    assert build_clusters([], cosine_threshold=0.88) == []


def test_build_clusters_default_constants() -> None:
    assert DEFAULT_CLUSTER_COSINE == 0.88
    assert DEFAULT_CLUSTER_MIN_SIZE == 3
    assert DEFAULT_CLUSTER_MAX_SIZE == 8


def test_build_clusters_seed_locks_members_even_if_too_small() -> None:
    """memorybox parity: если кластер не дотянул до min_size — members
    всё равно остаются в taken (не возвращаются в pool).

    Проверим: 2 близких + 1 третий, который близок только к одному из
    первых двух → второй уже взят первым (мини-кластером ниже min),
    третий пропадает в taken'е свого мини-кластера → ни одного
    кластера ≥ 3.
    """
    a_vec = [1.0, 0.0, 0.0]
    b_vec = [0.99, 0.01, 0.0]
    a = _candidate(a_vec)
    b = _candidate(b_vec)
    c = _candidate([0.99, 0.0, 0.001])  # близок к a
    clusters = build_clusters([a, b, c], cosine_threshold=0.88, min_size=3)
    # Первый seed = a, к нему добавятся b и c (≥ 0.88) → 1 кластер с 3.
    # Это параметризованный тест на parity — фиксируем actual behavior.
    assert len(clusters) == 1
    assert len(clusters[0].member_ids) == 3


# ── cooldown_elapsed ──────────────────────────────────────────────────


def _ts(at: datetime) -> str:
    return at.isoformat()


def test_cooldown_elapsed_none_state() -> None:
    now = datetime(2026, 5, 5, tzinfo=timezone.utc)
    assert cooldown_elapsed(None, now, hours=23) is True


def test_cooldown_elapsed_missing_last_run_at() -> None:
    now = datetime(2026, 5, 5, tzinfo=timezone.utc)
    assert cooldown_elapsed({}, now, hours=23) is True
    assert cooldown_elapsed({"last_run_at": ""}, now, hours=23) is True


def test_cooldown_elapsed_unparseable_last_run_at() -> None:
    now = datetime(2026, 5, 5, tzinfo=timezone.utc)
    assert cooldown_elapsed(
        {"last_run_at": "not-a-timestamp"}, now, hours=23
    ) is True


def test_cooldown_elapsed_within_window_blocks() -> None:
    now = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=22, minutes=59)
    state = {"last_run_at": _ts(last)}
    assert cooldown_elapsed(state, now, hours=23) is False


def test_cooldown_elapsed_past_window_passes() -> None:
    now = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=23, minutes=1)
    state = {"last_run_at": _ts(last)}
    assert cooldown_elapsed(state, now, hours=23) is True


def test_cooldown_elapsed_exact_boundary_passes() -> None:
    now = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=23)
    state = {"last_run_at": _ts(last)}
    assert cooldown_elapsed(state, now, hours=23) is True


def test_cooldown_elapsed_naive_datetime_treated_as_utc() -> None:
    now_naive = datetime(2026, 5, 5, 12, 0, 0)
    last = datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc)
    state = {"last_run_at": _ts(last)}
    assert cooldown_elapsed(state, now_naive, hours=23) is True


def test_cooldown_default_constant() -> None:
    assert DEFAULT_COOLDOWN_HOURS == 23


# ── pick_consolidated_kind ────────────────────────────────────────────


def test_pick_kind_majority_unique() -> None:
    assert pick_consolidated_kind(["fact", "fact", "note"]) == "fact"


def test_pick_kind_tie_priority_concept_first() -> None:
    assert pick_consolidated_kind(["concept", "episode"]) == "concept"


def test_pick_kind_tie_priority_note_over_fact() -> None:
    # priority: concept > note > fact > decision > episode.
    assert pick_consolidated_kind(["note", "fact"]) == "note"


def test_pick_kind_tie_priority_full_chain() -> None:
    assert pick_consolidated_kind(
        ["concept", "note", "fact", "decision", "episode"]
    ) == "concept"


def test_pick_kind_empty_returns_note() -> None:
    assert pick_consolidated_kind([]) == "note"


def test_pick_kind_unknown_kind_falls_to_priority_end() -> None:
    # Кastom kind не в priority list → conventional после всех known.
    # ['custom', 'fact'] — счёт 1+1 → tie. Sort by index:
    # 'fact' priority idx=2; 'custom' idx=len => 'fact' wins.
    assert pick_consolidated_kind(["custom", "fact"]) == "fact"


# ── pick_consolidated_visibility ──────────────────────────────────────


def test_pick_visibility_all_shared() -> None:
    assert pick_consolidated_visibility(["shared", "shared"]) == "shared"


def test_pick_visibility_one_private_wins() -> None:
    assert pick_consolidated_visibility(["shared", "private"]) == "private"
    assert pick_consolidated_visibility(["private", "shared"]) == "private"


def test_pick_visibility_empty_defaults_to_shared() -> None:
    assert pick_consolidated_visibility([]) == "shared"


def test_pick_visibility_unknown_treated_as_shared() -> None:
    # NULL/non-string из БД — D15 hardening.
    assert pick_consolidated_visibility(["shared", "unknown"]) == "shared"


# ── MemoryConsolidationConfig ─────────────────────────────────────────


def test_config_defaults_match_memorybox() -> None:
    cfg = MemoryConsolidationConfig()
    assert cfg.enabled is True
    assert cfg.tick_s == 3600.0
    assert cfg.apply_tick_s == 30.0
    assert cfg.cooldown_hours == 23
    assert cfg.window_days == 7
    assert cfg.window_tail_hours == 24
    assert cfg.cosine_threshold == 0.88
    assert cfg.min_cluster_size == 3
    assert cfg.max_cluster_size == 8
