"""Unit-тесты для selective gatekeeper (волна 17, port memorybox)."""

from __future__ import annotations

import uuid

from styx.engine.selective_gatekeeper import (
    Action,
    Candidate,
    GatekeeperConfig,
    _levenshtein_distance,
    _levenshtein_ratio,
    decide,
)


def _cand(*, distance: float, content: str = "neighbour", id_: uuid.UUID | None = None) -> Candidate:
    return Candidate(
        id=id_ or uuid.uuid4(),
        content=content,
        cosine_distance=distance,
    )


# ── Levenshtein ───────────────────────────────────────────────────────


def test_levenshtein_distance_identical_zero() -> None:
    assert _levenshtein_distance("abc", "abc") == 0


def test_levenshtein_distance_empty_left() -> None:
    assert _levenshtein_distance("", "abc") == 3


def test_levenshtein_distance_empty_right() -> None:
    assert _levenshtein_distance("abc", "") == 3


def test_levenshtein_distance_substitution() -> None:
    assert _levenshtein_distance("kitten", "sitting") == 3


def test_levenshtein_distance_completely_different() -> None:
    assert _levenshtein_distance("abc", "xyz") == 3


def test_levenshtein_ratio_identical_one() -> None:
    assert _levenshtein_ratio("abc", "abc") == 1.0


def test_levenshtein_ratio_empty_strings_one() -> None:
    assert _levenshtein_ratio("", "") == 1.0


def test_levenshtein_ratio_completely_different_zero() -> None:
    assert _levenshtein_ratio("abc", "xyz") == 0.0


def test_levenshtein_ratio_partial() -> None:
    # kitten ↔ sitting: distance 3, max_len 7 → ratio = 1 - 3/7 ≈ 0.571
    ratio = _levenshtein_ratio("kitten", "sitting")
    assert abs(ratio - (1.0 - 3.0 / 7.0)) < 1e-9


# ── decide(...) — все 4 ветки ────────────────────────────────────────


def test_disabled_short_circuits_to_store() -> None:
    config = GatekeeperConfig(enabled=False)
    decision = decide("anything", [_cand(distance=0.001)], config=config)
    assert decision.action == Action.STORE
    assert decision.existing_id is None


def test_noise_filter_skip_short_content() -> None:
    config = GatekeeperConfig(noise_filter=True, noise_min_length=10)
    decision = decide("да", [], config=config)
    assert decision.action == Action.SKIP


def test_noise_filter_off_keeps_short_content() -> None:
    config = GatekeeperConfig(noise_filter=False)
    decision = decide("да", [], config=config)
    assert decision.action == Action.STORE


def test_no_candidates_store() -> None:
    config = GatekeeperConfig()
    decision = decide("новая мысль про архитектуру", [], config=config)
    assert decision.action == Action.STORE
    assert decision.similarity is None


def test_high_similarity_merge() -> None:
    config = GatekeeperConfig(merge_threshold=0.92, supersede_threshold=0.85)
    existing_id = uuid.uuid4()
    cand = _cand(distance=0.05, content="существующий контент", id_=existing_id)
    decision = decide("новый похожий контент", [cand], config=config)
    assert decision.action == Action.MERGE
    assert decision.existing_id == existing_id
    assert decision.similarity is not None
    assert abs(decision.similarity - 0.95) < 1e-9


def test_supersede_zone_with_text_similarity_supersede() -> None:
    """sim=0.88 (в зоне supersede), Levenshtein ratio высокий (близкий
    текст) → supersede.
    """
    config = GatekeeperConfig(
        merge_threshold=0.92, supersede_threshold=0.85,
        levenshtein_threshold=0.3,
    )
    existing_id = uuid.uuid4()
    cand = _cand(
        distance=0.12,
        content="команда вернулась с встречи",
        id_=existing_id,
    )
    # 92% перекрытия — high lev ratio; sim 0.88 в зоне supersede.
    decision = decide(
        "команда вернулась с встречи!", [cand], config=config,
    )
    assert decision.action == Action.SUPERSEDE
    assert decision.existing_id == existing_id


def test_supersede_zone_with_low_levenshtein_store() -> None:
    """sim=0.88 (в зоне supersede), но Levenshtein ratio низкий → store.

    Это случай «semantic близок, но текст принципиально другой» — не
    «то же самое другими словами», а другая мысль.
    """
    config = GatekeeperConfig(
        merge_threshold=0.92, supersede_threshold=0.85,
        levenshtein_threshold=0.95,  # очень строгий порог
    )
    existing_id = uuid.uuid4()
    cand = _cand(
        distance=0.12,
        content="aaaaaaaaaaaaaaaa",
        id_=existing_id,
    )
    decision = decide("zzzzzzzzzzzzzzzz", [cand], config=config)
    assert decision.action == Action.STORE
    assert decision.similarity is not None
    assert abs(decision.similarity - 0.88) < 1e-9


def test_below_supersede_threshold_store() -> None:
    """Candidates с distance, выходящим за supersede-зону, не должны
    попадать в decide() от queries-уровня (фильтр на SQL). Но если
    попали — store, не merge / не supersede.
    """
    config = GatekeeperConfig(merge_threshold=0.92, supersede_threshold=0.85)
    cand = _cand(distance=0.20)  # similarity 0.80
    decision = decide("какой-то длинный текст", [cand], config=config)
    assert decision.action == Action.STORE


def test_top_one_wins_over_other_candidates() -> None:
    """В candidates приходит несколько; decide смотрит только на top-1."""
    config = GatekeeperConfig(merge_threshold=0.92, supersede_threshold=0.85)
    closest = _cand(distance=0.05, content="близкий вариант текста")
    further = _cand(distance=0.10, content="дальний вариант текста")
    decision = decide("новая запись содержательная", [closest, further], config=config)
    assert decision.action == Action.MERGE
    assert decision.existing_id == closest.id


def test_decide_returns_levenshtein_ratio_in_supersede_zone() -> None:
    """levenshtein_ratio populated в supersede и store-в-supersede-зоне."""
    config = GatekeeperConfig(
        merge_threshold=0.92, supersede_threshold=0.85,
        levenshtein_threshold=0.3,
    )
    cand = _cand(distance=0.12, content="текст один развёрнутый")
    decision = decide("текст два развёрнутый", [cand], config=config)
    assert decision.levenshtein_ratio is not None
