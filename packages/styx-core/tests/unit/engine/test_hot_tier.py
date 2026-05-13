"""Юнит-тесты hot_tier — put_many / scan_candidates / TTL / LRU / snapshot fence."""

from __future__ import annotations

import datetime as _dt
import math
import uuid

import pytest

from styx.engine import hot_tier
from styx.storage.queries import MemoryHit
from styx.turn_state import RecallSnapshot


@pytest.fixture(autouse=True)
def _reset_state():
    hot_tier.reset_all()
    yield
    hot_tier.reset_all()


def _unit(values: list[float]) -> list[float]:
    """Нормализация к unit length для воспроизводимых cosine."""
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _hit(
    *,
    embedding: list[float],
    kind_src: str = "subjective",
    agent_id: str = "agent-a",
    created_at: _dt.datetime | None = None,
    content: str = "memory content",
) -> MemoryHit:
    return MemoryHit(
        id=uuid.uuid4(),
        agent_id=agent_id,
        kind="subjective_dialogue",
        kind_src=kind_src,
        role="user",
        content=content,
        metadata={},
        created_at=created_at or _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc),
        score=0.5,
        match_score=0.5,
        embedding=embedding,
    )


# -- configure / get_state / reset ---------------------------------------


def test_get_state_none_initially() -> None:
    assert hot_tier.get_state("test-agent") is None


def test_configure_initialises_empty_state() -> None:
    hot_tier.configure("test-agent", ttl_s=120.0, lru_bound=10)
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert s.entries == {}
    assert s.ttl_s == 120.0
    assert s.lru_bound == 10


def test_configure_validates_ttl_positive() -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        hot_tier.configure("test-agent", ttl_s=0.0)


def test_configure_validates_lru_bound_positive() -> None:
    with pytest.raises(ValueError, match="lru_bound"):
        hot_tier.configure("test-agent", lru_bound=0)


def test_double_configure_resets() -> None:
    hot_tier.configure("test-agent")
    hot_tier.put_many("test-agent", [_hit(embedding=_unit([1.0, 0.0, 0.0]))])
    hot_tier.configure("test-agent", ttl_s=60.0, lru_bound=5)
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert s.entries == {}
    assert s.ttl_s == 60.0
    assert s.lru_bound == 5


def test_reset_clears_state() -> None:
    hot_tier.configure("test-agent")
    hot_tier.put_many("test-agent", [_hit(embedding=_unit([1.0, 0.0, 0.0]))])
    hot_tier.reset_all()
    assert hot_tier.get_state("test-agent") is None


# -- put_many -------------------------------------------------------------


def test_put_many_stores_entries() -> None:
    hot_tier.configure("test-agent")
    h1 = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    h2 = _hit(embedding=_unit([0.0, 1.0, 0.0]))
    hot_tier.put_many("test-agent", [h1, h2])
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert set(s.entries.keys()) == {h1.id, h2.id}


def test_put_skips_hit_without_embedding() -> None:
    """Без embedding'а scan невозможен — игнорируем."""
    hot_tier.configure("test-agent")
    h = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    h_no_embed = MemoryHit(
        id=h.id, agent_id=h.agent_id, kind=h.kind, kind_src=h.kind_src,
        role=h.role, content=h.content, metadata=h.metadata,
        created_at=h.created_at, score=h.score, match_score=h.match_score,
        embedding=None,
    )
    hot_tier.put_many("test-agent", [h_no_embed])
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert s.entries == {}


def test_put_skips_hit_without_kind_src() -> None:
    """Без kind_src snapshot fence неприменим."""
    hot_tier.configure("test-agent")
    h = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    h_no_kind_src = MemoryHit(
        id=h.id, agent_id=h.agent_id, kind=h.kind, kind_src=None,
        role=h.role, content=h.content, metadata=h.metadata,
        created_at=h.created_at, score=h.score, match_score=h.match_score,
        embedding=h.embedding,
    )
    hot_tier.put_many("test-agent", [h_no_kind_src])
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert s.entries == {}


def test_put_refreshes_evicted_at(monkeypatch: pytest.MonkeyPatch) -> None:
    hot_tier.configure("test-agent")
    h = _hit(embedding=_unit([1.0, 0.0, 0.0]))

    fake_clock = {"t": 1000.0}

    def _now() -> float:
        return fake_clock["t"]

    monkeypatch.setattr(hot_tier.time, "monotonic", _now)

    hot_tier.put_many("test-agent", [h])
    s = hot_tier.get_state("test-agent")
    assert s is not None
    initial_ts = s.entries[h.id].evicted_at
    assert initial_ts == 1000.0

    fake_clock["t"] = 1042.0
    hot_tier.put_many("test-agent", [h])
    refreshed_ts = s.entries[h.id].evicted_at
    assert refreshed_ts == 1042.0


def test_put_many_empty_no_op() -> None:
    hot_tier.configure("test-agent")
    hot_tier.put_many("test-agent", [])
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert s.entries == {}


def test_put_when_not_configured_silent() -> None:
    hot_tier.put_many("test-agent", [_hit(embedding=_unit([1.0, 0.0, 0.0]))])
    assert hot_tier.get_state("test-agent") is None


# -- LRU eviction ---------------------------------------------------------


def test_lru_overflow_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    hot_tier.configure("test-agent", lru_bound=3)

    fake_clock = {"t": 1000.0}
    monkeypatch.setattr(hot_tier.time, "monotonic", lambda: fake_clock["t"])

    h1 = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    h2 = _hit(embedding=_unit([0.0, 1.0, 0.0]))
    h3 = _hit(embedding=_unit([0.0, 0.0, 1.0]))
    h4 = _hit(embedding=_unit([1.0, 1.0, 0.0]))

    hot_tier.put_many("test-agent", [h1])
    fake_clock["t"] = 1010.0
    hot_tier.put_many("test-agent", [h2])
    fake_clock["t"] = 1020.0
    hot_tier.put_many("test-agent", [h3])
    fake_clock["t"] = 1030.0
    hot_tier.put_many("test-agent", [h4])  # overflow — h1 (oldest) выселяется

    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert h1.id not in s.entries
    assert {h2.id, h3.id, h4.id} == set(s.entries.keys())


def test_lru_keeps_refreshed_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh старого item'а в put_many двигает его вверх в LRU."""
    hot_tier.configure("test-agent", lru_bound=3)

    fake_clock = {"t": 1000.0}
    monkeypatch.setattr(hot_tier.time, "monotonic", lambda: fake_clock["t"])

    h1 = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    h2 = _hit(embedding=_unit([0.0, 1.0, 0.0]))
    h3 = _hit(embedding=_unit([0.0, 0.0, 1.0]))
    h4 = _hit(embedding=_unit([1.0, 1.0, 0.0]))

    hot_tier.put_many("test-agent", [h1])
    fake_clock["t"] = 1010.0
    hot_tier.put_many("test-agent", [h2])
    fake_clock["t"] = 1020.0
    hot_tier.put_many("test-agent", [h3])
    fake_clock["t"] = 1025.0
    hot_tier.put_many("test-agent", [h1])  # refresh h1 → у h1 evicted_at=1025
    fake_clock["t"] = 1030.0
    hot_tier.put_many("test-agent", [h4])  # overflow — h2 (oldest) выселяется

    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert h2.id not in s.entries
    assert {h1.id, h3.id, h4.id} == set(s.entries.keys())


# -- scan_candidates ------------------------------------------------------


def test_scan_returns_cosine_above_threshold() -> None:
    hot_tier.configure("test-agent")
    h = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    hot_tier.put_many("test-agent", [h])
    candidates = hot_tier.scan_candidates("test-agent", _unit([1.0, 0.0, 0.0]), min_score=0.5)
    assert len(candidates) == 1
    assert candidates[0].id == h.id
    assert math.isclose(candidates[0].score, 1.0, abs_tol=1e-9)
    assert math.isclose(candidates[0].match_score, 1.0, abs_tol=1e-9)


def test_scan_filters_below_threshold() -> None:
    hot_tier.configure("test-agent")
    h = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    hot_tier.put_many("test-agent", [h])
    # Ортогональный query → cosine=0 → ниже min_score=0.1 → отсеян.
    candidates = hot_tier.scan_candidates("test-agent", _unit([0.0, 1.0, 0.0]), min_score=0.1)
    assert candidates == []


def test_scan_sorts_by_cosine_desc() -> None:
    hot_tier.configure("test-agent")
    h_close = _hit(embedding=_unit([1.0, 0.0, 0.0]), content="close")
    h_far = _hit(embedding=_unit([0.5, 0.5, 0.0]), content="far")
    hot_tier.put_many("test-agent", [h_far, h_close])
    candidates = hot_tier.scan_candidates("test-agent", _unit([1.0, 0.0, 0.0]), min_score=0.0)
    assert candidates[0].id == h_close.id
    assert candidates[1].id == h_far.id


def test_scan_purges_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    hot_tier.configure("test-agent", ttl_s=10.0)
    fake_clock = {"t": 1000.0}
    monkeypatch.setattr(hot_tier.time, "monotonic", lambda: fake_clock["t"])
    h = _hit(embedding=_unit([1.0, 0.0, 0.0]))
    hot_tier.put_many("test-agent", [h])

    # В пределах TTL — возвращается.
    fake_clock["t"] = 1009.0
    assert len(hot_tier.scan_candidates("test-agent", _unit([1.0, 0.0, 0.0]), min_score=0.0)) == 1

    # За TTL — purged.
    fake_clock["t"] = 1011.0
    assert hot_tier.scan_candidates("test-agent", _unit([1.0, 0.0, 0.0]), min_score=0.0) == []
    s = hot_tier.get_state("test-agent")
    assert s is not None
    assert h.id not in s.entries


def test_scan_disabled_returns_empty() -> None:
    """Без configure() — scan возвращает пустоту."""
    assert hot_tier.scan_candidates("test-agent", _unit([1.0, 0.0, 0.0]), min_score=0.0) == []


# -- snapshot fence -------------------------------------------------------


def _snap(cycle_start: _dt.datetime, agent_id: str = "agent-a") -> RecallSnapshot:
    return RecallSnapshot(cycle_start=cycle_start, agent_id=agent_id)


def test_snapshot_keeps_objective_before_cycle_start() -> None:
    hot_tier.configure("test-agent")
    cycle = _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    h = _hit(
        embedding=_unit([1.0, 0.0, 0.0]),
        kind_src="objective",
        created_at=cycle - _dt.timedelta(minutes=5),
    )
    hot_tier.put_many("test-agent", [h])
    candidates = hot_tier.scan_candidates("test-agent", 
        _unit([1.0, 0.0, 0.0]), min_score=0.0, snapshot=_snap(cycle)
    )
    assert len(candidates) == 1


def test_snapshot_skips_objective_after_cycle_start() -> None:
    hot_tier.configure("test-agent")
    cycle = _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    h = _hit(
        embedding=_unit([1.0, 0.0, 0.0]),
        kind_src="objective",
        created_at=cycle + _dt.timedelta(minutes=5),
    )
    hot_tier.put_many("test-agent", [h])
    candidates = hot_tier.scan_candidates("test-agent", 
        _unit([1.0, 0.0, 0.0]), min_score=0.0, snapshot=_snap(cycle)
    )
    assert candidates == []


def test_snapshot_keeps_subjective_after_cycle_start_for_same_agent() -> None:
    """Subjective записи текущего агента видны вне snapshot'а («я положил, я помню»)."""
    hot_tier.configure("test-agent")
    cycle = _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    h = _hit(
        embedding=_unit([1.0, 0.0, 0.0]),
        kind_src="subjective",
        agent_id="agent-a",
        created_at=cycle + _dt.timedelta(minutes=5),
    )
    hot_tier.put_many("test-agent", [h])
    candidates = hot_tier.scan_candidates("test-agent", 
        _unit([1.0, 0.0, 0.0]),
        min_score=0.0,
        snapshot=_snap(cycle, agent_id="agent-a"),
    )
    assert len(candidates) == 1


def test_snapshot_skips_subjective_after_cycle_start_for_other_agent() -> None:
    hot_tier.configure("test-agent")
    cycle = _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    h = _hit(
        embedding=_unit([1.0, 0.0, 0.0]),
        kind_src="subjective",
        agent_id="agent-b",
        created_at=cycle + _dt.timedelta(minutes=5),
    )
    hot_tier.put_many("test-agent", [h])
    candidates = hot_tier.scan_candidates("test-agent", 
        _unit([1.0, 0.0, 0.0]),
        min_score=0.0,
        snapshot=_snap(cycle, agent_id="agent-a"),
    )
    assert candidates == []


def test_snapshot_subjective_tail_treated_like_subjective() -> None:
    hot_tier.configure("test-agent")
    cycle = _dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    h = _hit(
        embedding=_unit([1.0, 0.0, 0.0]),
        kind_src="subjective_tail",
        agent_id="agent-a",
        created_at=cycle + _dt.timedelta(minutes=5),
    )
    hot_tier.put_many("test-agent", [h])
    candidates = hot_tier.scan_candidates("test-agent", 
        _unit([1.0, 0.0, 0.0]),
        min_score=0.0,
        snapshot=_snap(cycle, agent_id="agent-a"),
    )
    assert len(candidates) == 1


# -- stats ----------------------------------------------------------------


def test_stats_disabled() -> None:
    assert hot_tier.stats("test-agent") == {"enabled": 0, "size": 0, "ttl_s": 0, "lru_bound": 0}


def test_stats_enabled() -> None:
    hot_tier.configure("test-agent", ttl_s=120.0, lru_bound=42)
    hot_tier.put_many("test-agent", [_hit(embedding=_unit([1.0, 0.0, 0.0]))])
    s = hot_tier.stats("test-agent")
    assert s["enabled"] == 1
    assert s["size"] == 1
    assert s["ttl_s"] == 120
    assert s["lru_bound"] == 42
