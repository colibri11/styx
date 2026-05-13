"""Юнит-тесты hot_tier.restore / snapshot (волна 13)."""

from __future__ import annotations

import datetime as _dt
import time
import uuid

import pytest

from styx.engine import hot_tier
from styx.engine.hot_tier import HotEntry


@pytest.fixture(autouse=True)
def _reset_state():
    hot_tier.reset_all()
    yield
    hot_tier.reset_all()


def _entry(*, agent_id: str = "agent-a", evicted_at: float | None = None) -> HotEntry:
    return HotEntry(
        id=uuid.uuid4(),
        agent_id=agent_id,
        kind="subjective_dialogue",
        kind_src="subjective",
        role="user",
        content="content",
        metadata={"k": "v"},
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
        embedding=[0.1, 0.2, 0.3],
        evicted_at=evicted_at if evicted_at is not None else time.monotonic(),
    )


def test_restore_without_configure_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="styx.engine.hot_tier")
    hot_tier.restore("test-agent", [_entry()])
    assert hot_tier.get_state("test-agent") is None
    assert any(
        "не configured" in r.message
        for r in caplog.records
    )


def test_restore_replaces_existing_entries() -> None:
    hot_tier.configure("test-agent", ttl_s=300.0, lru_bound=100)
    e1 = _entry()
    e2 = _entry()
    hot_tier.restore("test-agent", [e1, e2])
    state = hot_tier.get_state("test-agent")
    assert state is not None
    assert set(state.entries.keys()) == {e1.id, e2.id}


def test_restore_enforces_lru_bound() -> None:
    hot_tier.configure("test-agent", ttl_s=300.0, lru_bound=2)
    base = time.monotonic()
    entries = [
        _entry(evicted_at=base - 10),  # самый старый — выселится
        _entry(evicted_at=base - 5),
        _entry(evicted_at=base - 1),
    ]
    hot_tier.restore("test-agent", entries)
    state = hot_tier.get_state("test-agent")
    assert state is not None
    assert len(state.entries) == 2
    # Остались два самых свежих
    assert entries[1].id in state.entries
    assert entries[2].id in state.entries
    assert entries[0].id not in state.entries


def test_snapshot_empty_when_not_configured() -> None:
    assert hot_tier.snapshot("test-agent") == []


def test_snapshot_empty_state_returns_empty_list() -> None:
    hot_tier.configure("test-agent", ttl_s=300.0, lru_bound=100)
    assert hot_tier.snapshot("test-agent") == []


def test_snapshot_returns_entries_independent_of_state() -> None:
    hot_tier.configure("test-agent", ttl_s=300.0, lru_bound=100)
    e1 = _entry()
    e2 = _entry()
    hot_tier.restore("test-agent", [e1, e2])
    snap = hot_tier.snapshot("test-agent")
    assert {e.id for e in snap} == {e1.id, e2.id}
    # Mutating state.entries не должно затрагивать snap (shallow copy).
    state = hot_tier.get_state("test-agent")
    assert state is not None
    state.entries.clear()
    assert {e.id for e in snap} == {e1.id, e2.id}
