"""Юнит-тесты turn_state — observe / close / TTL / sticky semantics."""

from __future__ import annotations

import datetime as _dt

import pytest

from styx import turn_state


@pytest.fixture(autouse=True)
def _reset() -> None:
    turn_state.reset()
    yield
    turn_state.reset()


_BASE = _dt.datetime(2026, 5, 2, 12, 0, 0, tzinfo=_dt.timezone.utc)


def _t(seconds: int = 0) -> _dt.datetime:
    """Deterministic timestamp helper (seconds offset from _BASE)."""
    return _BASE + _dt.timedelta(seconds=seconds)


def test_peek_none_initially() -> None:
    assert turn_state.peek("alpha") is None
    assert turn_state.is_active("alpha") is False


def test_observe_first_call_opens_turn() -> None:
    turn_state.configure(ttl_s=60.0)
    snap = turn_state.observe("alpha", now=_t(0))
    assert snap.cycle_start == _t(0)
    assert snap.agent_id == "alpha"
    assert turn_state.is_active("alpha", now=_t(30)) is True


def test_observe_sticky_within_ttl() -> None:
    turn_state.configure(ttl_s=60.0)
    snap1 = turn_state.observe("alpha", now=_t(0))
    snap2 = turn_state.observe("alpha", now=_t(30))
    snap3 = turn_state.observe("alpha", now=_t(59))
    # cycle_start не меняется — sticky.
    assert snap1.cycle_start == snap2.cycle_start == snap3.cycle_start


def test_observe_opens_new_turn_after_ttl() -> None:
    turn_state.configure(ttl_s=60.0)
    snap1 = turn_state.observe("alpha", now=_t(0))
    snap2 = turn_state.observe("alpha", now=_t(120))  # > 60s от _t(0)
    assert snap2.cycle_start != snap1.cycle_start
    assert snap2.cycle_start == _t(120)


def test_close_clears_turn() -> None:
    turn_state.configure(ttl_s=60.0)
    turn_state.observe("alpha", now=_t(0))
    assert turn_state.is_active("alpha", now=_t(10)) is True
    turn_state.close("alpha")
    assert turn_state.is_active("alpha", now=_t(10)) is False


def test_observe_after_close_opens_new_turn() -> None:
    turn_state.configure(ttl_s=60.0)
    snap1 = turn_state.observe("alpha", now=_t(0))
    turn_state.close("alpha")
    snap2 = turn_state.observe("alpha", now=_t(5))
    assert snap2.cycle_start == _t(5)
    assert snap2.cycle_start != snap1.cycle_start


def test_close_idempotent_on_unknown_agent() -> None:
    turn_state.close("never-seen")  # не падает


def test_concurrent_agents_independent() -> None:
    turn_state.configure(ttl_s=60.0)
    snap_a = turn_state.observe("alpha", now=_t(0))
    snap_b = turn_state.observe("beta", now=_t(10))
    # Каждый имеет свой cycle_start.
    assert snap_a.cycle_start == _t(0)
    assert snap_b.cycle_start == _t(10)
    turn_state.close("alpha")
    # beta активен.
    assert turn_state.is_active("beta", now=_t(15)) is True
    assert turn_state.is_active("alpha", now=_t(15)) is False


def test_configure_resets_states() -> None:
    turn_state.configure(ttl_s=60.0)
    turn_state.observe("alpha", now=_t(0))
    turn_state.configure(ttl_s=30.0)  # пересоздание
    assert turn_state.peek("alpha") is None


def test_configure_validates_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        turn_state.configure(ttl_s=0)
    with pytest.raises(ValueError, match="ttl_s"):
        turn_state.configure(ttl_s=-1.0)


def test_reset_clears_all_states() -> None:
    turn_state.configure(ttl_s=60.0)
    turn_state.observe("alpha", now=_t(0))
    turn_state.observe("beta", now=_t(0))
    turn_state.reset()
    assert turn_state.peek("alpha") is None
    assert turn_state.peek("beta") is None
