"""Юнит-тесты focus_tracker.restore / snapshot (волна 13)."""

from __future__ import annotations

import pytest

from styx.engine import focus_tracker


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    focus_tracker.reset_all()
    yield
    focus_tracker.reset_all()


def test_restore_without_configure_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="styx.engine.focus_tracker")
    focus_tracker.restore("test-agent", 
        window=[[1.0, 2.0]],
        cached_salient={"role": "user", "content": "x"},
        epoch_id=5,
    )
    assert focus_tracker.get_state("test-agent") is None
    assert any(
        "не configured" in r.message
        for r in caplog.records
    )


def test_restore_replaces_window_and_cached_salient() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    focus_tracker.restore("test-agent", 
        window=[[0.1] * 3, [0.2] * 3, [0.3] * 3],
        cached_salient={"role": "user", "content": "salient"},
        epoch_id=7,
    )
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.window == [[0.1] * 3, [0.2] * 3, [0.3] * 3]
    assert state.cached_salient == {"role": "user", "content": "salient"}
    assert state.epoch_id == 7


def test_restore_trims_window_to_current_size_keeping_tail() -> None:
    """ENV ``STYX_FOCUS_WINDOW_SIZE`` уменьшился между restart'ами — оставляем хвост."""
    focus_tracker.configure("test-agent", window_size=2, drift_threshold=0.4)
    focus_tracker.restore("test-agent", 
        window=[[1.0], [2.0], [3.0], [4.0]],
        cached_salient=None,
        epoch_id=0,
    )
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.window == [[3.0], [4.0]]


def test_restore_clamps_negative_epoch_to_zero() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    focus_tracker.restore("test-agent", window=[], cached_salient=None, epoch_id=-1)
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.epoch_id == 0


def test_snapshot_returns_none_without_configure() -> None:
    assert focus_tracker.snapshot("test-agent") is None


def test_snapshot_empty_state() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    snap = focus_tracker.snapshot("test-agent")
    assert snap == ([], None, 0)


def test_snapshot_after_observe_and_cache() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    focus_tracker.observe("test-agent", [1.0, 0.0, 0.0])
    focus_tracker.observe("test-agent", [1.0, 0.0, 0.0])
    focus_tracker.set_cached("test-agent", {"role": "user", "content": "X"})
    snap = focus_tracker.snapshot("test-agent")
    assert snap is not None
    window, cached, epoch = snap
    assert len(window) == 2
    assert cached == {"role": "user", "content": "X"}
    assert epoch >= 1


def test_snapshot_independent_of_subsequent_mutations() -> None:
    """Snapshot не должен видеть последующие observe/set_cached."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    focus_tracker.observe("test-agent", [1.0, 0.0, 0.0])
    focus_tracker.set_cached("test-agent", {"role": "user", "content": "X"})
    snap = focus_tracker.snapshot("test-agent")
    assert snap is not None
    window_copy, cached_copy, _ = snap

    focus_tracker.observe("test-agent", [0.0, 1.0, 0.0])
    focus_tracker.set_cached("test-agent", {"role": "user", "content": "Y"})

    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert len(window_copy) == 1
    assert cached_copy == {"role": "user", "content": "X"}
    assert len(state.window) == 2
    assert state.cached_salient == {"role": "user", "content": "Y"}


def test_restore_preserves_cached_salient_independence() -> None:
    """Изменение оригинального dict cached_salient не должно мутировать state."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    salient = {"role": "user", "content": "X"}
    focus_tracker.restore("test-agent", window=[], cached_salient=salient, epoch_id=0)
    salient["content"] = "Y"
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.cached_salient == {"role": "user", "content": "X"}
