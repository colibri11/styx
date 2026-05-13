"""Юнит-тесты focus_tracker — observe / drift signal, FIFO, cache, reset."""

from __future__ import annotations

import math

import pytest

from styx.engine import focus_tracker


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    focus_tracker.reset_all()
    yield
    focus_tracker.reset_all()


def _unit(values: list[float]) -> list[float]:
    """Нормализация к unit length для воспроизводимых cosine."""
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


# -- configure / get_state / reset ----------------------------------------


def test_get_state_none_initially() -> None:
    assert focus_tracker.get_state("test-agent") is None


def test_configure_initializes_empty_state() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.window == []
    assert state.cached_salient is None
    assert state.epoch_id == 0


def test_configure_validates_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        focus_tracker.configure("test-agent", window_size=0)


def test_configure_validates_drift_threshold() -> None:
    with pytest.raises(ValueError, match="drift_threshold"):
        focus_tracker.configure("test-agent", drift_threshold=1.5)
    with pytest.raises(ValueError, match="drift_threshold"):
        focus_tracker.configure("test-agent", drift_threshold=-0.1)


def test_double_configure_resets_state() -> None:
    focus_tracker.configure("test-agent")
    focus_tracker.observe("test-agent", _unit([1.0, 0.0, 0.0]))
    focus_tracker.set_cached("test-agent", {"role": "user", "content": "X"})
    focus_tracker.configure("test-agent")
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.window == []
    assert state.cached_salient is None
    assert state.epoch_id == 0


def test_reset_clears_state() -> None:
    focus_tracker.configure("test-agent")
    focus_tracker.observe("test-agent", _unit([1.0, 0.0, 0.0]))
    focus_tracker.reset_all()
    assert focus_tracker.get_state("test-agent") is None


# -- observe / drift signal -----------------------------------------------


def test_observe_empty_window_is_drift() -> None:
    """Первый observe всегда возвращает drift=True (нет фокуса для сравнения)."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    drift = focus_tracker.observe("test-agent", _unit([1.0, 0.0, 0.0]))
    assert drift is True
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert len(state.window) == 1
    assert state.epoch_id == 1


def test_observe_same_embed_no_drift() -> None:
    """Идентичные embed'ы → cosine=1 → no drift."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e = _unit([1.0, 0.0, 0.0])
    focus_tracker.observe("test-agent", e)  # warm-up: первый всегда drift
    drift = focus_tracker.observe("test-agent", e)
    assert drift is False


def test_observe_orthogonal_embed_is_drift() -> None:
    """Ортогональные embed'ы → cosine=0 → drift (0 < 0.4)."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e1 = _unit([1.0, 0.0, 0.0])
    e2 = _unit([0.0, 1.0, 0.0])
    focus_tracker.observe("test-agent", e1)  # warm-up
    drift = focus_tracker.observe("test-agent", e2)
    assert drift is True


def test_observe_close_embed_below_threshold_is_drift() -> None:
    """Cosine ~0.3 < threshold 0.4 → drift."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e1 = _unit([1.0, 0.0, 0.0])
    # cos(e1, e2) = 0.3
    e2 = _unit([0.3, math.sqrt(1.0 - 0.09), 0.0])
    focus_tracker.observe("test-agent", e1)
    drift = focus_tracker.observe("test-agent", e2)
    assert drift is True


def test_observe_close_embed_above_threshold_no_drift() -> None:
    """Cosine ~0.7 > threshold 0.4 → no drift."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e1 = _unit([1.0, 0.0, 0.0])
    # cos(e1, e2) = 0.7
    e2 = _unit([0.7, math.sqrt(1.0 - 0.49), 0.0])
    focus_tracker.observe("test-agent", e1)
    drift = focus_tracker.observe("test-agent", e2)
    assert drift is False


def test_observe_uses_centroid_not_last_embed() -> None:
    """Centroid пересчитывается по окну, не по последнему embed'у."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e1 = _unit([1.0, 0.0, 0.0])
    focus_tracker.observe("test-agent", e1)  # window=[e1]
    focus_tracker.observe("test-agent", e1)  # window=[e1, e1], centroid=e1
    focus_tracker.observe("test-agent", e1)  # window=[e1, e1, e1], centroid=e1
    # Теперь centroid=[1,0,0]. Новый ортогональный embed → drift.
    e2 = _unit([0.0, 1.0, 0.0])
    drift = focus_tracker.observe("test-agent", e2)
    assert drift is True


def test_observe_fifo_window_drops_oldest() -> None:
    """После K+1 observe'ов окно содержит K последних embed'ов."""
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e_x = _unit([1.0, 0.0, 0.0])
    e_y = _unit([0.0, 1.0, 0.0])
    # 4 observe'а: [x, x, x, y] → окно после = [x, x, y]
    for _ in range(3):
        focus_tracker.observe("test-agent", e_x)
    focus_tracker.observe("test-agent", e_y)
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert len(state.window) == 3
    # Последний — y, два предыдущих — x.
    last = state.window[-1]
    assert all(abs(a - b) < 1e-9 for a, b in zip(last, e_y))
    assert all(abs(a - b) < 1e-9 for a, b in zip(state.window[0], e_x))


def test_observe_increments_epoch_only_on_drift() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e1 = _unit([1.0, 0.0, 0.0])
    focus_tracker.observe("test-agent", e1)  # epoch 1 (warm-up)
    focus_tracker.observe("test-agent", e1)  # no drift
    focus_tracker.observe("test-agent", e1)  # no drift
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.epoch_id == 1

    e2 = _unit([0.0, 1.0, 0.0])
    focus_tracker.observe("test-agent", e2)  # drift
    assert state.epoch_id == 2


def test_observe_when_not_configured_returns_drift() -> None:
    """observe на None state — fallback drift=True (caller сам должен проверить get_state)."""
    assert focus_tracker.get_state("test-agent") is None
    drift = focus_tracker.observe("test-agent", _unit([1.0, 0.0, 0.0]))
    assert drift is True


# -- set_cached / get_state ------------------------------------------------


def test_set_cached_persists_in_state() -> None:
    focus_tracker.configure("test-agent")
    salient = {"role": "user", "content": "[Styx] some memory"}
    focus_tracker.set_cached("test-agent", salient)
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.cached_salient is salient


def test_set_cached_none_invalidates() -> None:
    focus_tracker.configure("test-agent")
    focus_tracker.set_cached("test-agent", {"role": "user", "content": "X"})
    focus_tracker.set_cached("test-agent", None)
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.cached_salient is None


def test_set_cached_when_not_configured_is_silent() -> None:
    """Без configure() — set_cached не падает, не создаёт state."""
    focus_tracker.set_cached("test-agent", {"role": "user", "content": "X"})
    assert focus_tracker.get_state("test-agent") is None
