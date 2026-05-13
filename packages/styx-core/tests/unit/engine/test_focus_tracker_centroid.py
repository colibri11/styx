"""Юнит-тесты focus_tracker.get_centroid (волна 12)."""

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
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def test_get_centroid_none_when_not_configured() -> None:
    assert focus_tracker.get_centroid("test-agent") is None


def test_get_centroid_none_when_window_empty() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    assert focus_tracker.get_centroid("test-agent") is None


def test_get_centroid_equals_single_embed() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e = _unit([1.0, 0.0, 0.0])
    focus_tracker.observe("test-agent", e)
    centroid = focus_tracker.get_centroid("test-agent")
    assert centroid is not None
    assert all(abs(a - b) < 1e-9 for a, b in zip(centroid, e))


def test_get_centroid_is_mean_of_window() -> None:
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    e1 = _unit([1.0, 0.0, 0.0])
    e2 = _unit([0.0, 1.0, 0.0])
    focus_tracker.observe("test-agent", e1)
    focus_tracker.observe("test-agent", e2)
    centroid = focus_tracker.get_centroid("test-agent")
    assert centroid is not None
    expected = [(a + b) / 2.0 for a, b in zip(e1, e2)]
    assert all(abs(a - b) < 1e-9 for a, b in zip(centroid, expected))
