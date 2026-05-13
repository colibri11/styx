"""Юнит-тесты `engine/cache_stats.py` — per-agent counter (волна 29 Phase E)."""

from __future__ import annotations

import threading

import pytest

from styx.engine.cache_stats import (
    get_cache_stats,
    record_cache_stats,
    reset_cache_stats,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_cache_stats()
    yield
    reset_cache_stats()


def test_unknown_agent_returns_zero() -> None:
    s = get_cache_stats("unknown")
    assert s == {
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "samples": 0,
    }


def test_record_accumulates_per_agent() -> None:
    record_cache_stats("alpha", cache_read_tokens=100, cache_creation_tokens=20)
    record_cache_stats("alpha", cache_read_tokens=50, cache_creation_tokens=10)
    s = get_cache_stats("alpha")
    assert s["cache_read_tokens"] == 150
    assert s["cache_creation_tokens"] == 30
    assert s["samples"] == 2


def test_record_isolated_per_agent() -> None:
    record_cache_stats("alpha", cache_read_tokens=10, cache_creation_tokens=5)
    record_cache_stats("beta", cache_read_tokens=200, cache_creation_tokens=0)
    a = get_cache_stats("alpha")
    b = get_cache_stats("beta")
    assert a["cache_read_tokens"] == 10
    assert b["cache_read_tokens"] == 200
    assert a["samples"] == 1
    assert b["samples"] == 1


def test_record_zero_tokens_still_increments_samples() -> None:
    """Cache miss = 0/0 — still recorded; sample count важен для ratio."""
    record_cache_stats("alpha", cache_read_tokens=0, cache_creation_tokens=0)
    s = get_cache_stats("alpha")
    assert s["samples"] == 1
    assert s["cache_read_tokens"] == 0


def test_record_negative_clamped_to_zero() -> None:
    """Negative values — guard against caller bug."""
    record_cache_stats("alpha", cache_read_tokens=-5, cache_creation_tokens=10)
    s = get_cache_stats("alpha")
    assert s["cache_read_tokens"] == 0
    assert s["cache_creation_tokens"] == 10


def test_record_empty_agent_id_skipped() -> None:
    """Пустой agent_id — silent no-op (defensive against initialization race)."""
    record_cache_stats("", cache_read_tokens=100, cache_creation_tokens=10)
    record_cache_stats("alpha", cache_read_tokens=5, cache_creation_tokens=0)
    s = get_cache_stats("alpha")
    assert s["samples"] == 1


def test_reset_specific_agent() -> None:
    record_cache_stats("alpha", cache_read_tokens=10, cache_creation_tokens=5)
    record_cache_stats("beta", cache_read_tokens=20, cache_creation_tokens=10)
    reset_cache_stats("alpha")
    a = get_cache_stats("alpha")
    b = get_cache_stats("beta")
    assert a["samples"] == 0
    assert b["samples"] == 1


def test_thread_safe_concurrent_increments() -> None:
    """Lock защищает от race при параллельных push'ах."""
    def _push():
        for _ in range(100):
            record_cache_stats("alpha", cache_read_tokens=1, cache_creation_tokens=2)

    threads = [threading.Thread(target=_push) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = get_cache_stats("alpha")
    assert s["samples"] == 1000
    assert s["cache_read_tokens"] == 1000
    assert s["cache_creation_tokens"] == 2000
