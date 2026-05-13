"""Unit-тесты для reinterpret engine (волна 22).

Pure-функции `blend_embeddings` + `reinterpret_cooldown` без
Postgres'а через _FakeQueries stub.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from styx.engine.reinterpret import (
    DEFAULT_BLEND_WEIGHT,
    REINTERPRET_COOLDOWN_S,
    BlendError,
    CooldownCheck,
    ReinterpretConfig,
    blend_embeddings,
    reinterpret_cooldown,
)


# ── blend_embeddings ──────────────────────────────────────────────────


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def test_blend_default_weight_05_l2_normalised() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    out = blend_embeddings(a, b)
    # (1-0.5)*a + 0.5*b = (0.5, 0.5, 0); norm = sqrt(0.5)
    expected = [0.5 / math.sqrt(0.5), 0.5 / math.sqrt(0.5), 0.0]
    for got, exp in zip(out, expected):
        assert got == pytest.approx(exp, abs=1e-9)
    assert _norm(out) == pytest.approx(1.0, abs=1e-9)


def test_blend_weight_zero_returns_normalised_prev() -> None:
    a = [3.0, 4.0, 0.0]  # ||a|| = 5
    b = [1.0, 0.0, 0.0]
    out = blend_embeddings(a, b, weight=0.0)
    # result = a / ||a||
    assert out == pytest.approx([0.6, 0.8, 0.0], abs=1e-9)


def test_blend_weight_one_returns_normalised_next() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 3.0, 4.0]
    out = blend_embeddings(a, b, weight=1.0)
    assert out == pytest.approx([0.0, 0.6, 0.8], abs=1e-9)


def test_blend_invalid_weight_raises() -> None:
    with pytest.raises(BlendError) as exc:
        blend_embeddings([1.0], [1.0], weight=1.5)
    assert exc.value.code == "invalid_weight"

    with pytest.raises(BlendError) as exc:
        blend_embeddings([1.0], [1.0], weight=-0.1)
    assert exc.value.code == "invalid_weight"

    with pytest.raises(BlendError) as exc:
        blend_embeddings([1.0], [1.0], weight=float("nan"))
    assert exc.value.code == "invalid_weight"


def test_blend_empty_vector_raises() -> None:
    with pytest.raises(BlendError) as exc:
        blend_embeddings([], [1.0])
    assert exc.value.code == "empty_vector"

    with pytest.raises(BlendError) as exc:
        blend_embeddings([1.0], [])
    assert exc.value.code == "empty_vector"


def test_blend_dim_mismatch_raises() -> None:
    with pytest.raises(BlendError) as exc:
        blend_embeddings([1.0, 2.0], [1.0])
    assert exc.value.code == "dim_mismatch"


def test_blend_antipodal_at_w05_raises_zero_result() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    with pytest.raises(BlendError) as exc:
        blend_embeddings(a, b, weight=0.5)
    assert exc.value.code == "zero_result"


def test_blend_default_weight_constant() -> None:
    assert DEFAULT_BLEND_WEIGHT == 0.5


# ── reinterpret_cooldown ──────────────────────────────────────────────


class _FakeQueries:
    def __init__(
        self,
        *,
        pending_id: int | None = None,
        last_at: datetime | None = None,
    ) -> None:
        self._pending_id = pending_id
        self._last_at = last_at
        self.calls_pending: list[uuid.UUID] = []
        self.calls_latest: list[uuid.UUID] = []

    def find_pending_reinterpret_application(
        self, memory_id: uuid.UUID
    ) -> int | None:
        self.calls_pending.append(memory_id)
        return self._pending_id

    def latest_reinterpretation_at(
        self, memory_id: uuid.UUID
    ) -> datetime | None:
        self.calls_latest.append(memory_id)
        return self._last_at


def _now() -> datetime:
    return datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)


def test_cooldown_no_history_returns_ok() -> None:
    q = _FakeQueries()
    mid = uuid.uuid4()
    res = reinterpret_cooldown(q, mid, now=_now())
    assert res.ok is True
    assert res.reason is None
    # Только partial-pending запрашивается + latest.
    assert q.calls_pending == [mid]
    assert q.calls_latest == [mid]


def test_cooldown_pending_takes_priority_over_recent() -> None:
    # Даже если последняя revision была давно, pending-row блокирует.
    long_ago = _now() - timedelta(days=30)
    q = _FakeQueries(pending_id=42, last_at=long_ago)
    res = reinterpret_cooldown(q, uuid.uuid4(), now=_now())
    assert res.ok is False
    assert res.reason == "pending"
    assert res.pending_application_id == 42
    # Latest даже не запрашивается — short-circuit.
    assert q.calls_latest == []


def test_cooldown_recent_within_24h_blocks() -> None:
    last = _now() - timedelta(hours=23)
    q = _FakeQueries(last_at=last)
    res = reinterpret_cooldown(q, uuid.uuid4(), now=_now())
    assert res.ok is False
    assert res.reason == "recent"
    assert res.last_at == last
    assert res.next_at == last + timedelta(seconds=REINTERPRET_COOLDOWN_S)


def test_cooldown_24h_boundary_returns_ok() -> None:
    # Ровно 24h назад → elapsed == cooldown → ok=True.
    last = _now() - timedelta(seconds=REINTERPRET_COOLDOWN_S)
    q = _FakeQueries(last_at=last)
    res = reinterpret_cooldown(q, uuid.uuid4(), now=_now())
    assert res.ok is True


def test_cooldown_naive_datetime_treated_as_utc() -> None:
    last_naive = _now().replace(tzinfo=None) - timedelta(hours=10)
    q = _FakeQueries(last_at=last_naive)
    res = reinterpret_cooldown(q, uuid.uuid4(), now=_now())
    assert res.ok is False
    assert res.reason == "recent"


def test_cooldown_custom_cooldown_s() -> None:
    last = _now() - timedelta(hours=2)
    q = _FakeQueries(last_at=last)
    # 1h cooldown — 2h > 1h → ok.
    res = reinterpret_cooldown(q, uuid.uuid4(), now=_now(), cooldown_s=3600)
    assert res.ok is True
    # 4h cooldown — 2h < 4h → blocked.
    res2 = reinterpret_cooldown(q, uuid.uuid4(), now=_now(), cooldown_s=14400)
    assert res2.ok is False


# ── ReinterpretConfig defaults ────────────────────────────────────────


def test_reinterpret_config_defaults() -> None:
    cfg = ReinterpretConfig()
    assert cfg.enabled is True
    assert cfg.apply_tick_s == 30.0
    assert cfg.cooldown_s == 86400
    assert cfg.blend_weight == 0.5


# ── CooldownCheck factory ─────────────────────────────────────────────


def test_cooldown_check_make_ok() -> None:
    c = CooldownCheck.make_ok()
    assert c.ok is True
    assert c.reason is None
    assert c.last_at is None
    assert c.next_at is None
    assert c.pending_application_id is None


def test_cooldown_check_make_pending() -> None:
    c = CooldownCheck.make_pending(pending_application_id=7)
    assert c.ok is False
    assert c.reason == "pending"
    assert c.pending_application_id == 7
