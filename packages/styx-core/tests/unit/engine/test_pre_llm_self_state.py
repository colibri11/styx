"""Юнит-тесты channel self_state — все skip-условия + 8 октантов."""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from styx.emotional.state import EmotionalVector
from styx.engine.pre_llm_channels import self_state
from styx.engine.pre_llm_channels.self_state import OCTANTS, channel_self_state
from styx.engine.pre_llm_inject import ChannelHandle


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _entry(
    v: float, a: float, d: float, age_s: float = 0.0
) -> tuple[EmotionalVector, _dt.datetime]:
    at = _now() - _dt.timedelta(seconds=age_s)
    return (EmotionalVector(v, a, d), at)


class _StubQueries:
    """Queries stub с настраиваемым результатом get_last_emotional_state."""

    def __init__(
        self,
        latest: tuple[EmotionalVector, _dt.datetime] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._latest = latest
        self._raise = raise_exc
        self.calls = 0

    def get_last_emotional_state(self):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._latest


def _handle(queries, **overrides) -> ChannelHandle:
    base = dict(
        queries=queries,
        self_state_enabled=True,
        self_state_min_norm=0.2,
        self_state_max_age_s=900.0,
    )
    base.update(overrides)
    return ChannelHandle(**base)


# -- skip conditions ------------------------------------------------------


def test_skip_when_channel_disabled() -> None:
    h = _handle(_StubQueries(latest=_entry(0.9, 0.8, 0.7)),
                self_state_enabled=False)
    assert channel_self_state(h, {}) is None
    assert h.queries.calls == 0  # query даже не делается


def test_skip_when_no_state_recorded() -> None:
    h = _handle(_StubQueries(latest=None))
    assert channel_self_state(h, {}) is None


def test_skip_when_query_fails(caplog) -> None:
    h = _handle(_StubQueries(raise_exc=RuntimeError("conn dead")))
    with caplog.at_level("WARNING"):
        assert channel_self_state(h, {}) is None
    assert any("self_state" in rec.getMessage() for rec in caplog.records)


def test_skip_when_state_too_stale(caplog) -> None:
    """age > self_state_max_age_s — safety net на мёртвый воркер, WARNING."""
    h = _handle(_StubQueries(latest=_entry(0.9, 0.8, 0.7, age_s=901.0)))
    with caplog.at_level("WARNING"):
        assert channel_self_state(h, {}) is None
    assert any("self_state" in rec.getMessage() for rec in caplog.records)


def test_does_not_skip_when_state_fresh_enough() -> None:
    """age чуть меньше max_age_s — НЕ должен скипать по staleness."""
    h = _handle(_StubQueries(latest=_entry(0.9, 0.8, 0.7, age_s=899.0)))
    assert channel_self_state(h, {}) is not None


def test_skip_when_norm_below_threshold() -> None:
    # norm = sqrt(0.05² + 0.05² + 0.05²) ≈ 0.087, < 0.2
    h = _handle(_StubQueries(latest=_entry(0.05, 0.05, 0.05)))
    assert channel_self_state(h, {}) is None


def test_skip_when_norm_well_below_threshold() -> None:
    # Заметно ниже threshold (norm ≈ 0.141 против порога 0.2) — не
    # точная граница; см. отдельные boundary-тесты ниже для случая
    # norm == self_state_min_norm ровно.
    h = _handle(_StubQueries(latest=_entry(0.1, 0.1, 0.0)),
                self_state_min_norm=0.2)
    norm = math.sqrt(0.1**2 + 0.1**2)
    assert norm < 0.2  # sanity
    assert channel_self_state(h, {}) is None


def test_boundary_norm_exactly_at_min_norm_does_not_skip() -> None:
    """(a) norm == self_state_min_norm (0.2 ровно). ``channel_self_state``
    скипает по норме через строгое ``norm < handle.self_state_min_norm``
    — значит ровно на границе (norm == порог) скипа быть не должно."""
    h = _handle(_StubQueries(latest=_entry(0.2, 0.0, 0.0)),
                self_state_min_norm=0.2)
    norm = math.sqrt(0.2**2 + 0.0**2 + 0.0**2)
    assert norm == 0.2  # sanity: граница ровно совпадает, не приближение
    assert channel_self_state(h, {}) is not None


def test_boundary_age_exactly_at_max_age_does_not_skip(monkeypatch) -> None:
    """(б) age == self_state_max_age_s (900.0 ровно). ``channel_self_state``
    скипает по возрасту через строгое ``age_s > handle.self_state_max_age_s``
    — значит ровно на границе (age == порог) скипа быть не должно.

    ``channel_self_state`` берёт "now" через ``datetime.datetime.now()``
    в момент вызова; реальный wall-clock между подготовкой fixture'ы и
    вызовом канала дал бы age чуть БОЛЬШЕ 900.0 (флаки-риск), поэтому
    "now" фиксируется через monkeypatch на модульный ``_dt.datetime``
    внутри ``self_state`` — так age_s получается детерминированно
    равным ровно 900.0.
    """
    fixed_now = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    at = fixed_now - _dt.timedelta(seconds=900.0)

    class _FrozenDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(self_state._dt, "datetime", _FrozenDatetime)

    h = _handle(_StubQueries(latest=(EmotionalVector(0.9, 0.8, 0.7), at)))
    assert channel_self_state(h, {}) is not None


# -- octants --------------------------------------------------------------


@pytest.mark.parametrize(
    "vad,expected_phrase",
    [
        # pos+pos+pos
        ((0.5, 0.5, 0.5), "воодушевлённо и уверенно"),
        # pos+pos+neg
        ((0.5, 0.5, -0.5), "взволнованно и радостно"),
        # pos+neg+pos
        ((0.5, -0.5, 0.5), "спокойно и удовлетворённо"),
        # pos+neg+neg
        ((0.5, -0.5, -0.5), "умиротворённо и расслабленно"),
        # neg+pos+pos
        ((-0.5, 0.5, 0.5), "напряжённо и собранно"),
        # neg+pos+neg
        ((-0.5, 0.5, -0.5), "тревожно и взволнованно"),
        # neg+neg+pos
        ((-0.5, -0.5, 0.5), "тяжело и непреклонно"),
        # neg+neg+neg
        ((-0.5, -0.5, -0.5), "устало и подавленно"),
    ],
)
def test_eight_octants_resolve_phrase(
    vad: tuple[float, float, float], expected_phrase: str,
) -> None:
    h = _handle(_StubQueries(latest=_entry(*vad)))
    out = channel_self_state(h, {})
    assert out is not None
    assert expected_phrase in out
    assert out.startswith("Тебе сейчас")
    assert out.endswith(".")


def test_octants_table_is_complete() -> None:
    """Все 8 ключей покрыты."""
    expected_keys = {
        f"{a}{b}{c}"
        for a in ("pos", "neg")
        for b in ("pos", "neg")
        for c in ("pos", "neg")
    }
    assert set(OCTANTS.keys()) == expected_keys


def test_zero_treated_as_pos() -> None:
    """sign(0) = pos (как в memorybox)."""
    # norm чуть выше threshold чтобы channel сработал
    h = _handle(_StubQueries(latest=_entry(0.0, 0.3, 0.0)),
                self_state_min_norm=0.2)
    out = channel_self_state(h, {})
    assert out is not None
    # 0+0.3+0 → pospospos
    assert "воодушевлённо и уверенно" in out
