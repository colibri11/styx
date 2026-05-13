"""Юнит-тесты channel B (peer_vad) — все skip-условия + 8 октантов."""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from styx.engine.pre_llm_channels.peer_vad import OCTANTS, channel_peer_vad
from styx.engine.pre_llm_inject import ChannelHandle


class _StubQueries:
    """Queries stub с настраиваемым результатом get_latest_hot_sentiment."""

    def __init__(
        self,
        latest: tuple[tuple[float, float, float], _dt.datetime] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._latest = latest
        self._raise = raise_exc
        self.calls: list[float] = []

    def get_latest_hot_sentiment(self, *, within_seconds: float):
        self.calls.append(within_seconds)
        if self._raise is not None:
            raise self._raise
        return self._latest


def _handle(queries, **overrides) -> ChannelHandle:
    base = dict(
        queries=queries,
        peer_vad_enabled=True,
        peer_vad_min_norm=0.2,
        peer_vad_ttl_s=60.0,
    )
    base.update(overrides)
    return ChannelHandle(**base)


# -- skip conditions ------------------------------------------------------


def test_skip_when_channel_disabled() -> None:
    h = _handle(_StubQueries(latest=((0.9, 0.8, 0.7), _dt.datetime.now())),
                peer_vad_enabled=False)
    assert channel_peer_vad(h, {}) is None
    assert h.queries.calls == []  # query даже не делается


def test_skip_when_no_recent_entry() -> None:
    h = _handle(_StubQueries(latest=None))
    assert channel_peer_vad(h, {}) is None


def test_skip_when_query_fails(caplog) -> None:
    h = _handle(_StubQueries(raise_exc=RuntimeError("conn dead")))
    with caplog.at_level("WARNING"):
        assert channel_peer_vad(h, {}) is None
    assert any("peer_vad" in rec.getMessage() for rec in caplog.records)


def test_skip_when_norm_below_threshold() -> None:
    # norm = sqrt(0.05² + 0.05² + 0.05²) ≈ 0.087, < 0.2
    h = _handle(_StubQueries(latest=((0.05, 0.05, 0.05), _dt.datetime.now())))
    assert channel_peer_vad(h, {}) is None


def test_skip_when_norm_exactly_at_threshold_minus_epsilon() -> None:
    # точно ниже threshold
    h = _handle(_StubQueries(latest=((0.1, 0.1, 0.0), _dt.datetime.now())),
                peer_vad_min_norm=0.2)
    norm = math.sqrt(0.1**2 + 0.1**2)
    assert norm < 0.2  # sanity
    assert channel_peer_vad(h, {}) is None


def test_passes_ttl_to_queries() -> None:
    h = _handle(_StubQueries(latest=None), peer_vad_ttl_s=120.0)
    channel_peer_vad(h, {})
    assert h.queries.calls == [120.0]


# -- octants --------------------------------------------------------------


@pytest.mark.parametrize(
    "vad,expected_phrase",
    [
        # pos+pos+pos
        ((0.5, 0.5, 0.5), "оживлённо и уверенно"),
        # pos+pos+neg
        ((0.5, 0.5, -0.5), "взволнованно и радостно"),
        # pos+neg+pos
        ((0.5, -0.5, 0.5), "спокойно и удовлетворённо"),
        # pos+neg+neg
        ((0.5, -0.5, -0.5), "мягко и расслабленно"),
        # neg+pos+pos
        ((-0.5, 0.5, 0.5), "напряжённо и собранно"),
        # neg+pos+neg
        ((-0.5, 0.5, -0.5), "тревожно и взволнованно"),
        # neg+neg+pos
        ((-0.5, -0.5, 0.5), "сдержанно и тяжело"),
        # neg+neg+neg
        ((-0.5, -0.5, -0.5), "устало и подавленно"),
    ],
)
def test_eight_octants_resolve_phrase(
    vad: tuple[float, float, float], expected_phrase: str,
) -> None:
    h = _handle(_StubQueries(latest=(vad, _dt.datetime.now())))
    out = channel_peer_vad(h, {})
    assert out is not None
    assert expected_phrase in out
    assert out.startswith("Peer прозвучал:")
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
    h = _handle(_StubQueries(latest=((0.0, 0.3, 0.0), _dt.datetime.now())),
                peer_vad_min_norm=0.2)
    out = channel_peer_vad(h, {})
    assert out is not None
    # 0+0.3+0 → pospospos
    assert "оживлённо и уверенно" in out
