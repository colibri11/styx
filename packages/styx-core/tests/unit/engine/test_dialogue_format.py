"""Unit-тесты для ``engine/dialogue_format.py`` (волна 24).

Pure-функция format_transcript_line: speaker mapping, ISO timestamp
без 'T' и без миллисекунд.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from styx.engine.dialogue_format import format_transcript_line


def test_user_role_maps_to_human() -> None:
    ts = _dt.datetime(2026, 5, 5, 12, 34, 56)
    line = format_transcript_line("user", "привет", ts)
    assert line == "[2026-05-05 12:34:56] Human: привет"


def test_assistant_role_maps_to_agent() -> None:
    ts = _dt.datetime(2026, 5, 5, 12, 35, 1)
    line = format_transcript_line("assistant", "и тебе привет", ts)
    assert line == "[2026-05-05 12:35:01] Agent: и тебе привет"


def test_unknown_role_raises() -> None:
    ts = _dt.datetime(2026, 5, 5, 12, 0, 0)
    with pytest.raises(ValueError, match="unsupported role"):
        format_transcript_line("system", "noop", ts)
    with pytest.raises(ValueError):
        format_transcript_line("tool", "noop", ts)
    with pytest.raises(ValueError):
        format_transcript_line("summary", "noop", ts)


def test_microseconds_truncated() -> None:
    ts = _dt.datetime(2026, 5, 5, 12, 34, 56, 123456)
    line = format_transcript_line("user", "x", ts)
    # microsecond=0 → отсутствие .123456 в выводе.
    assert line == "[2026-05-05 12:34:56] Human: x"


def test_aware_timestamp_converted_to_utc() -> None:
    # +03:00 → UTC 09:34:56
    msk = _dt.timezone(_dt.timedelta(hours=3))
    ts = _dt.datetime(2026, 5, 5, 12, 34, 56, tzinfo=msk)
    line = format_transcript_line("user", "x", ts)
    assert line == "[2026-05-05 09:34:56] Human: x"


def test_separator_is_space_not_T() -> None:
    ts = _dt.datetime(2026, 1, 1, 0, 0, 0)
    line = format_transcript_line("assistant", "ok", ts)
    assert "T" not in line.split("] ", 1)[0]


def test_content_with_newlines_preserved() -> None:
    ts = _dt.datetime(2026, 5, 5, 12, 0, 0)
    line = format_transcript_line("user", "первая\nвторая", ts)
    assert line == "[2026-05-05 12:00:00] Human: первая\nвторая"
