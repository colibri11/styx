"""Unit-тесты для RecallTracker."""

from __future__ import annotations

import uuid

import pytest

from styx.providers.recall_tracker import RecallTracker


def test_append_and_take_basic() -> None:
    t = RecallTracker()
    sid = uuid.uuid4()
    t.append(sid, [1, 2, 3])
    assert t.take(sid) == [1, 2, 3]


def test_take_empties_buffer() -> None:
    t = RecallTracker()
    sid = uuid.uuid4()
    t.append(sid, [1, 2])
    t.take(sid)
    assert t.take(sid) == []


def test_append_extends() -> None:
    t = RecallTracker()
    sid = uuid.uuid4()
    t.append(sid, [1, 2])
    t.append(sid, [3])
    assert t.take(sid) == [1, 2, 3]


def test_max_per_session_truncates_oldest() -> None:
    t = RecallTracker(max_per_session=3)
    sid = uuid.uuid4()
    t.append(sid, [1, 2, 3, 4, 5])
    assert t.take(sid) == [3, 4, 5]


def test_multi_session_isolation() -> None:
    t = RecallTracker()
    a = uuid.uuid4()
    b = uuid.uuid4()
    t.append(a, [1, 2])
    t.append(b, [10, 20, 30])
    assert t.take(a) == [1, 2]
    assert t.take(b) == [10, 20, 30]
    # take не должен пересекаться.
    assert t.take(a) == []


def test_append_empty_is_noop() -> None:
    t = RecallTracker()
    sid = uuid.uuid4()
    t.append(sid, [])
    assert t.peek(sid) == []


def test_peek_does_not_drain() -> None:
    t = RecallTracker()
    sid = uuid.uuid4()
    t.append(sid, [7, 8])
    assert t.peek(sid) == [7, 8]
    assert t.peek(sid) == [7, 8]
    assert t.take(sid) == [7, 8]


def test_invalid_max_raises() -> None:
    with pytest.raises(ValueError):
        RecallTracker(max_per_session=0)
