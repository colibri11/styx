"""Unit-тесты для emotional/state.py — pure + DB."""

from __future__ import annotations

import datetime as _dt
import math
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.emotional.state import (
    EMOTIONAL_AXIS_MAX,
    EMOTIONAL_AXIS_MIN,
    INSTANT_DECAY_EPSILON,
    INSTANT_DECAY_PER_MINUTE,
    EmotionalVector,
    NEUTRAL_VECTOR,
    append_emotional_state,
    apply_decay,
    apply_instant_decay,
    clamp_axis,
    clamp_vector,
    decay_factor,
    list_active_agent_ids,
    max_abs,
    read_last_state,
)


# ── Pure functions ────────────────────────────────────────────────────


def test_clamp_axis_basic() -> None:
    assert clamp_axis(0.5) == 0.5
    assert clamp_axis(-2.0) == EMOTIONAL_AXIS_MIN
    assert clamp_axis(5.0) == EMOTIONAL_AXIS_MAX


def test_clamp_vector() -> None:
    v = EmotionalVector(2.0, -3.0, 0.5)
    out = clamp_vector(v)
    assert out.valence == 1.0
    assert out.arousal == -1.0
    assert out.dominance == 0.5


def test_max_abs() -> None:
    v = EmotionalVector(0.3, -0.7, 0.2)
    assert max_abs(v) == 0.7


def test_decay_factor() -> None:
    """factor = 0.95^minutes."""
    assert math.isclose(decay_factor(1), INSTANT_DECAY_PER_MINUTE)
    assert math.isclose(decay_factor(2), INSTANT_DECAY_PER_MINUTE ** 2)


def test_apply_decay() -> None:
    v = EmotionalVector(0.5, -0.4, 0.3)
    out = apply_decay(v, 10.0)
    f = INSTANT_DECAY_PER_MINUTE ** 10
    assert math.isclose(out.valence, 0.5 * f, rel_tol=1e-9)
    assert math.isclose(out.arousal, -0.4 * f, rel_tol=1e-9)


# ── DB-side ────────────────────────────────────────────────────────────


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM emotional_state WHERE agent_id LIKE 'state-test-%'")
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'state-test-%'")
    conn.commit()
    conn.close()


def test_read_last_state_empty_returns_none(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    assert read_last_state(db, agent) is None


def test_append_emotional_state_neutral_base(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    delta = EmotionalVector(0.1, 0.2, -0.3)
    out = append_emotional_state(db, agent, delta, source="hot_sentiment")
    db.commit()
    assert math.isclose(out.valence, 0.1)
    assert math.isclose(out.arousal, 0.2)
    assert math.isclose(out.dominance, -0.3)

    # И из БД достанем то же значение.
    last = read_last_state(db, agent)
    assert last is not None
    vec, _ = last
    assert math.isclose(vec.valence, 0.1, abs_tol=1e-6)


def test_append_emotional_state_accumulates(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    append_emotional_state(db, agent, EmotionalVector(0.3, 0, 0))
    db.commit()
    out = append_emotional_state(db, agent, EmotionalVector(0.4, 0, 0))
    db.commit()
    assert math.isclose(out.valence, 0.7, abs_tol=1e-6)


def test_append_emotional_state_clamps(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    append_emotional_state(db, agent, EmotionalVector(0.8, 0, 0))
    db.commit()
    out = append_emotional_state(db, agent, EmotionalVector(0.5, 0, 0))
    db.commit()
    # 0.8 + 0.5 = 1.3 → clamp до 1.0.
    assert out.valence == 1.0


# ── Decay ──────────────────────────────────────────────────────────────


def test_apply_instant_decay_no_history_noop(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    out = apply_instant_decay(db, agent)
    assert out.decayed is False
    assert out.point is None


def test_apply_instant_decay_recent_point_noop(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    append_emotional_state(db, agent, EmotionalVector(0.5, 0, 0))
    db.commit()
    out = apply_instant_decay(db, agent)
    # at = now() в БД, прошло < 1 минуты → no decay.
    assert out.decayed is False


def test_apply_instant_decay_below_epsilon_noop(db) -> None:
    """В пределах epsilon — decay не пишется (журнал не раздувается)."""
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    # Прямой INSERT с малым значением и старым at.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state (agent_id, at, valence, arousal, dominance) "
            "VALUES (%s, now() - interval '10 minutes', %s, %s, %s)",
            (agent, INSTANT_DECAY_EPSILON / 2, 0.0, 0.0),
        )
    db.commit()
    out = apply_instant_decay(db, agent)
    assert out.decayed is False


def test_apply_instant_decay_writes_decayed_point(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    # Точка 10 минут назад, valence=0.5.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state (agent_id, at, valence, arousal, dominance, source) "
            "VALUES (%s, now() - interval '10 minutes', 0.5, 0.0, 0.0, 'hot_sentiment')",
            (agent,),
        )
    db.commit()

    out = apply_instant_decay(db, agent)
    db.commit()
    assert out.decayed is True
    assert out.point is not None
    # 0.5 * 0.95^10 ≈ 0.299
    expected = 0.5 * (INSTANT_DECAY_PER_MINUTE ** 10)
    assert math.isclose(out.point.valence, expected, rel_tol=0.01)

    # В БД должно быть две точки.
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT count(*)::int AS n, "
            "       count(*) FILTER (WHERE source = 'decay')::int AS decay_n "
            "  FROM emotional_state WHERE agent_id = %s",
            (agent,),
        )
        row = cur.fetchone()
    assert row["n"] == 2
    assert row["decay_n"] == 1


# ── list_active_agent_ids ─────────────────────────────────────────────


def test_list_active_agent_ids_returns_distinct(db) -> None:
    agent_a = f"state-test-A-{uuid.uuid4().hex[:6]}"
    agent_b = f"state-test-B-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content) VALUES "
            "(%s, 'user', 'x'), (%s, 'user', 'y'), (%s, 'assistant', 'z')",
            (agent_a, agent_a, agent_b),
        )
    db.commit()
    ids = list_active_agent_ids(db)
    assert agent_a in ids
    assert agent_b in ids
