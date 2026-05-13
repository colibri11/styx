"""Unit-тесты для emotional/baseline.py."""

from __future__ import annotations

import datetime as _dt
import math
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.emotional.baseline import (
    BASELINE_EMA_ALPHA,
    EmotionalBaseline,
    read_baseline_for_scoring,
    recompute_baseline,
)


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM emotional_state WHERE agent_id LIKE 'baseline-test-%'")
        cur.execute("DELETE FROM emotional_baseline WHERE agent_id LIKE 'baseline-test-%'")
    conn.commit()
    conn.close()


def _insert_state(
    conn: psycopg.Connection, agent_id: str, *, v: float, a: float, d: float, minutes_ago: float = 0
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "  (agent_id, at, valence, arousal, dominance) "
            "VALUES (%s, now() - make_interval(secs => %s), %s, %s, %s)",
            (agent_id, minutes_ago * 60, v, a, d),
        )
    conn.commit()


# ── recompute_baseline ────────────────────────────────────────────────


def test_recompute_baseline_skips_when_no_instant(db) -> None:
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    out = recompute_baseline(db, agent)
    assert out.skipped is True
    assert out.baseline is None
    assert out.sample_size == 0


def test_recompute_baseline_starts_from_zero(db) -> None:
    """Без current baseline → α × 0 + (1-α) × mean."""
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    _insert_state(db, agent, v=0.5, a=0.3, d=0.0, minutes_ago=10)
    _insert_state(db, agent, v=0.3, a=0.1, d=-0.2, minutes_ago=5)

    out = recompute_baseline(db, agent)
    db.commit()
    assert out.skipped is False
    assert out.sample_size == 2

    # mean = (0.4, 0.2, -0.1); next = 0.02 × mean (1-α=0.02)
    expected_v = (1 - BASELINE_EMA_ALPHA) * 0.4
    assert out.baseline is not None
    assert math.isclose(out.baseline.valence, expected_v, rel_tol=0.01)


def test_recompute_baseline_updates_existing(db) -> None:
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    # Initial baseline = (0.5, 0, 0).
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_baseline (agent_id, valence, arousal, dominance) "
            "VALUES (%s, 0.5, 0, 0)",
            (agent,),
        )
    db.commit()

    _insert_state(db, agent, v=0.0, a=0.0, d=0.0, minutes_ago=5)

    out = recompute_baseline(db, agent)
    db.commit()
    assert out.skipped is False
    # next = α × 0.5 + 0.02 × 0 = 0.49
    expected = BASELINE_EMA_ALPHA * 0.5
    assert out.baseline is not None
    assert math.isclose(out.baseline.valence, expected, rel_tol=0.01)


def test_recompute_baseline_preserves_mood_active(db) -> None:
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_baseline "
            "  (agent_id, valence, arousal, dominance, mood_active) "
            "VALUES (%s, 0.0, 0.0, 0.0, true)",
            (agent,),
        )
    db.commit()
    _insert_state(db, agent, v=0.5, a=0, d=0, minutes_ago=2)

    recompute_baseline(db, agent)
    db.commit()

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT mood_active FROM emotional_baseline WHERE agent_id = %s",
            (agent,),
        )
        row = cur.fetchone()
    assert row["mood_active"] is True


def test_recompute_baseline_excludes_outside_window(db) -> None:
    """Точки старше окна 60min не попадают в среднее."""
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    _insert_state(db, agent, v=1.0, a=1.0, d=1.0, minutes_ago=120)  # вне окна
    _insert_state(db, agent, v=0.1, a=0.0, d=0.0, minutes_ago=10)  # внутри

    out = recompute_baseline(db, agent)
    db.commit()
    assert out.sample_size == 1
    # mean = (0.1, 0, 0); next = 0.02 × 0.1
    assert out.baseline is not None
    assert math.isclose(out.baseline.valence, 0.02 * 0.1, rel_tol=0.05)


# ── read_baseline_for_scoring ─────────────────────────────────────────


def test_read_baseline_for_scoring_none_when_missing(db) -> None:
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    assert read_baseline_for_scoring(db, agent) is None


def test_read_baseline_for_scoring_returns_dataclass(db) -> None:
    agent = f"baseline-test-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_baseline (agent_id, valence, arousal, dominance) "
            "VALUES (%s, 0.3, -0.1, 0.0)",
            (agent,),
        )
    db.commit()

    out = read_baseline_for_scoring(db, agent)
    assert out == EmotionalBaseline(0.30000001192092896, -0.10000000149011612, 0.0) or (
        # real в pgvector — float32, не float64; допускаем погрешность
        math.isclose(out.valence, 0.3, abs_tol=1e-6)
        and math.isclose(out.arousal, -0.1, abs_tol=1e-6)
        and out.dominance == 0.0
    )


def test_read_baseline_for_scoring_none_for_empty_agent_id(db) -> None:
    assert read_baseline_for_scoring(db, None) is None
    assert read_baseline_for_scoring(db, "") is None
