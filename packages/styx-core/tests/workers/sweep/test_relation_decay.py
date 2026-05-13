"""Tests для run_relation_decay (волна 21).

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries
from styx.workers.sweep.relation_decay import (
    DEFAULT_DECAY_RATE,
    DEFAULT_IDLE_THRESHOLD_DAYS,
    run_relation_decay,
)


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _seed_co_retrieved(
    conn: psycopg.Connection,
    *,
    weight: float,
    last_reinforced_age_days: int,
) -> uuid.UUID:
    """INSERT'ит co_retrieved ребро с заданным weight и last_reinforced
    в прошлом на ``last_reinforced_age_days`` дней.
    """
    src, tgt = uuid.uuid4(), uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relations "
            "  (source_type, source_id, target_type, target_id, relation, "
            "   weight, metadata) "
            "VALUES ('memory', %s, 'memory', %s, 'co_retrieved', %s, "
            "        jsonb_build_object('last_reinforced', "
            "            (now() - make_interval(days => %s))::text)) "
            "RETURNING id",
            (src, tgt, weight, last_reinforced_age_days),
        )
        return cur.fetchone()[0]


def test_decays_old_high_weight_edges(conn: psycopg.Connection) -> None:
    """weight > 1.0 + старее idle_threshold_days → weight уменьшен."""
    relation_id = _seed_co_retrieved(
        conn, weight=1.5, last_reinforced_age_days=20,
    )
    conn.commit()

    result = run_relation_decay(conn, decay_rate=0.05, idle_threshold_days=14)
    conn.commit()

    assert result.decayed == 1
    with conn.cursor() as cur:
        cur.execute("SELECT weight FROM relations WHERE id=%s", (relation_id,))
        weight = cur.fetchone()[0]
    assert abs(weight - 1.45) < 1e-9


def test_floor_clamps_at_1(conn: psycopg.Connection) -> None:
    """weight 1.02, decay 0.05 → не уходит ниже 1.0."""
    relation_id = _seed_co_retrieved(
        conn, weight=1.02, last_reinforced_age_days=20,
    )
    conn.commit()
    run_relation_decay(conn, decay_rate=0.05, idle_threshold_days=14)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT weight FROM relations WHERE id=%s", (relation_id,))
        weight = cur.fetchone()[0]
    assert abs(weight - 1.0) < 1e-9


def test_skips_recent_edges(conn: psycopg.Connection) -> None:
    """last_reinforced моложе threshold → не трогаем."""
    relation_id = _seed_co_retrieved(
        conn, weight=1.5, last_reinforced_age_days=5,
    )
    conn.commit()
    result = run_relation_decay(conn, decay_rate=0.05, idle_threshold_days=14)
    conn.commit()
    assert result.decayed == 0
    with conn.cursor() as cur:
        cur.execute("SELECT weight FROM relations WHERE id=%s", (relation_id,))
        weight = cur.fetchone()[0]
    assert abs(weight - 1.5) < 1e-9


def test_skips_baseline_weight(conn: psycopg.Connection) -> None:
    """weight = 1.0 (baseline never-reinforced) → не трогаем."""
    relation_id = _seed_co_retrieved(
        conn, weight=1.0, last_reinforced_age_days=30,
    )
    conn.commit()
    result = run_relation_decay(conn, decay_rate=0.05, idle_threshold_days=14)
    conn.commit()
    assert result.decayed == 0
    with conn.cursor() as cur:
        cur.execute("SELECT weight FROM relations WHERE id=%s", (relation_id,))
        weight = cur.fetchone()[0]
    assert abs(weight - 1.0) < 1e-9


def test_skips_other_relation_types(conn: psycopg.Connection) -> None:
    """related_to / supersedes — decay'ить нельзя."""
    src, tgt = uuid.uuid4(), uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relations "
            "  (source_type, source_id, target_type, target_id, relation, "
            "   weight, metadata) "
            "VALUES ('memory', %s, 'memory', %s, 'related_to', 1.5, "
            "        jsonb_build_object('last_reinforced', "
            "            (now() - interval '30 days')::text))",
            (src, tgt),
        )
    conn.commit()
    result = run_relation_decay(conn, decay_rate=0.05, idle_threshold_days=14)
    conn.commit()
    assert result.decayed == 0


def test_default_constants_match_memorybox() -> None:
    assert DEFAULT_DECAY_RATE == 0.05
    assert DEFAULT_IDLE_THRESHOLD_DAYS == 14
