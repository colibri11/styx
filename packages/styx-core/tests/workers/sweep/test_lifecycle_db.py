"""Тесты lifecycle sweep'а на migrated_db с фикстурными memories."""

from __future__ import annotations

import datetime as _dt
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.workers.sweep.lifecycle import (
    DEFAULT_TRANSITION_BATCH_SIZE,
    LIFECYCLE_STATE_KEY,
    apply_fresh_to_settled,
    apply_settled_to_dormant,
    compute_distribution,
    lifecycle_refresh,
    resolve_autotune_config,
)
from styx.workers.sweep.state import get_state, set_state


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    # Cleanup в конце каждого теста чтобы не путать соседей.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'sweep-test-%'")
        cur.execute("DELETE FROM consolidation_state WHERE key = %s", (LIFECYCLE_STATE_KEY,))
    conn.commit()
    conn.close()


def _insert_memory(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    content: str = "x",
    lifecycle: str = "fresh",
    age_days: float = 0,
    idle_days: float | None = None,
) -> uuid.UUID:
    """INSERT memory с заданным created_at и last_accessed_at."""
    created_at_offset = age_days * 86400
    if idle_days is None:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories "
                "  (agent_id, role, content, lifecycle, created_at) "
                "VALUES (%s, 'user', %s, %s, now() - make_interval(secs => %s)) "
                "RETURNING id",
                (agent_id, content, lifecycle, created_at_offset),
            )
            row = cur.fetchone()
    else:
        last_accessed_offset = idle_days * 86400
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories "
                "  (agent_id, role, content, lifecycle, created_at, last_accessed_at) "
                "VALUES (%s, 'user', %s, %s, "
                "        now() - make_interval(secs => %s), "
                "        now() - make_interval(secs => %s)) "
                "RETURNING id",
                (agent_id, content, lifecycle, created_at_offset, last_accessed_offset),
            )
            row = cur.fetchone()
    conn.commit()
    return row[0]


# ── compute_distribution ─────────────────────────────────────────────


def test_compute_distribution_cold_start(db) -> None:
    """Меньше min_population_for_tuning — quantiles None."""
    cfg = resolve_autotune_config(None)
    # Постгрес чистая — total=0
    dist = compute_distribution(db, cfg)
    assert dist["total"] == 0
    assert dist["fresh_quantile"] is None
    assert dist["idle_quantile"] is None


def test_compute_distribution_with_data(db) -> None:
    """Достаточно данных — quantile считается."""
    cfg = resolve_autotune_config(
        {"min_population_for_tuning": 5, "fresh_share": 0.4, "dormant_share": 0.4}
    )
    agent = "sweep-test-dist"
    # 10 memories с разными возрастами (0..9 дней)
    for i in range(10):
        _insert_memory(db, agent_id=agent, age_days=float(i), idle_days=float(i))

    dist = compute_distribution(db, cfg)
    assert dist["total"] >= 10
    assert dist["fresh_quantile"] is not None
    assert dist["idle_quantile"] is not None


# ── apply transitions ────────────────────────────────────────────────


def test_apply_fresh_to_settled_only_old(db) -> None:
    """Старые fresh идут в settled, молодые остаются."""
    agent = "sweep-test-fts"
    young = _insert_memory(db, agent_id=agent, age_days=0.5, lifecycle="fresh")
    old = _insert_memory(db, agent_id=agent, age_days=10, lifecycle="fresh")

    n = apply_fresh_to_settled(db, age_days=5.0, batch_size=100)
    assert n == 1

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, lifecycle FROM memories WHERE id = ANY(%s::uuid[])", ([young, old],))
        rows = {r["id"]: r["lifecycle"] for r in cur.fetchall()}
    assert rows[young] == "fresh"
    assert rows[old] == "settled"


def test_apply_settled_to_dormant_uses_idle(db) -> None:
    """Settled с большим idle (или без last_accessed_at + старые) → dormant."""
    agent = "sweep-test-std"
    fresh_idle = _insert_memory(
        db, agent_id=agent, age_days=20, idle_days=2, lifecycle="settled"
    )
    stale = _insert_memory(
        db, agent_id=agent, age_days=20, idle_days=15, lifecycle="settled"
    )
    # Без last_accessed_at но с created_at=20d → idle = 20d
    no_access = _insert_memory(
        db, agent_id=agent, age_days=20, idle_days=None, lifecycle="settled"
    )

    n = apply_settled_to_dormant(db, idle_days=10.0, batch_size=100)
    # fresh_idle=2d остаётся; stale=15d → dormant; no_access=20d → dormant.
    assert n == 2

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, lifecycle FROM memories WHERE id = ANY(%s::uuid[])",
            ([fresh_idle, stale, no_access],),
        )
        rows = {r["id"]: r["lifecycle"] for r in cur.fetchall()}
    assert rows[fresh_idle] == "settled"
    assert rows[stale] == "dormant"
    assert rows[no_access] == "dormant"


def test_apply_skips_superseded(db) -> None:
    """superseded_by NOT NULL — не трогаем."""
    agent = "sweep-test-super"
    a = _insert_memory(db, agent_id=agent, age_days=10, lifecycle="fresh")
    b = _insert_memory(db, agent_id=agent, age_days=10, lifecycle="fresh")
    # Помечаем a как superseded by b
    with db.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by = %s WHERE id = %s",
            (b, a),
        )
    db.commit()

    n = apply_fresh_to_settled(db, age_days=1.0, batch_size=100)
    assert n == 1  # только b — a исключён через superseded_by IS NULL

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, lifecycle FROM memories WHERE id = %s", (a,))
        assert cur.fetchone()["lifecycle"] == "fresh"


# ── lifecycle_refresh ───────────────────────────────────────────────


def test_lifecycle_refresh_cold_start_uses_bounds_min(db) -> None:
    """Cold-start — пороги = bounds.min, persist в state."""
    agent = "sweep-test-cold"
    # Несколько свежих и одна старая.
    for _ in range(3):
        _insert_memory(db, agent_id=agent, age_days=0.5, lifecycle="fresh")
    old_fresh = _insert_memory(
        db, agent_id=agent, age_days=2, lifecycle="fresh"
    )

    summary = lifecycle_refresh(db)
    assert summary["fresh_to_settled_age_days"] == 1.0  # bounds.min
    assert summary["settled_to_dormant_idle_days"] == 7.0

    state = get_state(db, LIFECYCLE_STATE_KEY)
    assert state is not None
    assert state["fresh_to_settled_age_days"] == 1.0
    assert state["settled_to_dormant_idle_days"] == 7.0

    # 2-дневный fresh должен перейти в settled (порог 1d).
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT lifecycle FROM memories WHERE id = %s", (old_fresh,))
        assert cur.fetchone()["lifecycle"] == "settled"


def test_lifecycle_refresh_smoothing_against_previous(db) -> None:
    """С previous threshold'ом next = prev + α × (current - prev)."""
    agent = "sweep-test-smooth"
    for _ in range(3):
        _insert_memory(db, agent_id=agent, age_days=0.1)

    # Заранее положим previous threshold.
    set_state(
        db,
        LIFECYCLE_STATE_KEY,
        {"fresh_to_settled_age_days": 5.0, "settled_to_dormant_idle_days": 100.0},
    )
    db.commit()

    summary = lifecycle_refresh(db)
    # cold-start (мало данных) → "current" будет bounds.min = 1.0.
    # next = 5 + 0.3 * (1 - 5) = 5 - 1.2 = 3.8.
    assert abs(summary["fresh_to_settled_age_days"] - 3.8) < 1e-6
    assert abs(summary["settled_to_dormant_idle_days"] - (100 + 0.3 * (7 - 100))) < 1e-6


def test_lifecycle_refresh_persists_thresholds_before_apply(db) -> None:
    """Пороги в state записываются до apply — даже если apply упадёт,
    они зафиксированы."""
    agent = "sweep-test-persist"
    _insert_memory(db, agent_id=agent, age_days=0.1)

    lifecycle_refresh(db)
    state = get_state(db, LIFECYCLE_STATE_KEY)
    assert state is not None
    assert "computed_at" in state
    assert "total_population" in state
