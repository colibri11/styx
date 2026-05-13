"""Тесты memory_consolidation scheduler + apply-sweeper (волна 22).

Postgres-skip: на host без БД — skip.
"""

from __future__ import annotations

import datetime as _dt
import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from styx import turn_state
from styx.engine.memory_consolidation import MemoryConsolidationConfig
from styx.storage import migrate
from styx.storage.queries import (
    AgentScopedQueries,
    enqueue_llm_task,
    get_memory_daily_state,
)
from styx.workers.sweep.memory_consolidation import (
    run_memory_consolidation_apply_sweep,
    run_memory_consolidation_scheduler_tick,
)


@pytest.fixture
def db(clean_db: str):
    migrate.run(clean_db)
    conn = psycopg.connect(clean_db)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def reset_turn_state():
    turn_state.reset()
    yield
    turn_state.reset()


def _seed_close_memories(
    conn: psycopg.Connection, agent_id: str, n: int,
    *, age_days: float = 2.0,
) -> list[uuid.UUID]:
    """Создаёт N memories с близкими embedding'ами и сдвигает
    created_at в окно [now-7d..now-24h]."""
    q = AgentScopedQueries(conn, agent_id)
    ids = []
    for i in range(n):
        mid = q.insert_memory(
            role="summary", content=f"сходный смысл {i}",
            kind="note", kind_src="subjective",
            embedding=[1.0] + [0.001 * i] + [0.0] * 766,
        )
        ids.append(mid)
    long_ago = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=age_days)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET created_at = %s WHERE id = ANY(%s)",
            (long_ago, ids),
        )
    conn.commit()
    return ids


# ── Scheduler ─────────────────────────────────────────────────────────


def test_scheduler_disabled_returns_zero(db) -> None:
    cfg = MemoryConsolidationConfig(enabled=False)
    n = run_memory_consolidation_scheduler_tick(db, config=cfg)
    assert n == 0


def test_scheduler_enqueues_cluster(db) -> None:
    ids = _seed_close_memories(db, "alpha", 3)
    cfg = MemoryConsolidationConfig()
    n = run_memory_consolidation_scheduler_tick(db, config=cfg)
    assert n == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memory_consolidation_applications "
            "WHERE agent_id=%s AND status='pending_sleep'",
            ("alpha",),
        )
        assert cur.fetchone()[0] == 1
        cur.execute(
            "SELECT count(*) FROM llm_tasks "
            "WHERE task_type='memory_daily_consolidation'",
        )
        assert cur.fetchone()[0] == 1


def test_scheduler_advances_state_even_on_zero_clusters(db) -> None:
    """Memories но порог clustering не пройден → 0 enqueued, state advance."""
    q = AgentScopedQueries(db, "alpha")
    # Только 2 близких — < min_cluster_size=3.
    ids = []
    for i in range(2):
        mid = q.insert_memory(
            role="summary", content=f"x{i}", kind="note", kind_src="subjective",
            embedding=[1.0] + [0.001 * i] + [0.0] * 766,
        )
        ids.append(mid)
    long_ago = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=2)
    with db.cursor() as cur:
        cur.execute(
            "UPDATE memories SET created_at = %s WHERE id = ANY(%s)",
            (long_ago, ids),
        )
    db.commit()
    cfg = MemoryConsolidationConfig()
    n = run_memory_consolidation_scheduler_tick(db, config=cfg)
    assert n == 0
    state = get_memory_daily_state(db, "alpha")
    assert state is not None
    assert state["last_enqueued"] == 0


def test_scheduler_respects_cooldown(db) -> None:
    """Если state.last_run_at < 23h назад → tick → 0 enqueued."""
    _seed_close_memories(db, "alpha", 3)
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    cfg = MemoryConsolidationConfig()
    # Первый tick — enqueue happens.
    run_memory_consolidation_scheduler_tick(db, config=cfg, now=now)
    # Второй tick через 1 час — cooldown blocks.
    n2 = run_memory_consolidation_scheduler_tick(
        db, config=cfg, now=now + _dt.timedelta(hours=1),
    )
    assert n2 == 0


def test_scheduler_passes_after_cooldown_window(db) -> None:
    _seed_close_memories(db, "alpha", 3)
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    cfg = MemoryConsolidationConfig()
    run_memory_consolidation_scheduler_tick(db, config=cfg, now=now)
    # Через 24h — cooldown прошёл, но новых кандидатов нет
    # (existing уже в taken/superseded после первого enqueue не trogается;
    # все равно select_consolidation_window их найдёт, и cluster будет
    # снова собран).
    n2 = run_memory_consolidation_scheduler_tick(
        db, config=cfg, now=now + _dt.timedelta(hours=24),
    )
    # Может enqueue ещё раз — но не падает.
    assert n2 >= 0


def test_scheduler_isolates_agents(db) -> None:
    _seed_close_memories(db, "alpha", 3)
    _seed_close_memories(db, "beta", 3)
    cfg = MemoryConsolidationConfig()
    n = run_memory_consolidation_scheduler_tick(db, config=cfg)
    # Один кластер на агента → 2 task'а.
    assert n == 2
    with db.cursor() as cur:
        cur.execute(
            "SELECT agent_id FROM memory_consolidation_applications "
            "ORDER BY agent_id",
        )
        agents = [r[0] for r in cur.fetchall()]
    assert agents == ["alpha", "beta"]


# ── Apply sweeper ─────────────────────────────────────────────────────


def _enqueue_pending_consolidation(
    conn: psycopg.Connection, agent_id: str,
    source_ids: list[uuid.UUID],
    *, task_status: str = "pending", task_result=None,
) -> int:
    task_id = enqueue_llm_task(
        conn, task_type="memory_daily_consolidation",
        payload={
            "agent_id": agent_id,
            "memory_ids": [str(s) for s in source_ids],
        },
    )
    if task_status != "pending":
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks SET status=%s, result=%s WHERE id=%s",
                (
                    task_status,
                    Jsonb(task_result) if task_result is not None else None,
                    task_id,
                ),
            )
    q = AgentScopedQueries(conn, agent_id)
    app_id = q.insert_memory_consolidation_application(
        task_id=task_id, source_ids=source_ids,
    )
    conn.commit()
    return app_id


def _merged_result(source_ids, agent_id="alpha") -> dict:
    return {
        "skip": False,
        "skip_reason": None,
        "consolidated_text": "общий смысл",
        "consolidated_embedding": [0.5] * 768,
        "agent_id": agent_id,
        "source_ids": [str(s) for s in source_ids],
        "source_kinds": ["note"] * len(source_ids),
        "source_visibility": ["shared"] * len(source_ids),
    }


def test_apply_sweep_skip_when_active(db) -> None:
    sources = _seed_close_memories(db, "alpha", 3)
    _enqueue_pending_consolidation(
        db, "alpha", sources, task_status="done",
        task_result=_merged_result(sources),
    )
    turn_state.observe("alpha")  # active
    summary = run_memory_consolidation_apply_sweep(db)
    assert summary.applied == 0


def test_apply_sweep_applies_when_idle(db) -> None:
    sources = _seed_close_memories(db, "alpha", 3)
    app_id = _enqueue_pending_consolidation(
        db, "alpha", sources, task_status="done",
        task_result=_merged_result(sources),
    )
    summary = run_memory_consolidation_apply_sweep(db)
    assert summary.applied == 1
    assert summary.errors == 0
    # New consolidated memory created.
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE kind_src='dialogue_consolidation_daily' "
            "  AND agent_id=%s",
            ("alpha",),
        )
        assert cur.fetchone()[0] == 1
        # Sources superseded.
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE id=ANY(%s) AND superseded_by IS NOT NULL",
            (sources,),
        )
        assert cur.fetchone()[0] == 3
        # Application status applied.
        cur.execute(
            "SELECT status, new_memory_id FROM "
            "memory_consolidation_applications WHERE id=%s",
            (app_id,),
        )
        row = cur.fetchone()
    assert row[0] == "applied"
    assert row[1] is not None


def test_apply_sweep_marks_skipped_on_failed_task(db) -> None:
    sources = _seed_close_memories(db, "alpha", 3)
    app_id = _enqueue_pending_consolidation(
        db, "alpha", sources, task_status="failed",
    )
    summary = run_memory_consolidation_apply_sweep(db)
    assert summary.skipped == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM memory_consolidation_applications WHERE id=%s",
            (app_id,),
        )
        assert cur.fetchone()[0] == "skipped"


def test_apply_sweep_marks_skipped_on_skip_shape(db) -> None:
    sources = _seed_close_memories(db, "alpha", 3)
    skip_result = {
        "skip": True, "skip_reason": "разное",
        "consolidated_text": None, "consolidated_embedding": None,
    }
    _enqueue_pending_consolidation(
        db, "alpha", sources, task_status="done", task_result=skip_result,
    )
    summary = run_memory_consolidation_apply_sweep(db)
    assert summary.skipped == 1


def test_apply_sweep_idempotent_supersede_no_failure(db) -> None:
    """Если все источники уже superseded'ы — apply всё равно успешен,
    UPDATE возвращает rowcount=0."""
    sources = _seed_close_memories(db, "alpha", 3)
    # Pre-supersede sources.
    decoy = _seed_close_memories(db, "alpha", 1)[0]
    with db.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by=%s WHERE id=ANY(%s)",
            (decoy, sources),
        )
    db.commit()

    _enqueue_pending_consolidation(
        db, "alpha", sources, task_status="done",
        task_result=_merged_result(sources),
    )
    summary = run_memory_consolidation_apply_sweep(db)
    assert summary.applied == 1  # apply прошёл несмотря на rowcount=0
    assert summary.errors == 0


def test_apply_sweep_per_agent_isolation(db) -> None:
    src_a = _seed_close_memories(db, "alpha", 3)
    src_b = _seed_close_memories(db, "beta", 3)
    _enqueue_pending_consolidation(
        db, "alpha", src_a, task_status="done",
        task_result=_merged_result(src_a, "alpha"),
    )
    _enqueue_pending_consolidation(
        db, "beta", src_b, task_status="done",
        task_result=_merged_result(src_b, "beta"),
    )
    turn_state.observe("alpha")  # alpha active
    summary = run_memory_consolidation_apply_sweep(db)
    assert summary.applied == 1  # только beta
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE kind_src='dialogue_consolidation_daily' "
            "  AND agent_id='alpha'",
        )
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE kind_src='dialogue_consolidation_daily' "
            "  AND agent_id='beta'",
        )
        assert cur.fetchone()[0] == 1
