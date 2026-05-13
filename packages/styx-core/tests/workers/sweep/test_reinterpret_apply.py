"""Тесты reinterpret apply-sweeper (волна 22).

Postgres-skip: на host без БД — skip; в Docker integration suite —
прогон полный. Проверяет write-gate через `turn_state.is_active`.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from styx import turn_state
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries, enqueue_llm_task
from styx.workers.sweep.reinterpret_apply import run_reinterpret_apply_sweep


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


def _seed_memory(
    conn: psycopg.Connection, agent_id: str = "alpha",
    content: str = "old",
    embedding: list[float] | None = None,
) -> uuid.UUID:
    q = AgentScopedQueries(conn, agent_id)
    mid = q.insert_memory(
        role="summary", content=content, kind="note",
        kind_src="subjective",
        embedding=embedding or ([1.0, 0.0] + [0.0] * 766),
    )
    conn.commit()
    return mid


def _enqueue_pending(
    conn: psycopg.Connection, agent_id: str, mid: uuid.UUID,
    *, task_status: str = "pending", task_result=None,
) -> int:
    """Создаёт llm_task + reinterpret_application в нужном статусе."""
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge",
        payload={"agent_id": agent_id, "new_understanding_text": "x"},
        memory_id=mid,
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
    app_id = q.insert_reinterpret_application(task_id=task_id, memory_id=mid)
    conn.commit()
    return app_id


def _merged_result(prev_text: str = "old") -> dict:
    return {
        "skip": False,
        "skip_reason": None,
        "merged_text": "merged content",
        "merged_embedding": [0.7, 0.7] + [0.0] * 766,
        "previous_text": prev_text,
        "previous_embedding": [1.0, 0.0] + [0.0] * 766,
        "new_understanding_text": "новое понимание",
        "weight_applied": 0.5,
        "agent_id": "alpha",
    }


# ── Fast-path skip when agent active ────────────────────────────────


def test_sweep_skips_when_agent_active(db) -> None:
    """is_active(agent) == True → ни одна строка не обрабатывается."""
    mid = _seed_memory(db, "alpha")
    _enqueue_pending(db, "alpha", mid, task_status="done",
                     task_result=_merged_result())
    # Open active turn.
    turn_state.observe("alpha")

    summary = run_reinterpret_apply_sweep(db)
    assert summary.applied == 0
    assert summary.deferred == 0  # fast-path skip — не доходим до per-row
    assert summary.errors == 0
    # Memory не изменена.
    with db.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id=%s", (mid,))
        assert cur.fetchone()[0] == "old"


# ── Apply path ──────────────────────────────────────────────────────


def test_sweep_applies_when_agent_idle(db) -> None:
    mid = _seed_memory(db, "alpha")
    app_id = _enqueue_pending(
        db, "alpha", mid, task_status="done", task_result=_merged_result(),
    )
    summary = run_reinterpret_apply_sweep(db)
    assert summary.applied == 1
    assert summary.skipped == 0
    assert summary.deferred == 0
    assert summary.errors == 0

    with db.cursor() as cur:
        cur.execute(
            "SELECT content, status FROM memories m "
            "  CROSS JOIN reinterpret_applications a "
            " WHERE m.id=%s AND a.id=%s",
            (mid, app_id),
        )
        row = cur.fetchone()
    assert row[0] == "merged content"
    assert row[1] == "applied"


def test_sweep_inserts_audit_row(db) -> None:
    mid = _seed_memory(db, "alpha", content="прежний текст")
    _enqueue_pending(
        db, "alpha", mid, task_status="done",
        task_result=_merged_result(prev_text="прежний текст"),
    )
    run_reinterpret_apply_sweep(db)
    with db.cursor() as cur:
        cur.execute(
            "SELECT previous_text, merged_text, weight_applied "
            "FROM memory_reinterpretations WHERE memory_id=%s",
            (mid,),
        )
        row = cur.fetchone()
    assert row[0] == "прежний текст"
    assert row[1] == "merged content"
    assert float(row[2]) == 0.5


# ── Skip / mark_skipped paths ───────────────────────────────────────


def test_sweep_marks_skipped_on_failed_task(db) -> None:
    mid = _seed_memory(db, "alpha")
    app_id = _enqueue_pending(db, "alpha", mid, task_status="failed")
    summary = run_reinterpret_apply_sweep(db)
    assert summary.skipped == 1
    assert summary.applied == 0
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM reinterpret_applications WHERE id=%s",
            (app_id,),
        )
        assert cur.fetchone()[0] == "skipped"


def test_sweep_marks_skipped_on_skip_shape(db) -> None:
    mid = _seed_memory(db, "alpha")
    skip_result = {
        "skip": True, "skip_reason": "тавтология", "merged_text": None,
    }
    app_id = _enqueue_pending(
        db, "alpha", mid, task_status="done", task_result=skip_result,
    )
    summary = run_reinterpret_apply_sweep(db)
    assert summary.skipped == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM reinterpret_applications WHERE id=%s",
            (app_id,),
        )
        assert cur.fetchone()[0] == "skipped"


def test_sweep_marks_skipped_on_legacy_skipped_shape(db) -> None:
    """{'skipped': 'memory_gone'} тоже skip-shape."""
    mid = _seed_memory(db, "alpha")
    app_id = _enqueue_pending(
        db, "alpha", mid, task_status="done",
        task_result={"skipped": "memory_gone"},
    )
    summary = run_reinterpret_apply_sweep(db)
    assert summary.skipped == 1


def test_sweep_marks_skipped_on_unknown_shape(db) -> None:
    """Defensive: неожиданная shape → skip."""
    mid = _seed_memory(db, "alpha")
    app_id = _enqueue_pending(
        db, "alpha", mid, task_status="done",
        task_result={"random": "junk"},
    )
    summary = run_reinterpret_apply_sweep(db)
    assert summary.skipped == 1


# ── Defer paths ─────────────────────────────────────────────────────


def test_sweep_defers_on_pending_task(db) -> None:
    mid = _seed_memory(db, "alpha")
    _enqueue_pending(db, "alpha", mid, task_status="pending")
    summary = run_reinterpret_apply_sweep(db)
    assert summary.deferred == 1
    assert summary.applied == 0


def test_sweep_defers_on_running_task(db) -> None:
    mid = _seed_memory(db, "alpha")
    _enqueue_pending(db, "alpha", mid, task_status="running")
    summary = run_reinterpret_apply_sweep(db)
    assert summary.deferred == 1
    assert summary.applied == 0


# ── Per-agent isolation ─────────────────────────────────────────────


def test_sweep_isolates_agents(db) -> None:
    """Один агент active → его apps deferred; другой idle → applied."""
    mid_a = _seed_memory(db, "alpha")
    mid_b = _seed_memory(db, "beta")
    _enqueue_pending(db, "alpha", mid_a, task_status="done",
                     task_result=_merged_result())
    _enqueue_pending(db, "beta", mid_b, task_status="done",
                     task_result={**_merged_result(), "agent_id": "beta"})
    turn_state.observe("alpha")  # alpha active

    summary = run_reinterpret_apply_sweep(db)
    assert summary.applied == 1  # только beta
    assert summary.errors == 0
    with db.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id=%s", (mid_a,))
        assert cur.fetchone()[0] == "old"  # alpha не тронута
        cur.execute("SELECT content FROM memories WHERE id=%s", (mid_b,))
        assert cur.fetchone()[0] == "merged content"
