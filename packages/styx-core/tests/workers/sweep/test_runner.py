"""Unit-тесты для sweep runner'а — advisory lock + per-task try/except."""

from __future__ import annotations

import threading
import time
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.workers.sweep.runner import (
    SWEEP_ADVISORY_LOCK_KEY,
    SweepResult,
    SweepTask,
    run_sweep,
)


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


def test_run_sweep_empty_task_list_success(migrated_db: str) -> None:
    result = run_sweep(migrated_db, tasks=[])
    assert result.status == "success"
    assert result.skipped is False
    assert result.summary == {}
    assert result.errors == []
    assert result.sweep_id is not None


def test_run_sweep_writes_sweep_run_row(migrated_db: str, db) -> None:
    result = run_sweep(migrated_db, tasks=[])
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT status, summary, errors FROM sweep_runs WHERE id = %s",
            (result.sweep_id,),
        )
        row = cur.fetchone()
    assert row["status"] == "success"
    assert row["summary"] == {}
    assert row["errors"] == []


def test_run_sweep_skipped_when_lock_held(migrated_db: str, db) -> None:
    """Один holds advisory lock → второй sweep skip'ает."""
    # Берём lock на отдельной connection и держим.
    holder = psycopg.connect(migrated_db)
    with holder.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s::bigint)", (SWEEP_ADVISORY_LOCK_KEY,))
    holder.commit()

    try:
        result = run_sweep(migrated_db, tasks=[])
        assert result.skipped is True
        assert result.status == "success"
        assert result.sweep_id is None
        # sweep_runs не пополняется при skip.
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM sweep_runs WHERE finished_at IS NULL"
            )
            assert cur.fetchone()[0] == 0
    finally:
        with holder.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(%s::bigint)", (SWEEP_ADVISORY_LOCK_KEY,)
            )
        holder.commit()
        holder.close()


def test_run_sweep_partial_status_when_some_tasks_fail(migrated_db: str) -> None:
    """Один task падает, второй проходит → status='partial'."""
    def good(conn: psycopg.Connection) -> dict:
        return {"affected": 0}

    def bad(conn: psycopg.Connection) -> dict:
        raise RuntimeError("boom")

    result = run_sweep(
        migrated_db,
        tasks=[
            SweepTask(name="good", fn=good),
            SweepTask(name="bad", fn=bad),
        ],
    )
    assert result.status == "partial"
    assert result.summary["good"] == {"affected": 0}
    assert "error" in result.summary["bad"]
    assert any(e["task"] == "bad" for e in result.errors)


def test_run_sweep_failed_when_all_tasks_fail(migrated_db: str) -> None:
    def bad(conn: psycopg.Connection) -> dict:
        raise RuntimeError("boom")

    result = run_sweep(
        migrated_db,
        tasks=[SweepTask(name="bad", fn=bad)],
    )
    assert result.status == "failed"
    assert len(result.errors) == 1


def test_run_sweep_releases_lock_after_finish(migrated_db: str) -> None:
    """После окончания второй run_sweep подряд должен залочиться без skip."""
    r1 = run_sweep(migrated_db, tasks=[])
    assert r1.skipped is False
    r2 = run_sweep(migrated_db, tasks=[])
    assert r2.skipped is False  # lock уже отпущен после r1
