"""Unit-тесты для sweep_runs ledger'а."""

from __future__ import annotations

import datetime as _dt

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.workers.sweep.runs import finish_run, start_run


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


def test_start_run_inserts_running(db) -> None:
    sweep_id = start_run(db, _dt.datetime.now(tz=_dt.timezone.utc))
    db.commit()
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT status, started_at, finished_at FROM sweep_runs WHERE id = %s", (sweep_id,))
        row = cur.fetchone()
    assert row["status"] == "running"
    assert row["started_at"] is not None
    assert row["finished_at"] is None


def test_finish_run_updates(db) -> None:
    sweep_id = start_run(db, _dt.datetime.now(tz=_dt.timezone.utc))
    db.commit()
    summary = {"lifecycle_refresh": {"transitions_total": 5}}
    errors: list[dict] = []
    finish_run(db, sweep_id, "success", summary, errors)
    db.commit()
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT status, summary, errors, finished_at FROM sweep_runs WHERE id = %s",
            (sweep_id,),
        )
        row = cur.fetchone()
    assert row["status"] == "success"
    assert row["finished_at"] is not None
    assert row["summary"] == summary
    assert row["errors"] == []
