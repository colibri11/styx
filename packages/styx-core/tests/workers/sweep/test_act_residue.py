"""Pure lifecycle tests for the Wave 38 retry sweeper."""

from __future__ import annotations

import types
import uuid

import pytest

import styx.workers.sweep.act_residue as subject
from styx.storage.act_reduction import ActReductionConflict


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)


class _Transaction:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Connection:
    def __init__(self, rows):
        self.cursor_value = _Cursor(rows)

    def cursor(self, **_kwargs):
        return self.cursor_value

    def transaction(self):
        return _Transaction()


def _row(
    *,
    attempts: int,
    task_id=None,
    status: str = "retryable",
    task_status: str = "done",
    task_retry_count: int = 0,
    act_status: str = "completed",
):
    return {
        "agent_id": "agent-a",
        "act_id": uuid.uuid4(),
        "reducer_version": "act_residue_v1",
        "input_hash": "a" * 64,
        "task_id": task_id or uuid.uuid4(),
        "status": status,
        "task_status": task_status,
        "task_retry_count": task_retry_count,
        "attempt_count": attempts,
        "act_status": act_status,
    }


def test_retry_sweep_requeues_below_budget(monkeypatch) -> None:
    row = _row(attempts=1)
    calls = []

    def schedule(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(duplicate=False)

    monkeypatch.setattr(subject, "schedule_act_reduction", schedule)
    conn = _Connection([row])
    summary = subject.run_act_residue_retry_sweep(conn, max_attempts=3)

    assert (summary.scanned, summary.requeued) == (1, 1)
    assert summary.terminalized == summary.errors == 0
    assert calls[0][1]["retry"] is True


def test_retry_sweep_terminalizes_exhausted(monkeypatch) -> None:
    row = _row(attempts=3)
    calls = []
    monkeypatch.setattr(
        subject,
        "mark_act_reduction_terminal_failure",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    conn = _Connection([row])
    summary = subject.run_act_residue_retry_sweep(conn, max_attempts=3)

    assert summary.terminalized == 1
    assert summary.requeued == summary.errors == 0
    assert calls[0][1]["error_code"] == "retry_exhausted"


def test_retry_sweep_treats_concurrent_transition_as_race(monkeypatch) -> None:
    row = _row(attempts=1)
    monkeypatch.setattr(
        subject,
        "schedule_act_reduction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ActReductionConflict("won elsewhere")
        ),
    )
    summary = subject.run_act_residue_retry_sweep(
        _Connection([row]), max_attempts=3
    )
    assert summary.raced == 1
    assert summary.errors == 0


def test_retry_sweep_reconciles_failed_task_orphan_before_requeue(
    monkeypatch,
) -> None:
    row = _row(
        attempts=0,
        status="pending",
        task_status="failed",
        task_retry_count=1,
    )
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        subject,
        "mark_act_reduction_retryable",
        lambda *args, **kwargs: calls.append(("retryable", args, kwargs)),
    )
    monkeypatch.setattr(
        subject,
        "schedule_act_reduction",
        lambda *args, **kwargs: (
            calls.append(("schedule", args, kwargs))
            or types.SimpleNamespace(duplicate=False)
        ),
    )

    summary = subject.run_act_residue_retry_sweep(
        _Connection([row]), max_attempts=3
    )

    assert summary.reconciled == summary.requeued == 1
    assert summary.terminalized == summary.errors == 0
    assert [call[0] for call in calls] == ["retryable", "schedule"]
    assert calls[0][2]["error_code"] == "orphan_failed_task"


def test_retry_sweep_terminalizes_exhausted_failed_task_orphan(
    monkeypatch,
) -> None:
    row = _row(
        attempts=1,
        status="running",
        task_status="failed",
        task_retry_count=3,
    )
    calls = []
    monkeypatch.setattr(
        subject,
        "mark_act_reduction_terminal_failure",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    summary = subject.run_act_residue_retry_sweep(
        _Connection([row]), max_attempts=3
    )

    assert summary.reconciled == summary.terminalized == 1
    assert summary.requeued == summary.errors == 0
    assert calls[0][1]["error_code"] == "orphan_retry_exhausted"


def test_retry_sweep_sums_rolled_back_task_attempt_into_ledger_budget(
    monkeypatch,
) -> None:
    row = _row(
        attempts=2,
        status="pending",
        task_status="failed",
        task_retry_count=1,
    )
    calls = []
    monkeypatch.setattr(
        subject,
        "mark_act_reduction_terminal_failure",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    conn = _Connection([row])
    summary = subject.run_act_residue_retry_sweep(conn, max_attempts=3)
    assert summary.terminalized == summary.reconciled == 1
    assert calls[0][1]["error_code"] == "orphan_retry_exhausted"
    reconcile = [
        params for sql, params in conn.cursor_value.executed
        if "SET attempt_count=%s" in sql
    ]
    assert reconcile[0][0] == 3


def test_retry_sweep_resolves_failed_no_task_without_llm_requeue(monkeypatch) -> None:
    row = _row(
        attempts=0,
        task_id=None,
        status="retryable",
        task_status=None,
        act_status="failed",
    )
    row["task_id"] = None
    calls = []
    monkeypatch.setattr(
        subject,
        "schedule_act_reduction",
        lambda *args, **kwargs: (
            calls.append((args, kwargs))
            or types.SimpleNamespace(status="no_residue", duplicate=False)
        ),
    )
    summary = subject.run_act_residue_retry_sweep(
        _Connection([row]), max_attempts=3
    )
    assert summary.reconciled == 1
    assert summary.requeued == summary.terminalized == summary.errors == 0
    assert calls[0][1]["retry"] is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": True}, "max_attempts"),
        ({"limit": 0}, "limit"),
        ({"limit": 129}, "limit"),
    ],
)
def test_retry_sweep_rejects_unbounded_configuration(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        subject.run_act_residue_retry_sweep(_Connection([]), **kwargs)
