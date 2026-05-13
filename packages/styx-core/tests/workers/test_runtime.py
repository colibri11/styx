"""Unit-тесты для LlmWorker runtime."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

import psycopg
import pytest

from styx.llm import (
    LLMRateLimiter,
    OllamaChatClient,
    OllamaTerminalError,
    OllamaTransientError,
)
from styx.workers.runtime import (
    HandlerContext,
    HandlerResult,
    LlmTask,
    LlmWorker,
    fetch_task_status,
    insert_pending_task,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def fake_llm() -> OllamaChatClient:
    """Реальный класс с фейковым endpoint'ом — handler'ы в этих тестах не
    зовут chat_json, но для конструктора нужен инстанс."""
    return OllamaChatClient(base_url="http://x", model="m", max_attempts=1)


@pytest.fixture
def rate_limit() -> LLMRateLimiter:
    return LLMRateLimiter(capacity=4, refill_per_second=10.0)


@pytest.fixture
def worker(
    migrated_db: str, fake_llm: OllamaChatClient, rate_limit: LLMRateLimiter
) -> LlmWorker:
    w = LlmWorker(
        dsn=migrated_db,
        llm=fake_llm,
        rate_limit=rate_limit,
        # короткие таймауты — тесты не должны висеть
        poll_interval_s=0.05,
        reap_interval_s=0.0,  # reap на каждый цикл
        stale_running_threshold_s=1.0,
    )
    yield w
    # Гарантированный cleanup на случай если тест не вызвал stop()/run().
    w._close_conn()


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


# ── process_one ────────────────────────────────────────────────────────


def test_process_one_empty_queue(worker: LlmWorker) -> None:
    assert worker.process_one() is False


def test_process_one_marks_done_on_success(worker: LlmWorker, db) -> None:
    task_id = insert_pending_task(db, task_type="dummy_ok", payload={"k": "v"})

    seen: dict[str, Any] = {}

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        seen["task_id"] = task.id
        seen["payload"] = task.payload
        return HandlerResult(result={"ok": True, "score": 0.42})

    worker.register_handler("dummy_ok", handler)

    assert worker.process_one() is True
    assert seen["task_id"] == task_id
    assert seen["payload"] == {"k": "v"}

    row = fetch_task_status(db, task_id)
    assert row["status"] == "done"
    assert row["result"] == {"ok": True, "score": 0.42}
    assert row["error"] is None
    assert row["completed_at"] is not None
    assert row["retry_count"] == 0
    assert worker.metrics.processed == 1
    assert worker.metrics.processed_by_type["dummy_ok"] == 1


def test_process_one_marks_failed_on_handler_exception(
    worker: LlmWorker, db
) -> None:
    task_id = insert_pending_task(db, task_type="dummy_fail")

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        raise OllamaTerminalError("schema_mismatch: bla")

    worker.register_handler("dummy_fail", handler)

    assert worker.process_one() is True

    row = fetch_task_status(db, task_id)
    assert row["status"] == "failed"
    assert "schema_mismatch" in (row["error"] or "")
    assert row["retry_count"] == 1
    assert worker.metrics.failed == 1


def test_process_one_handles_transient_error_as_failed(
    worker: LlmWorker, db
) -> None:
    """Transient тоже считается failed (retry — отдельная политика)."""
    task_id = insert_pending_task(db, task_type="dummy_transient")

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        raise OllamaTransientError("timeout")

    worker.register_handler("dummy_transient", handler)

    assert worker.process_one() is True
    row = fetch_task_status(db, task_id)
    assert row["status"] == "failed"
    assert row["retry_count"] == 1


def test_process_one_marks_no_handler(worker: LlmWorker, db) -> None:
    task_id = insert_pending_task(db, task_type="unknown_type")

    assert worker.process_one() is True

    row = fetch_task_status(db, task_id)
    assert row["status"] == "failed"
    assert row["error"] == "no_handler"
    assert worker.metrics.skipped_no_handler == 1


def test_process_one_skipped_by_llm_metric(worker: LlmWorker, db) -> None:
    task_id = insert_pending_task(db, task_type="dummy_skip")

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        return HandlerResult(
            result={"skip": True, "reason": "too short"}, skipped_by_llm=True
        )

    worker.register_handler("dummy_skip", handler)

    assert worker.process_one() is True
    row = fetch_task_status(db, task_id)
    assert row["status"] == "done"
    assert row["result"] == {"skip": True, "reason": "too short"}
    assert worker.metrics.processed == 1
    assert worker.metrics.skipped_by_llm["dummy_skip"] == 1


def test_process_one_fifo_order(worker: LlmWorker, db) -> None:
    """ORDER BY created_at — первая INSERT'нутая обрабатывается первой."""
    a = insert_pending_task(db, task_type="dummy_fifo", payload={"n": 1})
    time.sleep(0.01)  # гарантированно разные created_at
    b = insert_pending_task(db, task_type="dummy_fifo", payload={"n": 2})

    order: list[uuid.UUID] = []

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        order.append(task.id)
        return HandlerResult(result={"n": task.payload["n"]})

    worker.register_handler("dummy_fifo", handler)
    worker.process_one()
    worker.process_one()

    assert order == [a, b]


def test_drain_processes_all_pending(worker: LlmWorker, db) -> None:
    for i in range(5):
        insert_pending_task(db, task_type="dummy_drain", payload={"i": i})

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        return HandlerResult(result={"i": task.payload["i"]})

    worker.register_handler("dummy_drain", handler)
    worker.drain()

    assert worker.metrics.processed == 5


# ── Crash recovery ──────────────────────────────────────────────────────


def test_run_resets_running_on_startup(
    migrated_db: str, fake_llm: OllamaChatClient, rate_limit: LLMRateLimiter, db
) -> None:
    """Strange-state row (running на старте) сбрасывается в pending."""
    # Симулируем падение прошлого worker'а: ставим status='running' вручную.
    task_id = insert_pending_task(db, task_type="dummy_recover")
    with db.cursor() as cur:
        cur.execute(
            "UPDATE llm_tasks SET status='running', started_at=now(), retry_count=2 "
            " WHERE id = %s",
            (task_id,),
        )
    db.commit()

    # Worker создаётся, открывает соединение, делает reset.
    w = LlmWorker(
        dsn=migrated_db,
        llm=fake_llm,
        rate_limit=rate_limit,
        poll_interval_s=0.05,
    )
    w._open_conn()
    n = w._reset_running_on_startup()
    w._close_conn()

    assert n == 1
    row = fetch_task_status(db, task_id)
    assert row["status"] == "pending"
    assert row["started_at"] is None
    # Reset на старте — без bump (это краш, не повтор).
    assert row["retry_count"] == 2


def test_reap_stale_running_bumps_retry(
    migrated_db: str, fake_llm: OllamaChatClient, rate_limit: LLMRateLimiter, db
) -> None:
    """Runtime reap бампит retry_count в отличие от startup-reset."""
    task_id = insert_pending_task(db, task_type="dummy_stale")
    # Ставим started_at в прошлое больше threshold'а.
    with db.cursor() as cur:
        cur.execute(
            "UPDATE llm_tasks "
            "   SET status='running', started_at=now() - interval '10 seconds', "
            "       retry_count=0 "
            " WHERE id = %s",
            (task_id,),
        )
    db.commit()

    w = LlmWorker(
        dsn=migrated_db,
        llm=fake_llm,
        rate_limit=rate_limit,
        stale_running_threshold_s=1.0,  # 1 секунда — наша строка точно старше
    )
    n = w.reap_stale_running()
    w._close_conn()

    assert n == 1
    row = fetch_task_status(db, task_id)
    assert row["status"] == "pending"
    assert row["retry_count"] == 1
    assert w.metrics.stale_reaped == 1


def test_reap_stale_running_skips_fresh(
    migrated_db: str, fake_llm: OllamaChatClient, rate_limit: LLMRateLimiter, db
) -> None:
    """Свежий running (started_at = now()) не reap'ится."""
    task_id = insert_pending_task(db, task_type="dummy_fresh")
    with db.cursor() as cur:
        cur.execute(
            "UPDATE llm_tasks SET status='running', started_at=now() WHERE id = %s",
            (task_id,),
        )
    db.commit()

    w = LlmWorker(
        dsn=migrated_db,
        llm=fake_llm,
        rate_limit=rate_limit,
        stale_running_threshold_s=60.0,
    )
    n = w.reap_stale_running()
    w._close_conn()

    assert n == 0
    row = fetch_task_status(db, task_id)
    assert row["status"] == "running"


# ── run() / stop() ─────────────────────────────────────────────────────


def test_run_blocks_and_stops_cleanly(
    migrated_db: str, fake_llm: OllamaChatClient, rate_limit: LLMRateLimiter, db
) -> None:
    """run() блокируется до stop(); затем drain'ит все pending (нет, не
    drain'ит — выходит сразу). Главное — корректный exit без зависаний."""
    insert_pending_task(db, task_type="dummy_run")

    processed: list[uuid.UUID] = []

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        processed.append(task.id)
        return HandlerResult(result={"done": True})

    w = LlmWorker(
        dsn=migrated_db,
        llm=fake_llm,
        rate_limit=rate_limit,
        poll_interval_s=0.05,
        reap_interval_s=0.0,
        stale_running_threshold_s=10.0,
    )
    w.register_handler("dummy_run", handler)

    runner = threading.Thread(target=w.run, daemon=True)
    runner.start()
    # Дать time на старт + обработку одной задачи.
    deadline = time.monotonic() + 2.0
    while not processed and time.monotonic() < deadline:
        time.sleep(0.05)
    w.stop()
    runner.join(timeout=2.0)

    assert not runner.is_alive(), "run() не завершился по stop()"
    assert len(processed) == 1


# ── periodic ───────────────────────────────────────────────────────────


def test_periodic_runs_at_interval(
    migrated_db: str, fake_llm: OllamaChatClient, rate_limit: LLMRateLimiter
) -> None:
    """register_periodic_task → fn вызывается несколько раз за окно."""
    calls = []

    def fn(conn: psycopg.Connection) -> None:
        calls.append(time.monotonic())

    w = LlmWorker(
        dsn=migrated_db,
        llm=fake_llm,
        rate_limit=rate_limit,
        poll_interval_s=10.0,
    )
    w.register_periodic_task("test_periodic", interval_s=0.1, fn=fn)
    w.start_periodic()

    time.sleep(0.5)
    w.stop()
    # Дать треду заметить stop.
    time.sleep(0.2)

    # При интервале 0.1 за 0.5s ожидаем 3-6 вызовов.
    assert len(calls) >= 3
    assert len(calls) <= 7


def test_periodic_invalid_interval(worker: LlmWorker) -> None:
    with pytest.raises(ValueError, match="interval_s"):
        worker.register_periodic_task("bad", interval_s=0, fn=lambda c: None)
