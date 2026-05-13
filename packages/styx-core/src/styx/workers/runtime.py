"""Worker runtime — drain loop ``llm_tasks`` + periodic tasks.

Архитектура: одна persistent connection в основном треде (drain loop
``llm_tasks``); каждая periodic-task запускается в собственном
daemon-треде с собственным psycopg-соединением (открывается на каждой
итерации и закрывается перед сном). Stop-сигнал — общий
``threading.Event``.

Один процесс ``styx-worker`` == один advisory-lock-holder для
consolidation sweep'ов (волна 7b). Multi-instance scaling в волне 7a
не предусмотрен; ``FOR UPDATE SKIP LOCKED`` на claim'е делает его
безопасным, но startup-reset (``status='running' → 'pending'`` без
bump retry_count) перестанет быть безопасным при > 1 worker'е — это
ставится явным TODO в момент когда понадобится horizontal scaling.

Контракт handler'а:

- Получает ``LlmTask`` + ``HandlerContext`` (conn, llm, rate_limit, logger).
- Возвращает ``HandlerResult(result, skipped_by_llm)`` либо raise:
  - ``OllamaTransientError`` → status=``failed``, retry_count+1.
  - ``OllamaTerminalError`` или любое другое → status=``failed``,
    retry_count+1, error содержит сообщение.
- Никаких commit'ов/rollback'ов внутри handler'а — runtime коммитит
  после успеха или rollback'ит после исключения.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from styx.embedding import EmbeddingClient
from styx.llm import LLMRateLimiter, OllamaChatClient

log = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

POLL_INTERVAL_S = 1.5
"""Пауза между пустыми claim'ами (порт DEFAULT_POLL_INTERVAL_MS)."""

REAP_INTERVAL_S = 60.0
"""Минимальный интервал между runtime-reap'ами (DEFAULT_REAP_INTERVAL_MS)."""

STALE_RUNNING_THRESHOLD_S = 300.0
"""``running``-row старше N секунд считается зависшей (DEFAULT_STALE_RUNNING_THRESHOLD_MS)."""

SHUTDOWN_TIMEOUT_S = 5.0
"""Сколько ждать завершения текущего process_one'а на stop()."""


# ── Public types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlmTask:
    id: uuid.UUID
    task_type: str
    memory_id: uuid.UUID | None
    payload: dict[str, Any]
    retry_count: int


@dataclass(frozen=True)
class HandlerContext:
    """Контекст, передаваемый в handler.

    ``conn`` — та же persistent connection runtime'а. Handler
    использует её для всех чтений/записей; runtime коммитит/откатывает
    транзакцию после возврата.

    ``llm`` / ``rate_limit`` — общие для всех handlers. Один лимитер на
    весь worker, чтобы не давить Ollama сразу несколькими LLM-вызовами
    подряд.

    ``embedder`` (волна 17) — sync embed для subjective writers
    (dialogue_batch_consolidation, future memory-over-memory). None если
    worker сконфигурирован без embedder'а — handler в этом случае
    fallback'ит на legacy-поведение «embedding=NULL, ждём reembed»
    (без gatekeeper apply'я).
    """

    conn: psycopg.Connection
    llm: OllamaChatClient
    rate_limit: LLMRateLimiter
    logger: logging.Logger
    embedder: EmbeddingClient | None = None


@dataclass(frozen=True)
class HandlerResult:
    result: dict[str, Any] | None = None
    skipped_by_llm: bool = False


class Handler(Protocol):
    def __call__(
        self, task: LlmTask, ctx: HandlerContext
    ) -> HandlerResult: ...


@dataclass
class WorkerMetrics:
    processed: int = 0
    failed: int = 0
    skipped_no_handler: int = 0
    stale_reaped: int = 0
    processed_by_type: dict[str, int] = field(default_factory=dict)
    skipped_by_llm: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "failed": self.failed,
            "skipped_no_handler": self.skipped_no_handler,
            "stale_reaped": self.stale_reaped,
            "processed_by_type": dict(self.processed_by_type),
            "skipped_by_llm": dict(self.skipped_by_llm),
        }


# ── Worker ──────────────────────────────────────────────────────────────


@dataclass
class _PeriodicTask:
    name: str
    interval_s: float
    fn: Callable[[psycopg.Connection], None]
    thread: threading.Thread | None = None


class LlmWorker:
    """Drain ``llm_tasks`` + хост для periodic-tasks.

    Не запускает поток автоматически. Caller вызывает либо ``run()``
    (блокирует до stop()), либо ``process_one()`` / ``drain()`` для
    тестов и one-shot прогонов.
    """

    def __init__(
        self,
        *,
        dsn: str,
        llm: OllamaChatClient,
        rate_limit: LLMRateLimiter,
        embedder: EmbeddingClient | None = None,
        poll_interval_s: float = POLL_INTERVAL_S,
        reap_interval_s: float = REAP_INTERVAL_S,
        stale_running_threshold_s: float = STALE_RUNNING_THRESHOLD_S,
    ) -> None:
        self._dsn = dsn
        self._llm = llm
        self._rate_limit = rate_limit
        self._embedder = embedder
        self._poll = poll_interval_s
        self._reap_interval = reap_interval_s
        self._stale_threshold = stale_running_threshold_s

        self._handlers: dict[str, Handler] = {}
        self._periodic: list[_PeriodicTask] = []
        self._metrics = WorkerMetrics()

        self._stop = threading.Event()
        self._conn: psycopg.Connection | None = None
        self._last_reap_at = 0.0

        # Healthz сигналы (волна 16). Все в monotonic-секундах.
        # `_started_at` ставится в run() / drain(); пока 0.0 — uptime
        # вычисляется как 0. `_last_iteration_at` обновляется на каждой
        # итерации `_main_loop`; `_last_drain_progress_at` — когда
        # `process_one()` либо подобрал task, либо завершил его (т.е.
        # очередь не зависла).
        self._started_at: float = 0.0
        self._last_iteration_at: float = 0.0
        self._last_drain_progress_at: float = 0.0

    # -- registration ---------------------------------------------------

    def register_handler(self, task_type: str, handler: Handler) -> None:
        self._handlers[task_type] = handler

    def register_periodic_task(
        self,
        name: str,
        interval_s: float,
        fn: Callable[[psycopg.Connection], None],
    ) -> None:
        """Запустить ``fn`` каждые ``interval_s`` секунд в daemon-треде.

        ``fn`` получает свежее psycopg-соединение (открыто перед
        вызовом, закрывается после) — изоляция от main drain-loop'а.

        Поток стартует только в ``run()`` или ``start_periodic()``.
        """
        if interval_s <= 0:
            raise ValueError(f"interval_s > 0 (got {interval_s})")
        self._periodic.append(_PeriodicTask(name=name, interval_s=interval_s, fn=fn))

    # -- lifecycle ------------------------------------------------------

    def start_periodic(self) -> None:
        """Запустить daemon-треды для всех зарегистрированных periodic.

        Идемпотентно — повторный вызов после старта no-op.
        """
        for pt in self._periodic:
            if pt.thread is not None and pt.thread.is_alive():
                continue
            pt.thread = threading.Thread(
                target=self._periodic_loop,
                args=(pt,),
                name=f"styx-periodic-{pt.name}",
                daemon=True,
            )
            pt.thread.start()

    def run(self) -> None:
        """Блокирующий запуск worker'а до ``stop()``.

        Открывает persistent connection, делает startup-reset
        running-rows, запускает periodic threads, потом крутит drain
        loop.
        """
        now = time.monotonic()
        self._started_at = now
        self._last_iteration_at = now
        self._last_drain_progress_at = now
        self._open_conn()
        try:
            n = self._reset_running_on_startup()
            if n > 0:
                log.info("startup reset %d stale 'running' task(s)", n)
            self.start_periodic()
            self._main_loop()
        finally:
            self._close_conn()

    def stop(self) -> None:
        self._stop.set()

    # -- single-step ----------------------------------------------------

    def process_one(self) -> bool:
        """Обработать одну ``pending``-задачу. ``False`` если очередь
        пуста."""
        self._ensure_conn()
        assert self._conn is not None
        try:
            task = self._claim_next(self._conn)
        except psycopg.Error as exc:
            log.warning("claim failed: %s", exc)
            self._conn.rollback()
            return False
        if task is None:
            return False

        handler = self._handlers.get(task.task_type)
        if handler is None:
            self._mark_no_handler(self._conn, task.id)
            self._metrics.skipped_no_handler += 1
            return True

        try:
            ctx = HandlerContext(
                conn=self._conn,
                llm=self._llm,
                rate_limit=self._rate_limit,
                logger=log,
                embedder=self._embedder,
            )
            out = handler(task, ctx)
        except Exception as exc:  # noqa: BLE001 — нам нужен любой
            self._conn.rollback()
            self._mark_failed(self._conn, task.id, exc)
            self._metrics.failed += 1
            log.warning("task %s failed: %s", task.id, exc)
            return True

        self._mark_done(self._conn, task.id, out.result)
        self._metrics.processed += 1
        self._metrics.processed_by_type[task.task_type] = (
            self._metrics.processed_by_type.get(task.task_type, 0) + 1
        )
        if out.skipped_by_llm:
            self._metrics.skipped_by_llm[task.task_type] = (
                self._metrics.skipped_by_llm.get(task.task_type, 0) + 1
            )
        return True

    def drain(self) -> None:
        """Прогнать ``process_one`` пока очередь не опустеет."""
        self._ensure_conn()
        while not self._stop.is_set():
            if not self.process_one():
                return

    def reap_stale_running(self) -> int:
        """Reset'нуть зависшие ``running``-rows старше threshold'а."""
        self._ensure_conn()
        assert self._conn is not None
        threshold_s = max(1, int(self._stale_threshold))
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks "
                "   SET status='pending', started_at=NULL, "
                "       retry_count=retry_count+1 "
                " WHERE status='running' "
                "   AND started_at < now() - make_interval(secs => %s)",
                (threshold_s,),
            )
            n = cur.rowcount or 0
        self._conn.commit()
        if n > 0:
            self._metrics.stale_reaped += n
            log.warning("reaped %d stale 'running' task(s)", n)
        return n

    @property
    def metrics(self) -> WorkerMetrics:
        return self._metrics

    def health_snapshot(self) -> dict[str, Any]:
        """Snapshot для healthz collector'ов (волна 16).

        Все возрасты — monotonic-секунды от текущего момента. Если
        ``run()`` ещё не вызывался (worker создан но не стартован) —
        ages = 0.0, queue-метрики пустые.
        """
        now = time.monotonic()
        if self._started_at == 0.0:
            uptime = 0.0
            iter_age = 0.0
            drain_age = 0.0
        else:
            uptime = now - self._started_at
            iter_age = now - self._last_iteration_at
            drain_age = now - self._last_drain_progress_at
        return {
            "uptime_s": uptime,
            "last_iteration_age_s": iter_age,
            "last_drain_progress_age_s": drain_age,
            "queue": {
                "processed": self._metrics.processed,
                "failed": self._metrics.failed,
                "skipped_no_handler": self._metrics.skipped_no_handler,
                "stale_reaped": self._metrics.stale_reaped,
            },
        }

    # ── internals ──────────────────────────────────────────────────────

    def _open_conn(self) -> None:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn)

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _ensure_conn(self) -> None:
        if self._conn is None:
            self._open_conn()

    def _reset_running_on_startup(self) -> int:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks SET status='pending', started_at=NULL "
                " WHERE status='running'"
            )
            n = cur.rowcount or 0
        self._conn.commit()
        return n

    def _claim_next(self, conn: psycopg.Connection) -> LlmTask | None:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "UPDATE llm_tasks "
                "   SET status='running', started_at=now() "
                " WHERE id = ( "
                "     SELECT id FROM llm_tasks "
                "      WHERE status='pending' "
                "      ORDER BY created_at "
                "      FOR UPDATE SKIP LOCKED "
                "      LIMIT 1 "
                " ) "
                "RETURNING id, task_type, memory_id, payload, retry_count"
            )
            row = cur.fetchone()
        conn.commit()
        if row is None:
            return None
        return LlmTask(
            id=row["id"],
            task_type=row["task_type"],
            memory_id=row["memory_id"],
            payload=row["payload"] or {},
            retry_count=row["retry_count"],
        )

    def _mark_done(
        self, conn: psycopg.Connection, task_id: uuid.UUID, result: dict[str, Any] | None
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks SET status='done', result=%s, completed_at=now() "
                " WHERE id = %s",
                (Jsonb(result) if result is not None else None, task_id),
            )
        conn.commit()

    def _mark_failed(
        self, conn: psycopg.Connection, task_id: uuid.UUID, exc: BaseException
    ) -> None:
        msg = str(exc)[:2000]
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks "
                "   SET status='failed', error=%s, completed_at=now(), "
                "       retry_count=retry_count+1 "
                " WHERE id = %s",
                (msg, task_id),
            )
        conn.commit()

    def _mark_no_handler(
        self, conn: psycopg.Connection, task_id: uuid.UUID
    ) -> None:
        # status CHECK допускает только pending/running/done/failed.
        # 'no_handler' фиксируется в error поле.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks "
                "   SET status='failed', error='no_handler', completed_at=now() "
                " WHERE id = %s",
                (task_id,),
            )
        conn.commit()

    # ── main loop ──────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        while not self._stop.is_set():
            self._last_iteration_at = time.monotonic()
            self._maybe_reap()
            if self._stop.is_set():
                break
            try:
                did = self.process_one()
            except Exception as exc:  # noqa: BLE001
                log.warning("unexpected loop error: %s", exc)
                did = False
            # `process_one()` вернулся (True/False) → drain не висит.
            # False = очередь пустая = тоже прогресс (нет накопленного
            # долга). Зависание = process_one зависает на LLM-вызове.
            self._last_drain_progress_at = time.monotonic()
            if self._stop.is_set():
                break
            if not did:
                self._stop.wait(timeout=self._poll)

    def _maybe_reap(self) -> None:
        now = time.monotonic()
        if now - self._last_reap_at < self._reap_interval:
            return
        self._last_reap_at = now
        try:
            self.reap_stale_running()
        except Exception as exc:  # noqa: BLE001
            log.warning("reap failed: %s", exc)

    def _periodic_loop(self, pt: _PeriodicTask) -> None:
        while not self._stop.is_set():
            try:
                with psycopg.connect(self._dsn) as conn:
                    pt.fn(conn)
            except Exception as exc:  # noqa: BLE001
                log.warning("periodic %s failed: %s", pt.name, exc)
            # wait либо до stop'а, либо до следующего interval'а.
            if self._stop.wait(timeout=pt.interval_s):
                return


# ── helpers для тестов ──────────────────────────────────────────────────


def insert_pending_task(
    conn: psycopg.Connection,
    *,
    task_type: str,
    memory_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Test/dev helper — INSERT pending task без триггеров."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO llm_tasks (task_type, memory_id, payload) "
            " VALUES (%s, %s, %s) RETURNING id",
            (task_type, memory_id, Jsonb(payload or {})),
        )
        row = cur.fetchone()
    conn.commit()
    if row is None:
        raise RuntimeError("INSERT не вернул id")
    return row[0]


def fetch_task_status(
    conn: psycopg.Connection, task_id: uuid.UUID
) -> dict[str, Any]:
    """Test helper — SELECT текущей строки task'а."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT status, result, error, retry_count, started_at, completed_at "
            "  FROM llm_tasks WHERE id = %s",
            (task_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"task {task_id} не найден")
    return row
