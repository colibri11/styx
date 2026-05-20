"""Entry-point ``styx worker run`` — собирает компоненты и стартует
``LlmWorker`` до SIGTERM/SIGINT.

В этой волне (7a) регистрируется единственный handler — importance.
Дальнейшие волны добавляют:
- 7b: periodic task ``sweep`` (lifecycle).
- 7c: handler ``usage_classification``.
- 7d: periodic task ``emotional_tick``.

Логика:

1. Загружает StyxConfig (DSN, ollama-url, llm-параметры).
2. Создаёт ``OllamaChatClient`` + ``LLMRateLimiter`` из конфига.
3. Создаёт ``LlmWorker``.
4. Регистрирует handler'ы.
5. Перехватывает SIGTERM/SIGINT → worker.stop().
6. Вызывает ``worker.run()`` (либо ``drain()`` при ``--once``).
"""

from __future__ import annotations

import logging
import signal
import sys
import urllib.error
import urllib.request

import psycopg

from styx import __version__ as STYX_VERSION
from styx.config import StyxConfig, load as load_config
from styx.embedding import OllamaEmbeddingClient
from styx.emotional.baseline import recompute_baseline
from styx.emotional.state import apply_instant_decay, list_active_agent_ids
from styx.llm import LLMRateLimiter, OllamaChatClient
from styx.observability.logging import setup_logging
from styx.workers.handlers.importance import (
    IMPORTANCE_TASK_TYPE,
    create_importance_handler,
)
from styx.workers.handlers.usage_classification import (
    USAGE_CLASSIFICATION_TASK_TYPE,
    create_usage_classification_handler,
)
from styx.workers.runtime import LlmWorker
from styx.workers.sweep.runner import run_sweep

log = logging.getLogger(__name__)


def build_worker(config: StyxConfig) -> LlmWorker:
    """Собрать ``LlmWorker`` со всеми handlers из текущего конфига.

    Чистая функция от конфига — для тестов и для main().
    """
    llm = OllamaChatClient(
        base_url=config.llm_url,
        model=config.llm_model,
        timeout_s=config.llm_timeout_s,
        max_attempts=config.llm_max_attempts,
    )
    rate_limit = LLMRateLimiter(
        capacity=config.llm_rate_limit_capacity,
        refill_per_second=config.llm_rate_limit_refill_per_s,
    )
    # Волна 17 — embedder в worker'е для sync embed субъективных
    # writers (dialogue_batch_consolidation handler и далее).
    embedder = OllamaEmbeddingClient(
        base_url=config.ollama_url,
        model=config.embedding_model,
        dim=config.embedding_dim,
        timeout=config.embedding_timeout_s,
    )
    worker = LlmWorker(
        dsn=config.database_url,
        llm=llm,
        rate_limit=rate_limit,
        embedder=embedder,
    )

    # Wave 7a — importance handler.
    worker.register_handler(IMPORTANCE_TASK_TYPE, create_importance_handler())

    # Wave 7c — usage classification handler.
    worker.register_handler(
        USAGE_CLASSIFICATION_TASK_TYPE, create_usage_classification_handler()
    )

    # Wave 7b — periodic lifecycle sweep. Стартует свою connection
    # в run_sweep'е (advisory lock держится на той session); main
    # worker connection не используется.
    sweep_lock_timeout_s = config.sweep_lock_timeout_s

    def _periodic_sweep(_conn) -> None:  # _conn выдаётся runtime'ом, но мы
        # его не используем — run_sweep сам открывает connection (нужно
        # для advisory lock на session-level).
        run_sweep(config.database_url, lock_timeout_s=sweep_lock_timeout_s)

    worker.register_periodic_task(
        "sweep", interval_s=config.sweep_interval_s, fn=_periodic_sweep
    )

    # Wave 7d — emotional tick. Раз в 60s по дефолту: для каждого
    # активного agent_id apply_instant_decay + recompute_baseline.
    def _emotional_tick(conn) -> None:
        agent_ids = list_active_agent_ids(conn)
        for aid in agent_ids:
            try:
                apply_instant_decay(conn, aid)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("emotional_tick decay для %s упал: %s", aid, exc)
                conn.rollback()
            try:
                recompute_baseline(conn, aid)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("emotional_tick baseline для %s упал: %s", aid, exc)
                conn.rollback()

    worker.register_periodic_task(
        "emotional_tick",
        interval_s=config.emotional_tick_interval_s,
        fn=_emotional_tick,
    )

    # Wave 14 — dialogue batch consolidation handler + scheduler.
    from styx.emotional.sentiment_batch import SentimentBatchMetrics
    from styx.workers.handlers.dialogue_batch_consolidation import (
        DIALOGUE_BATCH_TASK_TYPE,
        create_dialogue_batch_handler,
    )
    from styx.workers.sweep.batch_consolidation import (
        BatchSchedulerConfig,
        schedule_batch_tick,
    )

    batch_metrics = SentimentBatchMetrics()
    worker.register_handler(
        DIALOGUE_BATCH_TASK_TYPE,
        create_dialogue_batch_handler(
            batch_sentiment_enabled=config.batch_sentiment_enabled,
            batch_sentiment_metrics=batch_metrics,
            gatekeeper_config=config.gatekeeper_config(),
            auto_link_config=config.auto_link_config(),
            store_routing_config=config.store_routing_config(),
        ),
    )

    batch_config = BatchSchedulerConfig(
        enabled=config.batch_consolidation_enabled,
        trigger_messages=config.batch_trigger_messages,
        trigger_interval_s=config.batch_trigger_interval_s,
        period_gap_s=config.batch_period_gap_s,
    )

    def _batch_consolidation_tick(conn) -> None:
        if batch_config.enabled:
            schedule_batch_tick(conn, config=batch_config)

    worker.register_periodic_task(
        "batch_consolidation_scheduler",
        interval_s=config.batch_tick_interval_s,
        fn=_batch_consolidation_tick,
    )

    # Wave 21 — Hebbian relation decay. Раз в час уменьшает weight
    # 'co_retrieved' рёбер с last_reinforced > idle_days. Floor 1.0.
    if config.relation_decay_enabled:
        from styx.workers.sweep.relation_decay import run_relation_decay

        decay_rate = config.relation_decay_rate
        idle_days = config.relation_decay_idle_days

        def _relation_decay_tick(conn) -> None:
            run_relation_decay(
                conn,
                decay_rate=decay_rate,
                idle_threshold_days=idle_days,
            )
            conn.commit()

        worker.register_periodic_task(
            "relation_decay",
            interval_s=config.relation_decay_interval_s,
            fn=_relation_decay_tick,
        )

    # Wave 22 — reinterpret_merge handler + apply-sweeper.
    if config.reinterpret_enabled:
        from styx.workers.handlers.reinterpret_merge import (
            REINTERPRET_MERGE_TASK_TYPE,
            create_reinterpret_merge_handler,
        )
        from styx.workers.sweep.reinterpret_apply import (
            run_reinterpret_apply_sweep,
        )

        worker.register_handler(
            REINTERPRET_MERGE_TASK_TYPE,
            create_reinterpret_merge_handler(
                blend_weight=config.reinterpret_blend_weight,
            ),
        )

        def _reinterpret_apply_tick(conn) -> None:
            run_reinterpret_apply_sweep(conn)
            conn.commit()

        worker.register_periodic_task(
            "reinterpret_apply_sweeper",
            interval_s=config.reinterpret_apply_tick_s,
            fn=_reinterpret_apply_tick,
        )

    # Wave 22 — memory_daily_consolidation handler + scheduler + apply-sweeper.
    if config.memory_consolidation_enabled:
        from styx.workers.handlers.memory_daily_consolidation import (
            MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
            create_memory_daily_consolidation_handler,
        )
        from styx.workers.sweep.memory_consolidation import (
            run_memory_consolidation_apply_sweep,
            run_memory_consolidation_scheduler_tick,
        )

        worker.register_handler(
            MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
            create_memory_daily_consolidation_handler(),
        )

        memory_consolidation_cfg = config.memory_consolidation_config()

        def _memory_consolidation_scheduler_tick(conn) -> None:
            run_memory_consolidation_scheduler_tick(
                conn, config=memory_consolidation_cfg,
            )
            conn.commit()

        worker.register_periodic_task(
            "memory_consolidation_scheduler",
            interval_s=config.memory_consolidation_tick_s,
            fn=_memory_consolidation_scheduler_tick,
        )

        def _memory_consolidation_apply_tick(conn) -> None:
            run_memory_consolidation_apply_sweep(conn)
            conn.commit()

        worker.register_periodic_task(
            "memory_consolidation_apply_sweeper",
            interval_s=config.memory_consolidation_apply_tick_s,
            fn=_memory_consolidation_apply_tick,
        )

    # Defect-fix A — async embed chunks большого документа. file-ingest
    # большого документа INSERT'ит chunks с embedding=NULL и enqueue'ит
    # эту задачу; handler embed'ит chunks (не LLM-задача — только
    # ctx.embedder).
    from styx.workers.handlers.document_chunk_embed import (
        DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        create_document_chunk_embed_handler,
    )

    worker.register_handler(
        DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        create_document_chunk_embed_handler(),
    )

    return worker


def run_sweep_once(dsn: str | None = None, lock_timeout_s: float | None = None) -> int:
    """``styx worker sweep --once`` — один проход sweep'а и выход."""
    config = load_config()
    setup_logging(format=config.log_format, level="INFO")
    dsn = dsn or config.database_url
    timeout = lock_timeout_s if lock_timeout_s is not None else config.sweep_lock_timeout_s
    result = run_sweep(dsn, lock_timeout_s=timeout)
    log.info(
        "sweep finished: status=%s, skipped=%s, summary=%s, errors=%s",
        result.status, result.skipped, result.summary, result.errors,
    )
    if result.status == "failed":
        return 1
    return 0


def _ping_postgres(dsn: str, *, timeout_s: float = 2.0) -> bool:
    """Лёгкий health-check Postgres'а — ``SELECT 1`` с тайм-аутом.

    Используется в healthz collector'ах. Не кешируется: K8s probe
    interval обычно 10–30s, накладные расходы низкие. Любое исключение
    → ``False`` (down).
    """
    try:
        with psycopg.connect(dsn, connect_timeout=int(timeout_s)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ping_ollama(url: str, *, timeout_s: float = 2.0) -> bool:
    """Лёгкий health-check Ollama — GET /. 200 → up."""
    if not url:
        return False
    try:
        req = urllib.request.Request(url.rstrip("/") + "/", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    except Exception:  # noqa: BLE001
        return False


def _make_liveness_collector(worker: LlmWorker, config: StyxConfig):
    """Собрать collector для ``/healthz``.

    Liveness = процесс жив. Критерии: ``last_iteration_age_s`` <
    threshold + Postgres ping ok. Иначе status="down".
    """

    threshold = config.healthz_liveness_threshold_s
    dsn = config.database_url

    def collect() -> dict:
        snap = worker.health_snapshot()
        iter_age = snap["last_iteration_age_s"]
        pg_ok = _ping_postgres(dsn)
        # Если worker не стартовал (started_at==0, uptime==0) — мы CLI
        # collector обернут в HealthzServer.start() который зовётся
        # после worker.run() готовности. На безопасной стороне:
        # uptime==0 → считаем "starting", но не down.
        if snap["uptime_s"] <= 0:
            status = "ok" if pg_ok else "down"
        else:
            status = "ok" if iter_age < threshold and pg_ok else "down"
        return {
            "status": status,
            "uptime_s": round(snap["uptime_s"], 2),
            "last_iteration_age_s": round(iter_age, 2),
            "postgres": "ok" if pg_ok else "down",
            "queue": snap["queue"],
        }

    return collect


def _make_readiness_collector(worker: LlmWorker, config: StyxConfig):
    """Собрать collector для ``/readyz``.

    Readiness = воркер готов делать работу. Критерии: drain прогресс
    < threshold + Ollama ping ok.
    """

    threshold = config.healthz_readiness_threshold_s
    ollama_url = config.ollama_url

    def collect() -> dict:
        snap = worker.health_snapshot()
        drain_age = snap["last_drain_progress_age_s"]
        ollama_ok = _ping_ollama(ollama_url)
        if snap["uptime_s"] <= 0:
            status = "ok" if ollama_ok else "down"
        else:
            status = "ok" if drain_age < threshold and ollama_ok else "down"
        return {
            "status": status,
            "uptime_s": round(snap["uptime_s"], 2),
            "last_drain_progress_age_s": round(drain_age, 2),
            "ollama": "ok" if ollama_ok else "down",
            "queue": snap["queue"],
        }

    return collect


def run(*, once: bool = False) -> int:
    """Точка входа CLI ``styx worker run``."""
    config = load_config()
    setup_logging(format=config.log_format, level="INFO")
    worker = build_worker(config)

    log.info(
        "styx-worker starting (model=%s, dsn=%s, version=%s)",
        config.llm_model,
        _redact_dsn(config.database_url),
        STYX_VERSION,
    )

    if once:
        # Один проход — для CI / тестов. Без healthz (он не нужен).
        worker._open_conn()
        try:
            n = worker._reset_running_on_startup()
            if n > 0:
                log.info("startup reset %d stale 'running' task(s)", n)
            worker.drain()
        finally:
            worker._close_conn()
        return 0

    # Полный режим — крутится до сигнала. Healthz переехал в FastAPI
    # (см. packages/styx-core/src/styx/http/routes/healthz.py, Phase C).
    # workers/main.run() остаётся для backwards-compat (styx daemon run
    # внутри использует те же worker threads).

    def _on_signal(signum: int, frame) -> None:  # noqa: ARG001
        log.info("received signal %d → stopping", signum)
        worker.stop()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        worker.run()
    finally:
        pass
    log.info("styx-worker exited")
    return 0


def _redact_dsn(dsn: str) -> str:
    """Скрыть пароль в логах."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    if "@" not in rest:
        return dsn
    creds, host = rest.rsplit("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        return f"{scheme}://{user}:***@{host}"
    return dsn


if __name__ == "__main__":
    sys.exit(run())
