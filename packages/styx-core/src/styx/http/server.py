"""uvicorn entry для ``styx daemon run``.

Один процесс — два слоя:

1. FastAPI HTTP API (FastAPI + uvicorn) на ``http_bind:http_port``.
2. ``LlmWorker`` background-thread с handlers (importance,
   usage_classification, dialogue_batch_consolidation) и periodic
   tasks (sweep, emotional_tick, batch_consolidation_scheduler).

Loopback rule (D7): если ``http_bind`` не loopback и ``http_token``
пустой — daemon отказывается стартовать.
"""

from __future__ import annotations

import logging
import threading

from styx.config import StyxConfig
from styx.config import load as load_config
from styx.http import registry
from styx.http.app import create_app
from styx.observability.logging import setup_logging
from styx.workers.main import build_worker
from styx.workers.runtime import LlmWorker

log = logging.getLogger(__name__)

_LOOPBACK_BINDS = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


def enforce_auth_or_loopback(config: StyxConfig) -> None:
    """Заявленная инвариант: non-loopback bind + пустой token = отказ."""
    bind = (config.http_bind or "").strip()
    if bind not in _LOOPBACK_BINDS and not config.http_token:
        raise SystemExit(
            f"STYX_HTTP_BIND={bind!r} != loopback требует STYX_HTTP_TOKEN. "
            "Запуск отказан — открытый HTTP API без auth небезопасен."
        )


def start_workers(config: StyxConfig) -> tuple[LlmWorker, threading.Thread]:
    """Поднять ``LlmWorker`` в фоновом thread'е.

    Worker строится через ``workers.main.build_worker`` (тот же конфиг
    handlers + periodic tasks что и legacy ``styx-worker`` sidecar). В
    daemon-режиме (Phase E TODO D15) worker крутится рядом с FastAPI в
    одном процессе, разделяет StyxConfig + Postgres DSN + Ollama URL.

    Возвращает ``(worker, thread)`` — caller обязан вызвать
    ``worker.stop()`` + ``thread.join()`` при shutdown'е.
    """
    worker = build_worker(config)
    thread = threading.Thread(
        target=worker.run,
        name="styx-daemon-worker",
        daemon=True,
    )
    thread.start()
    log.info(
        "styx-daemon worker pool started "
        "(handlers: importance, usage_classification, dialogue_batch_consolidation; "
        "periodic: sweep, emotional_tick, batch_consolidation_scheduler)"
    )
    return worker, thread


def run_daemon(*, bind: str | None = None, port: int | None = None) -> None:
    """``styx daemon run`` entry point — FastAPI + worker pool в одном процессе."""
    import uvicorn

    config = load_config()
    if bind is not None:
        config = _override(config, http_bind=bind)
    if port is not None:
        config = _override(config, http_port=port)

    setup_logging(format=config.log_format)
    enforce_auth_or_loopback(config)

    app = create_app(config)

    # D15: worker pool в том же процессе. /readyz читает worker
    # health_snapshot через app.state.worker.
    worker, worker_thread = start_workers(config)
    app.state.worker = worker

    log.info(
        "styx-daemon starting bind=%s port=%d auth=%s",
        config.http_bind,
        config.http_port,
        "bearer" if config.http_token else "loopback",
    )

    try:
        uvicorn.run(
            app,
            host=config.http_bind,
            port=config.http_port,
            log_config=None,
        )
    finally:
        # Graceful shutdown:
        # 1. Останавливаем worker (drain in-flight task, stop periodic).
        # 2. Закрываем agent sessions (registry).
        try:
            worker.stop()
            worker_thread.join(timeout=10.0)
            if worker_thread.is_alive():
                log.warning("worker thread did not stop within 10s")
        except Exception as exc:  # noqa: BLE001
            log.warning("worker stop failed: %s", exc)

        for agent_id in registry.all_agent_ids():
            session = registry.unregister(agent_id)
            if session is not None:
                try:
                    session.core.shutdown()
                except Exception:  # noqa: BLE001
                    log.warning("shutdown failed for agent %s", agent_id)


def healthcheck(url: str) -> int:
    """``styx daemon healthcheck`` — exit 0 если /healthz status=ok."""
    import json
    import urllib.error
    import urllib.request

    target = url.rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(target, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"healthcheck failed: {exc}")
        return 1

    status = payload.get("status")
    print(f"status={status} postgres={payload.get('postgres')} version={payload.get('version')}")
    return 0 if status == "ok" else 1


def _override(config: StyxConfig, **fields):
    """Frozen dataclass override через replace."""
    from dataclasses import replace
    return replace(config, **fields)
