"""Healthz / Readyz endpoints — без auth, всегда доступны.

Liveness: процесс жив, Postgres ping ok.
Readiness: drain progress + Ollama ping ok.

Заменяет старый ``observability/healthz.py`` (stdlib HTTP server),
который был удалён в Phase A.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

import psycopg
from fastapi import APIRouter, Request, Response

from styx import __version__
from styx.http.models import HealthzResponse, ReadyzResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthzResponse)
def healthz(request: Request, response: Response) -> HealthzResponse:
    config = request.app.state.config
    started_at: float = request.app.state.started_at
    uptime = time.monotonic() - started_at
    pg_ok = _ping_postgres(config.database_url)
    status = "ok" if pg_ok else "down"
    if not pg_ok:
        response.status_code = 503
    return HealthzResponse(
        status=status,
        uptime_s=round(uptime, 2),
        postgres="ok" if pg_ok else "down",
        version=__version__,
    )


@router.get("/readyz", response_model=ReadyzResponse)
def readyz(request: Request, response: Response) -> ReadyzResponse:
    config = request.app.state.config
    started_at: float = request.app.state.started_at
    uptime = time.monotonic() - started_at
    ollama_ok = _ping_ollama(config.ollama_url)

    # D15: worker запущен внутри daemon — health_snapshot есть.
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        snap = worker.health_snapshot()
        drain_age = round(snap.get("last_drain_progress_age_s", 0.0), 2)
        queue_state = snap.get("queue", {}) or {}
    else:
        # TestClient / standalone-app без worker'а (юнит-тесты HTTP API).
        drain_age = None
        queue_state = {}

    threshold = config.healthz_readiness_threshold_s
    drain_ok = drain_age is None or drain_age < threshold
    status = "ok" if ollama_ok and drain_ok else "down"
    if not (ollama_ok and drain_ok):
        response.status_code = 503

    return ReadyzResponse(
        status=status,
        uptime_s=round(uptime, 2),
        last_drain_progress_age_s=drain_age,
        ollama="ok" if ollama_ok else "down",
        queue=queue_state,
        version=__version__,
    )


def _ping_postgres(dsn: str, *, timeout_s: float = 2.0) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=int(timeout_s)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ping_ollama(url: str, *, timeout_s: float = 2.0) -> bool:
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
