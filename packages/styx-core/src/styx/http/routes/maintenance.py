"""POST /maintenance/reembed — HTTP-канал к backfill'у `memories.embedding` (волна 31).

Host-агент в контейнере ходит к Styx только по HTTP (docker.sock не
примонтирован, CLI/`docker exec` недоступны). Этот эндпоинт дёргает тот же
``run_reembed`` (``commands/reembed.py``, волна 7e), что и CLI ``styx reembed``,
чтобы агент сам чинил NULL-хвосты, а не только наблюдал
``pending_indexing.memories`` через ``/analytics``.

Sync ``def``-хендлер: FastAPI оффлоудит его в threadpool, поэтому
rate-limited backfill-loop не блокирует event loop. Конкурентность —
session-level advisory lock (key ``9876543211``, отдельный от sweep'а
``9876543210``); session-level переживает per-row ``conn.commit()`` внутри
``run_reembed`` (xact-level снялся бы первым же коммитом).
"""

from __future__ import annotations

import logging
import time

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request

from styx.commands.reembed import run_reembed
from styx.embedding import make_embedding_client
from styx.http.auth import require_auth
from styx.http.models import MaintenanceReembedRequest, MaintenanceReembedResponse

log = logging.getLogger(__name__)

router = APIRouter()

# Отдельный от sweep'а (9876543210) — чтобы reembed и sweep не блокировали
# друг друга, но два параллельных reembed'а — взаимоисключались.
REEMBED_ADVISORY_LOCK_KEY = 9876543211


@router.post(
    "/maintenance/reembed",
    response_model=MaintenanceReembedResponse,
    dependencies=[Depends(require_auth)],
)
def maintenance_reembed(
    req: MaintenanceReembedRequest, request: Request
) -> MaintenanceReembedResponse:
    config = request.app.state.config
    t0 = time.monotonic()

    # Отдельный короткоживущий conn (НЕ session.core conn — тот занят
    # request/worker-транзакцией). 503 если БД недоступна.
    try:
        conn = psycopg.connect(config.database_url)
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"db unavailable: {exc}") from exc

    try:
        with conn:
            # Session-level lock: переживает per-row conn.commit() внутри
            # run_reembed. xact-level (pg_try_advisory_xact_lock) снялся бы
            # первым же коммитом — потому именно session-level.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_try_advisory_lock(%s::bigint) AS ok",
                    (REEMBED_ADVISORY_LOCK_KEY,),
                )
                row = cur.fetchone()
            conn.commit()

            if not row or not row[0]:
                log.warning(
                    "reembed skipped: advisory lock held by another instance"
                )
                return MaintenanceReembedResponse(
                    processed=0,
                    failed=0,
                    would_process=0,
                    dry_run=req.dry_run,
                    elapsed_ms=int((time.monotonic() - t0) * 1000),
                    skipped=True,
                )

            # Embed-клиент строим только на реальном backfill-пути (не на
            # skip/503) — конструктор дешёвый, но порядок чище.
            embed = make_embedding_client(
                base_url=config.ollama_url,
                model=config.embedding_model,
                dim=config.embedding_dim,
                timeout=config.embedding_timeout_s,
            )
            try:
                result = run_reembed(
                    conn=conn,
                    embed_client=embed,
                    mode=req.mode,
                    agent_id=req.agent_id,
                    limit=req.limit,
                    dry_run=req.dry_run,
                    batch_size=req.batch_size,
                    rate_per_second=req.rate_per_second,
                )
            except ValueError as exc:
                # Defensive: Pydantic уже отсёк невалидный body; долетает
                # только если run_reembed добавит новую инварианту.
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            finally:
                # Двойная страховка: явный unlock + закрытие conn на выходе
                # из `with conn` (psycopg3 закрывает session → lock и так
                # снимется, но explicit unlock детерминированнее).
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT pg_advisory_unlock(%s::bigint)",
                            (REEMBED_ADVISORY_LOCK_KEY,),
                        )
                    conn.commit()
                except psycopg.Error as exc:
                    log.warning("reembed: advisory_unlock failed: %s", exc)
    finally:
        try:
            conn.close()
        except psycopg.Error:
            pass

    return MaintenanceReembedResponse(
        processed=result.processed,
        failed=result.failed,
        would_process=result.would_process,
        dry_run=result.dry_run,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
        skipped=False,
    )
