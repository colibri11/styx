"""FastAPI app factory для Styx core daemon."""

from __future__ import annotations

import time

from fastapi import FastAPI

from styx import __version__
from styx.config import StyxConfig
from styx.http.routes import (
    agent as agent_route,
    agent_state as agent_state_route,
    analytics as analytics_route,
    confirm_usage as confirm_usage_route,
    context as context_route,
    dialogue as dialogue_route,
    explain as explain_route,
    healthz as healthz_route,
    ingest as ingest_route,
    ingest_document as ingest_document_route,
    maintenance as maintenance_route,
    memory_store as memory_store_route,
    pre_llm as pre_llm_route,
    recall as recall_route,
    reinterpret as reinterpret_route,
    relations as relations_route,
    search_archive as search_archive_route,
    sync_turn as sync_turn_route,
)


def create_app(config: StyxConfig) -> FastAPI:
    """Build FastAPI app с конфигом и подключёнными route'ами.

    State который handler'ы читают через ``request.app.state``:

    - ``config`` — ``StyxConfig``
    - ``started_at`` — ``time.monotonic()`` при старте
    - ``worker_queue`` — словарь из health_snapshot worker'а (опционально,
      устанавливается в ``server.py`` если worker запущен)
    - ``last_drain_progress_age_s`` — для /readyz (опционально)
    """
    app = FastAPI(
        title="styx-core",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.config = config
    app.state.started_at = time.monotonic()

    app.include_router(healthz_route.router)
    app.include_router(agent_route.router)
    app.include_router(sync_turn_route.router)
    app.include_router(recall_route.router)
    app.include_router(context_route.router)
    app.include_router(pre_llm_route.router)
    app.include_router(agent_state_route.router)
    app.include_router(memory_store_route.router)
    app.include_router(relations_route.router)
    app.include_router(search_archive_route.router)
    app.include_router(reinterpret_route.router)
    app.include_router(ingest_route.router)
    app.include_router(ingest_document_route.router)
    app.include_router(dialogue_route.router)
    app.include_router(explain_route.router)
    app.include_router(analytics_route.router)
    app.include_router(confirm_usage_route.router)
    app.include_router(maintenance_route.router)
    return app
