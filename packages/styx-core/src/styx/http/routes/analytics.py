"""GET /analytics?agent_id=… — per-agent stats + global totals (волна 25).

Caller-scoped: ``agent_id`` обязателен в query string. memorybox
allowed enumerate-without-id для admin pool'а; в Styx без RLS это
бы leak'нуло metadata, поэтому запрещено.

Без Hermes wrapper'а (D10).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from styx.engine.cache_stats import get_cache_stats
from styx.engine.context import (
    get_styx_sanitized_blocks_by_tag,
    get_styx_sanitized_blocks_total,
)
from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import AnalyticsResponse

router = APIRouter()


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    dependencies=[Depends(require_auth)],
)
def analytics(agent_id: str = Query(..., min_length=1)) -> AnalyticsResponse:
    session = registry.get(agent_id)
    try:
        outcome = session.core.get_analytics()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AnalyticsResponse(
        agents=outcome.agents,
        global_=outcome.global_totals,
        pending_indexing=outcome.pending_indexing,
        # Волна 26.5 Fix 6 + волна 30 Phase F: production observability.
        # Daemon-wide cumulative — не agent-scoped. См.
        # `engine/context.py::_styx_sanitized_blocks_total` (агрегат)
        # и `_styx_sanitized_blocks_by_tag` (per-tag breakdown).
        styx_sanitized_blocks_total=get_styx_sanitized_blocks_total(),
        styx_sanitized_blocks_by_tag=get_styx_sanitized_blocks_by_tag(),
        # Волна 29 Phase E: per-agent cache hit/miss counters.
        # Push'ятся через POST /agent/cache_stats от Hermes
        # `StyxAnthropicTransport`. Cumulative с момента старта daemon'а.
        cache_stats=get_cache_stats(agent_id),
    )
