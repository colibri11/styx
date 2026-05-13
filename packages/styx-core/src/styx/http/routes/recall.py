"""POST /recall — on-demand recall по semantic+keyword similarity."""

from __future__ import annotations

import time
from dataclasses import replace as _replace

from fastapi import APIRouter, Depends, HTTPException

from styx import turn_state
from styx.http import registry
from styx.http._wrap import populate_llm_text, should_wrap_for_llm
from styx.http.auth import require_auth
from styx.http.models import RecallMemory, RecallRequest, RecallResponse
from styx.storage.recall import recall_full

router = APIRouter()


@router.post(
    "/recall",
    response_model=RecallResponse,
    dependencies=[Depends(require_auth)],
)
def recall(
    req: RecallRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> RecallResponse:
    session = registry.get(req.agent_id)
    core = session.core
    if core._queries is None or core._embedding is None:
        raise HTTPException(
            status_code=503,
            detail="agent not fully initialized (queries/embedding missing)",
        )

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    full_cfg = core._recall_config.full
    if isinstance(req.limit, int) and 1 <= req.limit <= 20:
        full_cfg = _replace(full_cfg, memory_limit=req.limit)
    if req.min_score is not None:
        full_cfg = _replace(full_cfg, min_score=float(req.min_score))

    snapshot = turn_state.observe(req.agent_id)

    started = time.monotonic()
    with session.write_lock:
        if core._queries is None or core._embedding is None:
            raise HTTPException(status_code=503, detail="agent shut down mid-call")
        result = recall_full(
            queries=core._queries,
            embed_client=core._embedding,
            query=query,
            full_config=full_cfg,
            session_id=req.session_id,
            snapshot=snapshot,
        )
        core._conn.commit()  # type: ignore[union-attr]

    # Treker recall_event_ids для последующего classifier'а (волна 7c).
    if core._session_id is not None:
        ids = [
            hit.recall_event_id
            for hit in result.memories
            if hit.recall_event_id is not None
        ]
        if ids:
            core._recall_tracker.append(core._session_id, ids)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    response = RecallResponse(
        memories=[
            RecallMemory(
                id=str(m.id),
                content=m.content,
                score=float(m.score),
                role=m.role,
                created_at=m.created_at,
            )
            for m in result.memories
        ],
        queried_count=result.queried_count,
        internal_duplicates_removed=result.internal_duplicates_removed,
        elapsed_ms=elapsed_ms,
    )
    return populate_llm_text(response, "recall", wrap=wrap)
