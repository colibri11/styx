"""POST /memory_store — subjective write через selective gatekeeper.

Волна 17. Тонкий wrapper вокруг ``StyxMemoryCore.memory_store(...)``;
каждое решение gatekeeper'а (skip / merge / supersede / store) — в
ответе как ``action`` с relevant ids.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import MemoryStoreRequest, MemoryStoreResponse

router = APIRouter()


@router.post(
    "/memory_store",
    response_model=MemoryStoreResponse,
    dependencies=[Depends(require_auth)],
)
def memory_store(req: MemoryStoreRequest) -> MemoryStoreResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.memory_store(
            content=req.content,
            kind=req.kind,
            kind_src=req.kind_src,
            session_id=req.session_id,
            metadata=req.metadata,
            importance_provisional=req.importance_provisional,
        )
    except ValueError as exc:
        # Невалидный kind / session_id и т.д. — 422.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MemoryStoreResponse(
        action=outcome.action,
        memory_id=outcome.memory_id,
        existing_id=outcome.existing_id,
        similarity=outcome.similarity,
        routed=outcome.routed,
        document_id=outcome.document_id,
        chunks_count=outcome.chunks_count,
    )
