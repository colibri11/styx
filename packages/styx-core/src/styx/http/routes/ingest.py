"""POST /ingest_experience — внешний канал для pipelines (волна 23).

Идемпотентен по ``content_hash``: повторный ingest того же payload'а от
того же агента возвращает существующий ``memory_id`` с
``deduplicated=True``, без побочных эффектов.

Pipeline-канал — без gatekeeper'а / auto-link'а / store-routing'а.
Длинные документы (> 2400 chars) → 422; pipeline разбивает сам.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import IngestExperienceRequest, IngestExperienceResponse

router = APIRouter()


@router.post(
    "/ingest_experience",
    response_model=IngestExperienceResponse,
    dependencies=[Depends(require_auth)],
)
def ingest_experience(req: IngestExperienceRequest) -> IngestExperienceResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.ingest_experience(
            content=req.content,
            kind=req.kind,
            kind_src=req.kind_src,
            metadata=req.metadata,
            importance_provisional=req.importance_provisional,
            content_hash=req.content_hash,
            pipeline_id=req.pipeline_id,
            pipeline_version=req.pipeline_version,
            content_ref=req.content_ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # provider не инициализирован или ingest API disabled.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestExperienceResponse(
        memory_id=outcome.memory_id,
        deduplicated=outcome.deduplicated,
        content_hash=outcome.used_hash,
    )
