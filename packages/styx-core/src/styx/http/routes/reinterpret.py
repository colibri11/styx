"""POST /reinterpret — explicit reinterpret memory (волна 22).

Enqueue-only: HTTP не ждёт LLM. Apply через
`reinterpret_apply_sweeper` (~30-60s после close turn'а).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import ReinterpretRequest, ReinterpretResponse

router = APIRouter()


@router.post(
    "/reinterpret",
    response_model=ReinterpretResponse,
    dependencies=[Depends(require_auth)],
)
def reinterpret(req: ReinterpretRequest) -> ReinterpretResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.reinterpret_enqueue(
            memory_id=req.memory_id,
            new_understanding_text=req.new_understanding_text,
            weight=req.weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # provider не инициализирован или reinterpret disabled.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if outcome.status == "memory_not_found":
        raise HTTPException(
            status_code=404,
            detail={
                "status": "memory_not_found",
                "memory_id": outcome.memory_id,
            },
        )
    if outcome.status in ("cooldown", "already_pending"):
        detail: dict = {"status": outcome.status, "memory_id": outcome.memory_id}
        if outcome.last_reinterpreted_at:
            detail["last_reinterpreted_at"] = outcome.last_reinterpreted_at
        if outcome.next_available_at:
            detail["next_available_at"] = outcome.next_available_at
        if outcome.pending_application_id is not None:
            detail["pending_application_id"] = outcome.pending_application_id
        raise HTTPException(status_code=409, detail=detail)

    # status == 'queued'
    return ReinterpretResponse(
        status="queued",
        memory_id=outcome.memory_id,
        task_id=outcome.task_id,
        application_id=outcome.application_id,
        message=(
            "reinterpret queued; will apply once current turn closes "
            "and the sweeper runs (~30-60s)"
        ),
    )
