"""POST /confirm_usage — explicit ``used_in_output=true`` (волна 25).

Cross-agent guard через JOIN на ``memories.agent_id``: memory_id
чужого агента не апдейтится, попадает в response.missing.

Без Hermes wrapper'а (D10).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import ConfirmUsageRequest, ConfirmUsageResponse

router = APIRouter()


@router.post(
    "/confirm_usage",
    response_model=ConfirmUsageResponse,
    dependencies=[Depends(require_auth)],
)
def confirm_usage(req: ConfirmUsageRequest) -> ConfirmUsageResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.confirm_usage(memory_ids=req.memory_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ConfirmUsageResponse(
        updated=outcome.updated,
        requested=outcome.requested,
        missing=outcome.missing,
    )
