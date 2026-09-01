"""POST /affect/observe_turn — finalized host-turn observation seam."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import AffectObserveTurnRequest, AffectObserveTurnResponse

router = APIRouter()


@router.post(
    "/affect/observe_turn",
    response_model=AffectObserveTurnResponse,
    dependencies=[Depends(require_auth)],
)
def observe_turn(req: AffectObserveTurnRequest) -> AffectObserveTurnResponse:
    """Delegate a bounded, idempotency-addressed cognitive act to core."""
    session = registry.get(req.agent_id)
    payload = req.model_dump(exclude={"agent_id"})
    result = session.core.observe_affective_turn(**payload)
    return AffectObserveTurnResponse.model_validate(result)
