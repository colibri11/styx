"""Atomic cognitive continuity endpoints (wave 37)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import (
    CognitionCommitRequest,
    CognitionCommitResponse,
    CognitionPreturnRequest,
    CognitionPreturnResponse,
)
from styx.storage.cognition import SnapshotReplayConflict

router = APIRouter()


@router.post(
    "/cognition/preturn",
    response_model=CognitionPreturnResponse,
    dependencies=[Depends(require_auth)],
)
def preturn(req: CognitionPreturnRequest) -> CognitionPreturnResponse:
    session = registry.get(req.agent_id)
    payload = req.model_dump(exclude={"agent_id"})
    try:
        result = session.core.cognition_preturn(**payload)
    except SnapshotReplayConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CognitionPreturnResponse.model_validate(result)


@router.post(
    "/cognition/commit",
    response_model=CognitionCommitResponse,
    dependencies=[Depends(require_auth)],
)
def commit(req: CognitionCommitRequest) -> CognitionCommitResponse:
    session = registry.get(req.agent_id)
    payload = req.model_dump(exclude={"agent_id"})
    return CognitionCommitResponse.model_validate(
        session.core.cognition_commit(**payload)
    )
