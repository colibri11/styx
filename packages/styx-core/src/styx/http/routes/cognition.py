"""Atomic cognitive continuity endpoints (wave 37)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import (
    CognitionCommitRequest,
    CognitionCommitResponse,
    CognitionObserveRequest,
    CognitionObserveResponse,
    CognitionPreturnRequest,
    CognitionPreturnResponse,
)
from styx.storage.cognition import CognitiveCommitConflict, SnapshotReplayConflict
from styx.storage.observations import ObservationBackpressure, ObservationConflict

router = APIRouter()


@router.post(
    "/cognition/observations",
    response_model=CognitionObserveResponse,
    dependencies=[Depends(require_auth)],
)
def observe(req: CognitionObserveRequest) -> CognitionObserveResponse:
    """Append one preliminary-reduced external difference."""
    session = registry.get(req.agent_id)
    payload = req.model_dump(exclude={"agent_id"})
    try:
        result = session.core.cognition_observe(**payload)
    except ObservationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ObservationBackpressure as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "observation_backpressure",
                "pending_count": exc.pending_count,
                "retry_after_s": exc.retry_after_s,
            },
            headers={"Retry-After": str(exc.retry_after_s)},
        ) from exc
    return CognitionObserveResponse.model_validate(result)


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
    try:
        result = session.core.cognition_commit(**payload)
    except CognitiveCommitConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CognitionCommitResponse.model_validate(result)
