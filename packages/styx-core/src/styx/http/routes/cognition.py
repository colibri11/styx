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
    CognitionReadyClaimRequest,
    CognitionReadyClaimResponse,
    CognitionReadyResolveRequest,
    CognitionReadyResolveResponse,
    CognitionReadySignalRequest,
    CognitionReadySignalResponse,
)
from styx.storage.cognition import CognitiveCommitConflict, SnapshotReplayConflict
from styx.storage.observations import ObservationBackpressure, ObservationConflict
from styx.storage.ready_events import ReadyEventBackpressure, ReadyEventConflict

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
    except (ObservationBackpressure, ReadyEventBackpressure) as exc:
        pending_count = getattr(exc, "pending_count", 0)
        retry_after_s = getattr(exc, "retry_after_s", 5)
        raise HTTPException(
            status_code=429,
            detail={
                "code": "observation_backpressure",
                "pending_count": pending_count,
                "retry_after_s": retry_after_s,
            },
            headers={"Retry-After": str(retry_after_s)},
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


@router.post(
    "/cognition/ready-events/claim",
    response_model=CognitionReadyClaimResponse,
    dependencies=[Depends(require_auth)],
)
def claim_ready(req: CognitionReadyClaimRequest) -> CognitionReadyClaimResponse:
    session = registry.get(req.agent_id)
    try:
        result = session.core.cognition_ready_claim(
            **req.model_dump(exclude={"agent_id"})
        )
    except ReadyEventBackpressure as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return CognitionReadyClaimResponse.model_validate(result)


@router.post(
    "/cognition/ready-events/resolve",
    response_model=CognitionReadyResolveResponse,
    dependencies=[Depends(require_auth)],
)
def resolve_ready(req: CognitionReadyResolveRequest) -> CognitionReadyResolveResponse:
    session = registry.get(req.agent_id)
    try:
        result = session.core.cognition_ready_resolve(
            **req.model_dump(exclude={"agent_id"})
        )
    except ReadyEventConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CognitionReadyResolveResponse.model_validate(result)


@router.post(
    "/cognition/ready-events/signal",
    response_model=CognitionReadySignalResponse,
    dependencies=[Depends(require_auth)],
)
def signal_ready(req: CognitionReadySignalRequest) -> CognitionReadySignalResponse:
    session = registry.get(req.agent_id)
    try:
        result = session.core.cognition_ready_signal(
            signal_generation=req.signal_generation
        )
    except ReadyEventBackpressure as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return CognitionReadySignalResponse.model_validate(result)
