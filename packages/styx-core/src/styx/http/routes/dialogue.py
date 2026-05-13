"""POST /dialogue/* — dialogue tools HTTP routes (волна 24).

5 endpoints поверх `memories WHERE role IN ('user','assistant')`:

- ``POST /dialogue/save``           — explicit ad-hoc save.
- ``POST /dialogue/search``         — hybrid (FTS+vector) либо pure-vector.
- ``POST /dialogue/recent``         — chronological retrieval (oldest first).
- ``POST /dialogue/sessions``       — list of sessions с counts.
- ``POST /dialogue/prepare_summary``— transcript конкретной session.

Не имеют Hermes wrapper'а (D10 в waves/24) — это OpenClaw plugin
track регистрирует их как LLM-tools.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http._wrap import populate_llm_text, should_wrap_for_llm
from styx.http.auth import require_auth
from styx.http.models import (
    DialoguePrepareSummaryRequest,
    DialoguePrepareSummaryResponse,
    DialogueRecentRequest,
    DialogueRecentResponse,
    DialogueRecentRowModel,
    DialogueSaveRequest,
    DialogueSaveResponse,
    DialogueSearchHitModel,
    DialogueSearchRequest,
    DialogueSearchResponse,
    DialogueSessionInfoModel,
    DialogueSessionsRequest,
    DialogueSessionsResponse,
)

router = APIRouter()


@router.post(
    "/dialogue/save",
    response_model=DialogueSaveResponse,
    dependencies=[Depends(require_auth)],
)
def dialogue_save(
    req: DialogueSaveRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> DialogueSaveResponse:
    session = registry.get(req.agent_id)
    try:
        memory_id = session.core.dialogue_save(
            role=req.role,
            content=req.content,
            session_id=req.session_id,
            metadata=req.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = DialogueSaveResponse(memory_id=str(memory_id))
    return populate_llm_text(response, "dialogue", wrap=wrap)


@router.post(
    "/dialogue/search",
    response_model=DialogueSearchResponse,
    dependencies=[Depends(require_auth)],
)
def dialogue_search(
    req: DialogueSearchRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> DialogueSearchResponse:
    session = registry.get(req.agent_id)
    try:
        results = session.core.dialogue_search(
            query=req.query,
            session_id=req.session_id,
            after=req.after,
            before=req.before,
            semantic_only=req.semantic_only,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = DialogueSearchResponse(
        results=[
            DialogueSearchHitModel(
                memory_id=r.memory_id,
                role=r.role,
                content=r.content,
                score=r.score,
                created_at=r.created_at,
                session_id=r.session_id,
            )
            for r in results
        ]
    )
    return populate_llm_text(response, "dialogue", wrap=wrap)


@router.post(
    "/dialogue/recent",
    response_model=DialogueRecentResponse,
    dependencies=[Depends(require_auth)],
)
def dialogue_recent(
    req: DialogueRecentRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> DialogueRecentResponse:
    session = registry.get(req.agent_id)
    try:
        rows = session.core.dialogue_recent(
            session_id=req.session_id,
            before=req.before,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = DialogueRecentResponse(
        rows=[
            DialogueRecentRowModel(
                memory_id=r.memory_id,
                role=r.role,
                content=r.content,
                created_at=r.created_at,
                session_id=r.session_id,
            )
            for r in rows
        ]
    )
    return populate_llm_text(response, "dialogue", wrap=wrap)


@router.post(
    "/dialogue/sessions",
    response_model=DialogueSessionsResponse,
    dependencies=[Depends(require_auth)],
)
def dialogue_sessions(
    req: DialogueSessionsRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> DialogueSessionsResponse:
    session = registry.get(req.agent_id)
    try:
        sessions = session.core.dialogue_list_sessions(limit=req.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = DialogueSessionsResponse(
        sessions=[
            DialogueSessionInfoModel(
                session_id=s.session_id,
                message_count=s.message_count,
                first_message_at=s.first_message_at,
                last_message_at=s.last_message_at,
            )
            for s in sessions
        ]
    )
    return populate_llm_text(response, "dialogue", wrap=wrap)


@router.post(
    "/dialogue/prepare_summary",
    response_model=DialoguePrepareSummaryResponse,
    dependencies=[Depends(require_auth)],
)
def dialogue_prepare_summary(
    req: DialoguePrepareSummaryRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> DialoguePrepareSummaryResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.dialogue_prepare_summary(
            session_id=req.session_id,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = DialoguePrepareSummaryResponse(
        session_id=outcome.session_id,
        message_count=outcome.message_count,
        first_message_at=outcome.first_message_at,
        last_message_at=outcome.last_message_at,
        transcript=outcome.transcript,
    )
    return populate_llm_text(response, "dialogue", wrap=wrap)
