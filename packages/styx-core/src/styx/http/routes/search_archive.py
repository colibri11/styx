"""POST /search_archive — pull-канал к архиву (волна 20).

FTS+vector hybrid query поверх `documents`/`chunks` (миграции 0005 +
0006) и `memories WHERE role IN ('user','assistant')`.

Pull-only: результаты возвращаются caller'у в response, никогда не
инжектятся в context. См. `.design/waves/20-search-archive.md` § D8.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.engine import search_archive as _engine
from styx.http import registry
from styx.http._wrap import populate_llm_text, should_wrap_for_llm
from styx.http.auth import require_auth
from styx.http.models import (
    SearchArchiveRequest,
    SearchArchiveResponse,
    SearchArchiveResultModel,
)
from styx.storage.queries import AgentScopedQueries

router = APIRouter()


_ALLOWED_SCOPES = {"documents", "chunks", "dialogue", "all"}


def _agent_session(agent_id: str):
    return registry.get(agent_id)


def _agent_queries(session) -> AgentScopedQueries:
    queries = getattr(session.core, "_queries", None)
    if queries is None:
        raise HTTPException(
            status_code=503, detail="agent core not initialized",
        )
    return queries


def _agent_embedder(session):
    embedder = getattr(session.core, "_embedding", None)
    if embedder is None:
        raise HTTPException(
            status_code=503, detail="agent embedder not initialized",
        )
    return embedder


@router.post(
    "/search_archive",
    response_model=SearchArchiveResponse,
    dependencies=[Depends(require_auth)],
)
def search_archive(
    req: SearchArchiveRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> SearchArchiveResponse:
    if req.scope not in _ALLOWED_SCOPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"invalid scope {req.scope!r}; allowed: "
                f"{sorted(_ALLOWED_SCOPES)}"
            ),
        )

    session = _agent_session(req.agent_id)
    queries = _agent_queries(session)
    embedder = _agent_embedder(session)
    config = session.core._config.search_archive_config()

    common = dict(
        queries=queries,
        embedder=embedder,
        query=req.query,
        limit=req.limit,
        date_from=req.date_from,
        date_to=req.date_to,
        snapshot_cycle_start=req.snapshot_cycle_start,
        config=config,
    )
    if req.scope == "documents":
        resp = _engine.search_documents(**common)
    elif req.scope == "chunks":
        resp = _engine.search_chunks(**common)
    elif req.scope == "dialogue":
        resp = _engine.search_dialogue(**common)
    else:  # 'all'
        resp = _engine.search_all(**common)

    response = SearchArchiveResponse(
        results=[
            SearchArchiveResultModel(
                scope=r.scope,
                text=r.text,
                snippet=r.snippet,
                score=r.score,
                document_id=r.document_id,
                chunk_position=r.chunk_position,
                chunk_positions=(
                    list(r.chunk_positions) if r.chunk_positions else None
                ),
                char_start=r.char_start,
                char_end=r.char_end,
                memory_id=r.memory_id,
                role=r.role,
                created_at=r.created_at,
            )
            for r in resp.results
        ],
        total_matched=resp.total_matched,
    )
    return populate_llm_text(response, "archive", wrap=wrap)
