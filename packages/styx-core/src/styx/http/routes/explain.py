"""POST /explain/{decompose,lifetime,topK} — observability surface
для скоринг pipeline'а (волна 25).

3 endpoints поверх ``StyxMemoryCore.explain_*``:

- ``POST /explain/decompose``      — 11-факторный breakdown.
- ``POST /explain/lifetime``       — lifecycle trace.
- ``POST /explain/topK``           — top-K с factor-breakdown'ами.

Без Hermes wrapper'ов (D10 в waves/25). Default consumer —
оператор/CLI; LLM может вызывать через ``styx_explain`` tool, но
ради рефлексии (debug собственных recall'ов), не как источник памяти
для цитирования. Поэтому wrap-channel = ``explain`` (волна 30 D6
указывает в скилле «never quote to user»).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http._wrap import populate_llm_text, should_wrap_for_llm
from styx.http.auth import require_auth
from styx.http.models import (
    ExplainDecomposeRequest,
    ExplainDecomposeResponse,
    ExplainLifetimeRequest,
    ExplainLifetimeResponse,
    ExplainTopKRequest,
    ExplainTopKResponse,
)

router = APIRouter()


@router.post(
    "/explain/decompose",
    response_model=ExplainDecomposeResponse,
    dependencies=[Depends(require_auth)],
)
def explain_decompose(
    req: ExplainDecomposeRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> ExplainDecomposeResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.explain_decompose(
            memory_id=req.memory_id,
            query=req.query,
            top_k_limit=req.top_k_limit,
            min_score=req.min_score,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = ExplainDecomposeResponse(
        mode="decompose",
        memory_id=outcome.memory_id,
        kind=outcome.kind,
        query=outcome.query,
        final_score=outcome.final_score,
        rank_in_result_set=outcome.rank_in_result_set,
        top_k_limit=outcome.top_k_limit,
        would_be_returned=outcome.would_be_returned,
        return_reason=outcome.return_reason,  # type: ignore[arg-type]
        not_returned_because=outcome.not_returned_because,
        factors=outcome.factors,
        computed_at=outcome.computed_at,
    )
    return populate_llm_text(response, "explain", wrap=wrap)


@router.post(
    "/explain/lifetime",
    response_model=ExplainLifetimeResponse,
    dependencies=[Depends(require_auth)],
)
def explain_lifetime(
    req: ExplainLifetimeRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> ExplainLifetimeResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.explain_lifetime(
            memory_id=req.memory_id,
            include_recall_history=req.include_recall_history,
            recall_history_limit=req.recall_history_limit,
            prune_min_relevance=req.prune_min_relevance,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = ExplainLifetimeResponse(
        mode="lifetime",
        memory_id=outcome.memory_id,
        content_preview=outcome.content_preview,
        kind=outcome.kind,
        agent_id=outcome.agent_id,
        visibility=outcome.visibility,
        created_at=outcome.created_at,
        updated_at=outcome.updated_at,
        age_days=outcome.age_days,
        importance=outcome.importance,
        lifecycle=outcome.lifecycle,
        access=outcome.access,
        relevance=outcome.relevance,
        usefulness=outcome.usefulness,
        decay=outcome.decay,
        recall_history=outcome.recall_history,
        co_retrieval_links=outcome.co_retrieval_links,
        computed_at=outcome.computed_at,
    )
    return populate_llm_text(response, "explain", wrap=wrap)


@router.post(
    "/explain/topK",
    response_model=ExplainTopKResponse,
    dependencies=[Depends(require_auth)],
)
def explain_topk(
    req: ExplainTopKRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> ExplainTopKResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.explain_topk(
            query=req.query,
            limit=req.limit,
            kinds=req.kinds,
            after=req.after,
            before=req.before,
            min_score=req.min_score,
            include_factors=req.include_factors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = ExplainTopKResponse(
        mode="top_k",
        query=outcome.query,
        limit=outcome.limit,
        total_candidates_considered=outcome.total_candidates_considered,
        items=outcome.items,
        computed_at=outcome.computed_at,
    )
    return populate_llm_text(response, "explain", wrap=wrap)
