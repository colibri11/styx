"""Relations / graph / link routes (волна 21)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http._wrap import populate_llm_text, should_wrap_for_llm
from styx.http.auth import require_auth
from styx.http.models import (
    GraphNode,
    GraphTraverseRequest,
    GraphTraverseResponse,
    LinkRequest,
    LinkResponse,
    RelationRow,
    RelationsQueryRequest,
    RelationsQueryResponse,
)
from styx.storage.queries import AgentScopedQueries

router = APIRouter()


def _agent_queries(agent_id: str) -> AgentScopedQueries:
    """AgentScopedQueries для зарегистрированного агента."""
    session = registry.get(agent_id)
    queries = getattr(session.core, "_queries", None)
    if queries is None:
        raise HTTPException(
            status_code=503,
            detail=f"agent_id={agent_id!r} core not initialized",
        )
    return queries


def _parse_uuid(raw: str | None) -> uuid.UUID | None:
    if raw is None:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"invalid UUID: {raw!r}",
        ) from None


@router.post(
    "/relations/query",
    response_model=RelationsQueryResponse,
    dependencies=[Depends(require_auth)],
)
def relations_query(
    req: RelationsQueryRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> RelationsQueryResponse:
    queries = _agent_queries(req.agent_id)
    source_uuid = _parse_uuid(req.source_id)
    target_uuid = _parse_uuid(req.target_id)
    rows = queries.query_relations(
        source_type=req.source_type,
        source_id=source_uuid,
        target_type=req.target_type,
        target_id=target_uuid,
        relation=req.relation,
        limit=req.limit,
    )
    out: list[RelationRow] = []
    for r in rows:
        out.append(RelationRow(
            id=str(r["id"]),
            source_type=r["source_type"],
            source_id=str(r["source_id"]),
            target_type=r["target_type"],
            target_id=str(r["target_id"]),
            relation=r["relation"],
            weight=float(r["weight"]),
            metadata=r["metadata"] or {},
            created_at=r.get("created_at"),
        ))
    response = RelationsQueryResponse(rows=out)
    return populate_llm_text(response, "relations", wrap=wrap)


@router.post(
    "/graph/traverse",
    response_model=GraphTraverseResponse,
    dependencies=[Depends(require_auth)],
)
def graph_traverse(
    req: GraphTraverseRequest,
    wrap: bool = Depends(should_wrap_for_llm),
) -> GraphTraverseResponse:
    queries = _agent_queries(req.agent_id)
    root_uuid = _parse_uuid(req.entity_id)
    if root_uuid is None:
        raise HTTPException(
            status_code=422, detail="entity_id is required"
        )

    # Detect root type + content_preview. В Styx все entities — memory
    # (одна таблица). После волны 19 расширится.
    with queries.conn.cursor() as cur:
        cur.execute(
            "SELECT LEFT(content, 100) FROM memories WHERE id = %s",
            (root_uuid,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"entity {req.entity_id!r} not found",
        )
    root_node = GraphNode(
        id=req.entity_id,
        type=req.entity_type or "memory",
        relation="",
        direction="outgoing",  # placeholder для root
        depth=0,
        weight=0.0,
        content_preview=row[0] or "",
    )

    nodes_raw = queries.traverse_graph(
        root_id=root_uuid,
        depth=req.depth,
        relation_filter=req.relation_filter,
        limit=req.limit,
    )
    nodes = [
        GraphNode(
            id=str(n["id"]),
            type=n["type"],
            relation=n["relation"],
            direction=n["direction"],
            depth=n["depth"],
            weight=n["weight"],
            content_preview=n.get("content_preview", ""),
        )
        for n in nodes_raw
    ]
    response = GraphTraverseResponse(root=root_node, nodes=nodes)
    return populate_llm_text(response, "relations", wrap=wrap)


@router.post(
    "/link",
    response_model=LinkResponse,
    dependencies=[Depends(require_auth)],
)
def link(req: LinkRequest) -> LinkResponse:
    session = registry.get(req.agent_id)
    core = session.core
    queries = getattr(core, "_queries", None)
    if queries is None:
        raise HTTPException(
            status_code=503,
            detail=f"agent_id={req.agent_id!r} core not initialized",
        )
    source_uuid = _parse_uuid(req.source_id)
    target_uuid = _parse_uuid(req.target_id)
    if source_uuid is None or target_uuid is None:
        raise HTTPException(
            status_code=422, detail="source_id and target_id required",
        )
    # Волна 34: insert_link+commit на постоянном per-agent _conn под
    # rollback-guard (на сбое conn не остаётся в aborted-state). write_lock
    # обязателен: guard зовёт rollback на shared _conn — без сериализации
    # параллельный write по тому же соединению был бы откачен.
    with session.write_lock:
        with core._guarded_write("link"):
            created = queries.insert_link(
                source_type=req.source_type,
                source_id=source_uuid,
                target_type=req.target_type,
                target_id=target_uuid,
                relation=req.relation,
                weight=req.weight,
                metadata=req.metadata,
            )
            queries.conn.commit()
    return LinkResponse(created=created)
