"""Agent lifecycle endpoints — /agent/initialize, /agent/shutdown.

Идемпотентный initialize (Q15): повторный вызов для зарегистрированного
agent_id возвращает существующую сессию (с обновлёнными tools).

shutdown — flush working_set, освобождает state, удаляет из registry.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from styx.engine import cache_stats as _cache_stats
from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import (
    AgentCacheStatsRequest,
    InitializeRequest,
    InitializeResponse,
    ShutdownRequest,
    ToolSchema,
)
from styx.providers.memory import StyxMemoryCore

router = APIRouter()


@router.post(
    "/agent/initialize",
    response_model=InitializeResponse,
    dependencies=[Depends(require_auth)],
)
def initialize(req: InitializeRequest, request: Request) -> InitializeResponse:
    existing = registry.get_optional(req.agent_id)
    if existing is not None:
        # Идемпотентный путь — обновляем session_id (если задан) и
        # возвращаем существующий tool schema. State полностью пересоздавать
        # не нужно — bridges уже configured, working_set restored, save
        # thread running. Это safe потому что MemoryCore методы не
        # завязаны на старый session_id (sync_turn принимает session per
        # call).
        if req.session_id:
            existing.core._session_id = _coerce_session(req.session_id)
        return InitializeResponse(
            agent_id=existing.agent_id,
            tools=[ToolSchema(**s) for s in existing.tool_schemas],
        )

    core = StyxMemoryCore(req.agent_id)
    core.initialize(
        session_id=req.session_id or "",
        agent_identity=req.agent_id,
        platform=req.platform,
        model=req.model,
    )
    tool_schemas = core.get_tool_schemas()
    session = registry.register(
        req.agent_id,
        core,
        write_lock=core._write_lock,
        tool_schemas=tool_schemas,
    )
    return InitializeResponse(
        agent_id=session.agent_id,
        tools=[ToolSchema(**s) for s in tool_schemas],
    )


@router.post(
    "/agent/shutdown",
    status_code=204,
    dependencies=[Depends(require_auth)],
)
def shutdown(req: ShutdownRequest) -> Response:
    session = registry.unregister(req.agent_id)
    if session is not None:
        try:
            session.core.shutdown()
        except Exception:  # noqa: BLE001 — fail-safe
            # Логирование внутри shutdown'а; не пропагандируем 500
            # клиенту, потому что registry уже очищена.
            pass
    return Response(status_code=204)


def _coerce_session(value: str | None):
    """Lazy-import чтобы избежать cycle при импорте route'ов."""
    from styx.providers.memory import _coerce_session_id
    return _coerce_session_id(value)


@router.post(
    "/agent/cache_stats",
    status_code=204,
    dependencies=[Depends(require_auth)],
)
def cache_stats(req: AgentCacheStatsRequest) -> Response:
    """Принять cache stats от Hermes (волна 29 Phase E).

    Hermes ``StyxAnthropicTransport`` после каждого LLM call'а зовёт
    ``extract_cache_stats(response)`` (Anthropic-specific
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens``) и
    шлёт результат в этот endpoint. Daemon аккумулирует per-agent
    кумулятивы; они доступны через ``GET /analytics?agent_id=…``
    (поле ``cache_stats``).

    Caller-scoped: agent_id обязателен. 204 No Content при successful
    record. Не валидируется существование agent'а в registry — это
    fire-and-forget metric channel, не lifecycle endpoint.
    """
    _cache_stats.record_cache_stats(
        req.agent_id,
        cache_read_tokens=req.cache_read_tokens,
        cache_creation_tokens=req.cache_creation_tokens,
    )
    return Response(status_code=204)
