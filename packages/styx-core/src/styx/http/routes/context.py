"""HTTP routes для context engine.

Включает:

- ``POST /context/build`` (волна 9-10) — прямой вызов StyxComposer.compress
  для in-process Hermes plugin'а.
- ``POST /context/{bootstrap,ingest,ingest_batch,dispose}`` (волна 26
  Phase B) — lifecycle endpoints для OpenClaw ContextEngine plugin.
  Тонкие wrapper'ы над ``StyxMemoryCore``: agent/initialize-style
  bootstrap, raw insert single message (ingest), pairwise sync_turn
  (ingest_batch), agent/shutdown-style dispose.
- ``POST /context/{assemble,compact,after_turn}`` (волна 26 Phase C) —
  оставшиеся OpenClaw lifecycle hooks. assemble — head+tail+salient
  через StyxComposer (форвардит /context/build паттерн в OpenClaw shape);
  compact — Phase C minimal (см. models.py); after_turn — fire-and-
  forget hook для совместимости.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from styx.engine.context import StyxComposer
from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import (
    ContextAfterTurnRequest,
    ContextAfterTurnResponse,
    ContextAssembleRequest,
    ContextAssembleResponse,
    ContextBootstrapRequest,
    ContextBootstrapResponse,
    ContextBuildRequest,
    ContextBuildResponse,
    ContextCompactRequest,
    ContextCompactResponse,
    ContextDisposeRequest,
    ContextDisposeResponse,
    ContextIngestBatchRequest,
    ContextIngestBatchResponse,
    ContextIngestRequest,
    ContextIngestResponse,
)
from styx.providers.memory import StyxMemoryCore

router = APIRouter()


@router.post(
    "/context/build",
    response_model=ContextBuildResponse,
    dependencies=[Depends(require_auth)],
)
def build_context(req: ContextBuildRequest) -> ContextBuildResponse:
    session = registry.get(req.agent_id)
    # StyxComposer держит in-memory счётчики (compression_count, last_*),
    # которые HTTP-stateless API не может persist'ить между запросами
    # без extra ceremony. На текущей итерации создаём fresh composer
    # на каждый /context/build, attach'им к session, чтобы счётчики
    # копились между вызовами для одного agent_id.
    composer = _get_or_create_composer(session)
    out = composer.compress(
        list(req.messages),
        current_tokens=req.current_tokens,
        focus_topic=req.focus_topic,
    )
    salient_injected = any(
        msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and "[Styx —" in msg.get("content", "")
        for msg in out
    )
    return ContextBuildResponse(
        messages=out,
        compression_count=composer.compression_count,
        salient_injected=salient_injected,
    )


@router.post(
    "/context/bootstrap",
    response_model=ContextBootstrapResponse,
    dependencies=[Depends(require_auth)],
)
def context_bootstrap(req: ContextBootstrapRequest) -> ContextBootstrapResponse:
    """Engine first sees the session — idempotent agent/initialize.

    Если ``agent_id`` уже зарегистрирован — initialized=False, никакой
    state не пересоздаётся (working_set / save-thread остаются прежними).
    Если новый — создаётся ``StyxMemoryCore``, регистрируется в registry,
    initialized=True. ``parent_session_id`` зарезервирован под subagent
    сценарии и сейчас игнорируется.
    """

    existing = registry.get_optional(req.agent_id)
    if existing is not None:
        if req.session_id:
            existing.core._session_id = _coerce_session(req.session_id)
        return ContextBootstrapResponse(ok=True, initialized=False)

    core = StyxMemoryCore(req.agent_id)
    core.initialize(
        session_id=req.session_id or "",
        agent_identity=req.agent_id,
    )
    registry.register(
        req.agent_id,
        core,
        write_lock=core._write_lock,
        tool_schemas=core.get_tool_schemas(),
    )
    return ContextBootstrapResponse(ok=True, initialized=True)


@router.post(
    "/context/ingest",
    response_model=ContextIngestResponse,
    dependencies=[Depends(require_auth)],
)
def context_ingest(req: ContextIngestRequest) -> ContextIngestResponse:
    """Single-message ingest: raw ``insert_message`` + embed-after-commit.

    Без полного sync_turn-stack'а (D5 Phase B) — gatekeeper / auto-link /
    classifier / sentiment применяются только в ``/context/ingest_batch``,
    где гарантирована (user, assistant) пара. heartbeat tick'и (без
    payload'а) пропускаются с ingested=False.
    """

    if req.is_heartbeat:
        return ContextIngestResponse(ok=True, ingested=False, memory_id=None)

    session = registry.get(req.agent_id)
    mid = session.core.ingest_single_message(
        role=req.message.role,
        content=req.message.content,
        session_id=req.session_id or "",
    )
    return ContextIngestResponse(
        ok=True,
        ingested=mid is not None,
        memory_id=str(mid) if mid is not None else None,
    )


@router.post(
    "/context/ingest_batch",
    response_model=ContextIngestBatchResponse,
    dependencies=[Depends(require_auth)],
)
def context_ingest_batch(req: ContextIngestBatchRequest) -> ContextIngestBatchResponse:
    """Целый turn после run'а — pairwise ``sync_turn``.

    Группируем ``messages`` в смежные (user, assistant) пары и для каждой
    зовём ``core.sync_turn``. Хвостовая user-реплика без assistant'а —
    ``sync_turn(user, "")``. Замыкающая assistant-реплика без user —
    ``sync_turn("", assistant)``. system / tool messages в Phase B
    пропускаются (memories хранят только role∈{user,assistant}; system/tool
    выйдут отдельным каналом в будущих волнах).

    ingested_count считает каждое не-пустое user/assistant сообщение.
    """

    session = registry.get(req.agent_id)
    target_session = req.session_id or ""

    pending_user: str | None = None
    ingested = 0

    for msg in req.messages:
        role = msg.role
        content = msg.content or ""
        if role not in ("user", "assistant"):
            continue
        if role == "user":
            if pending_user is not None:
                # Два user подряд — flush первого без assistant'а.
                session.core.sync_turn(
                    user_content=pending_user,
                    assistant_content="",
                    session_id=target_session,
                )
                ingested += 1
            pending_user = content
            continue
        # assistant
        if pending_user is not None:
            session.core.sync_turn(
                user_content=pending_user,
                assistant_content=content,
                session_id=target_session,
            )
            ingested += 2 if content else 1
            pending_user = None
        else:
            session.core.sync_turn(
                user_content="",
                assistant_content=content,
                session_id=target_session,
            )
            if content:
                ingested += 1

    if pending_user is not None:
        session.core.sync_turn(
            user_content=pending_user,
            assistant_content="",
            session_id=target_session,
        )
        ingested += 1

    return ContextIngestBatchResponse(ok=True, ingested_count=ingested)


@router.post(
    "/context/dispose",
    response_model=ContextDisposeResponse,
    dependencies=[Depends(require_auth)],
)
def context_dispose(req: ContextDisposeRequest) -> ContextDisposeResponse:
    """Release engine resources.

    Если задан ``agent_id`` — сводится к ``agent/shutdown`` (unregister +
    core.shutdown). Если оба null — no-op (per-engine cleanup; cache'ей
    в core между агентами нет, future-extension под plugin reload /
    gateway stop).
    """

    if req.agent_id:
        session = registry.unregister(req.agent_id)
        if session is not None:
            try:
                session.core.shutdown()
            except Exception:  # noqa: BLE001 — fail-safe, registry уже очищена
                pass
    return ContextDisposeResponse(ok=True)


@router.post(
    "/context/assemble",
    response_model=ContextAssembleResponse,
    dependencies=[Depends(require_auth)],
)
def context_assemble(req: ContextAssembleRequest) -> ContextAssembleResponse:
    """Сборка геометрии входа на model run для **runtime** channel'а.

    Использует ``StyxComposer.assemble_for_runtime`` (волна 26.7) —
    eviction-normalize messages БЕЗ inject'а salient в messages array;
    salient идёт отдельно в ``system_prompt_addition`` как уже
    обёрнутая строка ``<styx-salient>...</styx-salient>`` (волна 30).

    Зачем расщеплено (ADR § 41.10):
    - Salient как ``role=user`` message между existing user и assistant
      нарушал strict alternation OpenAI Responses API (codex backend).
      API silently отбрасывал второй consecutive user → boевые агенты
      не видели salient в input'е.
    - ``systemPromptAddition`` — задокументированный channel в OpenClaw
      runtime (cf. ``selection-*.js: if (assembled.systemPromptAddition)
      systemPromptText = prependSystemPromptAddition(...)``) без
      alternation requirements.

    Отличие от ``/context/build`` (Hermes path):
    - ``/context/build`` продолжает использовать ``compress()`` — там
      salient инжектится в messages, alternation issue не наблюдалось,
      cache invariant 26.5 на messages-inject держится.
    - ``/context/assemble`` (этот endpoint) — для OpenClaw embedded:
      salient через system_prompt_addition.

    Симметричный fix для Hermes runtime path — отложен на волну 29
    (Hermes parity recheck), к тому моменту будет понятно, ломается
    ли там alternation тоже.
    """

    session = registry.get(req.agent_id)
    composer = _get_or_create_composer(session)
    result = composer.assemble_for_runtime(
        list(req.messages),
        current_tokens=req.token_budget,
        focus_topic=req.prompt,
    )
    out_messages = result["messages"]
    salient_text = result["salient_text"]
    return ContextAssembleResponse(
        messages=out_messages,
        estimated_tokens=_rough_token_estimate(out_messages),
        system_prompt_addition=salient_text,
        prompt_authority="assembled",
    )


@router.post(
    "/context/compact",
    response_model=ContextCompactResponse,
    dependencies=[Depends(require_auth)],
)
def context_compact(req: ContextCompactRequest) -> ContextCompactResponse:
    """Phase C minimal — возвращает {ok:true, compacted:false}.

    Реальное семантическое сжатие (memory_consolidation, ADR § 37) уже
    идёт async через workers/handlers/memory_daily_consolidation.
    /context/compact приходит от runtime'а на slash /compact / overflow
    recovery; engine с ownsCompaction:true НЕ должен блокировать turn
    LLM-сжатием (это создаст user-visible latency). Возврат
    ``compacted=false`` означает "no in-place change", runtime
    продолжает с теми же messages, а sweepers доделают.
    """

    # Validate registry membership — без зарегистрированного agent'а
    # /context/compact не имеет смысла.
    registry.get(req.agent_id)
    return ContextCompactResponse(
        ok=True,
        compacted=False,
        reason="async-consolidation",
    )


@router.post(
    "/context/after_turn",
    response_model=ContextAfterTurnResponse,
    dependencies=[Depends(require_auth)],
)
def context_after_turn(req: ContextAfterTurnRequest) -> ContextAfterTurnResponse:
    """Phase C minimal — fire-and-forget hook, возвращает {ok:true}.

    Реальные post-turn операции (drift recompute, salient cache refresh,
    sweep ticks) выполняются автоматически через workers/sweepers
    (волны 10/11/22). Endpoint существует чтобы plugin TS мог реализовать
    ContextEngine.afterTurn — runtime ждёт async-метод.
    """

    registry.get(req.agent_id)
    return ContextAfterTurnResponse(ok=True)


def _rough_token_estimate(messages: list[dict[str, Any]]) -> int:
    """Грубое приближение token count: ~4 char per token.

    Используется для AssembleResult.estimatedTokens — runtime дальше
    смотрит на это число при overflow-precheck. Точная токенизация
    делается provider'ом, наша оценка ≤ realу.
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text") or ""
                    if isinstance(text, str):
                        total += len(text)
    return (total + 3) // 4


def _get_or_create_composer(session) -> StyxComposer:
    """Lazy attach composer к session.core."""
    composer = getattr(session, "_composer", None)
    if composer is None:
        config = session.core._config
        ctx_len = 0
        if config is not None:
            ctx_len = getattr(config, "context_length", 0) or 0
        composer = StyxComposer(
            session.agent_id,
            context_length=ctx_len,
        )
        session._composer = composer
    return composer


def _coerce_session(value: str | None):
    """Lazy-import чтобы избежать cycle."""
    from styx.providers.memory import _coerce_session_id
    return _coerce_session_id(value)
