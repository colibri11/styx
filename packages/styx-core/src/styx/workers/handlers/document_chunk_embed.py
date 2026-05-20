"""Handler ``document_chunk_embed`` — async embed chunks (Defect-fix A).

Большой документ (chunks больше порога ``document_ingest_async_chunk_
threshold``) ingest'ится так: ``/ingest_document`` синхронно парсит
файл, режет на chunks, INSERT'ит ``documents`` + ``chunks`` с
``embedding=NULL``, создаёт маркер акта (embed маркера — один вызов,
дёшево) и enqueue'ит ЭТУ задачу; endpoint возвращается быстро.

Handler здесь дочитывает chunks без embedding'а и embed'ит их через
``ctx.embedder``. Это не LLM-задача (qwen3 не зовётся) — только embed,
поэтому handler не использует ``ctx.llm``.

Идемпотентность: handler embed'ит только chunks с ``embedding IS
NULL``; повторный прогон (retry после частичного fail) подберёт
оставшиеся. Документ без NULL-chunks → no-op success.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from styx.embedding import EmbeddingError
from styx.llm import OllamaTerminalError
from styx.storage.queries import AgentScopedQueries
from styx.workers.runtime import (
    Handler,
    HandlerContext,
    HandlerResult,
    LlmTask,
)

log = logging.getLogger(__name__)


DOCUMENT_CHUNK_EMBED_TASK_TYPE = "document_chunk_embed"


def _validate_payload(raw: Any) -> tuple[str, uuid.UUID]:
    """Returns (agent_id, document_id). Raises ValueError на schema mismatch."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"payload должен быть object, получено {type(raw).__name__}"
        )
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("payload.agent_id обязателен (str non-empty)")
    raw_doc = raw.get("document_id")
    if not isinstance(raw_doc, str) or not raw_doc:
        raise ValueError("payload.document_id обязателен (str non-empty)")
    try:
        document_id = uuid.UUID(raw_doc)
    except ValueError as exc:
        raise ValueError(f"document_id: invalid UUID {raw_doc!r}") from exc
    return agent_id, document_id


def create_document_chunk_embed_handler() -> Handler:
    """Factory для ``document_chunk_embed`` handler'а."""

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        try:
            agent_id, document_id = _validate_payload(task.payload)
        except ValueError as exc:
            raise OllamaTerminalError(f"invalid_payload: {exc}", exc)

        if ctx.embedder is None:
            raise OllamaTerminalError(
                "embedder_unavailable: document_chunk_embed требует embed, "
                "ctx.embedder=None"
            )

        queries = AgentScopedQueries(ctx.conn, agent_id)
        pending = queries.chunks_without_embedding(document_id)

        if not pending:
            # Документ уже полностью embed'нут (повторный прогон или
            # race с inline-путём) — no-op success.
            return HandlerResult(
                result={
                    "agent_id": agent_id,
                    "document_id": str(document_id),
                    "embedded": 0,
                    "remaining": 0,
                }
            )

        embedded = 0
        for chunk_id, content in pending:
            try:
                vec = ctx.embedder.embed(content)
            except EmbeddingError as exc:
                # Частичный fail: уже embed'нутые chunks остаются
                # (caller — runtime — закоммитит то что прошло до
                # raise? нет: runtime rollback'ит при исключении).
                # Поэтому НЕ raise'им — логируем и оставляем chunk
                # NULL'ом; retry задачи (или следующий прогон)
                # подберёт. Документ деградирует частично, не падает.
                log.warning(
                    "document_chunk_embed: embed chunk %s упал: %s",
                    chunk_id, exc,
                )
                continue
            queries.update_chunk_embedding(chunk_id, vec)
            embedded += 1

        remaining = len(pending) - embedded
        log.info(
            "document_chunk_embed: document=%s embedded=%d remaining=%d",
            document_id, embedded, remaining,
        )
        return HandlerResult(
            result={
                "agent_id": agent_id,
                "document_id": str(document_id),
                "embedded": embedded,
                "remaining": remaining,
            }
        )

    return handler
