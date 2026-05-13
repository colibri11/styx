"""Store-routing orchestrator (волна 19).

Длинный content (`> store_routing.limit`, default 2400) разделяется на
chunks, пишется в `documents` + `chunks`, а в `memories` остаётся
короткая tail-memory с `archive_ref` и truncate-summary.

Pure orchestrator — все DB-операции делегируются ``AgentScopedQueries``,
embed — внешнему ``EmbeddingClient``-протоколу. Транзакцией управляет
caller (commit'ит после auto-link для tail-memory).

См. `.design/waves/19-documents-chunks.md` § «D5 Summary strategy»,
«D7 Транзакционность», «D9 archive_ref locator».
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from styx.storage.queries import AgentScopedQueries

from styx.engine.chunker import ChunkData, chunk_text


@dataclass(frozen=True)
class StoreRoutingConfig:
    """Конфиг store-routing'а (волна 19).

    Дефолты — port memorybox `tools/memory.ts:STORE_ROUTING_THRESHOLD`
    (2400) + `chunker.ts` (1600/320). ``summary_chars=1500`` — отступ
    от CHECK constraint'а memories.content ≤ 2400.
    """

    enabled: bool = True
    limit: int = 2400
    chunk_size: int = 1600
    chunk_overlap: int = 320
    summary_chars: int = 1500


@dataclass(frozen=True)
class RoutedWriteResult:
    """Результат `route_long_content` — id'шки нового document'а,
    числа chunks и опционально tail-memory.

    ``tail_memory_id`` / ``summary_embedding`` — None при
    ``create_tail_memory=False`` (file-ingest pipeline волны 28:
    pull-only архив, не subjective write).

    ``summary_embedding`` (когда не None) возвращается caller'у чтобы
    auto-link для tail-memory мог обойтись без повторного embed'а —
    он уже посчитан внутри ``route_long_content``.
    """

    tail_memory_id: uuid.UUID | None
    document_id: uuid.UUID
    chunks_count: int
    summary_embedding: list[float] | None


class _Embedder(Protocol):
    """Минимальный sync-protocol embed-клиента (см. styx.embedding)."""

    def embed(self, text: str) -> list[float]: ...


def make_tail_summary(content: str, *, limit: int) -> str:
    """Truncate content до ``limit`` chars не разрывая слово.

    Если content ≤ limit — возвращает as-is, без маркера.
    Если truncate срезался по середине слова — отступ до ближайшего
    whitespace; маркер ``…`` добавляется только при truncate'е.

    Для мультибайтовых символов длина считается в codepoint'ах (Python
    `len`), не в байтах. CHECK constraint memories.content_length тоже
    использует `length(content)` — codepoint'ы; consistency.
    """
    if len(content) <= limit:
        return content
    cut = content[:limit]
    if not cut[-1].isspace() and limit < len(content) and not content[limit].isspace():
        last_ws = cut.rfind(" ")
        if last_ws > 0:
            cut = cut[:last_ws]
    return cut.rstrip() + "…"


def route_long_content(
    queries: "AgentScopedQueries",
    embedder: _Embedder,
    *,
    content: str,
    kind: str,
    kind_src: str,
    role: str,
    session_id: uuid.UUID | str | None = None,
    metadata: dict[str, Any] | None = None,
    importance_provisional: float | None = None,
    config: StoreRoutingConfig,
    source: str,
    extra_archive_metadata: dict[str, Any] | None = None,
    create_tail_memory: bool = True,
    content_hash: str | None = None,
    file_path: str | None = None,
    original_name: str | None = None,
    mime_type: str | None = None,
    source_ref: str | None = None,
    size_bytes: int | None = None,
    visibility: str | None = None,
    document_metadata_extra: dict[str, Any] | None = None,
) -> RoutedWriteResult:
    """Route long content в documents + chunks + опц. tail-memory.

    Не делает commit. Транзакцией управляет caller.

    Аргументы:
    - ``content`` — оригинальный (длинный) текст.
    - ``kind`` / ``kind_src`` / ``role`` — поля для tail-memory. Caller
      решает: для memory_store обычно ``kind_src='subjective_tail'``,
      для batch consolidation — ``'dialogue_batch_consolidation'``.
      Игнорируются при ``create_tail_memory=False``.
    - ``session_id`` / ``metadata`` / ``importance_provisional`` —
      форвардятся в `insert_memory` для tail-memory. Игнорируются при
      ``create_tail_memory=False``.
    - ``source`` — литерал call-site'а (`'memory_store'`,
      `'insert_batch_memory'`, `'ingest_document'`, ...) для
      documents.source и audit'а.
    - ``extra_archive_metadata`` — дополнительные ключи в archive_ref
      tail-memory'и (например, оригинальный `archive_ref` от dialogue
      batch handler'а — D11). Не применяется при create_tail_memory=False.
    - ``create_tail_memory`` — False для pipeline channels (file-ingest
      волны 28, IAmBook §V «архив vs память»). При False tail-memory
      не создаётся, embed summary не делается, RoutedWriteResult
      возвращает ``tail_memory_id=None`` и ``summary_embedding=None``.
    - ``content_hash`` — опц. SHA256 от source bytes (file-ingest)
      либо явный hash от caller'а. Пишется в documents.content_hash,
      работает с partial UNIQUE constraint для idempotency.
    - ``file_path`` / ``original_name`` / ``mime_type`` / ``source_ref``
      / ``size_bytes`` / ``visibility`` — file-метаданные (волна 28).
      Идут в новые колонки documents (миграция 0007).
    - ``document_metadata_extra`` — дополнительные ключи в
      documents.metadata JSONB (page_count для PDF, sheet_names для
      XLSX, etc — output парсеров).

    Raises:
    - ``ValueError`` — если chunker вернул 0 chunks (degenerate input).
    - ``EmbeddingError`` — embed одного из chunks или summary упал;
      caller ловит и rollback'ит транзакцию.

    Возвращает ``RoutedWriteResult`` с document_id, chunks_count и опц.
    tail_memory_id + summary_embedding.
    """
    chunks = chunk_text(
        content,
        size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    if not chunks:
        raise ValueError(
            "store_routing: chunker вернул 0 chunks для content > limit "
            "(degenerate input — whitespace-only?)"
        )

    chunk_embeddings: list[list[float]] = [
        embedder.embed(c.text) for c in chunks
    ]

    summary: str | None = None
    summary_embedding: list[float] | None = None
    if create_tail_memory:
        summary = make_tail_summary(content, limit=config.summary_chars)
        summary_embedding = embedder.embed(summary)

    doc_metadata: dict[str, Any] = {"source_kind": kind}
    if extra_archive_metadata:
        doc_metadata.update(extra_archive_metadata)
    if document_metadata_extra:
        doc_metadata.update(document_metadata_extra)

    document_id = queries.insert_document(
        source=source,
        char_count=len(content),
        summary=summary,
        content_hash=content_hash,
        metadata=doc_metadata,
        file_path=file_path,
        original_name=original_name,
        mime_type=mime_type,
        source_ref=source_ref,
        size_bytes=size_bytes,
        visibility=visibility,
    )

    queries.insert_chunks_batch(
        document_id,
        [
            (i, ck.text, vec, ck.char_start, ck.char_end)
            for i, (ck, vec) in enumerate(zip(chunks, chunk_embeddings))
        ],
    )

    if not create_tail_memory:
        return RoutedWriteResult(
            tail_memory_id=None,
            document_id=document_id,
            chunks_count=len(chunks),
            summary_embedding=None,
        )

    assert summary is not None
    assert summary_embedding is not None

    archive_ref: dict[str, Any] = {
        "kind": "document",
        "id": str(document_id),
        "locator": f"styx://store/{document_id}",
        "snippet": chunks[0].text[:1000],
    }
    if extra_archive_metadata:
        archive_ref["extra"] = extra_archive_metadata

    tail_id = queries.insert_memory(
        role=role,
        content=summary,
        kind=kind,
        kind_src=kind_src,
        session_id=session_id,
        embedding=summary_embedding,
        metadata=metadata or {},
        importance_provisional=importance_provisional,
        archive_ref=archive_ref,
    )

    return RoutedWriteResult(
        tail_memory_id=tail_id,
        document_id=document_id,
        chunks_count=len(chunks),
        summary_embedding=summary_embedding,
    )
