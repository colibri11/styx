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

    ``chunks_embedded_inline`` (Defect-fix A) — False означает, что
    chunks записаны с ``embedding=NULL`` (async ingest большого
    документа); caller обязан enqueue'ить embedding в worker pool.
    """

    tail_memory_id: uuid.UUID | None
    document_id: uuid.UUID
    chunks_count: int
    summary_embedding: list[float] | None
    chunks_embedded_inline: bool = True


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

    Используется для tail_mode='summary' — длинный *субъективный*
    write (memory_store / dialogue consolidation): tail сохраняет
    обрезок самого содержания. Для документов это неверно (документ ≠
    память) — там tail_mode='act_marker', см. ``make_act_marker``.
    """
    if len(content) <= limit:
        return content
    cut = content[:limit]
    if not cut[-1].isspace() and limit < len(content) and not content[limit].isspace():
        last_ws = cut.rfind(" ")
        if last_ws > 0:
            cut = cut[:last_ws]
    return cut.rstrip() + "…"


def make_act_marker(
    *,
    document_id: uuid.UUID,
    original_name: str | None,
    mime_type: str | None,
    char_count: int,
    source: str,
    source_ref: str | None = None,
    limit: int,
) -> str:
    """Собрать маркер акта архивации документа (Defect-fix A).

    По концепции (IAmBook §V): документ — артефакт, его место архив
    (``documents``+``chunks``). В память (``memories``) от документа
    входит не обрезок содержания, а **маркер акта** — след того, что
    «я положил в архив документ такого-то рода». Маркер несёт:
    тип/происхождение (mime, source), о чём (имя), объём, ссылку на
    архив (``locator``) — но НЕ содержание документа.

    Результат гарантированно ≤ ``limit`` chars (CHECK constraint
    memories.content). Имя файла усекается если маркер не влезает.
    """
    name = (original_name or "без имени").strip() or "без имени"
    mime = (mime_type or "неизвестный тип").strip() or "неизвестный тип"
    locator = f"styx://store/{document_id}"

    def _compose(display_name: str) -> str:
        parts = [
            f"Я положил в архив документ «{display_name}»",
            f"тип: {mime}",
            f"объём: {char_count} симв.",
            f"происхождение: {source}",
        ]
        if source_ref:
            parts.append(f"источник: {source_ref}")
        parts.append(f"архив: {locator}")
        return "; ".join(parts) + "."

    marker = _compose(name)
    if len(marker) <= limit:
        return marker
    # Маркер не влезает — усекаем имя файла (служебные поля важнее).
    overflow = len(marker) - limit
    trimmed_len = max(8, len(name) - overflow - 1)
    trimmed_name = name[:trimmed_len].rstrip() + "…"
    marker = _compose(trimmed_name)
    if len(marker) <= limit:
        return marker
    # Крайний случай (очень длинные служебные поля) — жёсткий рез.
    return marker[: limit - 1].rstrip() + "…"


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
    tail_mode: str = "summary",
    embed_chunks_inline: bool = True,
    async_chunk_threshold: int | None = None,
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
    - ``create_tail_memory`` — False → tail-memory не создаётся,
      RoutedWriteResult возвращает ``tail_memory_id=None`` и
      ``summary_embedding=None``.
    - ``tail_mode`` (Defect-fix A) — режим tail-memory при
      ``create_tail_memory=True``:
      * ``'summary'`` (default) — tail = обрезок самого содержания
        (``make_tail_summary``). Для длинного *субъективного* write'а
        (memory_store / dialogue consolidation) — содержание и есть
        память.
      * ``'act_marker'`` — tail = маркер акта архивации документа
        (``make_act_marker``): «я положил в архив документ такого-то
        рода» — тип/происхождение/о чём/ссылка, БЕЗ содержания
        (IAmBook §V: документ ≠ память; в линию `я` входит акт, не
        артефакт). Используется documents-каналом (file-ingest).
    - ``embed_chunks_inline`` (Defect-fix A) — True (default): chunks
      embed'ятся синхронно. False: chunks INSERT'ятся с
      ``embedding=NULL`` (caller потом enqueue'ит embedding в worker
      pool — async ingest большого документа). tail-memory (если
      создаётся) embed'ится всегда inline — это один embed, дёшево, и
      маркер акта должен быть recallable сразу.
    - ``async_chunk_threshold`` (Defect-fix A) — если задан и chunker
      вернул больше chunks, чем порог, ``embed_chunks_inline``
      принудительно становится False (большой документ → async). Это
      позволяет решить inline/async за один проход chunking'а, не
      дублируя его в caller'е.
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
    if tail_mode not in ("summary", "act_marker"):
        raise ValueError(
            f"route_long_content: tail_mode должен быть "
            f"'summary'|'act_marker', получено {tail_mode!r}"
        )

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

    # Async-решение за один проход chunking'а: большой документ
    # (chunks > порога) → embedding откладывается в worker pool.
    chunks_inline = embed_chunks_inline
    if (
        async_chunk_threshold is not None
        and len(chunks) > async_chunk_threshold
    ):
        chunks_inline = False

    # embed_chunks_inline=False → chunks INSERT'ятся с embedding=NULL,
    # caller enqueue'ит embedding в worker pool (async ingest).
    chunk_embeddings: list[list[float] | None]
    if chunks_inline:
        chunk_embeddings = [embedder.embed(c.text) for c in chunks]
    else:
        chunk_embeddings = [None] * len(chunks)

    doc_metadata: dict[str, Any] = {"source_kind": kind}
    if extra_archive_metadata:
        doc_metadata.update(extra_archive_metadata)
    if document_metadata_extra:
        doc_metadata.update(document_metadata_extra)

    # documents.summary в режиме act_marker оставляем None — это поле
    # хранит обрезок содержания (для summary-mode tail'а), а маркер
    # акта содержания не несёт.
    doc_summary: str | None = None
    if create_tail_memory and tail_mode == "summary":
        doc_summary = make_tail_summary(content, limit=config.summary_chars)

    document_id = queries.insert_document(
        source=source,
        char_count=len(content),
        summary=doc_summary,
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
            chunks_embedded_inline=chunks_inline,
        )

    # tail-memory: в summary-mode — обрезок содержания; в act_marker-
    # mode — маркер акта архивации (Defect-fix A, IAmBook §V).
    if tail_mode == "summary":
        assert doc_summary is not None
        tail_text = doc_summary
    else:
        tail_text = make_act_marker(
            document_id=document_id,
            original_name=original_name,
            mime_type=mime_type,
            char_count=len(content),
            source=source,
            source_ref=source_ref,
            limit=config.summary_chars,
        )
    summary_embedding = embedder.embed(tail_text)

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
        content=tail_text,
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
        chunks_embedded_inline=chunks_inline,
    )
