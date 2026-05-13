"""Document ingest orchestrator (волна 28).

File-ingest pipeline channel: parser → chunks → embed → INSERT document.
НЕ создаёт tail-memory в `memories` (pull-only архив, IAmBook §V).

Pure orchestrator — все DB-операции делегируются ``AgentScopedQueries``,
embed — ``EmbeddingClient``-протоколу. Транзакцией управляет caller
(commit после успешного return).

См. `.design/waves/28-documents-pipeline.md` § «D5 Tail-memory не
создаётся», «D9 content_hash SHA256», «D10 Path security», «D11 mime
detection».
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from styx.engine.document_parsers import (
    is_supported_extension,
    mime_for_extension,
    normalize_extension,
    parse,
)
from styx.engine.store_routing import (
    RoutedWriteResult,
    StoreRoutingConfig,
    route_long_content,
)

if TYPE_CHECKING:
    from styx.storage.queries import AgentScopedQueries


# Размер блока для streaming SHA256 (избегаем загрузки всего файла в память).
_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class DocumentIngestConfig:
    """Конфиг file-ingest pipeline (волна 28).

    ``allowed_roots`` — list абсолютных директорий-whitelist. Empty list
    (default) означает «no whitelist» (lab mode); production deploy
    должен задать STYX_INGEST_DOC_ROOTS.

    ``max_bytes`` — отказ если файл больше (default 50 MiB).
    """

    allowed_roots: list[Path] = field(default_factory=list)
    max_bytes: int = 50 * 1024 * 1024


@dataclass(frozen=True)
class IngestDocumentResult:
    """Результат file-ingest'а.

    ``deduplicated=True`` — повторный ingest того же файла (matched
    content_hash); document не парсился заново, ``chunks_count``
    отражает существующий ряд (caller может re-query при необходимости).
    """

    document_id: uuid.UUID
    deduplicated: bool
    chunks_count: int
    mime_type: str
    original_name: str
    size_bytes: int
    char_count: int
    content_hash: str


class _Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def validate_path(
    raw_path: str,
    config: DocumentIngestConfig,
) -> Path:
    """Валидация path под D10 (security).

    Pipeline:
        1. Absolute check.
        2. Resolve (strict=True) — кидает FileNotFoundError если
           файл не существует; защита от symlink escape (resolved
           path должен остаться в whitelist).
        3. Whitelist check (если непустой).
        4. Size guard.

    Все ошибки — ValueError с человекочитаемым detail (HTTP route
    маппит в 422).
    """
    p = Path(raw_path)
    if not p.is_absolute():
        raise ValueError(f"path must be absolute, got: {raw_path!r}")

    try:
        resolved = p.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {raw_path}") from exc

    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")

    if config.allowed_roots:
        ok = False
        for root in config.allowed_roots:
            try:
                resolved.relative_to(root)
                ok = True
                break
            except ValueError:
                continue
        if not ok:
            raise ValueError(
                f"path outside allowed roots: {resolved} "
                f"(roots: {[str(r) for r in config.allowed_roots]})"
            )

    size = resolved.stat().st_size
    if size > config.max_bytes:
        raise ValueError(
            f"file too large: {size} bytes > max {config.max_bytes}"
        )

    return resolved


def compute_content_hash(path: Path) -> str:
    """SHA256 от file bytes (streaming, 1 MiB blocks)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_HASH_CHUNK_BYTES)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ingest_document(
    queries: "AgentScopedQueries",
    embedder: _Embedder,
    *,
    raw_path: str,
    config: DocumentIngestConfig,
    store_routing: StoreRoutingConfig,
    source_ref: str | None = None,
    visibility: str | None = None,
    metadata: dict | None = None,
    content_hash: str | None = None,
) -> IngestDocumentResult:
    """File-ingest entry point.

    Pipeline:
        1. validate_path (absolute, exists, whitelist, size).
        2. SHA256 (или explicit ``content_hash``).
        3. find_document_by_content_hash — idempotency check.
           Hit → return с deduplicated=True.
        4. mime detect + magic bytes verify (в parse).
        5. parse → ParsedDocument.
        6. empty text → ValueError «empty document».
        7. route_long_content с create_tail_memory=False.
        8. return IngestDocumentResult.

    Не делает commit. Транзакцией управляет caller.

    Raises:
        ValueError: path invalid / unsupported extension / mime
            mismatch / encrypted PDF / empty document / chunker
            degenerate input.
    """
    path = validate_path(raw_path, config)

    ext = normalize_extension(path)
    if not is_supported_extension(ext):
        raise ValueError(
            f"unsupported extension: {ext or '<none>'} "
            f"(supported: .pdf, .docx, .xlsx, .md, .markdown, .txt, .text)"
        )

    effective_hash = content_hash or compute_content_hash(path)
    existing_id = queries.find_document_by_content_hash(effective_hash)
    if existing_id is not None:
        # Idempotency — повторный ingest того же файла того же agent_id.
        # chunks_count / mime / etc через query на existing — out of
        # scope orchestrator'а; возвращаем 0 chunks как маркер «look up
        # existing». HTTP route может расширить ответ при необходимости.
        return IngestDocumentResult(
            document_id=existing_id,
            deduplicated=True,
            chunks_count=0,
            mime_type=mime_for_extension(ext) or "",
            original_name=path.name,
            size_bytes=path.stat().st_size,
            char_count=0,
            content_hash=effective_hash,
        )

    parsed = parse(path)

    if not parsed.text:
        raise ValueError(
            "empty document (no extractable text — image-only PDF / "
            "scan / empty source file?)"
        )

    mime = mime_for_extension(ext) or ""
    size = path.stat().st_size

    doc_extra: dict = {}
    doc_extra.update(parsed.metadata)
    if metadata:
        doc_extra.update(metadata)

    result: RoutedWriteResult = route_long_content(
        queries,
        embedder,
        content=parsed.text,
        kind="document",
        kind_src="file_ingest",
        role="system",
        config=store_routing,
        source="ingest_document",
        create_tail_memory=False,
        content_hash=effective_hash,
        file_path=str(path),
        original_name=path.name,
        mime_type=mime,
        source_ref=source_ref,
        size_bytes=size,
        visibility=visibility,
        document_metadata_extra=doc_extra,
    )

    return IngestDocumentResult(
        document_id=result.document_id,
        deduplicated=False,
        chunks_count=result.chunks_count,
        mime_type=mime,
        original_name=path.name,
        size_bytes=size,
        char_count=len(parsed.text),
        content_hash=effective_hash,
    )
