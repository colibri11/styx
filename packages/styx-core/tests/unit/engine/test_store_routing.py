"""Unit-тесты для engine.store_routing (волна 19).

Покрытия:
- ``make_tail_summary``: truncate at word boundary, no truncate when
  ≤ limit, ellipsis marker only при truncate'е.
- ``route_long_content``: количество chunks, embed call'ов, archive_ref
  shape, kind_src + role + metadata projected в tail-memory.
- Degenerate input (chunker возвращает 0) → ValueError.
- Embed-fail в любом chunk'е / summary поднимается наверх (нет
  partial state — caller rollback'ит транзакцию).

Работает на mock'ах AgentScopedQueries + EmbeddingClient — без
Postgres'а / Ollama.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from styx.embedding import EmbeddingError
from styx.engine.store_routing import (
    RoutedWriteResult,
    StoreRoutingConfig,
    make_tail_summary,
    route_long_content,
)


# ── make_tail_summary ─────────────────────────────────────────────────


def test_summary_under_limit_returns_as_is() -> None:
    text = "Короткий текст."
    assert make_tail_summary(text, limit=100) == text


def test_summary_truncates_at_word_boundary() -> None:
    """limit=15 разрывает слово 'абвгдеж' посередине → отступ до
    предыдущего пробела."""
    text = "Первое слово абвгдеж" * 5
    out = make_tail_summary(text, limit=15)
    # Должен обрезать на пробеле, не разрывая слово.
    assert out.endswith("…")
    body = out[:-1].rstrip()
    assert body == "Первое слово"
    # Полный текст содержит обрезанную часть.
    assert text.startswith(body)


def test_summary_at_exact_limit_no_marker() -> None:
    text = "x" * 100
    out = make_tail_summary(text, limit=100)
    assert out == text
    assert not out.endswith("…")


def test_summary_just_above_limit_appends_marker() -> None:
    text = "слово " * 50
    out = make_tail_summary(text, limit=20)
    assert out.endswith("…")
    assert len(out) <= 21  # body ≤ limit + 1 char для маркера


def test_summary_no_word_boundary_falls_back_to_hard_cut() -> None:
    """Если в первых ``limit`` chars нет пробела — отступ к whitespace
    rfind возвращает -1; делаем грубый rstrip + marker."""
    text = "x" * 50 + " разделитель"
    out = make_tail_summary(text, limit=10)
    assert out.endswith("…")
    # Body — обрезанные xxx, без падения на assertion.
    assert "x" in out


# ── route_long_content (mock embedder/queries) ────────────────────────


@dataclass
class _StubEmbedder:
    """Возвращает детерминированные «vectors» (768 нулевых float'ов)
    + считает количество вызовов и аргументы."""

    dim: int = 768
    fail_on_text: str | None = None
    calls: list[str] = field(default_factory=list)

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail_on_text is not None and self.fail_on_text in text:
            raise EmbeddingError(f"stub embed-fail для {self.fail_on_text!r}")
        return [0.0] * self.dim


class _StubQueries:
    """Имитирует AgentScopedQueries для route_long_content.

    Запоминает аргументы в insert_document/insert_chunks_batch/
    insert_memory; возвращает фиксированные UUID'ы.
    """

    def __init__(self) -> None:
        self.document_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        self.tail_memory_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
        self.insert_document_args: dict[str, Any] | None = None
        self.insert_chunks_batch_args: tuple[uuid.UUID, list[Any]] | None = None
        self.insert_memory_args: dict[str, Any] | None = None

    def insert_document(self, **kwargs: Any) -> uuid.UUID:
        self.insert_document_args = kwargs
        return self.document_id

    def insert_chunks_batch(
        self, document_id: uuid.UUID, chunks: list[Any]
    ) -> None:
        self.insert_chunks_batch_args = (document_id, chunks)

    def insert_memory(self, **kwargs: Any) -> uuid.UUID:
        self.insert_memory_args = kwargs
        return self.tail_memory_id


_LONG_CONTENT = (
    "Длинный текст про важные дела. " * 100  # ~3000 chars
)


def _config() -> StoreRoutingConfig:
    return StoreRoutingConfig(
        enabled=True,
        limit=2400,
        chunk_size=1000,
        chunk_overlap=200,
        summary_chars=200,
    )


def test_route_writes_document_and_chunks_and_tail() -> None:
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = _config()

    result = route_long_content(
        queries,  # type: ignore[arg-type]
        embedder,
        content=_LONG_CONTENT,
        kind="note",
        kind_src="subjective_tail",
        role="system",
        config=cfg,
        source="memory_store",
    )

    assert isinstance(result, RoutedWriteResult)
    assert result.document_id == queries.document_id
    assert result.tail_memory_id == queries.tail_memory_id
    assert result.chunks_count >= 2  # ≥ 2 chunk'а на 3000 chars при limit=1000

    # insert_document получил source + char_count + summary.
    doc_args = queries.insert_document_args
    assert doc_args is not None
    assert doc_args["source"] == "memory_store"
    assert doc_args["char_count"] == len(_LONG_CONTENT)
    assert doc_args["summary"] is not None
    assert len(doc_args["summary"]) <= cfg.summary_chars + 1

    # insert_chunks_batch получил document_id и список из chunks_count записей.
    chunks_args = queries.insert_chunks_batch_args
    assert chunks_args is not None
    doc_id, chunk_records = chunks_args
    assert doc_id == queries.document_id
    assert len(chunk_records) == result.chunks_count
    # Каждая запись — (position, content, embedding, char_start, char_end).
    for i, rec in enumerate(chunk_records):
        assert rec[0] == i  # position
        assert isinstance(rec[1], str)  # content
        assert isinstance(rec[2], list) and len(rec[2]) == embedder.dim
        assert isinstance(rec[3], int)
        assert isinstance(rec[4], int)
        assert rec[3] <= rec[4]

    # insert_memory: tail-memory с archive_ref правильной формы.
    tail_args = queries.insert_memory_args
    assert tail_args is not None
    assert tail_args["kind_src"] == "subjective_tail"
    assert tail_args["role"] == "system"
    assert tail_args["kind"] == "note"
    assert tail_args["embedding"] == [0.0] * embedder.dim
    archive_ref = tail_args["archive_ref"]
    assert archive_ref["kind"] == "document"
    assert archive_ref["id"] == str(queries.document_id)
    assert archive_ref["locator"] == f"styx://store/{queries.document_id}"
    assert isinstance(archive_ref["snippet"], str)
    assert len(archive_ref["snippet"]) <= 1000


def test_embedder_called_for_each_chunk_plus_summary() -> None:
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = _config()

    result = route_long_content(
        queries,  # type: ignore[arg-type]
        embedder,
        content=_LONG_CONTENT,
        kind="note",
        kind_src="subjective_tail",
        role="system",
        config=cfg,
        source="memory_store",
    )

    # N chunks + 1 summary embed = total embedder calls.
    assert len(embedder.calls) == result.chunks_count + 1


def test_zero_chunks_raises_value_error() -> None:
    """Whitespace-only content > limit → chunker [] → ValueError."""
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = _config()

    whitespace = " \n\n " * 1000  # > 2400 chars whitespace
    with pytest.raises(ValueError):
        route_long_content(
            queries,  # type: ignore[arg-type]
            embedder,
            content=whitespace,
            kind="note",
            kind_src="subjective_tail",
            role="system",
            config=cfg,
            source="memory_store",
        )


def test_embed_fail_on_chunk_propagates() -> None:
    queries = _StubQueries()
    # Будет fail на любом chunk'е (text содержит «дела»).
    embedder = _StubEmbedder(fail_on_text="дела")
    cfg = _config()

    with pytest.raises(EmbeddingError):
        route_long_content(
            queries,  # type: ignore[arg-type]
            embedder,
            content=_LONG_CONTENT,
            kind="note",
            kind_src="subjective_tail",
            role="system",
            config=cfg,
            source="memory_store",
        )

    # Caller увидит exception — partial state не коммит'ится (queries
    # вызывались, но caller обязан rollback'нуть транзакцию).


def test_extra_archive_metadata_projected_into_archive_ref() -> None:
    """Phase C: insert_batch_memory wire передаёт оригинальный
    archive_ref dialogue range; route_long_content переносит его в
    archive_ref.extra."""
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = _config()

    extra = {"dialogue_range": {"from": "2026-05-04T10:00:00Z", "to": "2026-05-04T10:30:00Z"}}

    route_long_content(
        queries,  # type: ignore[arg-type]
        embedder,
        content=_LONG_CONTENT,
        kind="episode",
        kind_src="dialogue_batch_consolidation",
        role="summary",
        config=cfg,
        source="insert_batch_memory",
        extra_archive_metadata=extra,
    )

    tail_args = queries.insert_memory_args
    assert tail_args is not None
    assert tail_args["archive_ref"]["extra"] == extra

    doc_args = queries.insert_document_args
    assert doc_args is not None
    # documents.metadata также содержит дополнительные ключи.
    assert doc_args["metadata"]["dialogue_range"] == extra["dialogue_range"]


def test_kind_src_and_role_forwarded() -> None:
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = _config()

    route_long_content(
        queries,  # type: ignore[arg-type]
        embedder,
        content=_LONG_CONTENT,
        kind="episode",
        kind_src="dialogue_batch_consolidation",
        role="summary",
        config=cfg,
        source="insert_batch_memory",
    )

    tail_args = queries.insert_memory_args
    assert tail_args is not None
    assert tail_args["kind"] == "episode"
    assert tail_args["kind_src"] == "dialogue_batch_consolidation"
    assert tail_args["role"] == "summary"


def test_session_id_and_importance_forwarded() -> None:
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = _config()
    sid = uuid.UUID("33333333-3333-3333-3333-333333333333")

    route_long_content(
        queries,  # type: ignore[arg-type]
        embedder,
        content=_LONG_CONTENT,
        kind="note",
        kind_src="subjective_tail",
        role="system",
        session_id=sid,
        importance_provisional=0.7,
        config=cfg,
        source="memory_store",
        metadata={"k": "v"},
    )

    tail_args = queries.insert_memory_args
    assert tail_args is not None
    assert tail_args["session_id"] == sid
    assert tail_args["importance_provisional"] == 0.7
    assert tail_args["metadata"] == {"k": "v"}
