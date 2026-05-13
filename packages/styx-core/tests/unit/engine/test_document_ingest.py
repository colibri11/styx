"""Unit-tests для document_ingest orchestrator (волна 28).

Mock-based: AgentScopedQueries + Embedder заменяются stub'ами. Реальный
INSERT в Postgres тестируется в `tests/integration/http/test_ingest_document.py`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from styx.engine.document_ingest import (
    DocumentIngestConfig,
    compute_content_hash,
    ingest_document,
    validate_path,
)
from styx.engine.store_routing import StoreRoutingConfig


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "documents"


# ── Mocks ───────────────────────────────────────────────────────────


class _StubEmbedder:
    """Возвращает детерминированный 8-dim вектор от длины текста."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text) % 7)] * 8


class _StubQueries:
    """Минимальный stub под `AgentScopedQueries` для orchestrator-тестов.

    Хранит вызовы; не делает реальной DB-работы.
    """

    def __init__(self, *, existing_hashes: set[str] | None = None) -> None:
        self.existing_hashes = existing_hashes or set()
        self.inserted_documents: list[dict] = []
        self.inserted_chunks: list[tuple[uuid.UUID, list]] = []

    def find_document_by_content_hash(
        self, content_hash: str
    ) -> uuid.UUID | None:
        if content_hash in self.existing_hashes:
            return uuid.UUID(int=0xABCDEF)
        return None

    def insert_document(self, **kwargs) -> uuid.UUID:
        new_id = uuid.uuid4()
        self.inserted_documents.append({"id": new_id, **kwargs})
        return new_id

    def insert_chunks_batch(
        self, document_id: uuid.UUID, chunks: list
    ) -> None:
        self.inserted_chunks.append((document_id, chunks))

    def insert_memory(self, **kwargs) -> uuid.UUID:
        # Не должно вызываться — file-ingest НЕ создаёт tail-memory.
        raise AssertionError(
            "insert_memory вызван в file-ingest pipeline — должен быть skip"
        )


def _store_cfg() -> StoreRoutingConfig:
    return StoreRoutingConfig(
        enabled=True, limit=2400, chunk_size=1600, chunk_overlap=320,
        summary_chars=1500,
    )


# ── validate_path ───────────────────────────────────────────────────


def test_validate_path_absolute_required(tmp_path: Path) -> None:
    cfg = DocumentIngestConfig()
    with pytest.raises(ValueError, match="must be absolute"):
        validate_path("relative/path.pdf", cfg)


def test_validate_path_file_not_found(tmp_path: Path) -> None:
    cfg = DocumentIngestConfig()
    with pytest.raises(ValueError, match="file not found"):
        validate_path(str(tmp_path / "missing.pdf"), cfg)


def test_validate_path_not_a_file(tmp_path: Path) -> None:
    cfg = DocumentIngestConfig()
    with pytest.raises(ValueError, match="not a regular file"):
        validate_path(str(tmp_path), cfg)


def test_validate_path_whitelist_enforced(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("hi")
    outside = tmp_path / "other"
    outside.mkdir()
    cfg = DocumentIngestConfig(allowed_roots=[outside])
    with pytest.raises(ValueError, match="outside allowed roots"):
        validate_path(str(target), cfg)


def test_validate_path_whitelist_ok(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    target = root / "file.txt"
    target.write_text("hi")
    cfg = DocumentIngestConfig(allowed_roots=[root])
    resolved = validate_path(str(target), cfg)
    assert resolved == target.resolve()


def test_validate_path_symlink_escape_blocked(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real = outside / "secret.txt"
    real.write_text("classified")
    link = inside / "link.txt"
    link.symlink_to(real)

    cfg = DocumentIngestConfig(allowed_roots=[inside])
    with pytest.raises(ValueError, match="outside allowed roots"):
        validate_path(str(link), cfg)


def test_validate_path_size_guard(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_bytes(b"x" * 100)
    cfg = DocumentIngestConfig(max_bytes=50)
    with pytest.raises(ValueError, match="file too large"):
        validate_path(str(target), cfg)


# ── compute_content_hash ────────────────────────────────────────────


def test_compute_content_hash_stable(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_bytes(b"Hello Styx wave 28")
    h1 = compute_content_hash(target)
    h2 = compute_content_hash(target)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex digest


def test_compute_content_hash_differs_on_change(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_bytes(b"first")
    h1 = compute_content_hash(target)
    target.write_bytes(b"second")
    h2 = compute_content_hash(target)
    assert h1 != h2


# ── ingest_document orchestrator ────────────────────────────────────


def test_ingest_document_plaintext_full_flow() -> None:
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = DocumentIngestConfig()

    result = ingest_document(
        queries,  # type: ignore[arg-type]
        embedder,
        raw_path=str(FIXTURE_DIR / "sample.txt"),
        config=cfg,
        store_routing=_store_cfg(),
    )

    assert not result.deduplicated
    assert result.mime_type == "text/plain"
    assert result.original_name == "sample.txt"
    assert result.size_bytes > 0
    assert result.char_count > 0
    assert result.chunks_count >= 1
    assert len(result.content_hash) == 64
    # Document INSERT — один.
    assert len(queries.inserted_documents) == 1
    doc = queries.inserted_documents[0]
    assert doc["source"] == "ingest_document"
    assert doc["file_path"].endswith("sample.txt")
    assert doc["mime_type"] == "text/plain"
    assert doc["size_bytes"] == result.size_bytes
    assert doc["content_hash"] == result.content_hash
    # Chunks INSERT — один batch.
    assert len(queries.inserted_chunks) == 1
    # Embedder зван по числу chunks (НЕ +1 для summary —
    # create_tail_memory=False).
    assert len(embedder.calls) == result.chunks_count


def test_ingest_document_markdown_flow() -> None:
    queries = _StubQueries()
    embedder = _StubEmbedder()
    cfg = DocumentIngestConfig()

    result = ingest_document(
        queries,  # type: ignore[arg-type]
        embedder,
        raw_path=str(FIXTURE_DIR / "sample.md"),
        config=cfg,
        store_routing=_store_cfg(),
    )
    assert result.mime_type == "text/markdown"
    assert result.original_name == "sample.md"


def test_ingest_document_idempotency_on_existing_hash() -> None:
    """Повторный ingest того же файла → deduplicated=True, без INSERT'ов."""
    queries_first = _StubQueries()
    embedder = _StubEmbedder()
    cfg = DocumentIngestConfig()
    src = FIXTURE_DIR / "sample.txt"

    # Первый прогон — посчитает hash сам, INSERT'нёт.
    first = ingest_document(
        queries_first,  # type: ignore[arg-type]
        embedder,
        raw_path=str(src),
        config=cfg,
        store_routing=_store_cfg(),
    )

    # Второй прогон — отдадим тот же hash как уже существующий.
    queries_second = _StubQueries(existing_hashes={first.content_hash})
    second = ingest_document(
        queries_second,  # type: ignore[arg-type]
        embedder,
        raw_path=str(src),
        config=cfg,
        store_routing=_store_cfg(),
    )
    assert second.deduplicated
    assert second.chunks_count == 0
    assert len(queries_second.inserted_documents) == 0
    assert len(queries_second.inserted_chunks) == 0


def test_ingest_document_empty_text_raises(tmp_path: Path) -> None:
    """Image-only PDF — empty text → 422 «empty document»."""
    target = tmp_path / "empty.txt"
    target.write_text("")  # empty file

    queries = _StubQueries()
    embedder = _StubEmbedder()
    with pytest.raises(ValueError, match="empty document"):
        ingest_document(
            queries,  # type: ignore[arg-type]
            embedder,
            raw_path=str(target),
            config=DocumentIngestConfig(),
            store_routing=_store_cfg(),
        )
    assert len(queries.inserted_documents) == 0


def test_ingest_document_unsupported_extension(tmp_path: Path) -> None:
    decoy = tmp_path / "foo.pptx"
    decoy.write_text("not really pptx")
    queries = _StubQueries()
    embedder = _StubEmbedder()
    with pytest.raises(ValueError, match="unsupported extension"):
        ingest_document(
            queries,  # type: ignore[arg-type]
            embedder,
            raw_path=str(decoy),
            config=DocumentIngestConfig(),
            store_routing=_store_cfg(),
        )


def test_ingest_document_no_tail_memory_check() -> None:
    """Защита: insert_memory НЕ вызывается (file-ingest pipeline)."""
    queries = _StubQueries()
    embedder = _StubEmbedder()
    # _StubQueries.insert_memory кидает AssertionError если вызвался.
    ingest_document(
        queries,  # type: ignore[arg-type]
        embedder,
        raw_path=str(FIXTURE_DIR / "sample.txt"),
        config=DocumentIngestConfig(),
        store_routing=_store_cfg(),
    )
    # Дошли сюда — значит insert_memory не звался.


def test_ingest_document_explicit_content_hash(tmp_path: Path) -> None:
    """Override content_hash из request — пишется в documents.content_hash."""
    target = tmp_path / "f.txt"
    target.write_text("body")
    queries = _StubQueries()
    embedder = _StubEmbedder()
    explicit = "deadbeef" * 8  # 64 chars
    result = ingest_document(
        queries,  # type: ignore[arg-type]
        embedder,
        raw_path=str(target),
        config=DocumentIngestConfig(),
        store_routing=_store_cfg(),
        content_hash=explicit,
    )
    assert result.content_hash == explicit
    assert queries.inserted_documents[0]["content_hash"] == explicit


def test_ingest_document_metadata_passthrough(tmp_path: Path) -> None:
    target = tmp_path / "f.md"
    target.write_text("# header\nbody text here")
    queries = _StubQueries()
    embedder = _StubEmbedder()
    ingest_document(
        queries,  # type: ignore[arg-type]
        embedder,
        raw_path=str(target),
        config=DocumentIngestConfig(),
        store_routing=_store_cfg(),
        source_ref="user-upload-42",
        visibility="private",
        metadata={"upload_source": "openclaw"},
    )
    doc = queries.inserted_documents[0]
    assert doc["source_ref"] == "user-upload-42"
    assert doc["visibility"] == "private"
    # metadata extras передаются в document_metadata_extra
    # (через ingest_document -> route_long_content).
    assert "upload_source" in doc["metadata"]
    assert doc["metadata"]["upload_source"] == "openclaw"
    # parsed.metadata (line_count) тоже должен попасть.
    assert "line_count" in doc["metadata"]
