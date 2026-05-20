"""Тесты document_chunk_embed handler (Defect-fix A).

Async-embed chunks большого документа: file-ingest INSERT'ит chunks
с embedding=NULL, этот handler их embed'ит.

Postgres-skip: на host без БД — skip; в Docker integration — полный.
"""

from __future__ import annotations

import logging
import uuid

import psycopg
import pytest

from styx.embedding import EmbeddingClient, EmbeddingError
from styx.llm import LLMRateLimiter, OllamaChatClient, OllamaTerminalError
from styx.storage.queries import AgentScopedQueries
from styx.workers.handlers.document_chunk_embed import (
    DOCUMENT_CHUNK_EMBED_TASK_TYPE,
    create_document_chunk_embed_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


class _StubEmbedder(EmbeddingClient):
    def __init__(self, vector: list[float] | Exception) -> None:
        self._vector = vector
        self.calls: list[str] = []

    def embed(self, content: str) -> list[float]:  # type: ignore[override]
        self.calls.append(content)
        if isinstance(self._vector, Exception):
            raise self._vector
        return list(self._vector)


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


@pytest.fixture
def rate_limit() -> LLMRateLimiter:
    return LLMRateLimiter(capacity=4, refill_per_second=10.0)


def _ctx(conn, rate_limit, embedder=None) -> HandlerContext:
    llm = OllamaChatClient(base_url="http://x", model="m", max_attempts=1)
    return HandlerContext(
        conn=conn, llm=llm, rate_limit=rate_limit,
        logger=logging.getLogger("test"), embedder=embedder,
    )


def _seed_document_with_null_chunks(
    conn: psycopg.Connection, agent_id: str, n_chunks: int,
) -> uuid.UUID:
    q = AgentScopedQueries(conn, agent_id)
    doc_id = q.insert_document(source="ingest_document", char_count=5000)
    q.insert_chunks_batch(
        doc_id,
        [(i, f"chunk {i}", None, i * 100, i * 100 + 50) for i in range(n_chunks)],
    )
    conn.commit()
    return doc_id


def _embed_vec() -> list[float]:
    v = [0.0] * 768
    v[0] = 0.5
    return v


def test_handler_embeds_all_null_chunks(db, rate_limit) -> None:
    doc_id = _seed_document_with_null_chunks(db, "alpha", 3)
    embedder = _StubEmbedder(_embed_vec())
    handler = create_document_chunk_embed_handler()
    task = LlmTask(
        id=uuid.uuid4(),
        task_type=DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        memory_id=None,
        payload={"agent_id": "alpha", "document_id": str(doc_id)},
        retry_count=0,
    )
    out = handler(task, _ctx(db, rate_limit, embedder))
    db.commit()

    assert out.result is not None
    assert out.result["embedded"] == 3
    assert out.result["remaining"] == 0
    assert len(embedder.calls) == 3
    # Все chunks теперь с embedding'ом.
    q = AgentScopedQueries(db, "alpha")
    assert q.chunks_without_embedding(doc_id) == []


def test_handler_noop_when_all_embedded(db, rate_limit) -> None:
    """Документ без NULL-chunks — no-op success (идемпотентность)."""
    q = AgentScopedQueries(db, "alpha")
    doc_id = q.insert_document(source="ingest_document", char_count=100)
    q.insert_chunks_batch(doc_id, [(0, "c", _embed_vec(), 0, 1)])
    db.commit()

    embedder = _StubEmbedder(_embed_vec())
    handler = create_document_chunk_embed_handler()
    task = LlmTask(
        id=uuid.uuid4(),
        task_type=DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        memory_id=None,
        payload={"agent_id": "alpha", "document_id": str(doc_id)},
        retry_count=0,
    )
    out = handler(task, _ctx(db, rate_limit, embedder))
    assert out.result is not None
    assert out.result["embedded"] == 0
    assert len(embedder.calls) == 0


def test_handler_partial_fail_leaves_remaining(db, rate_limit) -> None:
    """Embed-fail на chunk'е не роняет handler — chunk остаётся NULL,
    retry/следующий прогон подберёт."""
    doc_id = _seed_document_with_null_chunks(db, "alpha", 2)
    embedder = _StubEmbedder(EmbeddingError("ollama down"))
    handler = create_document_chunk_embed_handler()
    task = LlmTask(
        id=uuid.uuid4(),
        task_type=DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        memory_id=None,
        payload={"agent_id": "alpha", "document_id": str(doc_id)},
        retry_count=0,
    )
    out = handler(task, _ctx(db, rate_limit, embedder))
    db.commit()
    assert out.result is not None
    assert out.result["embedded"] == 0
    assert out.result["remaining"] == 2
    # chunks остались NULL.
    q = AgentScopedQueries(db, "alpha")
    assert len(q.chunks_without_embedding(doc_id)) == 2


def test_handler_invalid_payload_raises_terminal(db, rate_limit) -> None:
    handler = create_document_chunk_embed_handler()
    task = LlmTask(
        id=uuid.uuid4(),
        task_type=DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        memory_id=None,
        payload={"agent_id": "alpha"},  # нет document_id
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError):
        handler(task, _ctx(db, rate_limit, _StubEmbedder(_embed_vec())))


def test_handler_no_embedder_raises_terminal(db, rate_limit) -> None:
    doc_id = _seed_document_with_null_chunks(db, "alpha", 1)
    handler = create_document_chunk_embed_handler()
    task = LlmTask(
        id=uuid.uuid4(),
        task_type=DOCUMENT_CHUNK_EMBED_TASK_TYPE,
        memory_id=None,
        payload={"agent_id": "alpha", "document_id": str(doc_id)},
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError):
        handler(task, _ctx(db, rate_limit, embedder=None))
