"""Тесты reinterpret_merge handler (волна 22).

Postgres-skip (как test_dialogue_batch_consolidation): на host без
DSN — skip; в Docker integration suite прогон полный.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import psycopg
import pytest

from styx.embedding import EmbeddingClient
from styx.llm import LLMRateLimiter, OllamaChatClient, OllamaTerminalError
from styx.storage.queries import AgentScopedQueries
from styx.workers.handlers.reinterpret_merge import (
    REINTERPRET_MERGE_TASK_TYPE,
    SYSTEM_PROMPT,
    create_reinterpret_merge_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


# ── Stubs for LLM + embedder ──────────────────────────────────────────


class _ScriptedChat(OllamaChatClient):
    def __init__(self, response: Any) -> None:
        super().__init__(base_url="http://x", model="m", max_attempts=1)
        self._response = response
        self.calls = 0
        self.captured: list[list[dict]] = []

    def chat_json(self, messages, *, timeout_s=None, max_attempts=None):  # type: ignore[override]
        self.captured.append(list(messages))
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


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


def _ctx(conn, llm, rate_limit, embedder=None) -> HandlerContext:
    return HandlerContext(
        conn=conn, llm=llm, rate_limit=rate_limit,
        logger=logging.getLogger("test"), embedder=embedder,
    )


def _seed_memory(
    conn: psycopg.Connection, agent_id: str, *, content: str = "старая",
    embedding: list[float] | None = None,
) -> uuid.UUID:
    q = AgentScopedQueries(conn, agent_id)
    mid = q.insert_memory(
        role="summary", content=content, kind="note",
        kind_src="subjective",
        embedding=embedding if embedding else [0.1] * 768,
    )
    conn.commit()
    return mid


def _make_task(
    *, memory_id: uuid.UUID | None,
    agent_id: str = "alpha",
    text: str = "новое понимание",
    weight: float | None = None,
) -> LlmTask:
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "new_understanding_text": text,
    }
    if weight is not None:
        payload["weight"] = weight
    return LlmTask(
        id=uuid.uuid4(),
        task_type=REINTERPRET_MERGE_TASK_TYPE,
        memory_id=memory_id,
        payload=payload,
        retry_count=0,
    )


# ── Handler tests ─────────────────────────────────────────────────────


def test_handler_no_memory_id_returns_skipped_shape(db, rate_limit) -> None:
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({"never": "called"})
    out = handler(_make_task(memory_id=None), _ctx(db, llm, rate_limit))
    assert out.skipped_by_llm is True
    assert out.result == {"skipped": "no_memory_id"}
    assert llm.calls == 0


def test_handler_memory_gone_returns_skipped_shape(db, rate_limit) -> None:
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({"never": "called"})
    out = handler(
        _make_task(memory_id=uuid.uuid4()),
        _ctx(db, llm, rate_limit),
    )
    assert out.skipped_by_llm is True
    assert out.result == {"skipped": "memory_gone"}
    assert llm.calls == 0


def test_handler_agent_id_mismatch_terminal(db, rate_limit) -> None:
    mid = _seed_memory(db, "alpha")
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({"never": "called"})
    with pytest.raises(OllamaTerminalError, match="agent_id_mismatch"):
        handler(
            _make_task(memory_id=mid, agent_id="beta"),
            _ctx(db, llm, rate_limit),
        )


def test_handler_invalid_payload_terminal(db, rate_limit) -> None:
    """Payload без agent_id или с недопустимым weight — terminal."""
    mid = _seed_memory(db, "alpha")
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({})
    bad = LlmTask(
        id=uuid.uuid4(),
        task_type=REINTERPRET_MERGE_TASK_TYPE,
        memory_id=mid,
        payload={"agent_id": "alpha"},  # no new_understanding_text
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError, match="invalid_payload"):
        handler(bad, _ctx(db, llm, rate_limit))


def test_handler_skip_branch_returns_skip_result(db, rate_limit) -> None:
    mid = _seed_memory(db, "alpha")
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({
        "skip": True, "skip_reason": "тавтология", "merged_text": None,
    })
    out = handler(
        _make_task(memory_id=mid),
        _ctx(db, llm, rate_limit),
    )
    assert out.skipped_by_llm is True
    assert out.result["skip"] is True
    assert out.result["skip_reason"] == "тавтология"
    assert out.result["merged_text"] is None
    assert out.result["merged_embedding"] is None
    assert out.result["previous_text"] == "старая"
    assert isinstance(out.result["previous_embedding"], list)
    assert out.result["agent_id"] == "alpha"
    assert llm.calls == 1


def test_handler_merged_branch_blends_embeddings(db, rate_limit) -> None:
    """Happy path: skip=False → embed → blend → result merged."""
    prev_emb = [1.0, 0.0] + [0.0] * 766
    mid = _seed_memory(db, "alpha", embedding=prev_emb)
    handler = create_reinterpret_merge_handler(blend_weight=0.5)
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "merged_text": "новое уплотнённое понимание",
    })
    next_emb = [0.0, 1.0] + [0.0] * 766
    embedder = _StubEmbedder(next_emb)
    out = handler(
        _make_task(memory_id=mid),
        _ctx(db, llm, rate_limit, embedder=embedder),
    )
    assert out.skipped_by_llm is False
    res = out.result
    assert res["skip"] is False
    assert res["merged_text"] == "новое уплотнённое понимание"
    merged = res["merged_embedding"]
    assert len(merged) == 768
    # blend(prev, next, w=0.5) = (0.5, 0.5) → norm sqrt(0.5)
    import math
    assert merged[0] == pytest.approx(0.5 / math.sqrt(0.5), abs=1e-6)
    assert merged[1] == pytest.approx(0.5 / math.sqrt(0.5), abs=1e-6)
    assert res["weight_applied"] == 0.5
    assert res["previous_text"] == "старая"
    assert res["new_understanding_text"] == "новое понимание"
    assert embedder.calls == ["новое уплотнённое понимание"]


def test_handler_payload_weight_overrides_default(db, rate_limit) -> None:
    prev_emb = [1.0, 0.0] + [0.0] * 766
    mid = _seed_memory(db, "alpha", embedding=prev_emb)
    handler = create_reinterpret_merge_handler(blend_weight=0.5)
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None, "merged_text": "x",
    })
    embedder = _StubEmbedder([0.0, 1.0] + [0.0] * 766)
    out = handler(
        _make_task(memory_id=mid, weight=0.8),
        _ctx(db, llm, rate_limit, embedder=embedder),
    )
    assert out.result["weight_applied"] == 0.8


def test_handler_embedder_unavailable_terminal(db, rate_limit) -> None:
    mid = _seed_memory(db, "alpha")
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None, "merged_text": "x",
    })
    with pytest.raises(OllamaTerminalError, match="embedder_unavailable"):
        handler(
            _make_task(memory_id=mid),
            _ctx(db, llm, rate_limit, embedder=None),
        )


def test_handler_schema_mismatch_terminal(db, rate_limit) -> None:
    mid = _seed_memory(db, "alpha")
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({"unexpected": "shape"})
    with pytest.raises(OllamaTerminalError, match="schema_mismatch"):
        handler(_make_task(memory_id=mid), _ctx(db, llm, rate_limit))


def test_handler_previous_embedding_missing_terminal(db, rate_limit) -> None:
    """Memory без embedding — blend невозможен → terminal."""
    q = AgentScopedQueries(db, "alpha")
    mid = q.insert_memory(
        role="summary", content="без вектора",
        kind="note", kind_src="subjective",
    )
    db.commit()
    handler = create_reinterpret_merge_handler()
    llm = _ScriptedChat({"never": "called"})
    with pytest.raises(OllamaTerminalError, match="previous_embedding_missing"):
        handler(_make_task(memory_id=mid), _ctx(db, llm, rate_limit))


# ── System prompt sanity ──────────────────────────────────────────────


def test_system_prompt_describes_reinterpret_contract() -> None:
    assert "previous" in SYSTEM_PROMPT
    assert "new_understanding" in SYSTEM_PROMPT
    assert "merged_text" in SYSTEM_PROMPT
    assert "skip" in SYSTEM_PROMPT
