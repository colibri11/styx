"""Тесты memory_daily_consolidation handler (волна 22).

Postgres-skip: на host без БД — skip; в Docker integration — полный.
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
from styx.workers.handlers.memory_daily_consolidation import (
    MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
    SYSTEM_PROMPT,
    create_memory_daily_consolidation_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


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


def _seed_memories(
    conn: psycopg.Connection, agent_id: str, n: int,
    *, kind: str = "note", visibility: str = "private",
) -> list[uuid.UUID]:
    """Schema default visibility='private'; для shared/private — UPDATE."""
    q = AgentScopedQueries(conn, agent_id)
    out = []
    for i in range(n):
        mid = q.insert_memory(
            role="summary", content=f"мемка {i}",
            kind=kind, kind_src="subjective",
            embedding=[1.0] + [0.0] * 767,
        )
        out.append(mid)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET visibility=%s WHERE id = ANY(%s)",
            (visibility, out),
        )
    conn.commit()
    return out


def _make_task(
    *, agent_id: str, memory_ids: list[uuid.UUID],
) -> LlmTask:
    return LlmTask(
        id=uuid.uuid4(),
        task_type=MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
        memory_id=None,
        payload={
            "agent_id": agent_id,
            "memory_ids": [str(m) for m in memory_ids],
        },
        retry_count=0,
    )


# ── Validators / payload ─────────────────────────────────────────────


def test_handler_invalid_payload_terminal(db, rate_limit) -> None:
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({})
    bad = LlmTask(
        id=uuid.uuid4(),
        task_type=MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
        memory_id=None,
        payload={"agent_id": "alpha"},  # no memory_ids
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError, match="invalid_payload"):
        handler(bad, _ctx(db, llm, rate_limit))


def test_handler_too_few_memory_ids_terminal(db, rate_limit) -> None:
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({})
    bad = LlmTask(
        id=uuid.uuid4(),
        task_type=MEMORY_DAILY_CONSOLIDATION_TASK_TYPE,
        memory_id=None,
        payload={
            "agent_id": "alpha",
            "memory_ids": [str(uuid.uuid4())],  # 1 < min 2
        },
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError, match="invalid_payload"):
        handler(bad, _ctx(db, llm, rate_limit))


# ── Empty cluster / superseded ───────────────────────────────────────


def test_handler_empty_cluster_skips_no_op(db, rate_limit) -> None:
    """Memory_ids в payload, но всех уже не существует (race
    supersede/delete между enqueue и claim) → skip no-op, НЕ terminal.

    Acceptance #3: пустой кластер после claim трактуется как успешный
    skip, а задача помечается done/skipped (return HandlerResult), а не
    failed (raise). LLM не вызывается — источников уже нет.
    """
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({})
    fake_ids = [uuid.uuid4(), uuid.uuid4()]
    out = handler(
        _make_task(agent_id="alpha", memory_ids=fake_ids),
        _ctx(db, llm, rate_limit),
    )
    assert out.skipped_by_llm is True
    assert out.result["skip"] is True
    assert "race supersede/delete" in out.result["skip_reason"]
    assert out.result["consolidated_text"] is None
    assert out.result["consolidated_embedding"] is None
    assert out.result["source_ids"] == []  # никто не выжил
    assert out.result["source_kinds"] is None
    assert out.result["source_visibility"] is None
    assert llm.calls == 0  # LLM не звался — консолидировать нечего


def test_handler_collapsed_cluster_one_survivor_skips_no_op(
    db, rate_limit,
) -> None:
    """Частичный коллапс: из 3 источников до claim выжил только 1 →
    skip no-op, source_ids содержит единственного выжившего."""
    ids = _seed_memories(db, "alpha", 3)
    # Удаляем 2 из 3 источников между enqueue и claim (race delete).
    with db.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE id = ANY(%s)", (ids[1:],))
    db.commit()
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({})
    out = handler(
        _make_task(agent_id="alpha", memory_ids=ids),
        _ctx(db, llm, rate_limit),
    )
    assert out.skipped_by_llm is True
    assert out.result["skip"] is True
    assert out.result["source_ids"] == [str(ids[0])]  # выживший источник
    assert out.result["consolidated_text"] is None
    assert out.result["consolidated_embedding"] is None
    assert out.result["source_kinds"] is None
    assert llm.calls == 0


def test_handler_some_superseded_terminal(db, rate_limit) -> None:
    ids = _seed_memories(db, "alpha", 3)
    # Superseded one of the sources between enqueue and claim.
    with db.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by=%s WHERE id=%s",
            (ids[1], ids[0]),
        )
    db.commit()
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({})
    with pytest.raises(OllamaTerminalError, match="some_source_already_superseded"):
        handler(
            _make_task(agent_id="alpha", memory_ids=ids),
            _ctx(db, llm, rate_limit),
        )


# ── Skip path ────────────────────────────────────────────────────────


def test_handler_skip_branch(db, rate_limit) -> None:
    ids = _seed_memories(db, "alpha", 3)
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({
        "skip": True, "skip_reason": "разное",
        "consolidated_text": None,
    })
    out = handler(
        _make_task(agent_id="alpha", memory_ids=ids),
        _ctx(db, llm, rate_limit),
    )
    assert out.skipped_by_llm is True
    assert out.result["skip"] is True
    assert out.result["consolidated_text"] is None
    assert out.result["consolidated_embedding"] is None
    assert sorted(out.result["source_ids"]) == sorted(str(s) for s in ids)
    assert out.result["source_kinds"] is None
    assert llm.calls == 1


# ── Merged path ──────────────────────────────────────────────────────


def test_handler_merged_branch_embeds_and_returns_full_result(
    db, rate_limit,
) -> None:
    ids = _seed_memories(db, "alpha", 3, visibility="shared")
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "consolidated_text": "общая направленность",
    })
    embedder = _StubEmbedder([0.5, 0.5] + [0.0] * 766)
    out = handler(
        _make_task(agent_id="alpha", memory_ids=ids),
        _ctx(db, llm, rate_limit, embedder=embedder),
    )
    assert out.skipped_by_llm is False
    res = out.result
    assert res["skip"] is False
    assert res["consolidated_text"] == "общая направленность"
    assert len(res["consolidated_embedding"]) == 768
    assert sorted(res["source_ids"]) == sorted(str(s) for s in ids)
    assert res["source_kinds"] == ["note", "note", "note"]
    assert res["source_visibility"] == ["shared", "shared", "shared"]
    assert embedder.calls == ["общая направленность"]


def test_handler_merged_collects_private_visibility(db, rate_limit) -> None:
    ids = _seed_memories(db, "alpha", 3, visibility="private")
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None, "consolidated_text": "x",
    })
    embedder = _StubEmbedder([1.0] + [0.0] * 767)
    out = handler(
        _make_task(agent_id="alpha", memory_ids=ids),
        _ctx(db, llm, rate_limit, embedder=embedder),
    )
    assert out.result["source_visibility"] == ["private", "private", "private"]


def test_handler_merged_no_embedder_terminal(db, rate_limit) -> None:
    ids = _seed_memories(db, "alpha", 3)
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None, "consolidated_text": "x",
    })
    with pytest.raises(OllamaTerminalError, match="embedder_unavailable"):
        handler(
            _make_task(agent_id="alpha", memory_ids=ids),
            _ctx(db, llm, rate_limit, embedder=None),
        )


def test_handler_schema_mismatch_terminal(db, rate_limit) -> None:
    ids = _seed_memories(db, "alpha", 3)
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({"random": "junk"})
    with pytest.raises(OllamaTerminalError, match="schema_mismatch"):
        handler(
            _make_task(agent_id="alpha", memory_ids=ids),
            _ctx(db, llm, rate_limit),
        )


# ── Cross-agent isolation ────────────────────────────────────────────


def test_handler_other_agent_sources_filtered_to_empty_cluster(
    db, rate_limit,
) -> None:
    """payload.agent_id='alpha' с memory_ids от 'beta' → cross-agent
    scope отфильтровывает всё → empty cluster → skip no-op (не terminal)."""
    ids_beta = _seed_memories(db, "beta", 3)
    handler = create_memory_daily_consolidation_handler()
    llm = _ScriptedChat({})
    out = handler(
        _make_task(agent_id="alpha", memory_ids=ids_beta),
        _ctx(db, llm, rate_limit),
    )
    assert out.skipped_by_llm is True
    assert out.result["skip"] is True
    assert out.result["source_ids"] == []
    assert llm.calls == 0


# ── System prompt sanity ─────────────────────────────────────────────


def test_system_prompt_describes_consolidation_contract() -> None:
    assert "consolidated_text" in SYSTEM_PROMPT
    assert "skip" in SYSTEM_PROMPT
    assert "1–3 предложения" in SYSTEM_PROMPT
