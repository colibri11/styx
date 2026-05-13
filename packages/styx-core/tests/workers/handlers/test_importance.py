"""Unit-тесты для LLM importance handler'а."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from styx.llm import LLMRateLimiter, OllamaChatClient, OllamaTerminalError
from styx.workers.handlers.importance import (
    IMPORTANCE_TASK_TYPE,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _validate_importance_response,
    create_importance_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


# ── Validator ──────────────────────────────────────────────────────────


def test_validator_skip_path() -> None:
    raw = {
        "skip": True,
        "skip_reason": "слишком короткое",
        "importance_score": None,
        "rationale": None,
        "signals": None,
    }
    parsed = _validate_importance_response(raw)
    assert parsed.skip is True
    assert parsed.skip_reason == "слишком короткое"
    assert parsed.importance_score is None
    assert parsed.signals is None


def test_validator_scored_path() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "importance_score": 0.75,
        "rationale": "значимое предпочтение",
        "signals": {
            "has_specific_facts": True,
            "self_reference": True,
            "emotional_weight": 0.2,
            "declarative": True,
            "abstraction_level": "concrete",
        },
    }
    parsed = _validate_importance_response(raw)
    assert parsed.skip is False
    assert parsed.importance_score == 0.75
    assert parsed.signals is not None
    assert parsed.signals.abstraction_level == "concrete"


def test_validator_skip_must_have_reason() -> None:
    raw = {
        "skip": True,
        "skip_reason": "",
        "importance_score": None,
        "rationale": None,
        "signals": None,
    }
    with pytest.raises(ValueError, match="skip_reason"):
        _validate_importance_response(raw)


def test_validator_skip_must_null_other_fields() -> None:
    raw = {
        "skip": True,
        "skip_reason": "ok",
        "importance_score": 0.5,  # должен быть null
        "rationale": None,
        "signals": None,
    }
    with pytest.raises(ValueError, match="importance_score"):
        _validate_importance_response(raw)


def test_validator_scored_score_out_of_range() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "importance_score": 1.5,
        "rationale": "x",
        "signals": {
            "has_specific_facts": True,
            "self_reference": False,
            "emotional_weight": 0.0,
            "declarative": True,
            "abstraction_level": "abstract",
        },
    }
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _validate_importance_response(raw)


def test_validator_scored_invalid_abstraction_level() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "importance_score": 0.5,
        "rationale": "x",
        "signals": {
            "has_specific_facts": True,
            "self_reference": False,
            "emotional_weight": 0.0,
            "declarative": True,
            "abstraction_level": "weird",
        },
    }
    with pytest.raises(ValueError, match="abstraction_level"):
        _validate_importance_response(raw)


def test_validator_top_level_not_object() -> None:
    with pytest.raises(ValueError, match="object"):
        _validate_importance_response("just a string")


def test_validator_skip_field_missing() -> None:
    with pytest.raises(ValueError, match="skip"):
        _validate_importance_response({})


# ── User prompt builder ──────────────────────────────────────────────


def test_build_user_prompt_shape() -> None:
    from datetime import datetime, timezone

    class _Mem:
        content = "Сергей выбрал qwen3:4b-local"
        kind = "fact"
        created_at = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
        importance_provisional = 0.7

    out = _build_user_prompt(_Mem())  # type: ignore[arg-type]
    assert "kind: fact" in out
    assert "provisional_importance: 0.70" in out
    assert "Сергей выбрал qwen3:4b-local" in out
    assert "2026-05-01" in out


# ── System prompt sanity ─────────────────────────────────────────────


def test_system_prompt_starts_with_known_phrase() -> None:
    """Никакой автогенерации — port буквальный."""
    assert "Ты оцениваешь важность записи" in SYSTEM_PROMPT
    assert "qwen3:4b-local" in SYSTEM_PROMPT  # пример из калибровки
    assert "Отвечай только JSON" in SYSTEM_PROMPT


# ── Handler integration с migrated_db ───────────────────────────────


class _ScriptedChat(OllamaChatClient):
    """OllamaChatClient, но chat_json возвращает заранее заданный JSON."""

    def __init__(self, response: dict[str, Any] | Exception) -> None:
        super().__init__(base_url="http://x", model="m", max_attempts=1)
        self._response = response
        self.calls = 0

    def chat_json(self, messages, *, timeout_s=None, max_attempts=None):  # type: ignore[override]
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


@pytest.fixture
def rate_limit() -> LLMRateLimiter:
    return LLMRateLimiter(capacity=4, refill_per_second=10.0)


def _insert_memory(conn: psycopg.Connection, content: str, kind: str = "episode") -> uuid.UUID:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content, kind) "
            " VALUES ('test_agent', 'user', %s, %s) RETURNING id",
            (content, kind),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def _ctx(conn, llm, rate_limit) -> HandlerContext:
    return HandlerContext(
        conn=conn, llm=llm, rate_limit=rate_limit, logger=logging.getLogger("test"),
    )


def test_handler_no_memory_id_skips(db, rate_limit) -> None:
    handler = create_importance_handler()
    llm = _ScriptedChat({})  # не должен быть вызван
    task = LlmTask(
        id=uuid.uuid4(), task_type=IMPORTANCE_TASK_TYPE, memory_id=None,
        payload={}, retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    assert out.skipped_by_llm is False
    assert out.result == {"skipped": "no_memory_id"}
    assert llm.calls == 0


def test_handler_memory_gone_skips(db, rate_limit) -> None:
    handler = create_importance_handler()
    llm = _ScriptedChat({})
    task = LlmTask(
        id=uuid.uuid4(), task_type=IMPORTANCE_TASK_TYPE,
        memory_id=uuid.uuid4(),  # никогда не существовавший
        payload={}, retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    assert out.result == {"skipped": "memory_gone"}
    assert llm.calls == 0


def test_handler_scored_writes_importance_final(db, rate_limit) -> None:
    handler = create_importance_handler()
    mid = _insert_memory(db, content="Сергей выбрал qwen3:4b-local за окно контекста")
    response = {
        "skip": False,
        "skip_reason": None,
        "importance_score": 0.82,
        "rationale": "устойчивое предпочтение",
        "signals": {
            "has_specific_facts": True,
            "self_reference": True,
            "emotional_weight": 0.1,
            "declarative": True,
            "abstraction_level": "concrete",
        },
    }
    llm = _ScriptedChat(response)

    task = LlmTask(
        id=uuid.uuid4(), task_type=IMPORTANCE_TASK_TYPE,
        memory_id=mid, payload={}, retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()  # хэндлер не коммитит сам — это делает runtime

    assert out.skipped_by_llm is False
    assert out.result is not None
    assert out.result["importance_score"] == 0.82
    assert llm.calls == 1

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT importance_final FROM memories WHERE id = %s", (mid,))
        row = cur.fetchone()
    assert row is not None
    assert abs(float(row["importance_final"]) - 0.82) < 1e-6


def test_handler_skip_keeps_importance_null(db, rate_limit) -> None:
    handler = create_importance_handler()
    mid = _insert_memory(db, content="ок", kind="episode")
    response = {
        "skip": True,
        "skip_reason": "односложная служебная реплика",
        "importance_score": None,
        "rationale": None,
        "signals": None,
    }
    llm = _ScriptedChat(response)

    task = LlmTask(
        id=uuid.uuid4(), task_type=IMPORTANCE_TASK_TYPE,
        memory_id=mid, payload={}, retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert out.skipped_by_llm is True
    assert out.result is not None
    assert out.result["skip"] is True
    assert out.result["skip_reason"] == "односложная служебная реплика"
    assert llm.calls == 1

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT importance_final FROM memories WHERE id = %s", (mid,))
        row = cur.fetchone()
    assert row["importance_final"] is None


def test_handler_schema_mismatch_raises_terminal(db, rate_limit) -> None:
    """Битый JSON от LLM → OllamaTerminalError."""
    handler = create_importance_handler()
    mid = _insert_memory(db, content="some text content of any length")
    response = {"score": 0.5}  # старый формат, не подходит под схему
    llm = _ScriptedChat(response)

    task = LlmTask(
        id=uuid.uuid4(), task_type=IMPORTANCE_TASK_TYPE,
        memory_id=mid, payload={}, retry_count=0,
    )

    with pytest.raises(OllamaTerminalError, match="schema_mismatch"):
        handler(task, _ctx(db, llm, rate_limit))


def test_handler_consumes_rate_limit_token(db) -> None:
    """Handler берёт ровно один токен."""
    handler = create_importance_handler()
    mid = _insert_memory(db, content="some content")
    response = {
        "skip": True,
        "skip_reason": "test",
        "importance_score": None,
        "rationale": None,
        "signals": None,
    }
    llm = _ScriptedChat(response)
    rate_limit = LLMRateLimiter(capacity=1, refill_per_second=0.1)

    task = LlmTask(
        id=uuid.uuid4(), task_type=IMPORTANCE_TASK_TYPE,
        memory_id=mid, payload={}, retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()
    # После handler'а rate-limit бюджет должен быть исчерпан.
    assert rate_limit.try_acquire() is False
