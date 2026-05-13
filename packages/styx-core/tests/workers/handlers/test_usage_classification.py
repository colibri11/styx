"""Unit-тесты для usage_classification handler'а."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.llm import LLMRateLimiter, OllamaChatClient, OllamaTerminalError
from styx.workers.handlers.usage_classification import (
    MAX_PAYLOAD_REPLY_CHARS,
    MAX_RECALL_EVENT_IDS,
    SYSTEM_PROMPT,
    USAGE_CLASSIFICATION_TASK_TYPE,
    _build_user_prompt,
    _RecallRow,
    _validate_payload,
    _validate_response,
    create_usage_classification_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


# ── Payload validator ─────────────────────────────────────────────────


def test_validate_payload_valid() -> None:
    ids, text, agent = _validate_payload(
        {
            "recall_event_ids": [1, 2, 3],
            "llm_output_text": "ответ агента",
            "agent_id": "alyona",
        }
    )
    assert ids == [1, 2, 3]
    assert text == "ответ агента"
    assert agent == "alyona"


def test_validate_payload_too_many_ids() -> None:
    payload = {
        "recall_event_ids": list(range(1, MAX_RECALL_EVENT_IDS + 2)),
        "llm_output_text": "x",
        "agent_id": "a",
    }
    with pytest.raises(ValueError, match="лимит"):
        _validate_payload(payload)


def test_validate_payload_empty_ids() -> None:
    with pytest.raises(ValueError, match="recall_event_ids"):
        _validate_payload({"recall_event_ids": [], "llm_output_text": "x", "agent_id": "a"})


def test_validate_payload_missing_agent_id() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        _validate_payload({"recall_event_ids": [1], "llm_output_text": "x"})


def test_validate_payload_text_too_large() -> None:
    payload = {
        "recall_event_ids": [1],
        "llm_output_text": "a" * (MAX_PAYLOAD_REPLY_CHARS + 1),
        "agent_id": "a",
    }
    with pytest.raises(ValueError, match="лимит"):
        _validate_payload(payload)


# ── Response validator ────────────────────────────────────────────────


def test_validate_response_skip() -> None:
    out = _validate_response(
        {"skip": True, "skip_reason": "пусто", "classifications": None}
    )
    assert out.skip is True
    assert out.classifications is None


def test_validate_response_scored() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "classifications": [
            {"memory_id": "m-1", "used": True, "reason": "опирается"},
            {"memory_id": "m-2", "used": False, "reason": "тема другая"},
        ],
    }
    out = _validate_response(raw)
    assert out.skip is False
    assert out.classifications is not None
    assert len(out.classifications) == 2
    assert out.classifications[0].used is True


def test_validate_response_skip_must_have_reason() -> None:
    with pytest.raises(ValueError, match="skip_reason"):
        _validate_response({"skip": True, "skip_reason": "", "classifications": None})


def test_validate_response_scored_classification_needs_fields() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "classifications": [{"memory_id": "m-1"}],  # нет used / reason
    }
    with pytest.raises(ValueError, match="used"):
        _validate_response(raw)


# ── User prompt builder ──────────────────────────────────────────────


def test_build_user_prompt_truncates_long_reply() -> None:
    rows = [_RecallRow(id=1, memory_id="m-1", content="x", used_in_output=False)]
    long_reply = "a" * 10_000
    prompt = _build_user_prompt(long_reply, rows)
    assert "[truncated]" in prompt
    # MAX_REPLY_CHARS=8000
    assert prompt.count("a") <= 8005


# ── Handler integration ──────────────────────────────────────────────


class _ScriptedChat(OllamaChatClient):
    def __init__(self, scenario: dict[str, Any] | Exception) -> None:
        super().__init__(base_url="http://x", model="m", max_attempts=1)
        self._scenario = scenario
        self.calls = 0

    def chat_json(self, messages, *, timeout_s=None, max_attempts=None):  # type: ignore[override]
        self.calls += 1
        if isinstance(self._scenario, Exception):
            raise self._scenario
        return self._scenario


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'classifier-test-%'")
    conn.commit()
    conn.close()


@pytest.fixture
def rate_limit() -> LLMRateLimiter:
    return LLMRateLimiter(capacity=4, refill_per_second=10.0)


def _seed(db: psycopg.Connection, agent: str, *, content: str) -> tuple[uuid.UUID, int]:
    """INSERT memory + recall_event и вернуть (memory_id, recall_event_id)."""
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content) "
            "VALUES (%s, 'user', %s) RETURNING id",
            (agent, content),
        )
        memory_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO recall_events "
            "  (memory_id, query_hash, match_score, used_in_output) "
            "VALUES (%s, %s, %s, false) RETURNING id",
            (memory_id, b"\x00" * 32, 0.5),
        )
        recall_id = cur.fetchone()[0]
    db.commit()
    return memory_id, recall_id


def _ctx(conn, llm, rate_limit) -> HandlerContext:
    return HandlerContext(
        conn=conn, llm=llm, rate_limit=rate_limit, logger=logging.getLogger("test")
    )


def test_handler_skips_when_no_recall_rows(db, rate_limit) -> None:
    handler = create_usage_classification_handler()
    llm = _ScriptedChat({})  # не должен звониться
    task = LlmTask(
        id=uuid.uuid4(), task_type=USAGE_CLASSIFICATION_TASK_TYPE,
        memory_id=None,
        payload={
            "recall_event_ids": [99999],  # не существует
            "llm_output_text": "ответ",
            "agent_id": "ghost",
        },
        retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    assert out.result == {"skipped": "no_recall_rows"}
    assert llm.calls == 0


def test_handler_skip_path_no_flips(db, rate_limit) -> None:
    agent = f"classifier-test-{uuid.uuid4().hex[:6]}"
    memory_id, recall_id = _seed(db, agent, content="что-то полезное")
    handler = create_usage_classification_handler()
    llm = _ScriptedChat({"skip": True, "skip_reason": "пустой ответ", "classifications": None})

    task = LlmTask(
        id=uuid.uuid4(), task_type=USAGE_CLASSIFICATION_TASK_TYPE,
        memory_id=None,
        payload={
            "recall_event_ids": [recall_id],
            "llm_output_text": "ок",
            "agent_id": agent,
        },
        retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert out.skipped_by_llm is True
    assert out.result is not None
    assert out.result["skip"] is True
    # used_in_output остался false.
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT used_in_output FROM recall_events WHERE id = %s", (recall_id,))
        assert cur.fetchone()["used_in_output"] is False


def test_handler_scored_flips_used_in_output(db, rate_limit) -> None:
    agent = f"classifier-test-{uuid.uuid4().hex[:6]}"
    m1, r1 = _seed(db, agent, content="qwen3:4b-local предпочтение")
    m2, r2 = _seed(db, agent, content="портфель пятница")

    handler = create_usage_classification_handler()
    llm = _ScriptedChat(
        {
            "skip": False,
            "skip_reason": None,
            "classifications": [
                {"memory_id": str(m1), "used": True, "reason": "опирается"},
                {"memory_id": str(m2), "used": False, "reason": "тема другая"},
            ],
        }
    )
    task = LlmTask(
        id=uuid.uuid4(), task_type=USAGE_CLASSIFICATION_TASK_TYPE,
        memory_id=None,
        payload={
            "recall_event_ids": [r1, r2],
            "llm_output_text": "qwen3:4b-local — выбираю за окно",
            "agent_id": agent,
        },
        retry_count=0,
    )
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert out.skipped_by_llm is False
    assert out.result is not None
    assert out.result["flipped"] == 1
    assert r1 in out.result["used_recall_ids"]

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, used_in_output FROM recall_events WHERE id = ANY(%s::bigint[])",
            ([r1, r2],),
        )
        rows = {int(r["id"]): r["used_in_output"] for r in cur.fetchall()}
    assert rows[r1] is True
    assert rows[r2] is False


def test_handler_reconcile_unknown_memory_id(db, rate_limit) -> None:
    agent = f"classifier-test-{uuid.uuid4().hex[:6]}"
    _, r1 = _seed(db, agent, content="some content")

    handler = create_usage_classification_handler()
    llm = _ScriptedChat(
        {
            "skip": False,
            "skip_reason": None,
            "classifications": [
                {
                    "memory_id": str(uuid.uuid4()),  # фантом
                    "used": True,
                    "reason": "x",
                }
            ],
        }
    )
    task = LlmTask(
        id=uuid.uuid4(), task_type=USAGE_CLASSIFICATION_TASK_TYPE,
        memory_id=None,
        payload={
            "recall_event_ids": [r1],
            "llm_output_text": "содержательный ответ",
            "agent_id": agent,
        },
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError, match="classification_mismatch"):
        handler(task, _ctx(db, llm, rate_limit))


def test_handler_reconcile_missing_memory_id(db, rate_limit) -> None:
    agent = f"classifier-test-{uuid.uuid4().hex[:6]}"
    m1, r1 = _seed(db, agent, content="aaa")
    m2, r2 = _seed(db, agent, content="bbb")

    handler = create_usage_classification_handler()
    # классификация только m1, m2 пропущен.
    llm = _ScriptedChat(
        {
            "skip": False,
            "skip_reason": None,
            "classifications": [
                {"memory_id": str(m1), "used": True, "reason": "x"},
            ],
        }
    )
    task = LlmTask(
        id=uuid.uuid4(), task_type=USAGE_CLASSIFICATION_TASK_TYPE,
        memory_id=None,
        payload={
            "recall_event_ids": [r1, r2],
            "llm_output_text": "x",
            "agent_id": agent,
        },
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError, match="missing memory_id"):
        handler(task, _ctx(db, llm, rate_limit))


def test_handler_bad_payload_terminal(db, rate_limit) -> None:
    handler = create_usage_classification_handler()
    llm = _ScriptedChat({})
    task = LlmTask(
        id=uuid.uuid4(), task_type=USAGE_CLASSIFICATION_TASK_TYPE,
        memory_id=None,
        payload={"foo": "bar"},  # битая
        retry_count=0,
    )
    with pytest.raises(OllamaTerminalError, match="bad_payload"):
        handler(task, _ctx(db, llm, rate_limit))


def test_system_prompt_starts_with_known_phrase() -> None:
    assert "Ты классифицируешь" in SYSTEM_PROMPT
    assert "qwen3:4b-local" in SYSTEM_PROMPT  # пример из калибровки
