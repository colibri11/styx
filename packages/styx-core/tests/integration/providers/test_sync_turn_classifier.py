"""Test: sync_turn после styx_recall enqueue'ит классификатор."""

from __future__ import annotations

import json
import uuid
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.embedding import FakeEmbeddingClient
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    # Sentiment off — иначе понадобится HTTP mock; здесь нас интересует
    # только classifier-enqueue.
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "0")
    yield migrated_db
    # Cleanup llm_tasks от этого теста чтобы не загрязнять очередь
    # последующим integration-тестам.
    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM llm_tasks "
                " WHERE task_type = 'usage_classification' "
                "   AND payload->>'agent_id' LIKE 'classifier-%'"
            )
        conn.commit()


def _provider_with_fake_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> StyxMemoryCore:
    embed = FakeEmbeddingClient(dim=768)
    monkeypatch.setattr(
        "styx.providers.memory.make_embedding_client", lambda **_: embed
    )
    return StyxMemoryCore()


def test_sync_turn_enqueues_classification_after_recall(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _provider_with_fake_embed(monkeypatch)
    sid = str(uuid.uuid4())
    agent = f"classifier-{uuid.uuid4().hex[:6]}"
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        # Сначала «накормим» базу одной memory с тем же текстом что query
        # — FakeEmbedding hash'ит full text → cosine sim высокая.
        p.sync_turn(
            "Сергей предпочитает qwen3:4b-local за окно контекста.",
            "Понял, запомнил.",
            session_id=sid,
        )

        # Recall на тот же текст — должен найти эту memory + создать
        # recall_event, ids которых попадут в RecallTracker.
        result_json = p.handle_tool_call(
            "styx_recall",
            {"query": "Сергей предпочитает qwen3:4b-local за окно контекста.", "limit": 6},
            session_id=sid,
        )
        result = json.loads(result_json)
        assert result["count"] >= 1

        # Следующий sync_turn — content >= 50 char → enqueue classifier.
        p.sync_turn(
            "Что Сергей думает про qwen3?",
            "Сергей предпочитает qwen3:4b-local за окно контекста — это устойчивое его предпочтение.",
            session_id=sid,
        )

        with psycopg.connect(provider_env) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT task_type, status, payload "
                    "  FROM llm_tasks "
                    " WHERE task_type = 'usage_classification' "
                    "   AND payload->>'agent_id' = %s",
                    (agent,),
                )
                tasks = cur.fetchall()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
            conn.commit()

        assert len(tasks) == 1
        payload = tasks[0]["payload"]
        assert payload["agent_id"] == agent
        assert len(payload["recall_event_ids"]) >= 1
        assert "qwen3" in payload["llm_output_text"]
    finally:
        p.shutdown()


def test_sync_turn_skips_classifier_for_short_assistant(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _provider_with_fake_embed(monkeypatch)
    sid = str(uuid.uuid4())
    agent = f"classifier-{uuid.uuid4().hex[:6]}"
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn("Какой-то длинный текст для записи.", "memory-content!", session_id=sid)
        p.handle_tool_call(
            "styx_recall",
            {"query": "Какой-то длинный текст для записи.", "limit": 6},
            session_id=sid,
        )
        # assistant_content < 50 char — classifier skip'ается.
        p.sync_turn("Длинный какой-то юзер вопрос.", "ок", session_id=sid)

        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM llm_tasks "
                    " WHERE task_type = 'usage_classification' "
                    "   AND payload->>'agent_id' = %s",
                    (agent,),
                )
                n = cur.fetchone()[0]
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
            conn.commit()
        assert n == 0
    finally:
        p.shutdown()


def test_sync_turn_skips_classifier_when_no_recall(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _provider_with_fake_embed(monkeypatch)
    sid = str(uuid.uuid4())
    agent = f"classifier-{uuid.uuid4().hex[:6]}"
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        # Без styx_recall — buffer пустой.
        p.sync_turn(
            "Длинный user вопрос для теста.",
            "Длинный assistant ответ который точно длиннее пятидесяти символов.",
            session_id=sid,
        )
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM llm_tasks "
                    " WHERE task_type = 'usage_classification' "
                    "   AND payload->>'agent_id' = %s",
                    (agent,),
                )
                n = cur.fetchone()[0]
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
            conn.commit()
        assert n == 0
    finally:
        p.shutdown()
