"""Tests for sentiment hot-path в StyxMemoryCore.sync_turn (волна 7d)."""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.embedding import FakeEmbeddingClient
from styx.emotional.sentiment import K_HOT, SentimentClient
from styx.emotional.state import EmotionalVector
from styx.llm import LLMRateLimiter, OllamaChatClient
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str) -> str:
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    return migrated_db


class _ScriptedChat(OllamaChatClient):
    def __init__(self, scenario: dict[str, Any] | Exception | None) -> None:
        super().__init__(base_url="http://x", model="m", max_attempts=1)
        self._scenario = scenario
        self.calls = 0

    def chat_json(self, messages, *, timeout_s=None, max_attempts=None):  # type: ignore[override]
        self.calls += 1
        if self._scenario is None:
            return {"valence": 0, "arousal": 0, "dominance": 0}
        if isinstance(self._scenario, Exception):
            raise self._scenario
        return self._scenario


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, scenario: dict[str, Any] | Exception | None
) -> tuple[StyxMemoryCore, _ScriptedChat]:
    embed = FakeEmbeddingClient(dim=768)
    chat = _ScriptedChat(scenario)
    rate = LLMRateLimiter(capacity=10, refill_per_second=10.0)
    sentiment = SentimentClient(llm=chat, rate_limit=rate)

    monkeypatch.setattr(
        "styx.providers.memory.make_embedding_client", lambda **_: embed
    )
    monkeypatch.setattr(
        "styx.providers.memory.make_sentiment_client", lambda **_: sentiment
    )
    return StyxMemoryCore(), chat


def test_sync_turn_appends_emotional_state(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sentiment вернул VAD → emotional_state получает новую точку с
    delta = K_HOT × VAD."""
    p, chat = _make_provider(
        monkeypatch, {"valence": 0.7, "arousal": 0.4, "dominance": 0.2}
    )
    agent = f"sentiment-{uuid.uuid4().hex[:6]}"
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn(
            "Я сегодня очень рад, всё получилось!",
            "Поздравляю!",
            session_id=sid,
        )
        with psycopg.connect(provider_env) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT valence, arousal, dominance, source "
                    "  FROM emotional_state WHERE agent_id = %s "
                    " ORDER BY at DESC, id DESC LIMIT 1",
                    (agent,),
                )
                row = cur.fetchone()
        assert row is not None
        assert row["source"] == "hot_sentiment"
        # base = NEUTRAL, delta = K_HOT × (0.7, 0.4, 0.2).
        assert abs(float(row["valence"]) - K_HOT * 0.7) < 1e-5
        assert abs(float(row["arousal"]) - K_HOT * 0.4) < 1e-5
        assert abs(float(row["dominance"]) - K_HOT * 0.2) < 1e-5
        assert chat.calls == 1
    finally:
        p.shutdown()


def test_sync_turn_skips_emotional_on_short_user(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """user_content короче 20 символов → extract_vad skip'ает,
    emotional_state не обновляется (но sync_turn проходит)."""
    p, chat = _make_provider(
        monkeypatch, {"valence": 0.7, "arousal": 0, "dominance": 0}
    )
    agent = f"sentiment-{uuid.uuid4().hex[:6]}"
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn("ок", "понял", session_id=sid)
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM emotional_state WHERE agent_id = %s",
                    (agent,),
                )
                count = cur.fetchone()[0]
        assert count == 0  # никаких точек
        assert chat.calls == 0  # LLM не звался
    finally:
        p.shutdown()


def test_sync_turn_continues_on_sentiment_error(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sentiment упал → sync_turn возвращается успешно (memories +
    embeddings записаны), emotional_state пустой."""
    from styx.llm import OllamaTransientError

    p, _ = _make_provider(monkeypatch, OllamaTransientError("timeout"))
    agent = f"sentiment-{uuid.uuid4().hex[:6]}"
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn(
            "Какой-то достаточно длинный user-текст для теста.",
            "Понял, сделаю.",
            session_id=sid,
        )
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM memories WHERE agent_id = %s",
                    (agent,),
                )
                memories = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(*) FROM emotional_state WHERE agent_id = %s",
                    (agent,),
                )
                states = cur.fetchone()[0]
        # 2 memories (user + assistant) записаны; emotional_state — пуст.
        assert memories == 2
        assert states == 0
    finally:
        p.shutdown()


def test_sentiment_disabled_via_config(
    monkeypatch: pytest.MonkeyPatch, migrated_db: str
) -> None:
    """STYX_SENTIMENT_ENABLED=0 — provider не создаёт sentiment client."""
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "0")
    embed = FakeEmbeddingClient(dim=768)
    monkeypatch.setattr(
        "styx.providers.memory.make_embedding_client", lambda **_: embed
    )

    # make_sentiment_client не должен звониться при disabled.
    def _shouldnt_be_called(**_: Any) -> Any:
        raise AssertionError("make_sentiment_client не должен вызываться")

    monkeypatch.setattr(
        "styx.providers.memory.make_sentiment_client", _shouldnt_be_called
    )

    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    agent = f"sentiment-disabled-{uuid.uuid4().hex[:6]}"
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        assert p._sentiment is None
        p.sync_turn(
            "Какой-то достаточно длинный текст для теста.",
            "Хорошо.",
            session_id=sid,
        )
    finally:
        p.shutdown()
