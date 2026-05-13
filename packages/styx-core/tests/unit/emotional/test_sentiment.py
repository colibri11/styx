"""Unit-тесты для emotional/sentiment.py."""

from __future__ import annotations

import math
from typing import Any

import pytest

from styx.emotional.sentiment import (
    K_HOT,
    MAX_PEER_REPLY_LENGTH,
    MIN_PEER_REPLY_LENGTH,
    SYSTEM_PROMPT,
    SentimentClient,
    _validate_vad,
    scale_hot_vad_delta,
)
from styx.emotional.state import EmotionalVector
from styx.llm import (
    LLMRateLimiter,
    OllamaChatClient,
    OllamaTerminalError,
    OllamaTransientError,
)


# ── Validator ──────────────────────────────────────────────────────────


def test_validate_vad_valid() -> None:
    out = _validate_vad({"valence": 0.5, "arousal": -0.2, "dominance": 0.0})
    assert out == EmotionalVector(0.5, -0.2, 0.0)


def test_validate_vad_out_of_range() -> None:
    with pytest.raises(ValueError, match="valence"):
        _validate_vad({"valence": 1.5, "arousal": 0, "dominance": 0})


def test_validate_vad_missing_axis() -> None:
    with pytest.raises(ValueError, match="dominance"):
        _validate_vad({"valence": 0.0, "arousal": 0.0})


def test_validate_vad_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        _validate_vad({"valence": float("nan"), "arousal": 0, "dominance": 0})


def test_validate_vad_not_dict() -> None:
    with pytest.raises(ValueError, match="dict"):
        _validate_vad([0.5, 0.5, 0.5])


# ── scale_hot_vad_delta ───────────────────────────────────────────────


def test_scale_hot_vad_delta() -> None:
    vad = EmotionalVector(1.0, -0.5, 0.0)
    out = scale_hot_vad_delta(vad)
    assert math.isclose(out.valence, K_HOT)
    assert math.isclose(out.arousal, -K_HOT * 0.5)
    assert math.isclose(out.dominance, 0.0)


# ── SentimentClient ────────────────────────────────────────────────────


class _ScriptedChat(OllamaChatClient):
    """OllamaChatClient с заранее заданным response/exception."""

    def __init__(self, scenario: dict[str, Any] | Exception) -> None:
        super().__init__(base_url="http://x", model="m", max_attempts=1)
        self._scenario = scenario
        self.calls = 0

    def chat_json(self, messages, *, timeout_s=None, max_attempts=None):  # type: ignore[override]
        self.calls += 1
        if isinstance(self._scenario, Exception):
            raise self._scenario
        return self._scenario


def _make_client(scenario: dict[str, Any] | Exception) -> SentimentClient:
    rate = LLMRateLimiter(capacity=10, refill_per_second=10.0)
    return SentimentClient(llm=_ScriptedChat(scenario), rate_limit=rate)


# Tests


def test_extract_vad_skip_too_short() -> None:
    client = _make_client({"valence": 0.5, "arousal": 0, "dominance": 0})
    out = client.extract_vad("ок")
    assert out is None
    assert client.metrics.skips_too_short == 1
    # LLM не звался.
    assert client._llm.calls == 0  # type: ignore[union-attr]


def test_extract_vad_skip_too_long() -> None:
    client = _make_client({"valence": 0, "arousal": 0, "dominance": 0})
    huge = "a" * (MAX_PEER_REPLY_LENGTH + 1)
    out = client.extract_vad(huge)
    assert out is None
    assert client.metrics.skips_too_long == 1


def test_extract_vad_valid_response() -> None:
    client = _make_client({"valence": 0.7, "arousal": 0.4, "dominance": 0.2})
    text = "Ура, всё получилось наконец-то!"  # >= 20 chars
    out = client.extract_vad(text)
    assert out == EmotionalVector(0.7, 0.4, 0.2)
    assert client.metrics.calls == 1


def test_extract_vad_timeout_returns_none() -> None:
    client = _make_client(OllamaTransientError("timeout"))
    out = client.extract_vad("Какой-то длинный текст для эмоций.")
    assert out is None
    assert client.metrics.timeouts == 1


def test_extract_vad_terminal_returns_none() -> None:
    """4xx / malformed JSON в content → None, fail-open."""
    client = _make_client(OllamaTerminalError("HTTP 400"))
    out = client.extract_vad("Ну спасибо, очень помог.")
    assert out is None
    assert client.metrics.transient_errors == 1


def test_extract_vad_schema_error_returns_none() -> None:
    """LLM вернула что-то не подходящее под VAD-схему."""
    client = _make_client({"score": 0.5})  # нет valence/arousal/dominance
    out = client.extract_vad("Какой-то длинный достаточно текст.")
    assert out is None
    assert client.metrics.schema_errors == 1


def test_extract_vad_rate_limit_exhausted_skips() -> None:
    """Если rate-limit пуст — extract_vad возвращает None, не блокирует."""
    rate = LLMRateLimiter(capacity=1, refill_per_second=0.1)
    rate.try_acquire()  # съели единственный токен
    client = SentimentClient(
        llm=_ScriptedChat({"valence": 0, "arousal": 0, "dominance": 0}),
        rate_limit=rate,
    )
    out = client.extract_vad("Какая-то достаточно длинная реплика.")
    assert out is None
    assert client.metrics.transient_errors == 1


def test_extract_vad_unexpected_exception_returns_none() -> None:
    """Любая неожиданная ошибка — fail-open."""
    client = _make_client(RuntimeError("неожиданная"))
    out = client.extract_vad("Какая-то достаточно длинная реплика.")
    assert out is None
    assert client.metrics.transient_errors == 1


# ── System prompt ──────────────────────────────────────────────────────


def test_system_prompt_starts_with_known_phrase() -> None:
    """Не автогенерим — port буквальный."""
    assert "valence: негативный" in SYSTEM_PROMPT
    assert "Ура, всё получилось" in SYSTEM_PROMPT
