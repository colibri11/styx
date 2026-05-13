"""Unit-тесты для LLM chat-клиента и rate limiter'а."""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import patch

import pytest

from styx.llm import (
    LLMRateLimiter,
    OllamaChatClient,
    OllamaTerminalError,
    OllamaTransientError,
)


# ── Test scaffolding (повторяет стиль test_embedding.py) ──────────────


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._buf.read()


def _ok_envelope(content: dict[str, Any]) -> bytes:
    """Эмулирует Ollama ``/api/chat`` envelope с JSON-content внутри."""
    return json.dumps(
        {
            "model": "qwen3:4b-local",
            "message": {
                "role": "assistant",
                "content": json.dumps(content),
            },
            "done": True,
        }
    ).encode("utf-8")


def _patch_urlopen_returning(payload: bytes):
    return patch.object(
        urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(payload)
    )


# ── OllamaChatClient: success path ─────────────────────────────────────


def test_chat_json_success() -> None:
    client = OllamaChatClient(base_url="http://x", model="qwen3:4b-local")
    expected = {"answer": 42, "ok": True}
    with _patch_urlopen_returning(_ok_envelope(expected)):
        out = client.chat_json([{"role": "user", "content": "hi"}])
    assert out == expected


def test_chat_json_strips_trailing_slash() -> None:
    client = OllamaChatClient(base_url="http://x/", model="m")
    captured: dict[str, Any] = {}

    def _cap(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = req.full_url
        return _FakeResponse(_ok_envelope({"x": 1}))

    with patch.object(urllib.request, "urlopen", _cap):
        client.chat_json([{"role": "user", "content": "hi"}])
    assert captured["url"] == "http://x/api/chat"


def test_chat_json_request_payload_shape() -> None:
    """В body — ``format=json``, ``stream=false``, нет ``options``."""
    client = OllamaChatClient(base_url="http://x", model="m")
    captured: dict[str, Any] = {}

    def _cap(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = req.data
        return _FakeResponse(_ok_envelope({"y": 0}))

    with patch.object(urllib.request, "urlopen", _cap):
        client.chat_json([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])

    body = json.loads(captured["body"].decode("utf-8"))
    assert body == {
        "model": "m",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        "format": "json",
        "stream": False,
    }
    assert "options" not in body


# ── Errors: terminal vs transient ──────────────────────────────────────


def test_chat_json_4xx_is_terminal() -> None:
    client = OllamaChatClient(
        base_url="http://x", model="m", max_attempts=3
    )
    calls = {"count": 0}

    def _raise(req: Any, timeout: float | None = None) -> None:
        calls["count"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b""))  # type: ignore[arg-type]

    with patch.object(urllib.request, "urlopen", _raise):
        with pytest.raises(OllamaTerminalError, match="HTTP 400"):
            client.chat_json([{"role": "user", "content": "hi"}])
    # Терминальная — не ретраим.
    assert calls["count"] == 1


def test_chat_json_5xx_is_transient_and_retries() -> None:
    client = OllamaChatClient(
        base_url="http://x", model="m", max_attempts=3
    )
    calls = {"count": 0}

    def _flaky(req: Any, timeout: float | None = None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "Bad Gateway", {}, io.BytesIO(b""))  # type: ignore[arg-type]
        return _FakeResponse(_ok_envelope({"recovered": True}))

    with patch.object(urllib.request, "urlopen", _flaky):
        out = client.chat_json([{"role": "user", "content": "hi"}])
    assert out == {"recovered": True}
    assert calls["count"] == 3


def test_chat_json_5xx_exhausts_attempts() -> None:
    client = OllamaChatClient(
        base_url="http://x", model="m", max_attempts=2
    )
    calls = {"count": 0}

    def _bad(req: Any, timeout: float | None = None) -> None:
        calls["count"] += 1
        raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", {}, io.BytesIO(b""))  # type: ignore[arg-type]

    with patch.object(urllib.request, "urlopen", _bad):
        with pytest.raises(OllamaTransientError, match="HTTP 502"):
            client.chat_json([{"role": "user", "content": "hi"}])
    assert calls["count"] == 2


def test_chat_json_url_error_is_transient() -> None:
    client = OllamaChatClient(base_url="http://x", model="m", max_attempts=1)

    def _raise(req: Any, timeout: float | None = None) -> None:
        raise urllib.error.URLError("connection refused")

    with patch.object(urllib.request, "urlopen", _raise):
        with pytest.raises(OllamaTransientError, match="unreachable"):
            client.chat_json([{"role": "user", "content": "hi"}])


def test_chat_json_timeout_is_transient() -> None:
    client = OllamaChatClient(base_url="http://x", model="m", max_attempts=1)

    def _raise(req: Any, timeout: float | None = None) -> None:
        raise TimeoutError("read timed out")

    with patch.object(urllib.request, "urlopen", _raise):
        with pytest.raises(OllamaTransientError, match="timeout"):
            client.chat_json([{"role": "user", "content": "hi"}])


# ── Parse errors: terminal ─────────────────────────────────────────────


def test_chat_json_envelope_not_json_is_terminal() -> None:
    client = OllamaChatClient(base_url="http://x", model="m")
    with _patch_urlopen_returning(b"<html>not json</html>"):
        with pytest.raises(OllamaTerminalError, match="envelope не-JSON"):
            client.chat_json([{"role": "user", "content": "hi"}])


def test_chat_json_missing_message_is_terminal() -> None:
    client = OllamaChatClient(base_url="http://x", model="m")
    with _patch_urlopen_returning(json.dumps({"done": True}).encode()):
        with pytest.raises(OllamaTerminalError, match="нет message"):
            client.chat_json([{"role": "user", "content": "hi"}])


def test_chat_json_empty_content_is_terminal() -> None:
    client = OllamaChatClient(base_url="http://x", model="m")
    payload = json.dumps(
        {"message": {"role": "assistant", "content": ""}}
    ).encode()
    with _patch_urlopen_returning(payload):
        with pytest.raises(OllamaTerminalError, match="пустой message.content"):
            client.chat_json([{"role": "user", "content": "hi"}])


def test_chat_json_content_not_json_is_terminal() -> None:
    client = OllamaChatClient(base_url="http://x", model="m")
    payload = json.dumps(
        {"message": {"role": "assistant", "content": "это не json"}}
    ).encode()
    with _patch_urlopen_returning(payload):
        with pytest.raises(OllamaTerminalError, match="content не-JSON"):
            client.chat_json([{"role": "user", "content": "hi"}])


def test_chat_json_empty_messages_is_terminal() -> None:
    client = OllamaChatClient(base_url="http://x", model="m")
    with pytest.raises(OllamaTerminalError, match="messages пустой"):
        client.chat_json([])


# ── Per-call overrides ─────────────────────────────────────────────────


def test_chat_json_per_call_max_attempts_overrides() -> None:
    """Hot-path sentiment передаёт ``max_attempts=1`` явно."""
    client = OllamaChatClient(base_url="http://x", model="m", max_attempts=5)
    calls = {"count": 0}

    def _bad(req: Any, timeout: float | None = None) -> None:
        calls["count"] += 1
        raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", {}, io.BytesIO(b""))  # type: ignore[arg-type]

    with patch.object(urllib.request, "urlopen", _bad):
        with pytest.raises(OllamaTransientError):
            client.chat_json(
                [{"role": "user", "content": "hi"}], max_attempts=1
            )
    assert calls["count"] == 1


def test_chat_json_per_call_timeout_overrides() -> None:
    client = OllamaChatClient(base_url="http://x", model="m")
    captured: dict[str, Any] = {}

    def _cap(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["timeout"] = timeout
        return _FakeResponse(_ok_envelope({"x": 1}))

    with patch.object(urllib.request, "urlopen", _cap):
        client.chat_json(
            [{"role": "user", "content": "hi"}], timeout_s=0.8
        )
    assert captured["timeout"] == 0.8


# ── LLMRateLimiter ─────────────────────────────────────────────────────


def test_rate_limiter_initial_burst() -> None:
    """capacity=4 → первые 4 try_acquire должны пройти мгновенно."""
    limiter = LLMRateLimiter(capacity=4, refill_per_second=1.0)
    for _ in range(4):
        assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False  # capacity исчерпан


def test_rate_limiter_refills() -> None:
    """После refill_per_second секунды доливается ровно один токен."""
    limiter = LLMRateLimiter(capacity=2, refill_per_second=10.0)  # 10/sec для теста
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    time.sleep(0.15)  # ≈ 1.5 токена должно дойти
    assert limiter.try_acquire() is True


def test_rate_limiter_acquire_blocks() -> None:
    """acquire() ждёт пока появится токен."""
    limiter = LLMRateLimiter(capacity=1, refill_per_second=10.0)
    assert limiter.try_acquire() is True
    started = time.monotonic()
    got = limiter.acquire(timeout_s=0.5)
    elapsed = time.monotonic() - started
    assert got is True
    # При 10/sec следующий токен через ~100ms.
    assert 0.05 <= elapsed <= 0.4


def test_rate_limiter_acquire_timeout() -> None:
    """acquire() с истёкшим таймаутом возвращает False."""
    limiter = LLMRateLimiter(capacity=1, refill_per_second=0.5)  # один токен в 2s
    assert limiter.try_acquire() is True
    started = time.monotonic()
    got = limiter.acquire(timeout_s=0.1)
    elapsed = time.monotonic() - started
    assert got is False
    assert elapsed < 0.3


def test_rate_limiter_thread_safe() -> None:
    """N тредов на acquire — каждый получает токен ровно один раз,
    общее число не превышает capacity + ожидаемый refill."""
    limiter = LLMRateLimiter(capacity=2, refill_per_second=20.0)
    successes: list[bool] = []
    lock = threading.Lock()

    def _try() -> None:
        ok = limiter.try_acquire()
        with lock:
            successes.append(ok)

    threads = [threading.Thread(target=_try) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Без явных пауз доступно не более capacity токенов.
    assert sum(successes) <= 2


def test_rate_limiter_invalid_args() -> None:
    with pytest.raises(ValueError, match="capacity"):
        LLMRateLimiter(capacity=0, refill_per_second=1.0)
    with pytest.raises(ValueError, match="refill"):
        LLMRateLimiter(capacity=1, refill_per_second=0.0)
