"""LLM chat client — обёртка над Ollama ``/api/chat`` для structured JSON.

Используется в двух местах:
- ``workers/handlers/*`` — LLM-handlers внутри styx-worker sidecar'а
  (importance scoring, recall classifier).
- ``emotional/sentiment.py`` — hot-path VAD extraction inline в
  ``sync_turn`` (волна 7d).

Sync API. Запрос идёт с ``format=json, stream=false``; Ollama
гарантирует валидный JSON в ``message.content`` если модель умеет
(qwen3:4b-local умеет — Modelfile ``temperature=0``).

Errors:

- ``OllamaTransientError`` — попытка повторить осмысленна
  (timeout, 5xx, network). Retry внутри ``max_attempts``.
- ``OllamaTerminalError`` — повтор не поможет (4xx, не-JSON
  в content, schema mismatch у caller'а). Bubble up наверх.

Никаких ``options`` в request body — Modelfile defaults (num_ctx=50000,
temperature=0) рулят (Rule 1 LLM-layer addendum memorybox'а).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from styx.observability.logging import log_event

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Базовый класс для всех ошибок Ollama-клиента."""


class OllamaTransientError(OllamaError):
    """Временная ошибка — retry имеет смысл."""


class OllamaTerminalError(OllamaError):
    """Терминальная ошибка — retry не поможет."""


class OllamaChatClient:
    """Sync HTTP клиент для Ollama ``/api/chat`` с ``format=json``.

    ``timeout_s`` — per-attempt таймаут urllib. ``max_attempts`` — общее
    число попыток (1 = no retry, 2 = одна повторная попытка).

    Между попытками маленькая фиксированная задержка (200ms × attempt)
    — без exponential backoff, потому что rate-limit держится снаружи
    через ``LLMRateLimiter``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts >= 1")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._max_attempts = max_attempts

    @property
    def model(self) -> str:
        return self._model

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_s: float | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """POST ``/api/chat`` с ``format=json`` и парсингом ответа.

        ``timeout_s`` / ``max_attempts`` overrides — для hot-path
        sentiment'а, который хочет 0.8s + 1 attempt.

        Возвращает уже распарсенный JSON-объект из ``message.content``.
        Не словарь — может быть list/scalar если LLM такое сгенерил;
        caller валидирует структуру.
        """
        if not messages:
            raise OllamaTerminalError("chat_json: messages пустой")

        eff_timeout = timeout_s if timeout_s is not None else self._timeout
        eff_attempts = max_attempts if max_attempts is not None else self._max_attempts

        url = f"{self._base_url}/api/chat"
        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "format": "json",
                "stream": False,
            }
        ).encode("utf-8")

        last_transient: Exception | None = None
        started = time.monotonic()
        for attempt in range(1, eff_attempts + 1):
            try:
                raw = self._request(url, payload, eff_timeout)
            except OllamaTransientError as exc:
                last_transient = exc
                if attempt < eff_attempts:
                    time.sleep(0.2 * attempt)
                    continue
                log_event(
                    log,
                    "ollama_call",
                    op="chat",
                    model=self._model,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    success=False,
                    error="transient",
                    attempts=attempt,
                )
                raise
            except OllamaTerminalError:
                log_event(
                    log,
                    "ollama_call",
                    op="chat",
                    model=self._model,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    success=False,
                    error="terminal",
                    attempts=attempt,
                )
                raise
            try:
                parsed = self._parse(raw)
            except OllamaTerminalError:
                log_event(
                    log,
                    "ollama_call",
                    op="chat",
                    model=self._model,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    success=False,
                    error="parse",
                    attempts=attempt,
                )
                raise
            log_event(
                log,
                "ollama_call",
                op="chat",
                model=self._model,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                success=True,
                attempts=attempt,
            )
            return parsed

        # Защитная строка — на самом деле выйдем из цикла либо через
        # return, либо через raise.
        if last_transient is not None:
            raise last_transient
        raise OllamaTransientError("chat_json: не дошли до запроса")

    # ── helpers ────────────────────────────────────────────────────────

    def _request(self, url: str, payload: bytes, timeout: float) -> bytes:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # 5xx — transient; 4xx — terminal (запрос битый, retry не
            # поможет).
            if 500 <= e.code <= 599:
                raise OllamaTransientError(
                    f"Ollama HTTP {e.code}: {e.reason}"
                ) from e
            raise OllamaTerminalError(
                f"Ollama HTTP {e.code}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise OllamaTransientError(
                f"Ollama unreachable: {e.reason}"
            ) from e
        except TimeoutError as e:
            raise OllamaTransientError(
                f"Ollama timeout after {timeout}s"
            ) from e

    def _parse(self, raw: bytes) -> dict[str, Any]:
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as e:
            raise OllamaTerminalError(
                f"Ollama envelope не-JSON: {raw[:200]!r}"
            ) from e
        message = envelope.get("message")
        if not isinstance(message, dict):
            raise OllamaTerminalError(
                f"Ollama: нет message в ответе: {envelope!r}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise OllamaTerminalError(
                f"Ollama: пустой message.content: {envelope!r}"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise OllamaTerminalError(
                f"Ollama content не-JSON: {content[:200]!r}"
            ) from e


# ── Rate limiter ────────────────────────────────────────────────────────


class LLMRateLimiter:
    """Token-bucket rate limiter для шейпинга вызовов LLM.

    Capacity = пиковый burst (мгновенно отпускаемые tokens).
    Refill = постоянная скорость доливки (tokens/sec).

    Sync, thread-safe. Используется в worker-runtime (волна 7a) и в
    sentiment hot-path (волна 7d, отдельный экземпляр).

    ``acquire(timeout_s=None)`` — блокирует до получения токена либо
    таймаута. ``try_acquire()`` — non-blocking, ``True`` если получили.
    """

    def __init__(self, capacity: int, refill_per_second: float) -> None:
        if capacity < 1:
            raise ValueError("capacity >= 1")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second > 0")
        self._capacity = float(capacity)
        self._refill = float(refill_per_second)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill)
            self._last_refill = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill_locked()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def acquire(self, timeout_s: float | None = None) -> bool:
        """Блокирует до получения токена. ``True`` если получили,
        ``False`` если истёк ``timeout_s``."""
        deadline = (
            time.monotonic() + timeout_s if timeout_s is not None else None
        )
        with self._cond:
            while True:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # Сколько ждать до следующего полного токена.
                deficit = 1.0 - self._tokens
                wait_for_token = deficit / self._refill
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=min(wait_for_token, remaining))
                else:
                    self._cond.wait(timeout=wait_for_token)
