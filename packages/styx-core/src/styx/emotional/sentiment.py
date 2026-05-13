"""Hot-path sentiment extraction — синхронный VAD из peer-реплики.

Прямой port из memorybox `emotional/sentiment.ts`. Числа K_HOT,
MIN/MAX_PEER_REPLY_LENGTH, SENTIMENT_TIMEOUT_S — буквально.

Контракт:

- Возвращает ``EmotionalVector`` ∈ [-1, +1] на каждой оси, либо ``None``
  если skip (длина вне диапазона, таймаут, schema-error, transient).
- Fail-open: любая ошибка → ``None``, никаких raise. Hot-path не должен
  блокировать active context path.
- ``timeout_s=0.8``, ``max_attempts=1`` (no retry на hot-path).
- Modelfile defaults рулят (no ``options`` в request).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from styx.emotional.state import (
    EMOTIONAL_AXIS_MAX,
    EMOTIONAL_AXIS_MIN,
    EmotionalVector,
)
from styx.llm import (
    LLMRateLimiter,
    OllamaChatClient,
    OllamaTerminalError,
    OllamaTransientError,
)

log = logging.getLogger(__name__)


# ── Constants (port memorybox sentiment.ts:29-38) ─────────────────────

K_HOT = 0.15
"""Scale factor для hot-path delta. ``delta = K_HOT × vad``."""

MIN_PEER_REPLY_LENGTH = 20
MAX_PEER_REPLY_LENGTH = 4000

SENTIMENT_TIMEOUT_S = 0.8
SENTIMENT_MAX_ATTEMPTS = 1


SYSTEM_PROMPT = """Ты оцениваешь эмоциональный тон одной реплики собеседника по трём осям VAD в [-1, +1]:
- valence: негативный (-1) ... позитивный (+1)
- arousal: спокойный (-1) ... возбуждённый (+1)
- dominance: подавленный (-1) ... доминирующий (+1)

Верни строгий JSON:
{"valence": number, "arousal": number, "dominance": number}

Без пояснений, без текста вокруг. Только JSON.

Примеры:
"Ура, всё получилось!" → {"valence": 0.85, "arousal": 0.7, "dominance": 0.6}
"Заебало вообще всё, три часа хрень какую-то делаю." → {"valence": -0.7, "arousal": 0.6, "dominance": -0.2}
"Ничего не хочу, уже не вывожу." → {"valence": -0.5, "arousal": -0.6, "dominance": -0.7}
"Понятно, ок." → {"valence": 0.1, "arousal": 0, "dominance": 0}
"Не знаю, как быть, совсем растерялся." → {"valence": -0.3, "arousal": 0.2, "dominance": -0.6}
"Делаем так — решил, идём дальше." → {"valence": 0.2, "arousal": 0.2, "dominance": 0.7}
"Ну спасибо, очень помог." → {"valence": -0.4, "arousal": 0.3, "dominance": -0.1}"""


# ── Validator ─────────────────────────────────────────────────────────


def _validate_vad(raw: object) -> EmotionalVector:
    if not isinstance(raw, dict):
        raise ValueError(f"ожидаем dict, получили {type(raw).__name__}")

    def _axis(name: str) -> float:
        val = raw.get(name)
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{name} должен быть finite number, получено {val!r}")
        v = float(val)
        if not EMOTIONAL_AXIS_MIN <= v <= EMOTIONAL_AXIS_MAX:
            raise ValueError(
                f"{name}={v} вне [{EMOTIONAL_AXIS_MIN}, {EMOTIONAL_AXIS_MAX}]"
            )
        return v

    return EmotionalVector(
        valence=_axis("valence"),
        arousal=_axis("arousal"),
        dominance=_axis("dominance"),
    )


# ── Metrics ───────────────────────────────────────────────────────────


@dataclass
class SentimentMetrics:
    calls: int = 0
    skips_too_short: int = 0
    skips_too_long: int = 0
    timeouts: int = 0
    schema_errors: int = 0
    transient_errors: int = 0
    applied: int = 0
    latency_samples_ms: list[float] = field(default_factory=list)
    _cap: int = 1000

    def record_latency(self, ms: float) -> None:
        self.latency_samples_ms.append(ms)
        if len(self.latency_samples_ms) > self._cap:
            self.latency_samples_ms = self.latency_samples_ms[-self._cap :]

    def snapshot(self) -> dict:
        s = sorted(self.latency_samples_ms)

        def pct(p: float) -> float | None:
            if not s:
                return None
            idx = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
            return s[idx]

        return {
            "calls": self.calls,
            "skips_too_short": self.skips_too_short,
            "skips_too_long": self.skips_too_long,
            "timeouts": self.timeouts,
            "schema_errors": self.schema_errors,
            "transient_errors": self.transient_errors,
            "applied": self.applied,
            "latency_p50_ms": pct(0.5),
            "latency_p95_ms": pct(0.95),
        }


# ── Client ─────────────────────────────────────────────────────────────


class SentimentClient:
    """Sync VAD-extractor для hot-path в ``sync_turn``.

    Внутри держит ссылку на ``OllamaChatClient`` (свой rate-limiter)
    и ``SentimentMetrics``. Thread-safe для metrics.
    """

    def __init__(
        self,
        llm: OllamaChatClient,
        rate_limit: LLMRateLimiter,
    ) -> None:
        self._llm = llm
        self._rate = rate_limit
        self._metrics = SentimentMetrics()
        self._lock = threading.Lock()

    @property
    def metrics(self) -> SentimentMetrics:
        return self._metrics

    def extract_vad(self, text: str) -> EmotionalVector | None:
        """Извлечь VAD из peer-реплики. ``None`` на любой ошибке/skip'е."""
        with self._lock:
            self._metrics.calls += 1

        trimmed = text.strip()
        length = len(trimmed)
        if length < MIN_PEER_REPLY_LENGTH:
            with self._lock:
                self._metrics.skips_too_short += 1
            return None
        if length > MAX_PEER_REPLY_LENGTH:
            with self._lock:
                self._metrics.skips_too_long += 1
            return None

        # Rate-limit: если токена нет, не ждём — пропускаем (лучше
        # пустить turn без свежей дельты, чем добавить latency).
        if not self._rate.try_acquire():
            with self._lock:
                self._metrics.transient_errors += 1
            return None

        started_ms = time.monotonic() * 1000
        try:
            raw = self._llm.chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": trimmed},
                ],
                timeout_s=SENTIMENT_TIMEOUT_S,
                max_attempts=SENTIMENT_MAX_ATTEMPTS,
            )
        except OllamaTransientError as exc:
            log.debug("sentiment transient: %s", exc)
            with self._lock:
                self._metrics.timeouts += 1
            return None
        except OllamaTerminalError as exc:
            log.debug("sentiment terminal: %s", exc)
            with self._lock:
                self._metrics.transient_errors += 1
            return None
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.debug("sentiment unexpected: %s", exc)
            with self._lock:
                self._metrics.transient_errors += 1
            return None

        elapsed_ms = time.monotonic() * 1000 - started_ms
        with self._lock:
            self._metrics.record_latency(elapsed_ms)

        try:
            vad = _validate_vad(raw)
        except ValueError as exc:
            log.debug("sentiment schema_error: %s (raw=%r)", exc, raw)
            with self._lock:
                self._metrics.schema_errors += 1
            return None

        return vad

    def increment_applied(self) -> None:
        with self._lock:
            self._metrics.applied += 1


def scale_hot_vad_delta(vad: EmotionalVector) -> EmotionalVector:
    """Multiply by K_HOT — final delta ∈ [-K_HOT, K_HOT] на каждой оси."""
    return EmotionalVector(
        valence=K_HOT * vad.valence,
        arousal=K_HOT * vad.arousal,
        dominance=K_HOT * vad.dominance,
    )


def make_sentiment_client(
    *,
    base_url: str,
    model: str,
    timeout_s: float = SENTIMENT_TIMEOUT_S,
) -> SentimentClient:
    """Factory: создаёт OllamaChatClient + LLMRateLimiter (узкий, capacity=2)
    и возвращает SentimentClient.

    Hot-path использует **отдельный** rate-limiter, не shared с worker'ом
    (worker — отдельный sidecar-процесс, межпроцессного rate-limit нет
    в волне 7d).
    """
    llm = OllamaChatClient(
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        max_attempts=SENTIMENT_MAX_ATTEMPTS,
    )
    # capacity=2, refill 2/sec — мягкий burst для редких реплик,
    # не давит Ollama.
    rate_limit = LLMRateLimiter(capacity=2, refill_per_second=2.0)
    return SentimentClient(llm=llm, rate_limit=rate_limit)
