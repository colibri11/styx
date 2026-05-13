"""Embedding client — обёртка над Ollama HTTP API.

Используется в двух местах:
- ``providers/memory.py`` — embed-after-commit hook в sync_turn.
- ``storage/recall.py`` — embed query при recall-tool.

Sync API — embed возвращает list[float] напрямую. Замер на
embeddinggemma:300m-qat-q8_0 показал latency 54-79 ms на сообщение
→ sync блокировка после COMMIT не вредит.

Errors поднимаются как ``EmbeddingError``. Caller'ы логируют и
оставляют ``embedding = NULL`` в БД — recall просто не подберёт такие
memories до следующего успешного embed.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from typing import Protocol

from styx.observability.logging import log_event

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Любой провал embed-вызова. Caller обрабатывает."""


class EmbeddingClient(Protocol):
    """Интерфейс embed-провайдера. Sync, no batching в волне 7."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbeddingClient:
    """HTTP клиент для Ollama ``/api/embed``.

    Endpoint и модель приходят из конфига. Timeout по умолчанию 30s
    (с запасом на cold start ≈ 700ms на embeddinggemma; warm ≈ 60ms).

    Retry по умолчанию выключен. Один промах = ``embedding = NULL`` для
    этой memory; recall на ней не сработает до следующего успешного
    embed. Это сознательно: sync блокировка на 200-500ms × N retry
    страшнее чем периодически потерянный вектор.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._timeout = timeout

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if not text:
            raise EmbeddingError("embed: пустой текст недопустим")

        url = f"{self._base_url}/api/embed"
        payload = json.dumps({"model": self._model, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            log_event(
                log,
                "ollama_call",
                op="embed",
                model=self._model,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                success=False,
                error=f"http_{e.code}",
            )
            raise EmbeddingError(f"Ollama HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            log_event(
                log,
                "ollama_call",
                op="embed",
                model=self._model,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                success=False,
                error="unreachable",
            )
            raise EmbeddingError(f"Ollama unreachable: {e.reason}") from e
        except TimeoutError as e:
            log_event(
                log,
                "ollama_call",
                op="embed",
                model=self._model,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                success=False,
                error="timeout",
            )
            raise EmbeddingError(f"Ollama timeout after {self._timeout}s") from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise EmbeddingError(f"Ollama вернул не-JSON: {raw[:100]!r}") from e

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingError(f"Ollama: нет embeddings в ответе: {data!r}")
        vec = embeddings[0]
        if not isinstance(vec, list):
            raise EmbeddingError(f"Ollama: embeddings[0] не list: {vec!r}")
        if len(vec) != self._dim:
            raise EmbeddingError(
                f"Ollama dim mismatch: ожидаем {self._dim}, получили {len(vec)}"
            )
        # NaN/Inf отсеиваем — иначе pgvector упадёт при INSERT'е.
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vec):
            raise EmbeddingError("Ollama: вектор содержит NaN/Inf")
        log_event(
            log,
            "ollama_call",
            op="embed",
            model=self._model,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            success=True,
        )
        return [float(x) for x in vec]


class FakeEmbeddingClient:
    """Детерминированный embed для тестов.

    sha256(text) → seed для random.Random → ``dim`` гауссовых отсчётов
    → нормализация к unit-length. Свойства:

    - Одинаковые тексты → одинаковые векторы.
    - Разные тексты → почти ортогональные (cosine ≈ 0).
    - Длина = 1, годится для cosine_similarity тестов.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if not text:
            raise EmbeddingError("embed: пустой текст недопустим")
        rng = random.Random(text)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            raise EmbeddingError("FakeEmbedding: нулевой вектор (невероятно)")
        return [x / norm for x in vec]


def make_embedding_client(
    *,
    base_url: str,
    model: str,
    dim: int,
    timeout: float = 30.0,
) -> EmbeddingClient:
    """Factory — текущая реализация = OllamaEmbeddingClient."""
    return OllamaEmbeddingClient(
        base_url=base_url, model=model, dim=dim, timeout=timeout
    )
