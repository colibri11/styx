"""Unit-тесты для embedding клиента."""

from __future__ import annotations

import io
import json
import math
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import patch

import pytest

from styx.embedding import (
    EmbeddingError,
    FakeEmbeddingClient,
    OllamaEmbeddingClient,
    make_embedding_client,
)


# ── FakeEmbeddingClient ────────────────────────────────────────────────


def test_fake_embed_dim_matches() -> None:
    client = FakeEmbeddingClient(dim=768)
    vec = client.embed("hello")
    assert len(vec) == 768
    assert client.dim == 768


def test_fake_embed_deterministic() -> None:
    a = FakeEmbeddingClient().embed("same text")
    b = FakeEmbeddingClient().embed("same text")
    assert a == b


def test_fake_embed_different_texts_different_vectors() -> None:
    a = FakeEmbeddingClient().embed("alpha")
    b = FakeEmbeddingClient().embed("beta")
    assert a != b


def test_fake_embed_unit_length() -> None:
    """Нормализованный вектор — длина = 1.0 (с float-погрешностью)."""
    vec = FakeEmbeddingClient().embed("test")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-9


def test_fake_embed_different_texts_almost_orthogonal() -> None:
    """768-dim случайные нормализованные → cosine sim ≈ 0 (с разбросом)."""
    a = FakeEmbeddingClient().embed("alpha topic")
    b = FakeEmbeddingClient().embed("totally unrelated subject")
    cos = sum(x * y for x, y in zip(a, b))
    assert abs(cos) < 0.2  # реально ≪ 0.1, ставим запас


def test_fake_embed_rejects_empty() -> None:
    with pytest.raises(EmbeddingError):
        FakeEmbeddingClient().embed("")


def test_fake_embed_custom_dim() -> None:
    client = FakeEmbeddingClient(dim=128)
    assert len(client.embed("x")) == 128


# ── OllamaEmbeddingClient ──────────────────────────────────────────────


class _FakeResponse:
    """Минимальный context-manager поверх bytes payload."""

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._buf.read()


def _patch_urlopen(payload: dict[str, Any]):
    """patch on urllib.request.urlopen для возврата JSON-ответа."""
    raw = json.dumps(payload).encode("utf-8")
    return patch.object(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(raw),
    )


def test_ollama_success() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://ollama:11434",
        model="embeddinggemma:300m-qat-q8_0",
        dim=768,
    )
    expected = [0.1] * 768
    with _patch_urlopen({"embeddings": [expected]}):
        out = client.embed("hello")
    assert out == expected


def test_ollama_strips_trailing_slash() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://ollama:11434/",
        model="m",
        dim=4,
    )
    captured: dict[str, Any] = {}

    def _capture(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = req.full_url
        return _FakeResponse(json.dumps({"embeddings": [[0.0, 0.0, 0.0, 0.0]]}).encode())

    with patch.object(urllib.request, "urlopen", _capture):
        client.embed("x")
    assert captured["url"] == "http://ollama:11434/api/embed"


def test_ollama_dim_mismatch_raises() -> None:
    client = OllamaEmbeddingClient(base_url="http://x", model="m", dim=768)
    with _patch_urlopen({"embeddings": [[0.1, 0.2, 0.3]]}):
        with pytest.raises(EmbeddingError, match="dim mismatch"):
            client.embed("hello")


def test_ollama_missing_embeddings_raises() -> None:
    client = OllamaEmbeddingClient(base_url="http://x", model="m", dim=768)
    with _patch_urlopen({"error": "no model"}):
        with pytest.raises(EmbeddingError, match="нет embeddings"):
            client.embed("hello")


def test_ollama_non_finite_raises() -> None:
    client = OllamaEmbeddingClient(base_url="http://x", model="m", dim=4)
    with _patch_urlopen({"embeddings": [[0.1, float("nan"), 0.3, 0.4]]}):
        with pytest.raises(EmbeddingError, match="NaN/Inf"):
            client.embed("hello")


def test_ollama_http_error_raises() -> None:
    client = OllamaEmbeddingClient(base_url="http://x", model="m", dim=768)

    def _raise(req: Any, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError(
            req.full_url, 500, "Server Error", {}, io.BytesIO(b"")  # type: ignore[arg-type]
        )

    with patch.object(urllib.request, "urlopen", _raise):
        with pytest.raises(EmbeddingError, match="HTTP 500"):
            client.embed("hello")


def test_ollama_url_error_raises() -> None:
    client = OllamaEmbeddingClient(base_url="http://no-such-host", model="m", dim=768)

    def _raise(req: Any, timeout: float | None = None) -> None:
        raise urllib.error.URLError("connection refused")

    with patch.object(urllib.request, "urlopen", _raise):
        with pytest.raises(EmbeddingError, match="unreachable"):
            client.embed("hello")


def test_ollama_malformed_json_raises() -> None:
    client = OllamaEmbeddingClient(base_url="http://x", model="m", dim=768)

    with patch.object(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(b"<html>not json</html>"),
    ):
        with pytest.raises(EmbeddingError, match="не-JSON"):
            client.embed("hello")


def test_ollama_rejects_empty_text() -> None:
    client = OllamaEmbeddingClient(base_url="http://x", model="m", dim=768)
    with pytest.raises(EmbeddingError):
        client.embed("")


def test_ollama_request_payload() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://x", model="my-model", dim=4
    )
    captured: dict[str, Any] = {}

    def _capture(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = req.data
        captured["headers"] = dict(req.header_items())
        return _FakeResponse(json.dumps({"embeddings": [[0.0] * 4]}).encode())

    with patch.object(urllib.request, "urlopen", _capture):
        client.embed("payload check")

    body = json.loads(captured["body"].decode("utf-8"))
    assert body == {"model": "my-model", "input": "payload check"}
    assert captured["headers"]["Content-type"] == "application/json"


# ── factory ────────────────────────────────────────────────────────────


def test_make_embedding_client_returns_ollama() -> None:
    client = make_embedding_client(
        base_url="http://x", model="m", dim=768, timeout=5.0
    )
    assert isinstance(client, OllamaEmbeddingClient)
    assert client.dim == 768
