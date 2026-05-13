"""HTTP API test fixtures — TestClient + fake StyxConfig.

Постгрес-зависимые операции в core (initialize/sync_turn/recall) этими
fixture'ами **не покрываются** — только handler-layer (auth, validation,
healthz). DB integration tests живут в Docker compose stack (Phase E).
"""

from __future__ import annotations

import time
from dataclasses import replace as _replace

import pytest
from fastapi.testclient import TestClient

from styx.config import StyxConfig
from styx.http import registry
from styx.http.app import create_app


def make_config(**overrides) -> StyxConfig:
    """Минимальный StyxConfig для unit-тестов HTTP API."""
    base = StyxConfig(
        database_url="postgresql://x:y@localhost:65535/test",
        ollama_url="http://127.0.0.1:65534",
        http_bind="127.0.0.1",
        http_port=8788,
        http_token=None,
    )
    if overrides:
        return _replace(base, **overrides)
    return base


@pytest.fixture
def client_no_auth() -> TestClient:
    registry.reset_all()
    app = create_app(make_config())
    return TestClient(app)


@pytest.fixture
def client_with_auth() -> TestClient:
    registry.reset_all()
    app = create_app(make_config(http_token="test-token-do-not-use-in-prod"))
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_registry_after():
    yield
    registry.reset_all()
