"""Юнит-тесты `POST /agent/cache_stats` + analytics extension (волна 29 Phase E)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from styx.engine.cache_stats import get_cache_stats, reset_cache_stats
from styx.http.routes.agent import router as agent_router


@pytest.fixture(autouse=True)
def _reset_stats():
    reset_cache_stats()
    yield
    reset_cache_stats()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_router)
    return TestClient(app)


def test_post_cache_stats_returns_204(client: TestClient) -> None:
    """Successful push → 204 No Content (fire-and-forget endpoint)."""
    r = client.post(
        "/agent/cache_stats",
        json={
            "agent_id": "alpha",
            "cache_read_tokens": 100,
            "cache_creation_tokens": 20,
        },
    )
    assert r.status_code == 204
    s = get_cache_stats("alpha")
    assert s["cache_read_tokens"] == 100
    assert s["cache_creation_tokens"] == 20
    assert s["samples"] == 1


def test_post_cache_stats_validates_agent_id(client: TestClient) -> None:
    """Пустой agent_id → 422 (Pydantic min_length=1)."""
    r = client.post(
        "/agent/cache_stats",
        json={"agent_id": "", "cache_read_tokens": 1, "cache_creation_tokens": 1},
    )
    assert r.status_code == 422


def test_post_cache_stats_validates_negative_tokens(client: TestClient) -> None:
    """Negative tokens → 422 (Pydantic ge=0)."""
    r = client.post(
        "/agent/cache_stats",
        json={
            "agent_id": "alpha",
            "cache_read_tokens": -1,
            "cache_creation_tokens": 0,
        },
    )
    assert r.status_code == 422


def test_post_cache_stats_zero_zero_recorded(client: TestClient) -> None:
    """Cache miss всё равно учитывается (sample++ важен для ratio)."""
    r = client.post(
        "/agent/cache_stats",
        json={
            "agent_id": "alpha",
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        },
    )
    assert r.status_code == 204
    s = get_cache_stats("alpha")
    assert s["samples"] == 1


def test_post_multiple_pushes_accumulate(client: TestClient) -> None:
    """Несколько push'ей за turn — суммируются."""
    for n in (1, 2, 3):
        client.post(
            "/agent/cache_stats",
            json={
                "agent_id": "alpha",
                "cache_read_tokens": n * 10,
                "cache_creation_tokens": n,
            },
        )
    s = get_cache_stats("alpha")
    assert s["cache_read_tokens"] == 60
    assert s["cache_creation_tokens"] == 6
    assert s["samples"] == 3
