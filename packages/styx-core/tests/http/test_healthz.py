"""Healthz / Readyz endpoints — schema, status code semantics."""

from __future__ import annotations


def test_healthz_returns_payload(client_no_auth):
    resp = client_no_auth.get("/healthz")
    # Postgres недоступен в unit-окружении → 503; payload должен быть.
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "uptime_s" in data
    assert "postgres" in data
    assert "version" in data


def test_healthz_503_when_postgres_down(client_no_auth):
    """Конфиг указывает на несуществующий Postgres — статус down."""
    resp = client_no_auth.get("/healthz")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "down"
    assert data["postgres"] == "down"


def test_readyz_returns_payload(client_no_auth):
    resp = client_no_auth.get("/readyz")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "ollama" in data
    assert "version" in data
