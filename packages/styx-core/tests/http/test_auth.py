"""Auth dependency: bearer token + loopback rule."""

from __future__ import annotations

import pytest

from styx.http.server import enforce_auth_or_loopback
from styx.http import registry
from styx.http.registry import AgentSession
import threading


def _register_dummy(agent_id: str = "alyona") -> None:
    """Подсунуть фиктивный AgentSession в registry чтобы handler не упал на 404."""
    session = AgentSession(
        agent_id=agent_id,
        core=None,
        write_lock=threading.Lock(),
        started_at=0.0,
    )
    registry._REGISTRY[agent_id] = session  # noqa: SLF001 — direct injection для unit-теста


def test_no_token_no_auth_required(client_no_auth):
    # Без http_token любой POST на не-healthz endpoint должен пройти
    # хотя бы auth-проверку (далее упадёт на 404 в registry).
    resp = client_no_auth.post(
        "/agent/shutdown", json={"agent_id": "alyona"}
    )
    # 204 (если бы был зарегистрирован) или 200 (текущий handler пишет
    # 204). registry пуст — handler возвращает 204 без ошибки.
    assert resp.status_code in (200, 204)


def test_token_required_when_configured(client_with_auth):
    resp = client_with_auth.post(
        "/agent/shutdown", json={"agent_id": "alyona"}
    )
    assert resp.status_code == 401
    assert "missing bearer" in resp.json()["detail"]


def test_invalid_token_rejected(client_with_auth):
    resp = client_with_auth.post(
        "/agent/shutdown",
        json={"agent_id": "alyona"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401
    assert "invalid bearer" in resp.json()["detail"]


def test_valid_token_accepted(client_with_auth):
    resp = client_with_auth.post(
        "/agent/shutdown",
        json={"agent_id": "alyona"},
        headers={"Authorization": "Bearer test-token-do-not-use-in-prod"},
    )
    assert resp.status_code in (200, 204)


def test_healthz_no_auth_required(client_with_auth):
    """/healthz должен быть доступен даже когда token задан."""
    resp = client_with_auth.get("/healthz")
    # 200 или 503 (Postgres недоступен в unit-тесте) — оба это "auth прошёл"
    assert resp.status_code in (200, 503)


def test_loopback_rule_blocks_open_bind_without_token():
    from styx.config import StyxConfig

    cfg = StyxConfig(
        database_url="postgresql://x:y@localhost/test",
        http_bind="0.0.0.0",
        http_token=None,
    )
    with pytest.raises(SystemExit, match="loopback"):
        enforce_auth_or_loopback(cfg)


def test_loopback_rule_allows_open_bind_with_token():
    from styx.config import StyxConfig

    cfg = StyxConfig(
        database_url="postgresql://x:y@localhost/test",
        http_bind="0.0.0.0",
        http_token="abcdef" * 8,
    )
    enforce_auth_or_loopback(cfg)  # не должно падать


def test_loopback_rule_allows_loopback_without_token():
    from styx.config import StyxConfig

    cfg = StyxConfig(
        database_url="postgresql://x:y@localhost/test",
        http_bind="127.0.0.1",
        http_token=None,
    )
    enforce_auth_or_loopback(cfg)  # не должно падать
