"""Bearer token auth dependency для Styx HTTP API.

Loopback-rule (D7): если ``http_bind`` ≠ loopback и ``http_token``
пустой — daemon не стартует (это проверка в ``http/server.py``, не
здесь). Если token задан — все non-healthz endpoint'ы требуют
``Authorization: Bearer <token>``, иначе 401.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency: проверяет bearer token если он задан в config.

    Если ``http_token`` в config'е пустой/None — auth disabled, любой
    вызов проходит. Иначе: ``Authorization: Bearer <token>`` обязателен,
    значение сравнивается через ``secrets.compare_digest`` для защиты
    от timing attacks.
    """
    config_token = getattr(request.app.state, "config", None)
    token: str | None = (
        getattr(config_token, "http_token", None) if config_token else None
    )
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    presented = authorization[len("Bearer "):]
    if not secrets.compare_digest(presented, token):
        raise HTTPException(status_code=401, detail="invalid bearer token")
