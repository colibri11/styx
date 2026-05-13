"""Agent registry для Styx HTTP daemon.

Хранит по одному ``AgentSession`` на ``agent_id``. Регистрация —
``/agent/initialize`` handler; deregister — ``/agent/shutdown`` или
shutdown daemon'а.

Контракт идемпотентного initialize (Q15 в design-doc):
повторный initialize для уже зарегистрированного agent_id обновляет
session_id и tools, не пересоздаёт state. Реальная re-configure
делается на стороне ``StyxMemoryCore.initialize()`` который вызывается
до registry.register.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException


@dataclass
class AgentSession:
    """Registry entry — одна live агентская сессия в core daemon."""

    agent_id: str
    core: Any  # StyxMemoryCore (Any чтобы не тащить cycle import)
    write_lock: threading.Lock
    started_at: float
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)


_REGISTRY: dict[str, AgentSession] = {}
_LOCK = threading.Lock()


def register(
    agent_id: str,
    core: Any,
    *,
    write_lock: threading.Lock | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> AgentSession:
    """Зарегистрировать агента. Идемпотентно: если уже есть — replace."""
    session = AgentSession(
        agent_id=agent_id,
        core=core,
        write_lock=write_lock or threading.Lock(),
        started_at=time.monotonic(),
        tool_schemas=list(tool_schemas or []),
    )
    with _LOCK:
        _REGISTRY[agent_id] = session
    return session


def unregister(agent_id: str) -> AgentSession | None:
    """Удалить из registry; возвращает старый session если был."""
    with _LOCK:
        return _REGISTRY.pop(agent_id, None)


def get(agent_id: str) -> AgentSession:
    """Получить session или 404 если не зарегистрирован."""
    session = _REGISTRY.get(agent_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"agent_id={agent_id!r} not initialized; "
                   "call POST /agent/initialize first",
        )
    return session


def get_optional(agent_id: str) -> AgentSession | None:
    return _REGISTRY.get(agent_id)


def all_agent_ids() -> list[str]:
    return list(_REGISTRY.keys())


def reset_all() -> None:
    """Очистить registry. Для тестов и daemon shutdown."""
    with _LOCK:
        _REGISTRY.clear()
