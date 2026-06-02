"""Module-global per-process state для styx-hermes plugin.

Hermes-обёртки ``StyxContextEngine`` / ``StyxOpenAITransport`` /
``on_pre_llm_call`` регистрируются ДО того как ``MemoryProvider.initialize``
известно agent_id. Этот модуль — точка передачи: после
``MemoryProvider.initialize`` сохраняется ``agent_id`` + ``client``,
и компоненты engine/transport/hook читают их через ``get_session``.

Контракт (Q14 в design-doc):
- ``set_session(agent_id, client)`` — вызывается из
  ``MemoryProvider.initialize`` после успешного HTTP /agent/initialize.
- ``get_session()`` — возвращает ``(agent_id, client)`` или None.
- ``clear_session()`` — вызывается из ``MemoryProvider.shutdown``.

Один Hermes-процесс == один агент == одна установка sesion (Q20).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from styx_hermes.client import StyxCoreClient

log = logging.getLogger(__name__)

_AGENT_ID: str | None = None
_CLIENT: "StyxCoreClient | None" = None
_LOCK = threading.Lock()


def set_session(agent_id: str, client: "StyxCoreClient") -> None:
    """Зафиксировать active session. Двойной вызов заменяет state.

    Замена на ОТЛИЧНЫЙ agent_id неожиданна при one-process-one-agent
    (Q20) — шумим warning'ом, но state всё равно заменяем (replace
    намеренный по дизайну). Повтор тем же id — тихо, idempotent.
    """
    global _AGENT_ID, _CLIENT
    with _LOCK:
        if _AGENT_ID is not None and _AGENT_ID != agent_id:
            log.warning(
                "replacing active session agent_id %r with %r — "
                "unexpected under one-process-one-agent (Q20)",
                _AGENT_ID,
                agent_id,
            )
        _AGENT_ID = agent_id
        _CLIENT = client


def get_session() -> "tuple[str, StyxCoreClient] | None":
    """Возвращает ``(agent_id, client)`` или None если не set."""
    if _AGENT_ID is None or _CLIENT is None:
        return None
    return (_AGENT_ID, _CLIENT)


def clear_session() -> None:
    """Сбросить active session. Идемпотентно."""
    global _AGENT_ID, _CLIENT
    with _LOCK:
        _AGENT_ID = None
        _CLIENT = None
