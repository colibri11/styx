"""Module-global per-process state для styx-hermes plugin.

Hermes-обёртки ``StyxOpenAITransport`` / ``on_pre_llm_call``
регистрируются ДО того как ``MemoryProvider.initialize`` известно
agent_id. Этот модуль — точка передачи: после
``MemoryProvider.initialize`` сохраняется ``agent_id`` + ``client``,
и компоненты transport/hook читают их через ``get_session``.

Контракт (Q14 в design-doc):
- ``set_session(agent_id, client)`` — вызывается из
  ``MemoryProvider.initialize`` после успешного HTTP /agent/initialize.
- ``get_session()`` — возвращает ``(agent_id, client)`` или None.
- ``clear_session()`` — вызывается из ``MemoryProvider.shutdown``.

Один Hermes-процесс == один агент == одна установка sesion (Q20).
"""

from __future__ import annotations

import logging
import hashlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from styx_hermes.client import StyxCoreClient

log = logging.getLogger(__name__)

_AGENT_ID: str | None = None
_CLIENT: "StyxCoreClient | None" = None
_LOCK = threading.Lock()
_TURN_KEYS: dict[str, str] = {}
_TURN_CONTENT_KEYS: dict[tuple[str, str], list[str]] = {}


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[bounded by styx]...\n"
    available = max(0, limit - len(marker))
    head = available // 2
    return value[:head] + marker + value[-(available - head):]


def _content_fingerprint(user_content: str, assistant_content: str) -> str:
    bounded = (
        _bounded_text(user_content, 20_000) + "\0"
        + _bounded_text(assistant_content, 40_000)
    )
    return hashlib.sha256(bounded.encode("utf-8")).hexdigest()


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
        _TURN_KEYS.clear()
        _TURN_CONTENT_KEYS.clear()


def remember_turn_key(
    session_id: str,
    idempotency_key: str,
    *,
    user_content: str = "",
    assistant_content: str = "",
) -> None:
    """Publish the finalized physical-turn key for MemoryProvider.sync_turn."""
    with _LOCK:
        _TURN_KEYS[session_id or ""] = idempotency_key
        fingerprint = _content_fingerprint(user_content, assistant_content)
        content_key = (session_id or "", fingerprint)
        keys = _TURN_CONTENT_KEYS.setdefault(content_key, [])
        if not keys or keys[-1] != idempotency_key:
            keys.append(idempotency_key)
        if len(_TURN_KEYS) > 256:
            oldest = next(iter(_TURN_KEYS))
            _TURN_KEYS.pop(oldest, None)
        while len(_TURN_CONTENT_KEYS) > 2_048:
            oldest_content = next(iter(_TURN_CONTENT_KEYS))
            _TURN_CONTENT_KEYS.pop(oldest_content, None)


def get_turn_key(
    session_id: str,
    *,
    user_content: str = "",
    assistant_content: str = "",
) -> str | None:
    """Return, but do not consume, the latest finalized key for a session.

    Keeping the key makes a host retry after an ambiguous HTTP outcome use the
    same durable identity. Hermes invokes post_llm_call before memory sync.
    """
    with _LOCK:
        session = session_id or ""
        exact = _TURN_CONTENT_KEYS.get((
            session,
            _content_fingerprint(user_content, assistant_content),
        ))
        if exact:
            # Preserve the last key for ambiguous transport retries. When two
            # distinct physical turns have identical content, consume them in
            # Hermes' ordered sync queue one by one.
            return exact.pop(0) if len(exact) > 1 else exact[0]
        return _TURN_KEYS.get(session)
