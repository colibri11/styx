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
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from styx_hermes.client import StyxCoreClient

log = logging.getLogger(__name__)

_AGENT_ID: str | None = None
_CLIENT: "StyxCoreClient | None" = None
_LOCK = threading.Lock()
_TURN_KEYS: dict[str, str] = {}
_TURN_CONTENT_KEYS: dict[tuple[str, str], list[str]] = {}
_PRETURN_BY_ACT: "OrderedDict[tuple[str, str], tuple[str | None, float]]" = OrderedDict()
_ACT_COORDINATES: "OrderedDict[tuple[str, str], tuple[str | None, str | None, float]]" = OrderedDict()
_LAST_ACT_KEYS: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
_COMMITTED_ACT_KEYS: "OrderedDict[str, float]" = OrderedDict()

_SNAPSHOT_TTL_S = 120.0
_STATE_TTL_S = 30.0 * 60.0
_MAX_ACT_STATE = 2_048


def _prune_act_state(now: float | None = None) -> None:
    current = time.monotonic() if now is None else now
    for key, (_snapshot, touched_at) in list(_PRETURN_BY_ACT.items()):
        if current - touched_at > _SNAPSHOT_TTL_S:
            _PRETURN_BY_ACT.pop(key, None)
    for key, (_parent, _snapshot, touched_at) in list(_ACT_COORDINATES.items()):
        if current - touched_at > _STATE_TTL_S:
            _ACT_COORDINATES.pop(key, None)
    for key, (_act_key, touched_at) in list(_LAST_ACT_KEYS.items()):
        if current - touched_at > _STATE_TTL_S:
            _LAST_ACT_KEYS.pop(key, None)
    for key, touched_at in list(_COMMITTED_ACT_KEYS.items()):
        if current - touched_at > _STATE_TTL_S:
            _COMMITTED_ACT_KEYS.pop(key, None)
    for mapping in (
        _PRETURN_BY_ACT,
        _ACT_COORDINATES,
        _LAST_ACT_KEYS,
        _COMMITTED_ACT_KEYS,
    ):
        while len(mapping) > _MAX_ACT_STATE:
            mapping.popitem(last=False)


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
        _PRETURN_BY_ACT.clear()
        _ACT_COORDINATES.clear()
        _LAST_ACT_KEYS.clear()
        _COMMITTED_ACT_KEYS.clear()


def remember_preturn_snapshot(
    session_id: str, snapshot_token: str | None, *, act_key: str | None = None
) -> None:
    """Remember a fence only when Hermes exposed its physical turn key.

    An unkeyed FIFO can attach a cancelled preturn to a later act, so it is
    intentionally discarded rather than guessed.
    """
    with _LOCK:
        if not act_key:
            return
        now = time.monotonic()
        _prune_act_state(now)
        coordinate = (session_id or "", act_key)
        _PRETURN_BY_ACT.pop(coordinate, None)
        _PRETURN_BY_ACT[coordinate] = (snapshot_token, now)
        _prune_act_state(now)


def declare_act(session_id: str, act_key: str) -> tuple[str | None, str | None]:
    """Return stable parent/snapshot coordinates for a terminal retry.

    Host order, not timestamps or response text, establishes ancestry.  The
    declaration is retained even after a failed HTTP attempt so a later turn
    still names the physical act it actually followed.
    """
    with _LOCK:
        now = time.monotonic()
        _prune_act_state(now)
        session = session_id or ""
        coordinate = (session, act_key)
        existing = _ACT_COORDINATES.pop(coordinate, None)
        if existing is not None:
            parent, snapshot, _ = existing
            _ACT_COORDINATES[coordinate] = (parent, snapshot, now)
            return parent, snapshot
        parent_entry = _LAST_ACT_KEYS.get(session)
        parent = parent_entry[0] if parent_entry is not None else None
        snapshot_entry = _PRETURN_BY_ACT.pop(coordinate, None)
        snapshot = snapshot_entry[0] if snapshot_entry is not None else None
        _ACT_COORDINATES[coordinate] = (parent, snapshot, now)
        _LAST_ACT_KEYS.pop(session, None)
        _LAST_ACT_KEYS[session] = (act_key, now)
        _prune_act_state(now)
        return parent, snapshot


def mark_cognition_committed(act_key: str) -> None:
    with _LOCK:
        now = time.monotonic()
        _prune_act_state(now)
        _COMMITTED_ACT_KEYS.pop(act_key, None)
        _COMMITTED_ACT_KEYS[act_key] = now
        _prune_act_state(now)


def cognition_committed(act_key: str | None) -> bool:
    if act_key is None:
        return False
    with _LOCK:
        _prune_act_state()
        touched_at = _COMMITTED_ACT_KEYS.pop(act_key, None)
        if touched_at is None:
            return False
        _COMMITTED_ACT_KEYS[act_key] = time.monotonic()
        return True


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
