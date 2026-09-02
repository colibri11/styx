"""Hermes preturn hook — one fenced cognitive-input envelope from core.

Регистрируется в ``styx_hermes.plugin.register`` через
``ctx.register_hook("pre_llm_call", on_pre_llm_call)``.

agent_id discovery: ``_agent_session`` set'ится при
``MemoryProvider.initialize``. Если session нет — возвращаем None
(Hermes не аппендит контекст).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from styx_hermes.provenance import execution_provenance

from styx_hermes import _agent_session
from styx_hermes.engine.post_llm_hook import (
    MAX_HISTORY_CONTENT_CHARS,
    MAX_SOURCE_MESSAGES,
    MAX_USER_MESSAGE_CHARS,
    _bounded,
    _bounded_identifier,
    _idempotency_key,
    _text_content,
)

log = logging.getLogger(__name__)

_MAX_CURRENT_EVENT_FIELDS = 6
_MAX_CURRENT_EVENT_TEXT = 64


def _bounded_preturn_messages(raw: Any) -> list[dict[str, str]]:
    """Normalize the host-visible transcript without dropping its window."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw[-MAX_SOURCE_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = _bounded(
            _text_content(item.get("content"), MAX_HISTORY_CONTENT_CHARS),
            MAX_HISTORY_CONTENT_CHARS,
        )
        message = {"role": role, "content": content}
        name = _bounded_identifier(item.get("name"), 256)
        tool_call_id = _bounded_identifier(
            item.get("tool_call_id") or item.get("toolCallId"), 256
        )
        if name:
            message["name"] = name
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        out.append(message)
    return out


def _bounded_current_event(values: dict[str, Any]) -> dict[str, Any]:
    """Keep host coordinates structured, small and free of arbitrary objects."""
    out: dict[str, Any] = {}
    for key, value in list(values.items())[:_MAX_CURRENT_EVENT_FIELDS]:
        if not isinstance(key, str):
            continue
        safe_key = key[:64]
        if isinstance(value, str):
            out[safe_key] = value[:_MAX_CURRENT_EVENT_TEXT]
        elif value is None or isinstance(value, bool):
            out[safe_key] = value
        elif isinstance(value, (int, float)) and (
            not isinstance(value, float) or math.isfinite(value)
        ):
            out[safe_key] = value
    return out


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def on_pre_llm_call(**hermes_kwargs: Any) -> dict[str, str] | None:
    """Hermes pre_llm_call hook — sync HTTP вызов в core daemon."""
    session = _agent_session.get_session()
    if session is None:
        return None
    agent_id, client = session

    user_message = hermes_kwargs.get("user_message")
    user_text = _bounded(
        _text_content(user_message, MAX_USER_MESSAGE_CHARS),
        MAX_USER_MESSAGE_CHARS,
    )
    is_first_turn = bool(hermes_kwargs.get("is_first_turn", False))
    current_event: dict[str, Any] = {"is_first_turn": is_first_turn}
    current_event.update({
        k: v
        for k, v in hermes_kwargs.items()
        if k not in {
            "user_message", "conversation_history", "messages",
            "is_first_turn", "session_id", "model", "platform",
        }
        and v is not None
    })
    session_id = _bounded_identifier(hermes_kwargs.get("session_id"), 256)
    turn_id = _bounded_identifier(hermes_kwargs.get("turn_id"), 256)
    host_key = _idempotency_key(session_id, turn_id) if turn_id else None
    messages = _bounded_preturn_messages(
        hermes_kwargs.get("conversation_history", hermes_kwargs.get("messages"))
    )
    if not messages and user_text:
        messages = [{"role": "user", "content": user_text}]

    try:
        resp = client.cognition_preturn(
            agent_id,
            host_key=host_key,
            parent_host_key=_agent_session.predecessor_act_key(
                session_id, host_key
            ),
            session_id=session_id,
            messages=messages,
            query=user_text or None,
            model=_bounded_identifier(hermes_kwargs.get("model"), 512) or None,
            platform=_bounded_identifier(hermes_kwargs.get("platform"), 64) or None,
            planned_execution_provenance=execution_provenance(hermes_kwargs),
            extra={
                "current_event": _bounded_current_event(current_event),
            },
        )
    except Exception as exc:  # noqa: BLE001 — fail-open
        if not _is_not_found(exc):
            log.warning("styx-core /cognition/preturn failed: %s", exc)
            return None
        # Mixed-version deployment only: the legacy endpoint has no snapshot
        # fence and must disappear once all cores implement cognition/preturn.
        try:
            resp = client.pre_llm_inject(
                agent_id,
                session_id=session_id,
                user_message=user_text,
                is_first_turn=is_first_turn,
                model=_bounded_identifier(hermes_kwargs.get("model"), 512) or None,
                platform=_bounded_identifier(hermes_kwargs.get("platform"), 64)
                or None,
                extra=_bounded_current_event(current_event),
            )
        except Exception as legacy_exc:  # noqa: BLE001 — fail-open
            log.warning("styx-core preturn fallback failed: %s", legacy_exc)
            return None

    snapshot = resp.get("snapshot_token")
    if isinstance(snapshot, str) and snapshot:
        _agent_session.remember_preturn_snapshot(
            session_id,
            snapshot,
            act_key=host_key,
        )
    context = resp.get("system_prompt_addition") or resp.get("context")
    if not context:
        return None
    return {"context": context}
