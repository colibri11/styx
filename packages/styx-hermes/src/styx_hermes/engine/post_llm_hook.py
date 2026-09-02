"""Hermes ``post_llm_call`` hook — bounded terminal cognition transport.

Hermes fires this hook once after the tool loop has produced the finalized
assistant response. Styx forwards the finalized channel projection plus the
ordered tool evidence that survived host reduction. The projection is not
claimed to be the whole cognitive act. The hook is deliberately fail-open.

This module owns transport only. The core endpoint evaluates and persists the
causal transition; the host adapter never turns evidence into a style command.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from styx_hermes import _agent_session

log = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 24
MAX_HISTORY_CONTENT_CHARS = 4_000
MAX_TOOL_EVENTS = 64
MAX_TOOL_EVENT_CONTENT_CHARS = 4_000
MAX_USER_MESSAGE_CHARS = 20_000
MAX_ASSISTANT_RESPONSE_CHARS = 40_000
MAX_SOURCE_MESSAGES = 256
MAX_CONTENT_PARTS = 128
MAX_SERIALIZED_ITEMS = 64
MAX_SERIALIZED_DEPTH = 3


def _text_content(value: Any, limit: int = MAX_ASSISTANT_RESPONSE_CHARS) -> str:
    """Извлечь только model-visible text из string/multimodal content."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    remaining = max(0, limit)
    # Slice before iteration: a hostile multimodal array cannot force an
    # unbounded preprocessing pass before transport limits are applied.
    for part in reversed(value[-MAX_CONTENT_PARTS:]):
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("text", "input_text", "output_text"):
            continue
        text = part.get("text")
        if isinstance(text, str) and text and remaining > 0:
            piece = text[-remaining:]
            parts.append(piece)
            remaining -= len(piece) + (1 if len(parts) > 1 else 0)
        if remaining <= 0:
            break
    parts.reverse()
    return "\n".join(parts)[:limit]


def _bounded(text: str, limit: int) -> str:
    """Bound text while retaining both the opening and latest outcome."""
    if len(text) <= limit:
        return text
    marker = "\n...[bounded by styx]...\n"
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def _bounded_identifier(value: Any, limit: int) -> str:
    """Keep ordinary host IDs readable and hash pathological oversized IDs."""
    if isinstance(value, str):
        text = value.strip()
    elif value is None:
        text = ""
    elif isinstance(value, (bool, int, float)):
        text = str(value).strip()
    else:
        text = f"<{type(value).__name__}>"
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    marker = f":sha256:{digest}"
    if len(marker) >= limit:
        return digest[:limit]
    return text[: limit - len(marker)] + marker


def _serialization_projection(value: Any, *, depth: int = 0) -> Any:
    """Project arbitrary tool payloads into a small JSON-safe structure."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded(value, MAX_TOOL_EVENT_CONTENT_CHARS)
    if depth >= MAX_SERIALIZED_DEPTH:
        return f"<{type(value).__name__}:bounded>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        # Islice-like key traversal without stringifying values or materializing
        # the whole mapping. Unknown key objects become a type marker.
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_SERIALIZED_ITEMS:
                out["...[bounded]"] = True
                break
            safe_key = key if isinstance(key, str) else f"<{type(key).__name__}>"
            out[_bounded(safe_key, 128)] = _serialization_projection(
                item, depth=depth + 1
            )
        return out
    if isinstance(value, (list, tuple)):
        out = [
            _serialization_projection(item, depth=depth + 1)
            for item in value[:MAX_SERIALIZED_ITEMS]
        ]
        if len(value) > MAX_SERIALIZED_ITEMS:
            out.append("...[bounded]")
        return out
    return f"<{type(value).__name__}>"


def _bounded_serialized(value: Any, limit: int) -> str:
    if isinstance(value, str):
        return _bounded(value, limit)
    try:
        text = json.dumps(
            _serialization_projection(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError):
        text = f'"<{type(value).__name__}:unserializable>"'
    return _bounded(text, limit)


def _idempotency_key(session_id: str, turn_id: str) -> str:
    raw = f"hermes:{session_id or '-'}:{turn_id}"
    if len(raw) <= 512:
        return raw
    return f"hermes:sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _bounded_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    # Scan newest-first and stop as soon as the output contract is full. Both
    # the source and per-message content work are therefore bounded up front.
    for message in reversed(raw[-MAX_SOURCE_MESSAGES:]):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        content = _text_content(
            message.get("content"), MAX_HISTORY_CONTENT_CHARS
        )
        if not content:
            continue
        item = {
            "role": role,
            "content": _bounded(content, MAX_HISTORY_CONTENT_CHARS),
        }
        name = message.get("name")
        if isinstance(name, str) and name:
            item["name"] = _bounded(name, 256)
        out.append(item)
        if len(out) >= MAX_HISTORY_MESSAGES:
            break
    out.reverse()
    return out


def _bounded_tool_events(raw: Any) -> list[dict[str, Any]]:
    """Extract bounded tool calls/results from Hermes conversation messages."""
    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    inspected_calls = 0
    # Work backwards so we can stop after the newest MAX_TOOL_EVENTS rather
    # than traversing every tool call in every source message.
    for message in reversed(raw[-MAX_SOURCE_MESSAGES:]):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            events.append(
                {
                    "kind": "error" if message.get("is_error") is True else "result",
                    "tool_event_id": _bounded_serialized(
                        message.get("tool_call_id") or "", 256
                    ),
                    "name": _bounded_serialized(message.get("name") or "", 256),
                    "content": _bounded(
                        _text_content(
                            message.get("content"),
                            MAX_TOOL_EVENT_CONTENT_CHARS,
                        ),
                        MAX_TOOL_EVENT_CONTENT_CHARS,
                    ),
                    "metadata": {},
                }
            )
        elif message.get("role") == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in reversed(calls[-MAX_TOOL_EVENTS:]):
                    inspected_calls += 1
                    if not isinstance(call, dict):
                        if inspected_calls >= MAX_TOOL_EVENTS:
                            break
                        continue
                    function = call.get("function")
                    function = function if isinstance(function, dict) else {}
                    name = function.get("name") or call.get("name") or ""
                    arguments = (
                        function.get("arguments") or call.get("arguments") or ""
                    )
                    event = {
                        "kind": "call",
                        "tool_event_id": _bounded_serialized(call.get("id") or "", 256),
                        "name": _bounded_serialized(name, 256),
                        "content": _bounded_serialized(
                            arguments, MAX_TOOL_EVENT_CONTENT_CHARS
                        ),
                        "metadata": {},
                    }
                    events.append(event)
                    if len(events) >= MAX_TOOL_EVENTS:
                        break
        if len(events) >= MAX_TOOL_EVENTS or inspected_calls >= MAX_TOOL_EVENTS:
            break
    events.reverse()
    return events


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def on_post_llm_call(**hermes_kwargs: Any) -> None:
    """Commit one finalized Hermes turn to ``POST /cognition/commit``."""
    session = _agent_session.get_session()
    if session is None:
        return None
    agent_id, client = session

    turn_id = _bounded_identifier(hermes_kwargs.get("turn_id"), 256)
    if not turn_id:
        log.warning("styx post_llm_call: turn_id missing; cognition commit skipped")
        return None

    session_id = _bounded_identifier(hermes_kwargs.get("session_id"), 256)
    history = hermes_kwargs.get("conversation_history")
    idempotency_key = _idempotency_key(session_id, turn_id)
    parent_host_key, snapshot_token = _agent_session.declare_act(
        session_id, idempotency_key
    )
    user_message = _bounded(
        _text_content(
            hermes_kwargs.get("user_message"), MAX_USER_MESSAGE_CHARS
        ),
        MAX_USER_MESSAGE_CHARS,
    )
    assistant_response = _bounded(
        _text_content(
            hermes_kwargs.get("assistant_response"),
            MAX_ASSISTANT_RESPONSE_CHARS,
        ),
        MAX_ASSISTANT_RESPONSE_CHARS,
    )
    _agent_session.remember_turn_key(
        session_id,
        idempotency_key,
        user_content=user_message,
        assistant_content=assistant_response,
    )

    history_payload = _bounded_history(history)
    tool_events = _bounded_tool_events(history)
    try:
        result = client.cognition_commit(
            agent_id,
            host_key=idempotency_key,
            parent_host_key=parent_host_key,
            session_id=session_id or None,
            snapshot_token=snapshot_token,
            status=("failed" if hermes_kwargs.get("success") is False else "completed"),
            user_message=user_message,
            assistant_response=assistant_response,
            conversation_history=history_payload,
            tool_events=tool_events,
            consequences=[],
            model=_bounded_identifier(hermes_kwargs.get("model"), 512) or None,
            platform=_bounded_identifier(hermes_kwargs.get("platform"), 64) or None,
            extra={
                **(
                    {"task_id": _bounded_identifier(
                        hermes_kwargs.get("task_id"), 256
                    )}
                    if _bounded_identifier(hermes_kwargs.get("task_id"), 256)
                    else {}
                ),
                "projection_scope": "finalized_channel_output",
            },
        )
        if result.get("committed") is True or result.get("duplicate") is True:
            _agent_session.mark_cognition_committed(idempotency_key)
    except Exception as exc:  # noqa: BLE001 — completed turn must stay successful
        if not _is_not_found(exc):
            log.warning("styx-core /cognition/commit failed: %s", exc)
            return None
        # Mixed-version deployment only. The provider's later sync_turn uses
        # the same host key, so the legacy affect + dialogue writes deduplicate.
        try:
            client.observe_affective_turn(
                agent_id,
                idempotency_key=idempotency_key,
                turn_id=turn_id,
                session_id=session_id or None,
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=history_payload,
                tool_events=[
                    {
                        "kind": event.get("kind"),
                        "tool_call_id": event.get("tool_event_id", ""),
                        "name": event.get("name", ""),
                        "content": event.get("content", ""),
                    }
                    for event in tool_events
                    if event.get("kind") in ("call", "result")
                ],
                task_id=_bounded_identifier(hermes_kwargs.get("task_id"), 256)
                or None,
                model=_bounded_identifier(hermes_kwargs.get("model"), 512) or None,
                platform=_bounded_identifier(hermes_kwargs.get("platform"), 64)
                or None,
            )
        except Exception as legacy_exc:  # noqa: BLE001 — fail-open
            log.warning("styx-core terminal fallback failed: %s", legacy_exc)
    return None
