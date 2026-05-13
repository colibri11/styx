"""Styx transport core — wire-log digest + cache key resolution.

Host-agnostic ядро транспорт-логики: чистая функция
``compute_prefix_digest`` (sha256-первых N сообщений), per-agent state
``configure(agent_id, prompt_cache_key, wire_log_head_messages)`` и
``styx_cache_key_override(agent_id, params)`` для resolution cache_key.

Hermes-наследники (``StyxOpenAITransport``, ``StyxCodexTransport``) живут
в styx-hermes и поверх этого ядра реализуют ``ChatCompletionsTransport`` /
``ResponsesApiTransport`` ABC. См.
``packages/styx-hermes/src/styx_hermes/engine/transport.py``.

Per-agent state: словарь ``agent_id → TransportState``. Один core daemon
обслуживает несколько агентов параллельно.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)
wire_log = logging.getLogger("styx.transport.wire")


@dataclass
class TransportState:
    prompt_cache_key: str | None = None
    wire_log_head_messages: int = 3


_STATES: dict[str, TransportState] = {}
_LOCK = threading.Lock()


def configure(
    agent_id: str,
    *,
    prompt_cache_key: str | None = None,
    wire_log_head_messages: int | None = None,
) -> None:
    """Привязка контекста транспортов к агенту.

    Двойной configure заменяет state. ``prompt_cache_key`` / ``wire_log
    _head_messages`` — None означает «не override'ить» (оставить
    значение по умолчанию или предыдущее).
    """
    with _LOCK:
        state = _STATES.get(agent_id)
        if state is None:
            state = TransportState()
            _STATES[agent_id] = state
        if prompt_cache_key is not None:
            state.prompt_cache_key = prompt_cache_key
        if wire_log_head_messages is not None:
            state.wire_log_head_messages = wire_log_head_messages


def reset(agent_id: str) -> None:
    """Сброс transport state одного агента."""
    with _LOCK:
        _STATES.pop(agent_id, None)


def reset_all() -> None:
    """Сбросить transport state всех агентов."""
    with _LOCK:
        _STATES.clear()


def _reset_for_test() -> None:
    """Алиас на ``reset_all`` — backwards-compat для тестов."""
    reset_all()


def styx_cache_key_override(agent_id: str, params: dict[str, Any]) -> str | None:
    """Per-call params → per-agent override → agent_id fallback.

    Возвращает None если ни один источник не задан — тогда транспорт
    оставляет поведение Hermes по умолчанию.
    """
    per_call = params.get("prompt_cache_key")
    if per_call:
        return str(per_call)
    state = _STATES.get(agent_id)
    if state is not None and state.prompt_cache_key:
        return state.prompt_cache_key
    if agent_id:
        return agent_id
    return None


# Backwards-compatible alias used by Hermes-side transport classes.
_styx_cache_key_override = styx_cache_key_override


def log_prefix_slice(agent_id: str, api_kwargs: dict[str, Any]) -> None:
    """sha256-digest первых N элементов payload (messages или input)."""
    state = _STATES.get(agent_id)
    head_n = state.wire_log_head_messages if state is not None else 3
    if head_n <= 0:
        return

    payload_source = api_kwargs.get("messages") or api_kwargs.get("input")
    if not payload_source:
        return

    head = payload_source[:head_n]
    try:
        head_bytes = json.dumps(
            head,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        wire_log.debug("wire log skipped: %s", exc)
        return

    digest = hashlib.sha256(head_bytes).hexdigest()[:16]
    # #10: slice_hex — plaintext содержимого диалога.
    # На INFO уровне пишем только digest + размер (безопасно для syslog/Sentry).
    # slice_hex доступен только при STYX_WIRE_LOG_RAW=1 (отладка) или на DEBUG.
    if os.environ.get("STYX_WIRE_LOG_RAW") == "1":
        wire_log.info(
            "prefix_slice agent=%s digest=%s payload_items=%d head_items=%d slice_hex=%s",
            agent_id,
            digest,
            len(payload_source),
            len(head),
            head_bytes.hex(),
        )
    else:
        wire_log.info(
            "prefix_slice agent=%s digest=%s payload_items=%d head_items=%d",
            agent_id,
            digest,
            len(payload_source),
            len(head),
        )
        wire_log.debug(
            "prefix_slice agent=%s slice_hex=%s",
            agent_id,
            head_bytes.hex(),
        )


# Backwards-compatible alias.
_log_prefix_slice = log_prefix_slice


def compute_prefix_digest(
    payload: list[dict[str, Any]], head_count: int = 3
) -> str:
    """Helper для тестов и продакшен-проверки — тот же digest что в wire-log."""
    head = payload[:head_count]
    blob = json.dumps(
        head,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
