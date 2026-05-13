"""Pre-LLM focus inject framework — multi-channel, ephemeral.

Волна 15. Открывает второй канал инжекта Styx'а в active context —
Hermes pre_llm_call hook (`run_agent.py:10218`). Hook срабатывает один
раз в начале turn'а перед tool-loop'ом; собранный из каналов context
аппендится Hermes'ом к current turn's user message через
`"\n\n".join`. Inject ephemeral — не пишется в session DB.

StyxComposer.compress() (волны 9-10) остаётся главным каналом для
**стабильного контекста** между turn'ами; pre_llm_inject — для
**острого ephemeral сигнала**, который не оправдывает места в suffix'е.

Каналы — pure functions `(handle, hermes_kwargs) → str | None`. Никакого
внутреннего state'а; framework state (queries handle, config flags)
передаётся через ``ChannelHandle``. Per-agent state — словарь
``agent_id → AgentChannels``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from styx.storage.queries import AgentScopedQueries

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelHandle:
    """Снимок ссылок и config'а, общий для всех каналов pre_llm_inject."""

    queries: AgentScopedQueries
    peer_vad_enabled: bool = True
    peer_vad_min_norm: float = 0.2
    peer_vad_ttl_s: float = 60.0


ChannelFn = Callable[[ChannelHandle, dict[str, Any]], "str | None"]


@dataclass
class AgentChannels:
    """Per-agent framework state."""

    handle: ChannelHandle
    channels: list[tuple[str, ChannelFn]] = field(default_factory=list)
    enabled: bool = True


_STATES: dict[str, AgentChannels] = {}
_LOCK = threading.Lock()


def configure(
    agent_id: str,
    *,
    handle: ChannelHandle,
    channels: list[tuple[str, ChannelFn]],
    enabled: bool = True,
) -> None:
    """Сконфигурировать framework для ``agent_id``. Двойной configure заменяет state."""
    with _LOCK:
        _STATES[agent_id] = AgentChannels(
            handle=handle,
            channels=list(channels),
            enabled=enabled,
        )


def get_handle(agent_id: str) -> ChannelHandle | None:
    state = _STATES.get(agent_id)
    return state.handle if state is not None else None


def is_enabled(agent_id: str) -> bool:
    state = _STATES.get(agent_id)
    return bool(state and state.enabled)


def reset(agent_id: str) -> None:
    """Полный сброс framework state'а одного агента. Вызывается из shutdown() и тестов."""
    with _LOCK:
        _STATES.pop(agent_id, None)


def reset_all() -> None:
    """Сбросить framework для всех агентов."""
    with _LOCK:
        _STATES.clear()


def on_pre_llm_call(agent_id: str, **hermes_kwargs: Any) -> dict[str, str] | None:
    """Hook callback для Hermes ``pre_llm_call``.

    Hermes invoke'ит этот callback и аппендит ``return["context"]`` к
    user message. Если все каналы вернули None или framework выключен —
    возвращаем None, Hermes не аппендит ничего.

    Fail-open: каждый канал в отдельном try/except. Падающий канал
    логируется WARNING'ом и пропускается.
    """
    state = _STATES.get(agent_id)
    if state is None or not state.enabled:
        return None

    parts: list[str] = []
    for name, channel in state.channels:
        try:
            text = channel(state.handle, hermes_kwargs)
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.warning(
                "pre_llm_call channel %s failed for agent_id=%s: %s",
                name, agent_id, exc,
            )
            continue
        if text:
            parts.append(text)

    if not parts:
        return None
    return {"context": "\n\n".join(parts)}
