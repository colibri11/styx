"""Bridge между MemoryCore и StyxComposer для relevance-aware eviction.

Волна 12. По образцу ``salient_bridge``: ``StyxComposer.compress()`` не
имеет прямого доступа к ``queries`` (БД) — handle выставляется
provider'ом через per-agent dict, compress читает через
``get_handle(agent_id)``.

``StyxMemoryCore.initialize()`` configure'ит после
``salient_bridge.configure(...)`` и ``focus_tracker.configure(...)`` если
``eviction_relevance_enabled``. ``shutdown()`` зовёт ``reset(agent_id)``.

Per-agent state: словарь ``agent_id → EvictionRelevanceHandle``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from styx.storage.queries import AgentScopedQueries

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvictionRelevanceHandle:
    """Снимок ссылок и параметров для ``apply_relevance_eviction``."""

    queries: AgentScopedQueries
    keep_k: int
    threshold: float
    agent_id: str = ""


_HANDLES: dict[str, EvictionRelevanceHandle] = {}
_LOCK = threading.Lock()


def configure(
    agent_id: str,
    *,
    queries: AgentScopedQueries,
    keep_k: int,
    threshold: float,
) -> None:
    """Привязать bridge к provider'скому стейту для ``agent_id``.

    Двойной configure заменяет handle. ``keep_k < 0`` или
    ``threshold ∉ [0, 1]`` — ValueError, чтобы поймать конфигурационные
    ошибки на старте, не в hot-pat'е.
    """
    if keep_k < 0:
        raise ValueError(f"keep_k должен быть >= 0, получено {keep_k}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold должен быть в [0, 1], получено {threshold}"
        )
    with _LOCK:
        _HANDLES[agent_id] = EvictionRelevanceHandle(
            queries=queries,
            keep_k=keep_k,
            threshold=threshold,
            agent_id=agent_id,
        )


def get_handle(agent_id: str) -> EvictionRelevanceHandle | None:
    return _HANDLES.get(agent_id)


def reset(agent_id: str) -> None:
    """Сброс handle одного агента. Вызывается из ``shutdown()`` и тестовых fixture'ов."""
    with _LOCK:
        _HANDLES.pop(agent_id, None)


def reset_all() -> None:
    """Сбросить eviction-relevance bridge для всех агентов."""
    with _LOCK:
        _HANDLES.clear()
