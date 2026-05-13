"""Bridge между MemoryCore и StyxComposer для salient memories.

Per-agent handle: словарь ``agent_id → SalientHandle``. Один core daemon
обслуживает несколько агентов параллельно.

``StyxMemoryCore.initialize()`` зовёт ``configure(agent_id, ...)`` сразу
после построения ``queries`` / ``embed_client`` / ``recall_config``.
``StyxComposer.compress()`` достаёт handle через ``get_handle(agent_id)``
и передаёт в ``build_salient_block``. ``shutdown()`` зовёт
``reset(agent_id)``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from styx.embedding import EmbeddingClient
from styx.storage.queries import AgentScopedQueries
from styx.storage.recall_config import RecallConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SalientHandle:
    """Снимок ссылок, нужных ``build_salient_block`` для recall'а."""

    queries: AgentScopedQueries
    embed_client: EmbeddingClient
    recall_config: RecallConfig
    timeout_s: float
    min_query_len: int
    agent_id: str = ""


_HANDLES: dict[str, SalientHandle] = {}
_LOCK = threading.Lock()


def configure(
    agent_id: str,
    *,
    queries: AgentScopedQueries,
    embed_client: EmbeddingClient,
    recall_config: RecallConfig,
    timeout_s: float = 1.0,
    min_query_len: int = 20,
) -> None:
    """Привязать bridge к provider'скому стейту для ``agent_id``.

    Двойной configure заменяет handle — ContextEngine увидит свежие
    ссылки на следующем compress'е.
    """
    with _LOCK:
        _HANDLES[agent_id] = SalientHandle(
            queries=queries,
            embed_client=embed_client,
            recall_config=recall_config,
            timeout_s=timeout_s,
            min_query_len=min_query_len,
            agent_id=agent_id,
        )


def get_handle(agent_id: str) -> SalientHandle | None:
    return _HANDLES.get(agent_id)


def reset(agent_id: str) -> None:
    """Сброс handle одного агента. Вызывается из ``shutdown()`` и тестовых fixture'ов."""
    with _LOCK:
        _HANDLES.pop(agent_id, None)


def reset_all() -> None:
    """Сбросить salient bridge для всех агентов."""
    with _LOCK:
        _HANDLES.clear()
