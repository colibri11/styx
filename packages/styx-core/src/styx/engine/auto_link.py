"""Auto-link при INSERT — port memorybox `auto-linking.ts` (волна 18).

После каждого `store` / `supersede`-write'а subjective memory (или
captured dialogue ряда в `sync_turn`) — находит ближайших соседей по
embedding'у и пишет рёбра ``relation='related_to'`` в ``relations``.

Зовётся **только** на ветках STORE/SUPERSEDE gatekeeper'а (на MERGE
новый ряд удалён, на SKIP — тоже). Из `sync_turn` зовётся для каждого
captured user/assistant ряда после embed-after-commit.

Cross-agent (D2 в wave-doc'е): SELECT соседей идёт без `agent_id`-
фильтра. Несколько агентов живут в одном PG; общий пул знаний строится
auto-link'ом — единственное cross-agent место в Styx.

Идемпотентность через UNIQUE constraint
``(source_type, source_id, target_type, target_id, relation)`` в
миграции 0004 + ON CONFLICT DO NOTHING в INSERT'е.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from styx.observability.logging import log_event

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoLinkConfig:
    """Параметры auto-link'а. Дефолты — из memorybox `auto-linking.ts`."""

    enabled: bool = True
    max_distance: float = 0.25  # cosine distance, similarity ≥ 0.75
    max_links: int = 3


@dataclass(frozen=True)
class AutoLinkNeighbor:
    """Сосед-target для auto-link ребра."""

    id: uuid.UUID
    cosine_distance: float


def auto_link_after_store(
    queries: "AgentScopedQueries",  # noqa: F821 — forward ref
    *,
    memory_id: uuid.UUID,
    embedding: list[float],
    config: AutoLinkConfig,
    agent_id: str,
    source: str,
) -> int:
    """Auto-link только что INSERT'нутого ряда. Возвращает links_created.

    ``source`` — где зовётся ('dialogue_batch_consolidation' /
    'memory_store' / 'sync_turn'); идёт в structured log event.
    """
    if not config.enabled:
        return 0

    neighbors = queries.find_auto_link_candidates(
        embedding,
        max_distance=config.max_distance,
        max_links=config.max_links,
        exclude_id=memory_id,
    )
    if not neighbors:
        log_event(
            log, "auto_link",
            agent_id=agent_id, memory_id=str(memory_id),
            links_created=0, source=source,
        )
        return 0

    queries.insert_auto_link_relations(memory_id, neighbors)

    log_event(
        log, "auto_link",
        agent_id=agent_id, memory_id=str(memory_id),
        links_created=len(neighbors), source=source,
    )
    return len(neighbors)
