"""Hebbian co-retrieval reinforcement (волна 21).

При recall'е nthnt'е N≥2 memories — на всех C(N, 2) парах укрепляется
ребро ``relation='co_retrieved'`` в ``relations``. «Нейроны,
активирующиеся вместе, связываются» — формируется weighted graph
ко-активаций, независимый от семантического сходства (которое уже
покрыто `related_to` рёбрами от auto-link волны 18).

Алгоритм (порт memorybox `tools/memory.ts:reinforceCoRetrieval`):

- Try UPSERT через UNIQUE constraint (от волны 18, миграция 0004):
  существующее ребро → ``weight = LEAST(weight + bump, weight_max)``,
  ``metadata.last_reinforced = now()``. Несуществующее → INSERT с
  initial_weight (1.1, не 1.0 — чтобы decay'нувшие cold links
  отличались от never-reinforced baseline).
- Caps: `weight ∈ [1.0, 2.0]`. Bump 0.1 → насыщение за 10 совместных
  recall'ов.
- Decay (отдельная periodic-task `workers/sweep/relation_decay.py`):
  раз в час `weight = GREATEST(1.0, weight - 0.05)` для рёбер с
  ``last_reinforced`` старше 14 дней.

Sync, не fire-and-forget (D2): один UPSERT per pair (~5-10 ms на
N=10), удерживается в той же транзакции что recall_event INSERT.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from styx.storage.queries import AgentScopedQueries

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HebbianConfig:
    """Параметры Hebbian reinforcement. Дефолты — порт memorybox."""

    enabled: bool = True
    weight_bump: float = 0.1
    initial_weight: float = 1.1
    weight_max: float = 2.0


def reinforce_co_retrieval(
    queries: "AgentScopedQueries",
    *,
    memory_ids: list[uuid.UUID],
    config: HebbianConfig,
    agent_id: str,
) -> int:
    """Bumps `co_retrieved` weight для всех C(N, 2) пар memory_ids.

    Возвращает кол-во обработанных пар (0 если disabled или N < 2).

    Pairs формируются как (i, j) for i < j — без направления
    (co_retrieved семантически ненаправленный). Memorybox делает то же
    с (source=results[i], target=results[j]) — порядок по indexу top-K.
    Сохраняем тот же порядок для тестируемости и совместимости traversal.

    Не делает commit — caller (StyxMemoryCore.handle_tool_call)
    управляет транзакцией.
    """
    if not config.enabled:
        return 0
    if len(memory_ids) < 2:
        return 0

    pairs_processed = 0
    for i in range(len(memory_ids)):
        for j in range(i + 1, len(memory_ids)):
            queries.upsert_co_retrieved_pair(
                source_id=memory_ids[i],
                target_id=memory_ids[j],
                initial_weight=config.initial_weight,
                weight_bump=config.weight_bump,
                weight_max=config.weight_max,
            )
            pairs_processed += 1

    log.debug(
        "hebbian: agent_id=%s pairs=%d k=%d",
        agent_id, pairs_processed, len(memory_ids),
    )
    return pairs_processed
