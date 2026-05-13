"""Hebbian relation decay sweep — port memorybox `relation-decay.ts`.

Periodic-task в worker'е. Раз в час: для рёбер ``relation='co_retrieved'``
с ``weight > 1.0`` и ``last_reinforced > idle_threshold_days`` уменьшает
weight на ``decay_rate``, floor 1.0.

Cold-link sediments на baseline 1.0, не исчезает (чтобы оставаться
отличимым от никогда-не-reinforced рёбер). Fresh links (auto-link
'related_to' с DEFAULT weight=1.0) — не трогаем (фильтр на
``relation='co_retrieved'`` + ``weight > 1.0``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg

log = logging.getLogger(__name__)


DEFAULT_DECAY_RATE = 0.05
DEFAULT_IDLE_THRESHOLD_DAYS = 14


@dataclass(frozen=True)
class RelationDecayResult:
    decayed: int
    decay_rate: float
    idle_threshold_days: int


def run_relation_decay(
    conn: psycopg.Connection,
    *,
    decay_rate: float = DEFAULT_DECAY_RATE,
    idle_threshold_days: int = DEFAULT_IDLE_THRESHOLD_DAYS,
) -> RelationDecayResult:
    """UPDATE co_retrieved рёбра у которых last_reinforced устарело.

    Не делает commit — caller (periodic-task wrapper в `LlmWorker`)
    управляет транзакцией. По соглашению periodic-task в Styx:
    fn получает свежий conn (открытый перед вызовом, закрывается
    после), коммит ответственность wrapper'а.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE relations SET "
            "  weight = GREATEST(1.0, weight - %s) "
            " WHERE relation = 'co_retrieved' "
            "   AND weight > 1.0 "
            "   AND (metadata->>'last_reinforced')::timestamptz "
            "       < now() - make_interval(days => %s)",
            (decay_rate, idle_threshold_days),
        )
        decayed = cur.rowcount or 0

    log.info(
        "relation_decay: decayed=%d decay_rate=%.3f idle_days=%d",
        decayed, decay_rate, idle_threshold_days,
    )
    return RelationDecayResult(
        decayed=decayed,
        decay_rate=decay_rate,
        idle_threshold_days=idle_threshold_days,
    )
