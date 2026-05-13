"""Recall pipeline — embed query → search → filter → dedup → format.

Прямой port из ``openclaw-memorybox/src/recall/full.ts recallFull`` и
``recall/format.ts`` (упрощённого без token-budget — это волна 8).

В волне 7 покрывает только memories-path. Dialogues и chunks отброшены
(см. § 17.4 decisions). Hot-tier — волна 8.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..embedding import EmbeddingClient, EmbeddingError
from ..observability.logging import log_event
from .importance import query_hash
from .internal_dedup import internal_dedup
from .queries import AgentScopedQueries, MemoryHit
from .recall_config import DEFAULT_RECALL_CONFIG, FullRecallConfig
from .scoring import EmotionalBaseline as ScoringEmotionalBaseline

if TYPE_CHECKING:
    from styx.turn_state import RecallSnapshot

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecallResult:
    """Что recall возвращает caller'у — список hits + мета-сведения."""

    memories: list[MemoryHit]
    queried_count: int
    internal_duplicates_removed: int


def recall_full(
    *,
    queries: AgentScopedQueries,
    embed_client: EmbeddingClient,
    query: str,
    query_vector: list[float] | None = None,
    full_config: FullRecallConfig = DEFAULT_RECALL_CONFIG.full,
    session_id: str | None = None,
    record_events: bool = True,
    snapshot: "RecallSnapshot | None" = None,
) -> RecallResult:
    """Полный recall pipeline.

    1. Embed query → vector. Если caller передал ``query_vector``
       (волна 10: focus_tracker уже посчитал embed для drift detection),
       embed-вызов пропускается — переиспользуем готовый вектор.
    2. Search top-K (limit × 2 — запас под filter+dedup, как memorybox).
    3. Filter по min_score (если > 0).
    4. internal_dedup по similarity (требует include_embedding=True).
    5. Slice до memory_limit.
    6. Запись recall_events (UPSERT по UNIQUE).

    На EmbeddingError — пустой результат + лог. Caller получит
    зависимость на retry (нет — recall просто пропускается, LLM
    видит "ничего не найдено").
    """
    started = time.monotonic()
    if query_vector is not None:
        vec = query_vector
    else:
        try:
            vec = embed_client.embed(query)
        except EmbeddingError as exc:
            log.warning("recall_full: embed query упал: %s", exc)
            return RecallResult(memories=[], queried_count=0, internal_duplicates_removed=0)

    # Прочитать baseline для emotional_resonance фактора. Лениво: импорт
    # внутри функции потому что emotional/ — отдельный package, не должен
    # импортироваться при чистой работе с storage без эмоциональной части.
    from styx.emotional.baseline import read_baseline_for_scoring

    baseline_obj = read_baseline_for_scoring(queries.conn, queries.agent_id)
    scoring_baseline = (
        ScoringEmotionalBaseline(
            valence=baseline_obj.valence,
            arousal=baseline_obj.arousal,
            dominance=baseline_obj.dominance,
        )
        if baseline_obj is not None
        else None
    )

    # Волна 11: hot supplement. Items, недавно прошедшие через recall,
    # доплываются как extra candidates до filter+dedup+slice. БД-версия
    # побеждает на collision id (там composite score актуальный).
    from styx.engine import hot_tier

    hot_candidates = hot_tier.scan_candidates(
        queries.agent_id, vec, min_score=full_config.min_score, snapshot=snapshot
    )

    db_hits = queries.search_similar(
        query_vector=vec,
        query_text=query,
        limit=full_config.memory_limit * 2,
        full_config=full_config,
        emotional_baseline=scoring_baseline,
        include_embedding=True,
        snapshot=snapshot,
    )
    seen_ids = {h.id for h in db_hits}
    raw_hits = list(db_hits) + [h for h in hot_candidates if h.id not in seen_ids]
    queried_count = len(raw_hits)

    if full_config.min_score > 0:
        filtered = [h for h in raw_hits if h.score >= full_config.min_score]
    else:
        filtered = raw_hits

    dedup = internal_dedup(filtered, full_config.internal_dedup_similarity)
    sliced = dedup.kept[: full_config.memory_limit]

    if record_events and sliced:
        from dataclasses import replace as _replace

        qhash = query_hash(query)
        recorded: list[MemoryHit] = []
        for hit in sliced:
            rec_id = queries.record_recall_event(
                memory_id=hit.id,
                query_hash=qhash,
                match_score=hit.match_score,
                session_id=session_id,
            )
            recorded.append(_replace(hit, recall_event_id=rec_id))
        sliced = recorded

    # Обновить last_accessed_at для возвращённых memories — это сигнал
    # для lifecycle-sweep (волна 7b), без которого settled memories
    # никогда не уходили бы в dormant. Делаем после record_recall_event,
    # чтобы запись recall'а была первичной (она важнее для отладки чем
    # последний touch).
    if sliced:
        queries.update_last_accessed_at([h.id for h in sliced])

    # Волна 11: put-on-success. Возвращённые items копируются в hot
    # для следующих recall'ов (в пределах TTL).
    if sliced:
        hot_tier.put_many(queries.agent_id, sliced)

    log_event(
        log,
        "recall",
        agent_id=queries.agent_id,
        query_hash=query_hash(query),
        results_count=len(sliced),
        queried_count=queried_count,
        hot_candidates=len(hot_candidates),
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )
    return RecallResult(
        memories=sliced,
        queried_count=queried_count,
        internal_duplicates_removed=dedup.removed,
    )


def format_recall_text(result: RecallResult) -> str:
    """Простое текстовое форматирование recall для tool result.

    Без token-budget'а (волна 8). Каждый hit — score + content. Если
    пусто — короткий маркер.
    """
    if not result.memories:
        return "<no memories matched>"
    parts: list[str] = []
    for hit in result.memories:
        parts.append(
            f"[score={hit.score:.3f} role={hit.role}] {hit.content}"
        )
    return "\n\n".join(parts)
