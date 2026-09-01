"""Recall pipeline — embed query → search → filter → dedup → format.

Прямой port из ``openclaw-memorybox/src/recall/full.ts recallFull`` и
``recall/format.ts`` (упрощённого без token-budget — это волна 8).

В волне 7 покрывает только memories-path. Dialogues и chunks отброшены
(см. § 17.4 decisions). Hot-tier — волна 8.
"""

from __future__ import annotations

import json
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


def _current_scoring_affect(
    queries: AgentScopedQueries,
) -> ScoringEmotionalBaseline | None:
    """Актуальный causal residue для emotional recall.

    Legacy scoring API исторически называет аргумент ``baseline``. Для
    выбора памяти нужен не медленный средний фон сам по себе, а состояние,
    в котором начинается текущий когнитивный акт. Уверенность нового
    transition определяет, насколько instant отклоняет fallback baseline.
    Legacy state без confidence не выдаём за достоверное evidence.
    """
    from styx.emotional.baseline import read_baseline_for_scoring
    from styx.emotional.state import read_last_state_record

    baseline = read_baseline_for_scoring(queries.conn, queries.agent_id)
    try:
        state = read_last_state_record(queries.conn, queries.agent_id)
    except Exception as exc:  # noqa: BLE001 - recall must remain fail-open
        log.warning("current affect unavailable for recall: %s", exc)
        state = None
    if state is None or state.confidence is None:
        if baseline is None:
            return None
        return ScoringEmotionalBaseline(
            valence=baseline.valence,
            arousal=baseline.arousal,
            dominance=baseline.dominance,
        )

    confidence = max(0.0, min(1.0, state.confidence))
    base_v = baseline.valence if baseline is not None else 0.0
    base_a = baseline.arousal if baseline is not None else 0.0
    base_d = baseline.dominance if baseline is not None else 0.0
    return ScoringEmotionalBaseline(
        valence=base_v + confidence * (state.vector.valence - base_v),
        arousal=base_a + confidence * (state.vector.arousal - base_a),
        dominance=base_d + confidence * (state.vector.dominance - base_d),
    )


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

    # Causal instant residue меняет выбор памяти до генерации. Медленный
    # baseline остаётся fallback и опорой при низкой confidence.
    scoring_baseline = _current_scoring_affect(queries)

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
        coordinates = ""
        provenance = hit.affective_provenance
        if provenance is not None:
            context_at = provenance.context_at
            if hasattr(context_at, "isoformat"):
                context_at = context_at.isoformat()
            evidence = {
                "state_id": provenance.state_id,
                "context_at": context_at,
                "vad": {
                    "valence": round(provenance.vad.valence, 6),
                    "arousal": round(provenance.vad.arousal, 6),
                    "dominance": round(provenance.vad.dominance, 6),
                },
                "confidence": (
                    round(provenance.confidence, 6)
                    if provenance.confidence is not None else None
                ),
                "causal_refs": [
                    {
                        "evidence_id": ref.evidence_id,
                        "source_ref": ref.source_ref,
                        "cause_class": ref.cause_class,
                        "cause_subject": ref.cause_subject,
                        "status_at_capture": ref.status_at_capture,
                        "intensity": ref.intensity,
                        "confidence": ref.confidence,
                        "observed_at": (
                            ref.observed_at.isoformat()
                            if hasattr(ref.observed_at, "isoformat") else None
                        ),
                        "lease_expires_at": (
                            ref.lease_expires_at.isoformat()
                            if hasattr(ref.lease_expires_at, "isoformat") else None
                        ),
                    }
                    for ref in provenance.causal_refs
                ],
            }
            coordinates = " affect_evidence=" + json.dumps(
                evidence,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        parts.append(
            f"[score={hit.score:.3f} role={hit.role}{coordinates}] "
            f"{hit.content}"
        )
    return "\n\n".join(parts)
