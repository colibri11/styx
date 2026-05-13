"""LLM dialogue batch consolidation handler — port memorybox 5b/26b.

Handler читает окно user/assistant пар из ``memories``, отдаёт его
qwen3:4b-local'у с промптом, который требует:
- 2-5 предложений summary в первом лице (от лица агента);
- архивные hints (короткие срезы окна для будущей навигации);
- интегральный VAD peer-части окна (волна 26b — batch sentiment
  piggyback).

INSERT новых memories с ``kind_src='dialogue_batch_consolidation'``
+ append emotional_state с ``source='sentiment:batch'`` (если VAD
вернулся) — атомарно в одной транзакции. State в
``consolidation_state`` обновляется тем же commit'ом.

Контракт фиксирован в ``.design/waves/14-dialogue-consolidation.md``
и буквально port'нут из memorybox
``llm-worker/handlers/dialogue-batch-consolidation.ts``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass
from typing import Any

from styx.emotional.sentiment_batch import (
    K_BATCH,
    SentimentBatchMetrics,
    average_vads,
    scale_batch_vad_delta,
)
from styx.embedding import EmbeddingError
from styx.emotional.state import (
    EMOTIONAL_AXIS_MAX,
    EMOTIONAL_AXIS_MIN,
    EmotionalVector,
    append_emotional_state,
)
from styx.engine.auto_link import AutoLinkConfig, auto_link_after_store
from styx.engine.selective_gatekeeper import (
    Action,
    GatekeeperConfig,
    decide,
)
from styx.engine.store_routing import (
    StoreRoutingConfig,
    route_long_content,
)
from styx.llm import OllamaTerminalError
from styx.observability.logging import log_event
from styx.storage.queries import (
    AgentScopedQueries,
    set_batch_state,
)
from styx.workers.runtime import Handler, HandlerContext, HandlerResult, LlmTask

log = logging.getLogger(__name__)


DIALOGUE_BATCH_TASK_TYPE = "dialogue_batch_consolidation"

# ── Constants (port memorybox) ────────────────────────────────────────

MAX_CONTENT_CHARS = 2400  # CHECK constraint memories_content_length_check
L_BATCH_CHARS = 85_000
OVERLAP_CHARS = 17_000
OVERLAP_MESSAGES = 30
FEEDBACK_MEMORIES_LIMIT = 30
FEEDBACK_WINDOW_HOURS = 24


# ── System prompt (port из memorybox dialogue-batch-consolidation.ts) ──

SYSTEM_PROMPT = """Ты — подкорка агента, которая ведёт короткие субъектные заметки по итогам разговоров.
На вход — окно диалога (реплики человека и агента) и список того, что агент уже недавно запомнил.

Твоя задача: сформулировать 2–5 предложений от лица агента, о чём в этом окне шла речь
и к чему пришли. Заметка должна:
— быть написана в первом лице («я», «мы»);
— сохранять ключевой смысл окна, а не пересказывать слово в слово;
— не дублировать то, что уже есть в "memory_context";
— при важном переходе/решении — зафиксировать его явно.

Если в окне ничего достойного запоминания не произошло (рутинный обмен, служебные реплики,
всё уже и так в memory_context) — верни skip=true с причиной.

Дополнительно: оцени интегральный эмоциональный тон РЕПЛИК ЧЕЛОВЕКА (role=user)
в окне по трём осям VAD в [-1, +1]:
- valence: негативный (-1) ... позитивный (+1)
- arousal: спокойный (-1) ... возбуждённый (+1)
- dominance: подавленный (-1) ... доминирующий (+1)

Реплики агента (role=assistant) в этой оценке НЕ учитывай — это его выход,
не стимул. Если в окне нет выраженного эмоционального тона (сухой технический обмен,
только уточнения без эмоции) — верни "vad": null. Не галлюцинируй.

Верни строгий JSON по схеме:
{
  "skip": <true | false>,
  "skip_reason": <строка | null>,
  "summary": <строка 2–5 предложений на языке диалога | null>,
  "archive_hints": [
    {
      "snippet": <короткий срез ~150 символов, характеризующий ключевое место окна>
    }
  ] | null,
  "vad": {"valence": number, "arousal": number, "dominance": number} | null
}

Когда skip=true:
  — skip_reason обязателен;
  — summary и archive_hints должны быть null;
  — vad можно тоже null.

Когда skip=false:
  — summary обязателен (2–5 предложений, до 1800 символов);
  — archive_hints 0–2 элемента;
  — vad заполняется тремя числами в [-1, +1], либо null если peer-часть окна эмоционально нейтральна.

Отвечай только JSON. Без markdown, без заголовков, без комментариев."""


# ── Output schema (validator-based, без Pydantic) ────────────────────


@dataclass(frozen=True)
class ArchiveHint:
    snippet: str


@dataclass(frozen=True)
class BatchParsed:
    """Discriminated union по полю ``skip``."""

    skip: bool
    skip_reason: str | None
    summary: str | None
    archive_hints: list[ArchiveHint] | None
    vad: EmotionalVector | None  # волна 26b


def _validate_batch_response(raw: Any) -> BatchParsed:
    """Discriminated union по полю ``skip``. Любая ошибка → ValueError."""
    if not isinstance(raw, dict):
        raise ValueError(f"ожидаем object, получили {type(raw).__name__}")

    skip = raw.get("skip")
    if not isinstance(skip, bool):
        raise ValueError(f"skip должен быть bool, получено {skip!r}")

    skip_reason = raw.get("skip_reason")
    summary = raw.get("summary")
    archive_hints_raw = raw.get("archive_hints")
    vad_raw = raw.get("vad")

    vad = _validate_optional_vad(vad_raw)

    if skip:
        if not isinstance(skip_reason, str) or not (1 <= len(skip_reason) <= 300):
            raise ValueError(
                "skip=true: skip_reason обязателен, строка 1..300"
            )
        if summary is not None:
            raise ValueError("skip=true: summary должен быть null")
        if archive_hints_raw is not None:
            raise ValueError("skip=true: archive_hints должен быть null")
        return BatchParsed(
            skip=True, skip_reason=skip_reason,
            summary=None, archive_hints=None, vad=vad,
        )

    # skip=false
    if skip_reason is not None:
        raise ValueError("skip=false: skip_reason должен быть null")
    if not isinstance(summary, str) or not (1 <= len(summary) <= 4000):
        raise ValueError(
            "skip=false: summary обязателен, строка 1..4000"
        )
    hints = _validate_hints(archive_hints_raw)
    return BatchParsed(
        skip=False, skip_reason=None,
        summary=summary, archive_hints=hints, vad=vad,
    )


def _validate_hints(raw: Any) -> list[ArchiveHint]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("archive_hints должен быть list")
    # Memorybox truncate'ит до 2 на стороне клиента — мы тоже.
    out: list[ArchiveHint] = []
    for i, item in enumerate(raw[:10]):  # paranoia limit
        if not isinstance(item, dict):
            raise ValueError(f"archive_hints[{i}] должен быть object")
        snippet = item.get("snippet")
        if not isinstance(snippet, str) or not (1 <= len(snippet) <= 500):
            raise ValueError(
                f"archive_hints[{i}].snippet — строка 1..500"
            )
        out.append(ArchiveHint(snippet=snippet))
    return out[:2]  # max 2


def _validate_optional_vad(raw: Any) -> EmotionalVector | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"vad должен быть object или null, получено {type(raw).__name__}")

    def axis(name: str) -> float:
        v = raw.get(name)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"vad.{name} должен быть finite number, получено {v!r}")
        f = float(v)
        if not EMOTIONAL_AXIS_MIN <= f <= EMOTIONAL_AXIS_MAX:
            raise ValueError(
                f"vad.{name}={f} вне [{EMOTIONAL_AXIS_MIN}, {EMOTIONAL_AXIS_MAX}]"
            )
        return f

    return EmotionalVector(
        valence=axis("valence"),
        arousal=axis("arousal"),
        dominance=axis("dominance"),
    )


# ── Payload validation ────────────────────────────────────────────────


@dataclass(frozen=True)
class BatchPayload:
    agent_id: str
    window_from: _dt.datetime | None  # None = с самого начала
    window_to: _dt.datetime
    with_overlap: bool


def _parse_payload(raw: dict) -> BatchPayload:
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("payload.agent_id обязателен")
    window_from_raw = raw.get("window_from")
    window_to_raw = raw.get("window_to")
    with_overlap = bool(raw.get("with_overlap", False))

    if not isinstance(window_to_raw, str) or not window_to_raw:
        raise ValueError("payload.window_to обязателен (ISO timestamp)")
    window_to = _dt.datetime.fromisoformat(window_to_raw)
    window_from = (
        _dt.datetime.fromisoformat(window_from_raw)
        if isinstance(window_from_raw, str) and window_from_raw
        else None
    )
    return BatchPayload(
        agent_id=agent_id,
        window_from=window_from,
        window_to=window_to,
        with_overlap=with_overlap,
    )


# ── Chunker ───────────────────────────────────────────────────────────


def chunk_window_text(text: str) -> list[str]:
    """Coarse char-based chunker с фиксированным overlap'ом.

    Если text ≤ L_BATCH_CHARS — один chunk. Иначе несколько с
    overlap'ом OVERLAP_CHARS между ними.
    """
    if len(text) <= L_BATCH_CHARS:
        return [text]
    step = L_BATCH_CHARS - OVERLAP_CHARS
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + L_BATCH_CHARS, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


# ── Formatting ────────────────────────────────────────────────────────


def format_window(rows: list[dict]) -> str:
    return "\n".join(f"[{r['role']}]: {r['content']}" for r in rows)


def format_memory_context(rows: list[dict]) -> str:
    if not rows:
        return "(пусто)"
    return "\n".join(f"- {r['kind_src']}: {r['content']}" for r in rows)


def build_user_prompt(window_text: str, memory_context: str) -> str:
    return f"window:\n---\n{window_text}\n---\n\nmemory_context:\n{memory_context}"


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_archive_ref(
    rows: list[dict],
    chunk_text: str,
    hints: list[ArchiveHint],
    chunk_index: int | None,
) -> dict:
    first = rows[0]["created_at"]
    last = rows[-1]["created_at"]
    range_str = f"{first.isoformat()}..{last.isoformat()}"
    if chunk_index is not None:
        range_str = f"{range_str}#{chunk_index}"
    snippet = (
        hints[0].snippet[:500] if hints else chunk_text[:300]
    )
    return {
        "kind": "dialogue_message",
        "id": None,
        "locator": f"dialogue_messages[{range_str}]",
        "snippet": snippet,
    }


# ── Internal data ─────────────────────────────────────────────────────


@dataclass
class ChunkOutcome:
    skip: bool
    summary: str | None = None
    archive_ref: dict | None = None
    skip_reason: str | None = None
    vad: EmotionalVector | None = None


# ── Handler factory ───────────────────────────────────────────────────


def create_dialogue_batch_handler(
    *,
    batch_sentiment_enabled: bool = True,
    batch_sentiment_metrics: SentimentBatchMetrics | None = None,
    gatekeeper_config: GatekeeperConfig | None = None,
    auto_link_config: AutoLinkConfig | None = None,
    store_routing_config: StoreRoutingConfig | None = None,
) -> Handler:
    """Factory для batch consolidation handler.

    ``batch_sentiment_enabled=False`` отключает только apply
    (memory всё равно создаётся, output VAD сохраняется).

    ``batch_sentiment_metrics`` — опционально для /healthz; если None,
    создаём свежий instance внутри handler'а (метрики не публикуются).

    ``gatekeeper_config`` (волна 17) — параметры selective gatekeeper'а.
    None → используем дефолтный config (enabled=True, memorybox-пороги).
    Gatekeeper срабатывает только если ``ctx.embedder`` доступен;
    иначе fallback на legacy embedding=NULL.

    ``auto_link_config`` (волна 18) — параметры auto-link'а. Зовётся
    только на STORE/SUPERSEDE ветках gatekeeper'а; на MERGE/SKIP не
    зовётся (новый ряд удалён или поглощён existing'ом).

    ``store_routing_config`` (волна 19) — параметры store-routing'а.
    Если LLM-summary > limit — content разделяется на chunks
    (documents+chunks); tail-memory с archive_ref сохраняет указатель
    на оригинал плюс metadata о dialogue range. None → дефолт
    (enabled=True, port memorybox).
    """
    metrics = batch_sentiment_metrics or SentimentBatchMetrics()
    gk_config = gatekeeper_config or GatekeeperConfig()
    al_config = auto_link_config or AutoLinkConfig()
    routing = store_routing_config or StoreRoutingConfig()

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        payload = _parse_payload(task.payload)
        agent_id = payload.agent_id
        queries = AgentScopedQueries(ctx.conn, agent_id)

        rows = queries.select_dialogue_window(
            window_from=payload.window_from
            or _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc),
            window_to=payload.window_to,
            with_overlap_messages=OVERLAP_MESSAGES if payload.with_overlap else 0,
        )
        if not rows:
            # System skip — окно пустое, не дёргали LLM. Продвигаем
            # state до window_to чтобы следующий tick не повторял.
            set_batch_state(
                ctx.conn, agent_id,
                {
                    "last_batch_at": _now_iso(),
                    "last_message_created_at": payload.window_to.isoformat(),
                    "last_message_id": None,
                    "last_window_end_at": payload.window_to.isoformat(),
                },
            )
            return HandlerResult(result={"skipped": "empty_window"})

        feedback = queries.select_recent_memories_for_consolidation(
            hours=FEEDBACK_WINDOW_HOURS, limit=FEEDBACK_MEMORIES_LIMIT,
        )

        window_text = format_window(rows)
        memory_context = format_memory_context(feedback)
        chunks = chunk_window_text(window_text)

        metrics.record_call()

        # Подготовить все chunk results ДО открытия транзакции — LLM
        # вызовы занимают секунды, не держим pg locks.
        outcomes: list[ChunkOutcome] = []
        for i, chunk in enumerate(chunks):
            user_prompt = build_user_prompt(chunk, memory_context)
            try:
                raw = ctx.llm.chat_json(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:
                log.warning(
                    "batch consolidation: LLM call упал на chunk %d/%d: %s",
                    i + 1, len(chunks), exc,
                )
                raise

            try:
                parsed = _validate_batch_response(raw)
            except ValueError as exc:
                metrics.record_schema_error()
                raise OllamaTerminalError(
                    f"schema_mismatch на chunk {i}: {exc}", exc,
                )

            if parsed.skip:
                outcomes.append(ChunkOutcome(
                    skip=True, skip_reason=parsed.skip_reason,
                    vad=parsed.vad,
                ))
                continue

            outcomes.append(ChunkOutcome(
                skip=False,
                # Full summary, без truncate'а — store-routing (волна 19)
                # роутит content > limit в documents+chunks; truncate
                # применяется только в legacy-ветке когда routing
                # disabled (см. loop ниже).
                summary=parsed.summary or "",
                archive_ref=build_archive_ref(
                    rows, chunk, parsed.archive_hints or [],
                    chunk_index=i if len(chunks) > 1 else None,
                ),
                vad=parsed.vad,
            ))

        # Average VAD across chunks (включая skip-chunks с непустым vad —
        # skip про память, не про эмоции).
        vads = [o.vad for o in outcomes if o.vad is not None]
        avg_vad = average_vads(vads)
        if avg_vad is None:
            metrics.record_skip_no_vad()

        # Atomic: VAD apply (если есть) → INSERT memories → state update.
        # Все три внутри одной транзакции; runtime коммитит после
        # возврата HandlerResult.
        latest_row = rows[-1]
        created_ids: list[str] = []
        skip_reasons: list[str] = []

        if avg_vad is not None and batch_sentiment_enabled:
            append_emotional_state(
                ctx.conn, agent_id,
                scale_batch_vad_delta(avg_vad),
                source="sentiment:batch",
                metadata={
                    "window_from": (
                        payload.window_from.isoformat()
                        if payload.window_from else None
                    ),
                    "window_to": payload.window_to.isoformat(),
                    "chunks": len(chunks),
                    "vad_samples": len(vads),
                    "k_batch": K_BATCH,
                },
            )
            metrics.record_applied()

        for o in outcomes:
            if o.skip:
                if o.skip_reason:
                    skip_reasons.append(o.skip_reason)
                continue
            full_content = o.summary or ""

            # Store-routing (волна 19): длинный summary разделяется на
            # chunks в documents/chunks; tail-memory с merged
            # archive_ref (document + dialogue_batch_archive_ref).
            # Требует ``ctx.embedder`` (без него skip — см. legacy
            # path ниже tоже скипает gatekeeper при NULL embedder'е).
            if (
                routing.enabled
                and ctx.embedder is not None
                and len(full_content) > routing.limit
            ):
                try:
                    routed = route_long_content(
                        queries, ctx.embedder,
                        content=full_content,
                        kind="episode",
                        kind_src="dialogue_batch_consolidation",
                        role="summary",
                        config=routing,
                        source="insert_batch_memory",
                        extra_archive_metadata=(
                            {"dialogue_batch_archive_ref": o.archive_ref}
                            if o.archive_ref else None
                        ),
                    )
                except (EmbeddingError, ValueError) as exc:
                    log.warning(
                        "batch consolidation: store-routing fail, skip "
                        "chunk: %s", exc,
                    )
                    skip_reasons.append("routing_failed")
                    continue
                if al_config.enabled:
                    auto_link_after_store(
                        queries,
                        memory_id=routed.tail_memory_id,
                        embedding=routed.summary_embedding,
                        config=al_config, agent_id=agent_id,
                        source="dialogue_batch_consolidation",
                    )
                log_event(
                    log, "store_routing",
                    agent_id=str(agent_id),
                    tail_memory_id=str(routed.tail_memory_id),
                    document_id=str(routed.document_id),
                    chunks_count=routed.chunks_count,
                    content_length=len(full_content),
                    source="dialogue_batch_consolidation",
                )
                created_ids.append(str(routed.tail_memory_id))
                continue

            # Legacy / disabled-routing path: truncate до CHECK
            # constraint'а memories.content (≤ 2400 chars).
            content = truncate(full_content, MAX_CONTENT_CHARS)
            mid = queries.insert_batch_memory(
                content=content,
                archive_ref=o.archive_ref or {},
            )

            # Sync embed (волна 17) — приводим writer к согласованной
            # sync-policy. ``ctx.embedder is None`` → legacy fallback:
            # embedding=NULL, ряд подберёт reembed CLI; gatekeeper
            # пропускается.
            if ctx.embedder is None:
                created_ids.append(str(mid))
                continue
            try:
                vec = ctx.embedder.embed(content)
            except EmbeddingError as exc:
                log.warning(
                    "batch consolidation: embed failed для memory %s: %s",
                    mid, exc,
                )
                created_ids.append(str(mid))
                continue
            queries.update_embedding(mid, vec)

            # Selective gatekeeper (волна 17) — apply решение прямо в
            # handler'е, в той же транзакции. Runtime коммитит после
            # возврата HandlerResult.
            if not gk_config.enabled:
                created_ids.append(str(mid))
                continue
            candidates = queries.find_gatekeeper_candidates(
                vec,
                max_cosine_distance=1.0 - gk_config.supersede_threshold,
                exclude_id=mid,
            )
            decision = decide(content, candidates, config=gk_config)
            log_event(
                log, "selective_decision",
                agent_id=str(agent_id),
                memory_id=str(mid),
                action=decision.action.value,
                existing_id=(
                    str(decision.existing_id)
                    if decision.existing_id else None
                ),
                similarity=decision.similarity,
                levenshtein_ratio=decision.levenshtein_ratio,
                source="dialogue_batch_consolidation",
            )

            if decision.action == Action.STORE:
                created_ids.append(str(mid))
                auto_link_after_store(
                    queries, memory_id=mid, embedding=vec,
                    config=al_config, agent_id=agent_id,
                    source="dialogue_batch_consolidation",
                )
            elif decision.action == Action.SKIP:
                queries.apply_gatekeeper_skip(mid)
                skip_reasons.append(f"gatekeeper_skip:{mid}")
            elif decision.action == Action.MERGE:
                assert decision.existing_id is not None
                queries.apply_gatekeeper_merge(
                    new_id=mid, existing_id=decision.existing_id,
                    new_content=content, new_embedding=vec,
                )
                # mid удалён, existing получил update; в created_ids
                # ничего не добавляем — существующий ряд просто обновлён.
            elif decision.action == Action.SUPERSEDE:
                assert decision.existing_id is not None
                queries.apply_gatekeeper_supersede(
                    new_id=mid, existing_id=decision.existing_id,
                    new_embedding=vec,
                )
                created_ids.append(str(mid))
                auto_link_after_store(
                    queries, memory_id=mid, embedding=vec,
                    config=al_config, agent_id=agent_id,
                    source="dialogue_batch_consolidation",
                )

        set_batch_state(
            ctx.conn, agent_id,
            {
                "last_batch_at": _now_iso(),
                "last_message_created_at": latest_row["created_at"].isoformat(),
                "last_message_id": str(latest_row["id"]),
                "last_window_end_at": payload.window_to.isoformat(),
            },
        )

        all_skipped = len(created_ids) == 0
        return HandlerResult(
            result={
                "chunks": len(chunks),
                "memories_created": created_ids,
                "skipped_reasons": skip_reasons,
            },
            skipped_by_llm=all_skipped,
        )

    return handler


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
