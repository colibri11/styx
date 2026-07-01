"""LLM handler `memory_daily_consolidation` — port memorybox 26 (волна 22).

Reads N (3..8) memories по `payload.memory_ids` под `agent_id`-scope,
asks qwen3:4b-local через `ctx.llm.chat_json` для consolidated_text
(или skip), embed'ит результат, сохраняет в `llm_tasks.result`.

Не делает UPDATE на memories или INSERT нового memory — это работа
`memory_consolidation_apply_sweeper` под write-gate'ом из волны 14.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from styx.embedding import EmbeddingError
from styx.llm import OllamaTerminalError
from styx.storage.queries import AgentScopedQueries, parse_vector
from styx.workers.runtime import (
    Handler,
    HandlerContext,
    HandlerResult,
    LlmTask,
)

log = logging.getLogger(__name__)


MEMORY_DAILY_CONSOLIDATION_TASK_TYPE = "memory_daily_consolidation"


# ── System prompt (port memorybox handlers/memory-daily-consolidation.ts) ──

SYSTEM_PROMPT = """Ты уплотняешь набор связанных памятей одного субъекта в одну.
Вход — список из 3–8 памятей: каждая с кратким id и content.
Задача — вернуть consolidated_text (1–3 предложения на русском),
в котором удержана смысловая направленность всего кластера:
повторяющаяся тема, общий вывод, единая конфигурация опыта.
Это не перечисление и не сумма; это переформулировка, ухватывающая то,
что остаётся, когда буквальности стираются.

Верни строгий JSON по схеме:
{
  "skip": <true | false>,
  "skip_reason": <строка на русском | null>,
  "consolidated_text": <строка 1–3 предложения на русском | null>
}

Когда skip=true:
  — skip_reason обязателен (короткая причина на русском);
  — consolidated_text должен быть null.
  Ставь skip=true, если:
    — памяти в кластере про разное, близость чисто лексическая;
    — общая формулировка была бы слишком абстрактной и не добавляет смысла к самим источникам;
    — кластер содержит противоречивые утверждения, которые нельзя свести к одной направленности без потери информации;
    — одна из памятей заметно перевешивает остальные и уплотнение просто продублирует её.
  Не придумывай consolidated_text, когда уплотнять нечего.

Когда skip=false:
  — skip_reason = null;
  — consolidated_text обязателен, 1–3 предложения, на русском,
    удерживает общую направленность кластера; сохраняет конкретику (имена, даты, факты),
    которые повторяются или определяют смысл; не добавляет того, чего не было.

Правила формулировки consolidated_text:
  — писать от первого лица субъекта, если источники в первом лице;
  — не начинать с «У меня было несколько памятей о…»; описывать саму мысль/состояние/наблюдение;
  — не суммировать как «в этих памятях…»;
  — не перечислять id источников;
  — держать язык источников (русский).

Отвечай только JSON. Без комментариев, без префиксов, без блоков кода."""


# ── Validators ────────────────────────────────────────────────────────


def _validate_payload(raw: Any) -> tuple[str, list[uuid.UUID]]:
    """Returns (agent_id, memory_ids). Raises ValueError на schema mismatch."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"payload должен быть object, получено {type(raw).__name__}"
        )
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("payload.agent_id обязателен (str non-empty)")
    raw_ids = raw.get("memory_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) < 2 or len(raw_ids) > 8:
        raise ValueError(
            f"payload.memory_ids — list[str] длиной 2..8, получено {raw_ids!r}"
        )
    out: list[uuid.UUID] = []
    for x in raw_ids:
        if not isinstance(x, str):
            raise ValueError(f"memory_ids: ожидаем UUID-строки, получено {x!r}")
        try:
            out.append(uuid.UUID(x))
        except ValueError as exc:
            raise ValueError(f"memory_ids: invalid UUID {x!r}") from exc
    return agent_id, out


def _validate_response(raw: Any) -> tuple[bool, str | None, str | None]:
    """Returns (skip, skip_reason, consolidated_text)."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"response должен быть object, получено {type(raw).__name__}"
        )
    skip = raw.get("skip")
    if not isinstance(skip, bool):
        raise ValueError(f"skip должен быть bool, получено {skip!r}")
    skip_reason = raw.get("skip_reason")
    text = raw.get("consolidated_text")
    if skip:
        if not isinstance(skip_reason, str) or not (
            1 <= len(skip_reason) <= 300
        ):
            raise ValueError(
                "skip=true: skip_reason обязателен (1..300 chars)"
            )
        if text is not None:
            raise ValueError(
                "skip=true: consolidated_text должен быть null"
            )
        return True, skip_reason, None
    if skip_reason is not None:
        raise ValueError("skip=false: skip_reason должен быть null")
    if not isinstance(text, str) or not (1 <= len(text) <= 2400):
        raise ValueError(
            "skip=false: consolidated_text — строка 1..2400"
        )
    return False, None, text


# ── Handler factory ───────────────────────────────────────────────────


def create_memory_daily_consolidation_handler() -> Handler:
    """Factory для memory_daily_consolidation handler'а."""

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        try:
            agent_id, memory_ids = _validate_payload(task.payload)
        except ValueError as exc:
            raise OllamaTerminalError(f"invalid_payload: {exc}", exc)

        queries = AgentScopedQueries(ctx.conn, agent_id)
        memories = queries.load_memories_for_consolidation(memory_ids)

        # Штатная гонка supersede/delete: между enqueue задачи и её
        # claim источники кластера успели быть слиты/удалены другим
        # apply-проходом. Консолидировать одинокую memory или пустой
        # кластер нечего — это ожидаемый no-op, а не сбой. Возвращаем
        # skip-результат (зеркалит LLM-skip ветку ниже), чтобы задача
        # пометилась done/skipped, а не failed терминально.
        if len(memories) < 2:
            surviving_ids = [str(m["id"]) for m in memories]
            skip_reason = (
                f"cluster схлопнулся до N<2 — race supersede/delete "
                f"({len(memories)} из {len(memory_ids)} источников выжили "
                "до claim)"
            )
            log.info(
                "memory_daily_consolidation: %s → skip no-op", skip_reason,
            )
            return HandlerResult(
                result={
                    "skip": True,
                    "skip_reason": skip_reason,
                    "consolidated_text": None,
                    "consolidated_embedding": None,
                    "agent_id": agent_id,
                    "source_ids": surviving_ids,
                    "source_kinds": None,
                    "source_visibility": None,
                },
                skipped_by_llm=True,
            )
        for m in memories:
            if m.get("superseded_by") is not None:
                raise OllamaTerminalError(
                    f"some_source_already_superseded: memory {m['id']} "
                    f"superseded_by={m['superseded_by']}"
                )

        source_ids = [m["id"] for m in memories]
        source_kinds = [m["kind"] for m in memories]
        source_visibility = [
            m.get("visibility") if m.get("visibility") in ("shared", "private")
            else "shared"
            for m in memories
        ]

        # LLM call.
        user_prompt = _build_user_prompt(memories)
        try:
            raw = ctx.llm.chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover — runtime drains
            log.warning(
                "memory_daily_consolidation: LLM call упал: %s", exc,
            )
            raise

        try:
            skip, skip_reason, consolidated_text = _validate_response(raw)
        except ValueError as exc:
            raise OllamaTerminalError(f"schema_mismatch: {exc}", exc)

        if skip:
            return HandlerResult(
                result={
                    "skip": True,
                    "skip_reason": skip_reason,
                    "consolidated_text": None,
                    "consolidated_embedding": None,
                    "agent_id": agent_id,
                    "source_ids": [str(s) for s in source_ids],
                    "source_kinds": None,
                    "source_visibility": None,
                },
                skipped_by_llm=True,
            )

        if ctx.embedder is None:
            raise OllamaTerminalError(
                "embedder_unavailable: memory_daily_consolidation требует "
                "embed для consolidated_text, ctx.embedder=None"
            )
        try:
            embedding = ctx.embedder.embed(consolidated_text)
        except EmbeddingError as exc:
            log.warning(
                "memory_daily_consolidation: embed упал: %s", exc,
            )
            raise

        return HandlerResult(
            result={
                "skip": False,
                "skip_reason": None,
                "consolidated_text": consolidated_text,
                "consolidated_embedding": embedding,
                "agent_id": agent_id,
                "source_ids": [str(s) for s in source_ids],
                "source_kinds": source_kinds,
                "source_visibility": source_visibility,
            }
        )

    return handler


def _build_user_prompt(memories: list[dict]) -> str:
    blocks = []
    for i, m in enumerate(memories):
        blocks.append(f"id{i + 1}: {m['content']}")
    body = "\n---\n".join(blocks)
    return f"cluster:\n---\n{body}\n---"
