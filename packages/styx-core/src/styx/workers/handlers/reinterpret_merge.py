"""LLM handler `reinterpret_merge` — port memorybox (волна 22).

Reads (previous_text, previous_embedding) from `memories` by
`task.memory_id`, asks qwen3:4b-local через `ctx.llm.chat_json` для
получения `merged_text` (или skip), embed'ит merged_text, blend'ит
embedding'и `(1-w)*prev + w*next` + L2-norm, и сохраняет результат
в `llm_tasks.result`.

Не делает UPDATE на memories или INSERT в audit — это работа
`reinterpret_apply_sweeper` под write-gate'ом из волны 14.
"""

from __future__ import annotations

import logging
from typing import Any

from styx.engine.reinterpret import (
    DEFAULT_BLEND_WEIGHT,
    BlendError,
    blend_embeddings,
)
from styx.embedding import EmbeddingError
from styx.llm import OllamaTerminalError
from styx.storage.queries import (
    parse_vector,
    worker_load_memory_for_reinterpret,
)
from styx.workers.runtime import (
    Handler,
    HandlerContext,
    HandlerResult,
    LlmTask,
)

log = logging.getLogger(__name__)


REINTERPRET_MERGE_TASK_TYPE = "reinterpret_merge"

MAX_NEW_UNDERSTANDING_CHARS = 2400


# ── System prompt (port memorybox handlers/reinterpret-merge.ts) ──────

SYSTEM_PROMPT = """Ты уплотняешь две формулировки одной и той же памяти в одну.
Вход — пара:
  previous — как запомнилось раньше;
  new_understanding — что добавилось в понимании.
Задача — вернуть компактный merged_text (1–3 предложения на русском),
в котором удержаны оба слоя смысла: прежнее и новое. Это не склейка
и не дополнение через запятую; это переформулировка, в которой новая
координата уже встроена.

Верни строгий JSON по схеме:
{
  "skip": <true | false>,
  "skip_reason": <строка на русском | null>,
  "merged_text": <строка 1–3 предложения на русском | null>
}

Когда skip=true:
  — skip_reason обязателен (короткая причина на русском);
  — merged_text должен быть null.
  Ставь skip=true, если:
    — new_understanding не добавляет новой координаты к previous
      (тавтология, перефразировка того же, уточнение без смены смысла);
    — new_understanding противоречит previous настолько, что «обе истины одновременно»
      бессмысленны (тогда нужен не reinterpret, а новая память или supersede);
    — previous или new_understanding — шум / служебная реплика / мусор.
  Не придумывай merged_text, когда сливать нечего — пустой выход лучше галлюцинации.

Когда skip=false:
  — skip_reason = null;
  — merged_text обязателен, 1–3 предложения, на русском,
    удерживает и прежнее понимание, и новую координату.

Правила формулировки merged_text:
  — не начинать с «Раньше я думал, а теперь»; не описывать сам факт переосмысления;
    описывать саму мысль, как она выглядит после переосмысления;
  — сохранять конкретику (имена, даты, факты) из обоих входов;
  — не выбрасывать эмоциональный оттенок, если он был в previous;
  — не добавлять того, чего не было ни в previous, ни в new_understanding.

Отвечай только JSON. Без комментариев, без префиксов, без блоков кода."""


# ── Validators ────────────────────────────────────────────────────────


def _validate_payload(raw: Any) -> tuple[str, str, float | None]:
    """Returns (agent_id, new_understanding_text, weight | None).
    Raises ValueError на schema mismatch."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"payload должен быть object, получено {type(raw).__name__}"
        )
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("payload.agent_id обязателен (str non-empty)")
    text = raw.get("new_understanding_text")
    if not isinstance(text, str) or not (
        1 <= len(text) <= MAX_NEW_UNDERSTANDING_CHARS
    ):
        raise ValueError(
            f"payload.new_understanding_text — строка 1..{MAX_NEW_UNDERSTANDING_CHARS}"
        )
    weight = raw.get("weight")
    if weight is not None:
        if not isinstance(weight, (int, float)):
            raise ValueError("payload.weight — number или null")
        weight = float(weight)
        if not (0.0 <= weight <= 1.0):
            raise ValueError(
                f"payload.weight={weight} вне [0, 1]"
            )
    return agent_id, text, weight


def _validate_response(raw: Any) -> tuple[bool, str | None, str | None]:
    """Returns (skip, skip_reason, merged_text). Raises ValueError на
    schema mismatch — handler конвертит в OllamaTerminalError."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"response должен быть object, получено {type(raw).__name__}"
        )
    skip = raw.get("skip")
    if not isinstance(skip, bool):
        raise ValueError(f"skip должен быть bool, получено {skip!r}")
    skip_reason = raw.get("skip_reason")
    merged_text = raw.get("merged_text")
    if skip:
        if not isinstance(skip_reason, str) or not (
            1 <= len(skip_reason) <= 300
        ):
            raise ValueError(
                "skip=true: skip_reason обязателен (1..300 chars)"
            )
        if merged_text is not None:
            raise ValueError("skip=true: merged_text должен быть null")
        return True, skip_reason, None
    # skip=false
    if skip_reason is not None:
        raise ValueError("skip=false: skip_reason должен быть null")
    if not isinstance(merged_text, str) or not (
        1 <= len(merged_text) <= MAX_NEW_UNDERSTANDING_CHARS
    ):
        raise ValueError(
            f"skip=false: merged_text — строка 1..{MAX_NEW_UNDERSTANDING_CHARS}"
        )
    return False, None, merged_text


# ── Handler factory ───────────────────────────────────────────────────


def create_reinterpret_merge_handler(
    *, blend_weight: float = DEFAULT_BLEND_WEIGHT,
) -> Handler:
    """Factory для reinterpret_merge handler'а.

    `blend_weight` — default weight blend'а если payload.weight отсутствует.
    """

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        # 1. Validate payload first — terminal на schema mismatch.
        try:
            agent_id, new_understanding_text, weight_payload = _validate_payload(
                task.payload
            )
        except ValueError as exc:
            raise OllamaTerminalError(f"invalid_payload: {exc}", exc)

        weight = weight_payload if weight_payload is not None else blend_weight

        # 2. memory_id обязателен — MCP tool/HTTP route его всегда
        # пишет. None означает manual enqueue без id, отдаём skip-shape.
        if task.memory_id is None:
            return HandlerResult(
                result={"skipped": "no_memory_id"}, skipped_by_llm=True,
            )

        # 3. Load memory by PK (cross-agent — agent_id сравнение далее).
        row = worker_load_memory_for_reinterpret(ctx.conn, task.memory_id)
        if row is None:
            # Race с CASCADE — memory удалена между enqueue и claim.
            return HandlerResult(
                result={"skipped": "memory_gone"}, skipped_by_llm=True,
            )

        if row["agent_id"] != agent_id:
            # MCP tool/HTTP route берёт agent_id из session, mismatch =
            # crafted payload или race с reassignment. Terminal.
            raise OllamaTerminalError(
                f"agent_id_mismatch: memory.agent_id={row['agent_id']!r} "
                f"payload.agent_id={agent_id!r}"
            )

        previous_embedding = parse_vector(row["embedding"])
        if not previous_embedding:
            # Memory без embedding — blend невозможен. Terminal:
            # reinterpret на не-embedded record — contract violation.
            raise OllamaTerminalError("previous_embedding_missing")

        previous_text = row["content"]

        # 4. LLM call.
        user_prompt = (
            f"previous:\n---\n{previous_text}\n---\n\n"
            f"new_understanding:\n---\n{new_understanding_text}\n---"
        )
        try:
            raw = ctx.llm.chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover — runtime drains
            log.warning("reinterpret_merge: LLM call упал: %s", exc)
            raise

        # 5. Validate LLM output.
        try:
            skip, skip_reason, merged_text = _validate_response(raw)
        except ValueError as exc:
            raise OllamaTerminalError(f"schema_mismatch: {exc}", exc)

        if skip:
            return HandlerResult(
                result={
                    "skip": True,
                    "skip_reason": skip_reason,
                    "merged_text": None,
                    "merged_embedding": None,
                    "previous_text": previous_text,
                    "previous_embedding": previous_embedding,
                    "new_understanding_text": new_understanding_text,
                    "weight_applied": None,
                    "agent_id": agent_id,
                },
                skipped_by_llm=True,
            )

        # skip=false → embed merged_text + blend.
        if ctx.embedder is None:
            raise OllamaTerminalError(
                "embedder_unavailable: reinterpret_merge требует embed для "
                "merged_text, ctx.embedder=None"
            )
        try:
            next_embedding = ctx.embedder.embed(merged_text)
        except EmbeddingError as exc:
            log.warning("reinterpret_merge: embed merged_text упал: %s", exc)
            raise

        try:
            merged_embedding = blend_embeddings(
                previous_embedding, next_embedding, weight=weight,
            )
        except BlendError as exc:
            raise OllamaTerminalError(
                f"blend_{exc.code}: {exc}", exc,
            )

        return HandlerResult(
            result={
                "skip": False,
                "skip_reason": None,
                "merged_text": merged_text,
                "merged_embedding": merged_embedding,
                "previous_text": previous_text,
                "previous_embedding": previous_embedding,
                "new_understanding_text": new_understanding_text,
                "weight_applied": weight,
                "agent_id": agent_id,
            }
        )

    return handler
