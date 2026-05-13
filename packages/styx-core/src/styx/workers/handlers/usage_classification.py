"""LLM usage classifier — post-hoc проставляет ``recall_events.used_in_output``.

Прямой port memorybox `llm-worker/handlers/usage-classification.ts`.
SYSTEM_PROMPT, MAX_REPLY_CHARS, MAX_MEMORY_CHARS, reconciliation
strict — буквально.

Scheduler (kто enqueue'ит task'ы) — собственный дизайн Styx, не port
(memorybox skeleton). См. ADR § 20.

Cross-agent: handler читает recall_events / memories по PK, без agent_id
фильтра (admin-tier, как importance handler).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from styx.llm import OllamaTerminalError
from styx.workers.runtime import Handler, HandlerContext, HandlerResult, LlmTask


USAGE_CLASSIFICATION_TASK_TYPE = "usage_classification"

MAX_REPLY_CHARS = 8000
MAX_MEMORY_CHARS = 1500
MAX_PAYLOAD_REPLY_CHARS = 80_000
MAX_RECALL_EVENT_IDS = 20


SYSTEM_PROMPT = """Ты классифицируешь, какие из предложенных воспоминаний (memories) агент реально использовал в своём ответе.

Тебе дан текст ответа агента и список воспоминаний с их id. Для каждого воспоминания реши, опирался ли на него агент при формулировании ответа. "Опирался" означает: тезис, факт, формулировка или логика ответа заметно соответствует содержанию воспоминания.

Верни строгий JSON по схеме:
{
  "skip": <true | false>,
  "skip_reason": <строка на русском | null>,
  "classifications": <[{memory_id, used, reason}] | null>
}

Когда skip=true:
  — skip_reason обязателен (короткая причина на русском);
  — classifications = null.
  Ставь skip=true, если ответ агента пустой, односложный, служебный или не содержит опоры на какой-либо материал.
  Пустой выход лучше галлюцинации.

Когда skip=false:
  — skip_reason = null;
  — classifications — массив на каждое входящее воспоминание; порядок произвольный, но каждый memory_id должен встретиться ровно один раз.
  — used: true если агент опирался; false если ответ не пересекается с воспоминанием по смыслу.
  — reason: 1 короткое предложение на русском, доказывающее решение (цитата/перефраз из ответа).

Строгость: used=true только если связь очевидна. Если сомневаешься — ставь used=false. Лучше пропустить реально использованное, чем ошибочно пометить ignored как used — эта метка влияет на дальнейший scoring памяти.

Примеры:

Пример 1.
Ответ агента: "ок"
Воспоминания: [{"memory_id": "m-1", "content": "пользователь любит qwen3:4b-local за большое контекстное окно"}]
Ответ:
{"skip": true, "skip_reason": "односложный служебный ответ, нет опоры ни на что", "classifications": null}

Пример 2.
Ответ агента: "Для этой задачи подойдёт qwen3:4b-local — у неё большое окно контекста, что у нас является приоритетом."
Воспоминания: [
  {"memory_id": "m-1", "content": "пользователь предпочитает qwen3:4b-local из-за большого контекстного окна в ущерб количеству параметров"},
  {"memory_id": "m-2", "content": "В прошлую пятницу обсуждали ребалансировку портфеля"}
]
Ответ:
{"skip": false, "skip_reason": null, "classifications": [
  {"memory_id": "m-1", "used": true, "reason": "агент напрямую опирается на предпочтение qwen3:4b-local за большое окно"},
  {"memory_id": "m-2", "used": false, "reason": "в ответе нет ни портфеля, ни пятницы — тема другая"}
]}

Отвечай только JSON. Без комментариев, без префиксов, без блоков кода."""


# ── Validators ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassificationItem:
    memory_id: str
    used: bool
    reason: str


@dataclass(frozen=True)
class UsageResultParsed:
    skip: bool
    skip_reason: str | None
    classifications: list[ClassificationItem] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skip": self.skip,
            "skip_reason": self.skip_reason,
            "classifications": (
                [
                    {
                        "memory_id": c.memory_id,
                        "used": c.used,
                        "reason": c.reason,
                    }
                    for c in self.classifications
                ]
                if self.classifications is not None
                else None
            ),
        }


def _validate_payload(raw: Any) -> tuple[list[int], str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"payload должен быть object, получено {type(raw).__name__}")
    ids = raw.get("recall_event_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("recall_event_ids должен быть непустым list")
    if len(ids) > MAX_RECALL_EVENT_IDS:
        raise ValueError(f"recall_event_ids превышает лимит {MAX_RECALL_EVENT_IDS}")
    out_ids: list[int] = []
    for i in ids:
        if not isinstance(i, int) or i <= 0:
            raise ValueError(f"recall_event_id должен быть positive int, получено {i!r}")
        out_ids.append(i)
    text = raw.get("llm_output_text")
    if not isinstance(text, str) or not text:
        raise ValueError("llm_output_text должен быть непустой строкой")
    if len(text) > MAX_PAYLOAD_REPLY_CHARS:
        raise ValueError(f"llm_output_text превышает лимит {MAX_PAYLOAD_REPLY_CHARS}")
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id должен быть непустой строкой")
    return out_ids, text, agent_id


def _validate_response(raw: Any) -> UsageResultParsed:
    if not isinstance(raw, dict):
        raise ValueError(f"ответ должен быть object, получено {type(raw).__name__}")
    skip = raw.get("skip")
    if not isinstance(skip, bool):
        raise ValueError(f"skip должен быть bool, получено {skip!r}")

    skip_reason = raw.get("skip_reason")
    classifications = raw.get("classifications")

    if skip:
        if not isinstance(skip_reason, str) or not (1 <= len(skip_reason) <= 300):
            raise ValueError("skip=true: skip_reason обязателен 1..300 chars")
        if classifications is not None:
            raise ValueError("skip=true: classifications должен быть null")
        return UsageResultParsed(True, skip_reason, None)

    # skip=false
    if skip_reason is not None:
        raise ValueError("skip=false: skip_reason должен быть null")
    if not isinstance(classifications, list) or not classifications:
        raise ValueError("skip=false: classifications должен быть непустым list")

    items: list[ClassificationItem] = []
    for c in classifications:
        if not isinstance(c, dict):
            raise ValueError("classifications[*] должен быть object")
        mid = c.get("memory_id")
        used = c.get("used")
        reason = c.get("reason")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"classifications.memory_id должен быть строкой, получено {mid!r}")
        if not isinstance(used, bool):
            raise ValueError(f"classifications.used должен быть bool, получено {used!r}")
        if not isinstance(reason, str) or not (1 <= len(reason) <= 500):
            raise ValueError("classifications.reason должен быть 1..500 chars")
        items.append(ClassificationItem(memory_id=mid, used=used, reason=reason))

    return UsageResultParsed(False, None, items)


# ── DB helpers ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RecallRow:
    id: int
    memory_id: str  # uuid as text
    content: str
    used_in_output: bool


def _load_recall_rows(
    conn: psycopg.Connection, ids: list[int]
) -> list[_RecallRow]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT re.id::bigint AS id, "
            "       re.memory_id::text AS memory_id, "
            "       m.content AS content, "
            "       re.used_in_output AS used_in_output "
            "  FROM recall_events re "
            "  JOIN memories m ON m.id = re.memory_id "
            " WHERE re.id = ANY(%s::bigint[])",
            (ids,),
        )
        rows = cur.fetchall()
    return [
        _RecallRow(
            id=int(r["id"]),
            memory_id=str(r["memory_id"]),
            content=r["content"],
            used_in_output=bool(r["used_in_output"]),
        )
        for r in rows
    ]


def _flip_used_in_output(
    conn: psycopg.Connection, recall_ids: list[int]
) -> int:
    """UPDATE used_in_output=true с guard'ом IS DISTINCT FROM. Возвращает rowcount.

    Guard ``used_in_output = false`` — чтобы explicit confirmation
    (если когда-нибудь будет — не реализовано в Styx) не overwrite'ить.
    """
    if not recall_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE recall_events "
            "   SET used_in_output = true "
            " WHERE id = ANY(%s::bigint[]) "
            "   AND used_in_output = false",
            (recall_ids,),
        )
        return cur.rowcount or 0


# ── Prompt builder ────────────────────────────────────────────────────


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…[truncated]"


def _build_user_prompt(reply: str, rows: list[_RecallRow]) -> str:
    truncated_reply = _truncate(reply, MAX_REPLY_CHARS)
    memories_payload = [
        {"memory_id": r.memory_id, "content": _truncate(r.content, MAX_MEMORY_CHARS)}
        for r in rows
    ]
    memories_json = json.dumps(memories_payload, ensure_ascii=False, indent=2)
    return (
        f"agent_reply:\n---\n{truncated_reply}\n---\n"
        f"\n"
        f"memories:\n---\n{memories_json}\n---"
    )


# ── Handler factory ───────────────────────────────────────────────────


def create_usage_classification_handler() -> Handler:
    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        try:
            ids, reply_text, agent_id = _validate_payload(task.payload)
        except ValueError as exc:
            # Bad payload — scheduler-bug, retry не поможет.
            raise OllamaTerminalError(f"bad_payload: {exc}") from exc

        rows = _load_recall_rows(ctx.conn, ids)
        if not rows:
            # Все recall_events исчезли (memories→CASCADE), не LLM-skip.
            return HandlerResult(result={"skipped": "no_recall_rows"})

        live_id_set = {r.id for r in rows}
        missing_ids = [i for i in ids if i not in live_id_set]

        ctx.rate_limit.acquire()

        user_prompt = _build_user_prompt(reply_text, rows)
        raw = ctx.llm.chat_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        try:
            parsed = _validate_response(raw)
        except ValueError as exc:
            raise OllamaTerminalError(f"schema_mismatch: {exc}") from exc

        if parsed.skip:
            return HandlerResult(
                result={
                    **parsed.to_dict(),
                    "missing_recall_ids": missing_ids,
                },
                skipped_by_llm=True,
            )

        # Reconcile: каждый live memory_id ровно один раз; никаких лишних.
        live_memory_id_set = {r.memory_id for r in rows}
        seen: set[str] = set()
        assert parsed.classifications is not None
        for c in parsed.classifications:
            if c.memory_id not in live_memory_id_set:
                raise OllamaTerminalError(
                    f"classification_mismatch: unknown memory_id {c.memory_id}"
                )
            if c.memory_id in seen:
                raise OllamaTerminalError(
                    f"classification_mismatch: duplicate memory_id {c.memory_id}"
                )
            seen.add(c.memory_id)
        for mid in live_memory_id_set:
            if mid not in seen:
                raise OllamaTerminalError(
                    f"classification_mismatch: missing memory_id {mid}"
                )

        used_memory_ids = {c.memory_id for c in parsed.classifications if c.used}
        used_recall_ids = [r.id for r in rows if r.memory_id in used_memory_ids]

        flipped = 0
        if used_recall_ids:
            flipped = _flip_used_in_output(ctx.conn, used_recall_ids)

        return HandlerResult(
            result={
                **parsed.to_dict(),
                "flipped": flipped,
                "used_recall_ids": used_recall_ids,
                "missing_recall_ids": missing_ids,
            },
        )

    return handler
