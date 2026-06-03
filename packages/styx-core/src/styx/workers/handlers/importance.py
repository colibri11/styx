"""LLM importance handler — оценка ``memories.importance_final``.

Триггер ``enqueue_importance_scoring`` (волна 7) ставит pending row в
``llm_tasks`` на каждый INSERT/UPDATE OF content в memories. Этот
handler drain'ит очередь:

1. Загружает memory по ``task.memory_id``.
2. Вызывает ``qwen3:4b-local`` через chat-API с system prompt'ом
   (буквальный port из memorybox importance-scoring-from-content.ts).
3. Парсит и валидирует JSON ответ по дискриминирующей схеме (skip vs
   scored).
4. Skip → ``importance_final`` остаётся NULL,
   ``llm_tasks.result.skip = true``, ``skipped_by_llm`` метрика +1.
5. Scored → ``worker_update_importance_final`` записывает score.

Schema mismatch (LLM вернул что-то не подходящее под схему) →
``OllamaTerminalError`` — runtime пометит task как failed, retry'ить
бесполезно при том же контракте.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from styx.llm import OllamaTerminalError
from styx.storage.queries import (
    _WorkerMemoryRow,
    worker_load_memory,
    worker_update_importance_final,
)
from styx.workers.runtime import Handler, HandlerContext, HandlerResult, LlmTask


IMPORTANCE_TASK_TYPE = "importance_scoring_from_content"


# ── System prompt — буквальный port из memorybox ──────────────────────

SYSTEM_PROMPT = """Ты оцениваешь важность записи в долговременной памяти агента.
Верни строгий JSON по схеме:
{
  "skip": <true | false>,
  "skip_reason": <строка на русском | null>,
  "importance_score": <число от 0 до 1 | null>,
  "rationale": <строка 1–3 предложения на русском | null>,
  "signals": {
    "has_specific_facts": <bool>,
    "self_reference":     <bool>,
    "emotional_weight":   <число от 0 до 1>,
    "declarative":        <bool>,
    "abstraction_level":  "concrete" | "mixed" | "abstract"
  } | null
}

Когда skip=true:
  — skip_reason обязателен (короткая причина на русском);
  — importance_score, rationale, signals должны быть null.
  Ставь skip=true, если материал слишком короткий, служебный или шумный для осмысленной оценки.
  Не придумывай signals и не выставляй произвольный importance_score, когда оценивать нечего —
  пустой выход лучше галлюцинации.

Когда skip=false:
  — skip_reason = null;
  — importance_score, rationale, signals обязательны.

Шкала importance_score:
  0.0–0.2  — шум, мелкая сервисная реплика, одноразовое событие без последствий
  0.2–0.5  — фон, общие рассуждения, контекст без конкретики
  0.5–0.7  — полезное, содержит факт/предпочтение/решение с ограниченной релевантностью
  0.7–0.9  — значимое: личные факты, устойчивые предпочтения, ключевые решения
  0.9–1.0  — критически важно сохранить надолго: identity-level факты, принципы

Факторы в пользу высокого score:
  — конкретные факты (даты, имена, числа, места);
  — касается самого пользователя или агента лично;
  — декларация/правило/предпочтение длительного действия;
  — решение с последствиями.

Факторы в пользу низкого score:
  — общие рассуждения без конкретики;
  — одноразовое событие без продолжения;
  — техническая ремарка процесса разговора.

Примеры:

Вход: "ок"
Ответ: {"skip": true, "skip_reason": "односложная служебная реплика", "importance_score": null, "rationale": null, "signals": null}

Вход: "напомни, какой сегодня день"
Ответ: {"skip": true, "skip_reason": "процессная реплика без содержания, запоминать нечего", "importance_score": null, "rationale": null, "signals": null}

Вход: "qwen3:4b-local выбран из-за большого контекстного окна в ущерб количеству параметров"
Ответ: {"skip": false, "skip_reason": null, "importance_score": 0.82, "rationale": "Декларация устойчивого технического предпочтения с обоснованием; пригодится при будущих решениях о выборе модели.", "signals": {"has_specific_facts": true, "self_reference": true, "emotional_weight": 0.1, "declarative": true, "abstraction_level": "concrete"}}

Отвечай только JSON. Без комментариев, без префиксов, без блоков кода."""


# ── Result schemas (manual validation вместо pydantic) ────────────────


_VALID_ABSTRACTION = {"concrete", "mixed", "abstract"}


@dataclass(frozen=True)
class ImportanceSignals:
    has_specific_facts: bool
    self_reference: bool
    emotional_weight: float
    declarative: bool
    abstraction_level: str


@dataclass(frozen=True)
class ImportanceParsed:
    """Результат валидации ответа LLM. ``skip=True`` ⇒ остальные поля
    не используются (см. правила контракта)."""

    skip: bool
    skip_reason: str | None
    importance_score: float | None
    rationale: str | None
    signals: ImportanceSignals | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skip": self.skip,
            "skip_reason": self.skip_reason,
            "importance_score": self.importance_score,
            "rationale": self.rationale,
            "signals": (
                {
                    "has_specific_facts": self.signals.has_specific_facts,
                    "self_reference": self.signals.self_reference,
                    "emotional_weight": self.signals.emotional_weight,
                    "declarative": self.signals.declarative,
                    "abstraction_level": self.signals.abstraction_level,
                }
                if self.signals is not None
                else None
            ),
        }


def _validate_importance_response(raw: Any) -> ImportanceParsed:
    """Discriminated union по полю ``skip``. Любая ошибка → ValueError."""
    if not isinstance(raw, dict):
        raise ValueError(f"ожидаем object, получили {type(raw).__name__}")

    skip = raw.get("skip")
    if not isinstance(skip, bool):
        raise ValueError(f"skip должен быть bool, получено {skip!r}")

    skip_reason = raw.get("skip_reason")
    importance_score = raw.get("importance_score")
    rationale = raw.get("rationale")
    signals = raw.get("signals")

    if skip:
        if not isinstance(skip_reason, str) or not (1 <= len(skip_reason) <= 300):
            raise ValueError(
                "skip=true: skip_reason обязателен, строка 1..300 символов"
            )
        if importance_score is not None:
            raise ValueError("skip=true: importance_score должен быть null")
        if rationale is not None:
            raise ValueError("skip=true: rationale должен быть null")
        if signals is not None:
            raise ValueError("skip=true: signals должен быть null")
        return ImportanceParsed(
            skip=True,
            skip_reason=skip_reason,
            importance_score=None,
            rationale=None,
            signals=None,
        )

    # skip=false
    if skip_reason is not None:
        raise ValueError("skip=false: skip_reason должен быть null")
    if not isinstance(importance_score, (int, float)) or not math.isfinite(importance_score):
        raise ValueError(
            f"skip=false: importance_score должен быть number, получено {importance_score!r}"
        )
    if not 0.0 <= float(importance_score) <= 1.0:
        raise ValueError(
            f"skip=false: importance_score должен быть в [0, 1], получено {importance_score}"
        )
    if not isinstance(rationale, str) or not (1 <= len(rationale) <= 1000):
        raise ValueError("skip=false: rationale обязателен, 1..1000 символов")
    if not isinstance(signals, dict):
        raise ValueError("skip=false: signals обязателен (object)")
    parsed_signals = _validate_signals(signals)
    return ImportanceParsed(
        skip=False,
        skip_reason=None,
        importance_score=float(importance_score),
        rationale=rationale,
        signals=parsed_signals,
    )


def _validate_signals(raw: dict[str, Any]) -> ImportanceSignals:
    has_specific_facts = raw.get("has_specific_facts")
    self_reference = raw.get("self_reference")
    emotional_weight = raw.get("emotional_weight")
    declarative = raw.get("declarative")
    abstraction_level = raw.get("abstraction_level")

    if not isinstance(has_specific_facts, bool):
        raise ValueError("signals.has_specific_facts должен быть bool")
    if not isinstance(self_reference, bool):
        raise ValueError("signals.self_reference должен быть bool")
    if not isinstance(declarative, bool):
        raise ValueError("signals.declarative должен быть bool")
    if not isinstance(emotional_weight, (int, float)) or not math.isfinite(
        emotional_weight
    ):
        raise ValueError("signals.emotional_weight должен быть number")
    if not 0.0 <= float(emotional_weight) <= 1.0:
        raise ValueError("signals.emotional_weight должен быть в [0, 1]")
    if abstraction_level not in _VALID_ABSTRACTION:
        raise ValueError(
            f"signals.abstraction_level должен быть одним из {_VALID_ABSTRACTION}, "
            f"получено {abstraction_level!r}"
        )

    return ImportanceSignals(
        has_specific_facts=has_specific_facts,
        self_reference=self_reference,
        emotional_weight=float(emotional_weight),
        declarative=declarative,
        abstraction_level=abstraction_level,
    )


# ── User prompt builder ────────────────────────────────────────────────


def _build_user_prompt(memory: _WorkerMemoryRow) -> str:
    created_at = memory.created_at
    iso = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
    return (
        f"kind: {memory.kind}\n"
        f"created_at: {iso}\n"
        f"provisional_importance: {memory.importance_provisional:.2f}\n"
        f"\n"
        f"content:\n---\n{memory.content}\n---"
    )


# ── Handler factory ────────────────────────────────────────────────────


def create_importance_handler() -> Handler:
    """Factory возвращает handler-функцию.

    LLM-клиент и rate-limiter приходят через ``HandlerContext`` от
    runtime'а, чтобы handler оставался stateless и тестировался без
    повторного wiring'а.
    """

    def handler(task: LlmTask, ctx: HandlerContext) -> HandlerResult:
        if task.memory_id is None:
            # Триггер всегда ставит memory_id; null означает что кто-то
            # руками вставил pending — оценивать нечего, не LLM-skip.
            return HandlerResult(result={"skipped": "no_memory_id"})

        memory = worker_load_memory(ctx.conn, task.memory_id)
        if memory is None:
            # Memory удалена между enqueue и claim (CASCADE мог снести и
            # task, но эта строка пережила race — отдельный вариант).
            return HandlerResult(result={"skipped": "memory_gone"})

        ctx.rate_limit.acquire()

        user_prompt = _build_user_prompt(memory)
        # Modelfile defaults рулят (num_ctx=50000, temperature=0).
        raw = ctx.llm.chat_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        try:
            parsed = _validate_importance_response(raw)
        except ValueError as exc:
            # Терминальная: контракт не выполнен, retry на той же модели
            # ничего не починит.
            raise OllamaTerminalError(f"schema_mismatch: {exc}") from exc

        if parsed.skip:
            # Validate — Legitimate LLM-skip. importance_final остаётся
            # NULL (грейс-период в scoring отрабатывает).
            return HandlerResult(
                result=parsed.to_dict(),
                skipped_by_llm=True,
            )

        assert parsed.importance_score is not None  # invariant of validator
        worker_update_importance_final(
            ctx.conn, task.memory_id, parsed.importance_score
        )
        return HandlerResult(result=parsed.to_dict())

    return handler
