"""Reinterpret — port memorybox `reinterpret/blend.ts` + `cooldown.ts` (волна 22).

Reinterpret — explicit caller-side tool: агент явно переосмысляет
существующую memory, добавляет координату смысла, не переписывая
историю. memory_id сохраняется, граф цел.

Этот модуль — pure-функции и data-классы, без I/O. SQL-driver методы
лежат в `storage/queries.py`. Apply под write-gate'ом
(`turn_state.is_active(agent_id) == False`) делает
`workers/sweep/reinterpret_apply.py` после того как LLM handler
посчитает merged_text + merged_embedding.

См. `.design/waves/22-structural-memory-updates.md` § D6, D7.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_BLEND_WEIGHT = 0.5
"""Default weight нового vector'а при blend'е. memorybox
`reinterpret/blend.ts:DEFAULT_BLEND_WEIGHT`."""

REINTERPRET_COOLDOWN_S = 24 * 3600
"""Минимальный интервал между reinterpret'ами одной memory (24h).
memorybox `reinterpret/cooldown.ts:REINTERPRET_COOLDOWN_MS`."""


# ── BlendError ────────────────────────────────────────────────────────


class BlendError(Exception):
    """Pure-function blend embeddings ошибки."""

    __slots__ = ("code",)

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ── blend_embeddings (pure) ───────────────────────────────────────────


def blend_embeddings(
    prev: list[float],
    next_: list[float],
    *,
    weight: float = DEFAULT_BLEND_WEIGHT,
) -> list[float]:
    """Blend two embeddings: ``(1-w)*prev + w*next``, then L2-normalise.

    Pure function. Raises BlendError на:
    - ``invalid_weight`` — w вне [0, 1] или не finite;
    - ``empty_vector`` — len(prev)==0 или len(next)==0;
    - ``dim_mismatch`` — len(prev) != len(next);
    - ``zero_result`` — взвешенная сумма коллапсирует в нулевой вектор
      (антиподы при w=0.5 — теоретический edge case).

    L2-нормализация явная: pgvector хранит ненормализованные вектора,
    но cosine similarity (`<=>`) нормирует под капотом — линейная
    комбинация двух нормализованных входов сокращает длину, и cosine
    до результата искажается. Нормируем на единичную сферу.
    """
    if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise BlendError(
            "invalid_weight",
            f"weight должен быть в [0, 1], получено {weight!r}",
        )
    if len(prev) == 0 or len(next_) == 0:
        raise BlendError(
            "empty_vector",
            f"empty embedding: prev.length={len(prev)} next.length={len(next_)}",
        )
    if len(prev) != len(next_):
        raise BlendError(
            "dim_mismatch",
            f"embedding dim mismatch: prev={len(prev)} next={len(next_)}",
        )

    n = len(prev)
    out = [0.0] * n
    sq = 0.0
    for i in range(n):
        v = (1.0 - weight) * prev[i] + weight * next_[i]
        out[i] = v
        sq += v * v

    if sq == 0.0:
        raise BlendError(
            "zero_result",
            "blended vector — нулевой (prev и next антиподы при w=0.5)",
        )

    norm = math.sqrt(sq)
    return [v / norm for v in out]


# ── Cooldown ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CooldownCheck:
    """Результат `reinterpret_cooldown`. Discriminated union по `ok`."""

    ok: bool
    # ok=False / reason='recent'
    last_at: datetime | None = None
    next_at: datetime | None = None
    # ok=False / reason='pending'
    pending_application_id: int | None = None

    @property
    def reason(self) -> str | None:
        if self.ok:
            return None
        if self.pending_application_id is not None:
            return "pending"
        return "recent"

    @classmethod
    def make_ok(cls) -> "CooldownCheck":
        return cls(ok=True)

    @classmethod
    def make_recent(
        cls, *, last_at: datetime, next_at: datetime
    ) -> "CooldownCheck":
        return cls(ok=False, last_at=last_at, next_at=next_at)

    @classmethod
    def make_pending(cls, *, pending_application_id: int) -> "CooldownCheck":
        return cls(ok=False, pending_application_id=pending_application_id)


def reinterpret_cooldown(
    queries: "AgentScopedQueries",  # noqa: F821 — forward ref
    memory_id: uuid.UUID,
    *,
    now: datetime | None = None,
    cooldown_s: int = REINTERPRET_COOLDOWN_S,
) -> CooldownCheck:
    """Можно ли reinterpret'нуть `memory_id` сейчас.

    Branch 1 (приоритет): partial-unique catch — есть ли pending_sleep
    application для memory_id. Если да → вернуть pending.

    Branch 2: 24h rate-limit — последняя revision из
    memory_reinterpretations. None → ok. Старше cooldown → ok.

    `now` — для deterministic тестов. Default UTC now.
    """
    now_dt = now if now is not None else datetime.now(tz=timezone.utc)

    pending_id = queries.find_pending_reinterpret_application(memory_id)
    if pending_id is not None:
        return CooldownCheck.make_pending(pending_application_id=pending_id)

    last = queries.latest_reinterpretation_at(memory_id)
    if last is None:
        return CooldownCheck.make_ok()

    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)

    elapsed = (now_dt - last).total_seconds()
    if elapsed >= cooldown_s:
        return CooldownCheck.make_ok()

    return CooldownCheck.make_recent(
        last_at=last,
        next_at=last + timedelta(seconds=cooldown_s),
    )


# ── Config ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReinterpretConfig:
    """Параметры reinterpret-фичи. Дефолты из memorybox."""

    enabled: bool = True
    apply_tick_s: float = 30.0
    cooldown_s: int = REINTERPRET_COOLDOWN_S
    blend_weight: float = DEFAULT_BLEND_WEIGHT
