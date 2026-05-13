"""Selective gatekeeper — port memorybox `selective.ts` (волна 17).

Решает на каждом subjective write что делать с новой memory относительно
её ближайших соседей в `memories`:

- ``store``     — без кандидатов или semantic-сходство ниже supersede
                  threshold'а; новая memory остаётся как есть.
- ``merge``     — semantic-сходство ≥ merge_threshold (0.92); existing
                  memory обновляется (если новый content длиннее) и
                  поглощает все relations с new_id; new удаляется.
- ``supersede`` — supersede_threshold ≤ similarity < merge_threshold и
                  Levenshtein ratio > levenshtein_threshold (текст
                  «то же самое другими словами»); existing получает
                  ``superseded_by = new_id``, INSERT'ится relation
                  ``supersedes``.
- ``skip``      — content короче ``noise_min_length`` и noise filter
                  включён; new memory целиком удаляется.

`decide(...)` — pure function, без I/O. Apply (UPDATE/DELETE/relations
maintenance) делается через `AgentScopedQueries.apply_gatekeeper_*`
методы — они не делают commit, caller (handler/route) обвёртывает всю
sequence в одну транзакцию.

Levenshtein helpers — two-row DP, O(m*n). Для content ≤ 2400 (CHECK
constraint) ~10-20 ms на CPython.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Action(str, Enum):
    """Решение gatekeeper'а для одной записи."""

    STORE = "store"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    SKIP = "skip"


@dataclass(frozen=True)
class GatekeeperConfig:
    """Параметры gatekeeper'а. Дефолты — из memorybox `selective.ts`."""

    enabled: bool = True
    merge_threshold: float = 0.92
    supersede_threshold: float = 0.85
    levenshtein_threshold: float = 0.3
    noise_filter: bool = True
    noise_min_length: int = 10


@dataclass(frozen=True)
class Candidate:
    """Сосед из ``memories`` для рассмотрения gatekeeper'ом.

    ``cosine_distance`` — pgvector ``<=>`` оператор (0 = identical,
    2 = opposite). similarity = 1 - cosine_distance.
    """

    id: uuid.UUID
    content: str
    cosine_distance: float


@dataclass(frozen=True)
class Decision:
    """Решение gatekeeper'а + контекст для логирования."""

    action: Action
    existing_id: uuid.UUID | None = None
    similarity: float | None = None
    levenshtein_ratio: float | None = None


def decide(
    content: str,
    candidates: Sequence[Candidate],
    *,
    config: GatekeeperConfig,
) -> Decision:
    """Чистая функция: content + candidates + config → Decision.

    Caller отвечает за фильтрацию candidates по
    ``cosine_distance < 1 - config.supersede_threshold`` (это делается на
    SQL-уровне в ``find_gatekeeper_candidates``). Если в ``candidates``
    попадает что-то выходящее за supersede-зону — всё равно работаем
    корректно, просто такие соседи не «пройдут» в merge/supersede ветки.

    Disabled config / noise — короткозамыкания до проверки соседей.
    """
    if not config.enabled:
        return Decision(action=Action.STORE)

    if config.noise_filter and len(content) < config.noise_min_length:
        return Decision(action=Action.SKIP)

    if not candidates:
        return Decision(action=Action.STORE)

    # candidates приходят отсортированными ORDER BY cosine_distance ASC,
    # created_at ASC (D6) — самый близкий + старший при tie. Берём top-1.
    best = candidates[0]
    similarity = 1.0 - best.cosine_distance

    if similarity > config.merge_threshold:
        return Decision(
            action=Action.MERGE,
            existing_id=best.id,
            similarity=similarity,
        )

    ratio = _levenshtein_ratio(content, best.content)
    if similarity >= config.supersede_threshold and ratio > config.levenshtein_threshold:
        return Decision(
            action=Action.SUPERSEDE,
            existing_id=best.id,
            similarity=similarity,
            levenshtein_ratio=ratio,
        )

    return Decision(
        action=Action.STORE,
        similarity=similarity,
        levenshtein_ratio=ratio,
    )


def _levenshtein_distance(a: str, b: str) -> int:
    """Edit distance, two-row DP. O(m*n) time, O(min(m,n)) space.

    Семантика — port memorybox `levenshteinDistance` (selective.ts:21).
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _levenshtein_ratio(a: str, b: str) -> float:
    """1.0 = identical strings, 0.0 = completely different.

    ``ratio = 1 - editDistance / max(len(a), len(b))``. Port memorybox
    `levenshteinRatio` (selective.ts:54).
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein_distance(a, b) / max_len
