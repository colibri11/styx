"""Memory-over-memory daily consolidation — port memorybox 26 (волна 22).

Pure functions + data-classes для clustering, cooldown'а, и
выбора kind/visibility итогового consolidated memory. SQL-driver
методы лежат в `storage/queries.py`. Scheduler/apply-sweeper —
`workers/sweep/memory_consolidation.py`.

См. `.design/waves/22-structural-memory-updates.md` § D4, D8, D14.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


# ── Constants (port memorybox memory-consolidation-scheduler.ts) ──────

DEFAULT_TICK_S = 3600  # 1 ч
DEFAULT_COOLDOWN_HOURS = 23
DEFAULT_WINDOW_DAYS = 7
DEFAULT_WINDOW_TAIL_HOURS = 24
DEFAULT_CLUSTER_COSINE = 0.88
DEFAULT_CLUSTER_MIN_SIZE = 3
DEFAULT_CLUSTER_MAX_SIZE = 8
DEFAULT_APPLY_TICK_S = 30


# ── Config ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryConsolidationConfig:
    """Параметры memory-over-memory consolidation. Дефолты — memorybox."""

    enabled: bool = True
    tick_s: float = float(DEFAULT_TICK_S)
    apply_tick_s: float = float(DEFAULT_APPLY_TICK_S)
    cooldown_hours: int = DEFAULT_COOLDOWN_HOURS
    window_days: int = DEFAULT_WINDOW_DAYS
    window_tail_hours: int = DEFAULT_WINDOW_TAIL_HOURS
    cosine_threshold: float = DEFAULT_CLUSTER_COSINE
    min_cluster_size: int = DEFAULT_CLUSTER_MIN_SIZE
    max_cluster_size: int = DEFAULT_CLUSTER_MAX_SIZE


# ── Data ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterCandidate:
    """Item для clustering — id + распарсенный embedding."""

    id: uuid.UUID
    embedding: list[float]


@dataclass(frozen=True)
class Cluster:
    """Результат clustering — список member_ids кластера."""

    member_ids: list[uuid.UUID]


# ── Pure: cosine ──────────────────────────────────────────────────────


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Возвращает 0 на пустых/неравных длинах
    (matches memorybox `buildClusters` cosine fallback)."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        mag_a += a[i] * a[i]
        mag_b += b[i] * b[i]
    denom = math.sqrt(mag_a) * math.sqrt(mag_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


# ── Pure: build_clusters ──────────────────────────────────────────────


def build_clusters(
    items: list[ClusterCandidate],
    *,
    cosine_threshold: float = DEFAULT_CLUSTER_COSINE,
    min_size: int = DEFAULT_CLUSTER_MIN_SIZE,
    max_size: int = DEFAULT_CLUSTER_MAX_SIZE,
) -> list[Cluster]:
    """Greedy clustering. Buklv'ный port memorybox `buildClusters`.

    Алгоритм:
    - Сортировка items уже сделана caller'ом (по `created_at ASC`).
    - taken: set[id] — занятые.
    - Для каждого seed (если не в taken): cluster = [seed]; taken += seed.
      Прогоняем кандидатов в порядке списка; cosine(seed, cand) ≥
      threshold и cluster < max_size — добавляем cand; taken += cand.
    - Если cluster ≥ min_size — публикуем; иначе бросаем (memorybox:
      members навсегда уходят в taken даже если кластер не дотянул;
      этот wave-doc D4 закрепил parity).

    Returns: список Cluster в порядке seed'ов.
    """
    clusters: list[Cluster] = []
    taken: set[uuid.UUID] = set()

    for seed in items:
        if seed.id in taken:
            continue
        cluster_members: list[uuid.UUID] = [seed.id]
        taken.add(seed.id)
        for cand in items:
            if cand.id in taken:
                continue
            if len(cluster_members) >= max_size:
                break
            sim = cosine(seed.embedding, cand.embedding)
            if sim >= cosine_threshold:
                cluster_members.append(cand.id)
                taken.add(cand.id)
        if len(cluster_members) >= min_size:
            clusters.append(Cluster(member_ids=cluster_members))

    return clusters


# ── Pure: cooldown_elapsed ────────────────────────────────────────────


def cooldown_elapsed(
    state: dict | None,
    now: datetime,
    *,
    hours: int = DEFAULT_COOLDOWN_HOURS,
) -> bool:
    """True если scheduler можно запускать (cooldown прошёл / нет state'а).

    state shape: {"last_run_at": "ISO timestamp", ...}.
    None / отсутствие last_run_at / parse error → True (первый запуск
    или corruption — лучше попробовать).
    `(now - last).hours >= cooldown_hours` → True.
    """
    if state is None:
        return True
    last_iso = state.get("last_run_at")
    if not isinstance(last_iso, str) or not last_iso:
        return True
    try:
        last = datetime.fromisoformat(last_iso)
    except (ValueError, TypeError):
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    elapsed_hours = (now - last).total_seconds() / 3600.0
    return elapsed_hours >= hours


# ── Pure: pick_consolidated_kind ──────────────────────────────────────


_KIND_PRIORITY = ("concept", "note", "fact", "decision", "episode")
"""Приоритет на ties — buklv'ный port memorybox `pickConsolidatedKind`."""


def pick_consolidated_kind(source_kinds: list[str]) -> str:
    """Majority vote по `kind`. Ties по priority. Default 'note'.

    memorybox `consolidation/memory-daily-apply.ts:pickConsolidatedKind`.
    """
    if not source_kinds:
        return "note"

    counts: dict[str, int] = {}
    for k in source_kinds:
        counts[k] = counts.get(k, 0) + 1
    max_count = max(counts.values())
    candidates = [k for k, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]

    # Tie — sort по priority list; non-priority candidates идут last.
    def _priority_key(kind: str) -> int:
        try:
            return _KIND_PRIORITY.index(kind)
        except ValueError:
            return len(_KIND_PRIORITY)

    candidates.sort(key=_priority_key)
    return candidates[0]


# ── Pure: pick_consolidated_visibility ────────────────────────────────


def pick_consolidated_visibility(source_visibility: list[str]) -> str:
    """Conservative: один private → результат private. Иначе shared.

    memorybox `consolidation/memory-daily-apply.ts:pickConsolidatedVisibility`.
    Hardening (D15): NULL/non-string трактуем как 'shared'.
    """
    for v in source_visibility:
        if v == "private":
            return "private"
    return "shared"
