"""Hot-tier — short-lived in-process store of recently-active memory items.

Реализует поверхность waves-v1 § «Tier связи» v2: «Hot-tier с TTL и
автоматическим переходом active → hot → long при eviction».

Контракт:

- ``put_many(agent_id, hits)`` — копирует sliced ``MemoryHit`` в
  ``HotEntry``, refresh ``evicted_at``. LRU eviction при overflow
  (вытесняется минимальный ``evicted_at``). Зовётся из ``recall_full``
  после успешного filter+dedup+slice.
- ``scan_candidates(agent_id, query_vector, min_score, snapshot)`` —
  cosine match по всем не-expired entries; snapshot fence applied на
  Python; возвращает ``list[MemoryHit]`` с ``score=match_score=cosine``.
  Зовётся из ``recall_full`` до ``search_similar``.
- TTL lazy-checked при чтении (single-thread Python — background sweep
  избыточен).

Per-agent state: словарь ``agent_id → HotState``. Один core daemon
обслуживает несколько агентов параллельно.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from styx.storage.queries import MemoryHit

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from styx.turn_state import RecallSnapshot


_SUBJECTIVE_KIND_SRCS = frozenset({"subjective", "subjective_tail"})


@dataclass(frozen=True)
class HotEntry:
    """Lightweight копия MemoryHit без полей актуальных только в момент recall'а.

    ``score``/``match_score``/``recall_event_id`` намеренно не хранятся —
    устаревают (now() меняется, recall_events растут, baseline дрейфует).
    На read-path'е score пересчитывается как cosine; БД-версия побеждает
    при collision id (там composite score актуальный).
    """

    id: uuid.UUID
    agent_id: str
    kind: str
    kind_src: str
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: Any
    embedding: list[float]
    evicted_at: float


@dataclass
class HotState:
    entries: dict[uuid.UUID, HotEntry] = field(default_factory=dict)
    ttl_s: float = 300.0
    lru_bound: int = 100


_STATES: dict[str, HotState] = {}
_LOCK = threading.Lock()


def configure(agent_id: str, *, ttl_s: float = 300.0, lru_bound: int = 100) -> None:
    """Создать (или пересоздать) hot state для ``agent_id``."""
    if ttl_s <= 0:
        raise ValueError(f"ttl_s должен быть > 0, получено {ttl_s}")
    if lru_bound <= 0:
        raise ValueError(f"lru_bound должен быть > 0, получено {lru_bound}")
    with _LOCK:
        _STATES[agent_id] = HotState(ttl_s=ttl_s, lru_bound=lru_bound)


def get_state(agent_id: str) -> HotState | None:
    """Read-only доступ для тестов и /healthz."""
    return _STATES.get(agent_id)


def put_many(agent_id: str, hits: list[MemoryHit]) -> None:
    """Скопировать hits в hot. Refresh ``evicted_at`` для уже хранящихся id.

    Игнорирует hits без embedding'а (без него scan невозможен) и hits
    без kind_src (snapshot fence требует kind_src). Это no-op в обоих
    случаях — без поля item не сможет быть возвращён, незачем хранить.
    """
    s = _STATES.get(agent_id)
    if s is None or not hits:
        return
    now = time.monotonic()
    for hit in hits:
        if hit.embedding is None or hit.kind_src is None:
            continue
        s.entries[hit.id] = HotEntry(
            id=hit.id,
            agent_id=hit.agent_id,
            kind=hit.kind,
            kind_src=hit.kind_src,
            role=hit.role,
            content=hit.content,
            metadata=hit.metadata,
            created_at=hit.created_at,
            embedding=list(hit.embedding),
            evicted_at=now,
        )
    _enforce_lru(s)


def scan_candidates(
    agent_id: str,
    query_vector: list[float],
    *,
    min_score: float,
    snapshot: "RecallSnapshot | None" = None,
) -> list[MemoryHit]:
    """Cosine match по hot. Возвращает ``MemoryHit`` со score=match_score=cosine.

    Lazy TTL purge. Snapshot fence applied на Python (повторяет SQL
    snapshot_clause в ``search_similar``). Сортировка по cosine DESC —
    caller обычно дальше объединяет с БД и применяет filter+dedup+slice.
    """
    s = _STATES.get(agent_id)
    if s is None or not s.entries:
        return []

    now = time.monotonic()
    expired: list[uuid.UUID] = []
    candidates: list[MemoryHit] = []

    for entry_id, entry in s.entries.items():
        if now - entry.evicted_at > s.ttl_s:
            expired.append(entry_id)
            continue
        if not _passes_snapshot(entry, snapshot):
            continue
        cosine = _cosine(query_vector, entry.embedding)
        if cosine < min_score:
            continue
        candidates.append(_to_hit(entry, cosine))

    for eid in expired:
        del s.entries[eid]

    candidates.sort(key=lambda h: h.score, reverse=True)
    return candidates


def reset(agent_id: str) -> None:
    """Полный сброс одного агента. Вызывается из shutdown() и тестов."""
    with _LOCK:
        _STATES.pop(agent_id, None)


def reset_all() -> None:
    """Сбросить hot tier для всех агентов. Используется в daemon shutdown / тестах."""
    with _LOCK:
        _STATES.clear()


def restore(agent_id: str, entries: list[HotEntry]) -> None:
    """Записать persisted entries поверх текущего state'а (волна 13).

    Требует ``configure(agent_id, ...)`` уже вызванным — иначе no-op +
    warning. После replace применяется ``_enforce_lru`` — на случай
    если между save и restart'ом ENV ``STYX_HOT_TIER_LRU_BOUND``
    уменьшился.

    Lazy TTL purge на entries не делается — это работа
    ``scan_candidates`` на read-path'е (entries c истёкшим
    ``evicted_at`` будут отброшены при первом scan'е).
    """
    s = _STATES.get(agent_id)
    if s is None:
        log.warning(
            "hot_tier.restore: state не configured для agent_id=%s — no-op",
            agent_id,
        )
        return
    s.entries = {e.id: e for e in entries}
    _enforce_lru(s)


def snapshot(agent_id: str) -> list[HotEntry]:
    """Shallow-копия текущих entries для save (волна 13).

    Возвращает list[HotEntry]; пустой список если state не configured
    (caller — save-thread — должен это интерпретировать как «нечего
    сохранять» по hot_tier'у). HotEntry frozen, embedding/metadata
    разделяются по reference — каждый entry immutable.
    """
    s = _STATES.get(agent_id)
    if s is None:
        return []
    return list(s.entries.values())


def stats(agent_id: str) -> dict[str, int]:
    """Снимок состояния для /healthz / диагностики."""
    s = _STATES.get(agent_id)
    if s is None:
        return {"enabled": 0, "size": 0, "ttl_s": 0, "lru_bound": 0}
    return {
        "enabled": 1,
        "size": len(s.entries),
        "ttl_s": int(s.ttl_s),
        "lru_bound": s.lru_bound,
    }


def _enforce_lru(s: HotState) -> None:
    overflow = len(s.entries) - s.lru_bound
    if overflow <= 0:
        return
    by_evicted = sorted(s.entries.items(), key=lambda kv: kv[1].evicted_at)
    for entry_id, _ in by_evicted[:overflow]:
        del s.entries[entry_id]


def _passes_snapshot(entry: HotEntry, snapshot: "RecallSnapshot | None") -> bool:
    if snapshot is None:
        return True
    if entry.created_at <= snapshot.cycle_start:
        return True
    return (
        entry.kind_src in _SUBJECTIVE_KIND_SRCS
        and entry.agent_id == snapshot.agent_id
    )


def _to_hit(entry: HotEntry, cosine: float) -> MemoryHit:
    return MemoryHit(
        id=entry.id,
        agent_id=entry.agent_id,
        kind=entry.kind,
        kind_src=entry.kind_src,
        role=entry.role,
        content=entry.content,
        metadata=dict(entry.metadata),
        created_at=entry.created_at,
        score=cosine,
        match_score=cosine,
        embedding=list(entry.embedding),
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
