"""Per-agent cache observability (волна 29 Phase E).

Принимает cache hit/miss tokens от Hermes (`extract_cache_stats(response)`
→ `{cached_tokens, creation_tokens}`) через `POST /agent/cache_stats`,
аккумулирует в module-global dict per-agent, экспонирует через
`GET /analytics`.

Cache_read = tokens read from prompt cache (high → good cache hit rate).
Cache_creation = tokens written to cache (high during cold start, then
should drop to ~0 if salient/transport invariants держатся).

Thread-safe (lock); no DB persistence — в memory only (cumulative с
момента старта daemon'а). При рестарте обнуляется. Operator-surface
metric, не часть линии `я`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

_LOCK = threading.Lock()


@dataclass
class _AgentCacheStats:
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    samples: int = 0


_PER_AGENT: dict[str, _AgentCacheStats] = {}


def record_cache_stats(
    agent_id: str,
    *,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> None:
    """Прибавляет stats к per-agent аггрегату. Tokens должны быть ≥ 0."""
    if not agent_id:
        return
    cr = max(0, int(cache_read_tokens))
    cc = max(0, int(cache_creation_tokens))
    with _LOCK:
        s = _PER_AGENT.get(agent_id)
        if s is None:
            s = _AgentCacheStats()
            _PER_AGENT[agent_id] = s
        s.cache_read_tokens += cr
        s.cache_creation_tokens += cc
        s.samples += 1


def get_cache_stats(agent_id: str) -> dict[str, int]:
    """Snapshot для analytics endpoint'а. Если агент не отправил ни
    одного sample — все нули + samples=0."""
    with _LOCK:
        s = _PER_AGENT.get(agent_id)
        if s is None:
            return {
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "samples": 0,
            }
        return {
            "cache_read_tokens": s.cache_read_tokens,
            "cache_creation_tokens": s.cache_creation_tokens,
            "samples": s.samples,
        }


def reset_cache_stats(agent_id: str | None = None) -> None:
    """Сбросить stats — для тестов; production не вызывает.
    None → reset всех агентов."""
    with _LOCK:
        if agent_id is None:
            _PER_AGENT.clear()
        else:
            _PER_AGENT.pop(agent_id, None)
