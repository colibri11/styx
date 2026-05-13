"""``styx reembed`` — backfill / re-embed `memories.embedding`.

Волна 7e. Утилита one-shot: при отказе Ollama hot-path оставляет
embedding=NULL, эта команда добивает их за раз. Также покрывает
сценарий смены модели через `--all`.

Не trigger'ит importance/lifecycle/classifier — чисто бэкфилл вектора
(см. wave-doc § «Что НЕ делает»).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from styx.embedding import EmbeddingClient, EmbeddingError
from styx.llm import LLMRateLimiter
from styx.storage.queries import (
    reembed_count_targets,
    reembed_iter_targets,
    reembed_update_embedding,
)

if TYPE_CHECKING:
    import psycopg

log = logging.getLogger(__name__)


REEMBED_MODE_NULL_ONLY = "null_only"
REEMBED_MODE_ALL = "all"
_VALID_MODES = (REEMBED_MODE_NULL_ONLY, REEMBED_MODE_ALL)


@dataclass(frozen=True)
class ReembedResult:
    processed: int
    failed: int
    would_process: int
    dry_run: bool


def run_reembed(
    *,
    conn: "psycopg.Connection",
    embed_client: EmbeddingClient,
    mode: str = REEMBED_MODE_NULL_ONLY,
    agent_id: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 50,
    rate_per_second: float = 5.0,
) -> ReembedResult:
    """Прогон бэкфилла. Не закрывает ``conn`` — caller-owned."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode должен быть одним из {_VALID_MODES}, получено {mode!r}")
    if batch_size < 1:
        raise ValueError("batch_size >= 1")
    if rate_per_second <= 0:
        raise ValueError("rate_per_second > 0")
    if limit is not None and limit < 0:
        raise ValueError("limit >= 0")

    null_only = mode == REEMBED_MODE_NULL_ONLY

    if dry_run:
        n = reembed_count_targets(conn, null_only=null_only, agent_id=agent_id)
        if limit is not None:
            n = min(n, limit)
        log.info("reembed dry-run: would process %d memories", n)
        return ReembedResult(processed=0, failed=0, would_process=n, dry_run=True)

    rate_limit = LLMRateLimiter(
        capacity=max(1, int(rate_per_second)),
        refill_per_second=rate_per_second,
    )

    processed = 0
    failed = 0
    started_at = time.monotonic()

    for memory_id, content in reembed_iter_targets(
        conn, null_only=null_only, agent_id=agent_id, batch_size=batch_size
    ):
        if limit is not None and processed + failed >= limit:
            break
        rate_limit.acquire()

        try:
            vec = embed_client.embed(content)
        except EmbeddingError as exc:
            failed += 1
            log.warning("reembed: embed упал для memory %s: %s", memory_id, exc)
            continue

        try:
            reembed_update_embedding(conn, memory_id, vec)
            conn.commit()
            processed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.warning(
                "reembed: UPDATE упал для memory %s: %s", memory_id, exc
            )
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            continue

        if (processed + failed) % batch_size == 0:
            log.info(
                "reembed progress: processed=%d failed=%d (%.1fs)",
                processed, failed, time.monotonic() - started_at,
            )

    elapsed = time.monotonic() - started_at
    log.info(
        "reembed finished: processed=%d failed=%d in %.1fs",
        processed, failed, elapsed,
    )
    return ReembedResult(
        processed=processed, failed=failed, would_process=0, dry_run=False
    )
