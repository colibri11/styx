"""Search archive orchestrator (волна 20).

Pull-канал к архиву документов и реплик agent'а. FTS+vector hybrid
query поверх `documents`/`chunks` (миграция 0005) и `memories WHERE
role IN ('user','assistant')` (Styx-семантика реплик).

Четыре scope'а:

- `documents` — chunks search'нуты, затем grouped по `document_id` и
  stitched в regions (engine.stitch). Региону достаётся `score = max`
  across stitched chunks. Use case: caller хочет крупные куски с
  context'ом.
- `chunks` — chunks search'нуты, без stitching'а. Каждый hit —
  индивидуальный chunk. Use case: точечные snippet'ы для citation.
- `dialogue` — search поверх `memories` с фильтром
  `role IN ('user','assistant')`.
- `all` — fair-share interleave между `documents` и `dialogue`
  (НЕ `chunks`). Memorybox-стиль `Promise.all` + alternating merge
  `[doc_0, dlg_0, doc_1, dlg_1, ...]`, slice `limit`. У Styx sync —
  два последовательных SQL.

Pure orchestrator: queries делегируются ``AgentScopedQueries``, embed
— через ``_Embedder`` protocol. Никаких изменений в транзакции
(search_archive — read-only).

См. `.design/waves/20-search-archive.md`.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from styx.engine.stitch import StitchChunk, stitch_chunks

if TYPE_CHECKING:
    from styx.storage.queries import AgentScopedQueries, ChunkHit, DialogueHit


SCOPE_DEFAULT = "all"
SNIPPET_CHARS = 300
"""Длина snippet'а — first SNIPPET_CHARS chars текста (без highlight)."""


@dataclass(frozen=True)
class SearchArchiveConfig:
    """Конфиг search archive (волна 20). Все 4 поля управляются через
    `STYX_SEARCH_ARCHIVE_*` ENV; getter — `StyxConfig.search_archive_config()`.

    `k_candidates_factor` × `limit` (clamp by `k_candidates_min`) —
    сколько raw chunks тянуть для documents-scope, чтобы у stitching'а
    были соседи. Port memorybox `Math.min(limit * 8, 80)`.
    """

    default_limit: int = 10
    max_limit: int = 50
    k_candidates_factor: int = 8
    k_candidates_min: int = 80


@dataclass(frozen=True)
class SearchArchiveResult:
    """Heterogeneous union одного результата.

    `scope` — discriminator. Каждый scope использует только relevant поля
    (остальные — None). См. `.design/waves/20-search-archive.md` § «D11
    Response shape».
    """

    scope: Literal["document", "chunk", "dialogue"]
    text: str
    snippet: str
    score: float
    document_id: str | None = None
    chunk_position: int | None = None
    chunk_positions: tuple[int, ...] | None = None
    char_start: int | None = None
    char_end: int | None = None
    memory_id: str | None = None
    role: str | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class SearchArchiveResponse:
    results: list[SearchArchiveResult] = field(default_factory=list)
    total_matched: int = 0


class _Embedder(Protocol):
    """Минимальный sync-protocol embed-клиента (см. styx.embedding)."""

    def embed(self, text: str) -> list[float]: ...


def _clamp_limit(limit: int | None, config: SearchArchiveConfig) -> int:
    if limit is None or limit <= 0:
        return config.default_limit
    return min(limit, config.max_limit)


def _snippet(text: str) -> str:
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS]


def _hit_to_chunk_result(hit: "ChunkHit") -> SearchArchiveResult:
    return SearchArchiveResult(
        scope="chunk",
        text=hit.content,
        snippet=_snippet(hit.content),
        score=hit.score,
        document_id=str(hit.document_id),
        chunk_position=hit.position,
        char_start=hit.char_start,
        char_end=hit.char_end,
    )


def _hit_to_dialogue_result(hit: "DialogueHit") -> SearchArchiveResult:
    created_iso: str | None
    if isinstance(hit.created_at, _dt.datetime):
        created_iso = hit.created_at.isoformat()
    else:
        created_iso = str(hit.created_at) if hit.created_at is not None else None
    return SearchArchiveResult(
        scope="dialogue",
        text=hit.content,
        snippet=_snippet(hit.content),
        score=hit.score,
        memory_id=str(hit.memory_id),
        role=hit.role,
        created_at=created_iso,
    )


def search_chunks(
    *,
    queries: "AgentScopedQueries",
    embedder: _Embedder,
    query: str,
    limit: int | None,
    date_from: _dt.datetime | None = None,
    date_to: _dt.datetime | None = None,
    snapshot_cycle_start: _dt.datetime | None = None,
    config: SearchArchiveConfig | None = None,
) -> SearchArchiveResponse:
    """Top-K raw chunks без stitching'а. Каждый hit — один chunk."""
    cfg = config or SearchArchiveConfig()
    if not query.strip():
        return SearchArchiveResponse()
    eff_limit = _clamp_limit(limit, cfg)
    qvec = embedder.embed(query)

    hits = queries.search_chunks_for_archive(
        query_vector=qvec,
        query_text=query,
        limit=eff_limit,
        date_from=date_from,
        date_to=date_to,
        snapshot_cycle_start=snapshot_cycle_start,
    )
    results = [_hit_to_chunk_result(h) for h in hits]
    return SearchArchiveResponse(results=results, total_matched=len(results))


def search_documents(
    *,
    queries: "AgentScopedQueries",
    embedder: _Embedder,
    query: str,
    limit: int | None,
    date_from: _dt.datetime | None = None,
    date_to: _dt.datetime | None = None,
    snapshot_cycle_start: _dt.datetime | None = None,
    config: SearchArchiveConfig | None = None,
) -> SearchArchiveResponse:
    """Stitched regions per-document. Группируем raw chunks по
    `document_id` (порядок stable — sorted by position), stitch'им,
    sort by region.score DESC, slice `limit`."""
    cfg = config or SearchArchiveConfig()
    if not query.strip():
        return SearchArchiveResponse()
    eff_limit = _clamp_limit(limit, cfg)

    # K_candidates = limit * factor (clamped к k_candidates_min, чтобы
    # stitching имел достаточно соседей даже при limit=1). Port memorybox
    # `Math.min(limit * 8, 80)`.
    k_candidates = max(eff_limit * cfg.k_candidates_factor, cfg.k_candidates_min)

    qvec = embedder.embed(query)
    raw = queries.search_chunks_for_archive(
        query_vector=qvec,
        query_text=query,
        limit=k_candidates,
        date_from=date_from,
        date_to=date_to,
        snapshot_cycle_start=snapshot_cycle_start,
    )
    if not raw:
        return SearchArchiveResponse()

    by_doc: dict[str, list] = {}
    for h in raw:
        key = str(h.document_id)
        by_doc.setdefault(key, []).append(h)

    regions_all: list[SearchArchiveResult] = []
    for doc_id_str, hits in by_doc.items():
        hits.sort(key=lambda h: h.position)
        stitch_input = [
            StitchChunk(
                position=h.position,
                content=h.content,
                char_start=h.char_start,
                char_end=h.char_end,
                score=h.score,
            )
            for h in hits
        ]
        regions = stitch_chunks(hits[0].document_id, stitch_input)
        for region in regions:
            regions_all.append(
                SearchArchiveResult(
                    scope="document",
                    text=region.text,
                    snippet=_snippet(region.text),
                    score=region.score,
                    document_id=doc_id_str,
                    chunk_positions=region.chunk_positions,
                    char_start=region.char_start,
                    char_end=region.char_end,
                )
            )

    regions_all.sort(key=lambda r: r.score, reverse=True)
    sliced = regions_all[:eff_limit]
    return SearchArchiveResponse(results=sliced, total_matched=len(sliced))


def search_dialogue(
    *,
    queries: "AgentScopedQueries",
    embedder: _Embedder,
    query: str,
    limit: int | None,
    date_from: _dt.datetime | None = None,
    date_to: _dt.datetime | None = None,
    snapshot_cycle_start: _dt.datetime | None = None,
    config: SearchArchiveConfig | None = None,
) -> SearchArchiveResponse:
    """Top-K реплик из `memories WHERE role IN ('user','assistant')`."""
    cfg = config or SearchArchiveConfig()
    if not query.strip():
        return SearchArchiveResponse()
    eff_limit = _clamp_limit(limit, cfg)
    qvec = embedder.embed(query)

    hits = queries.search_dialogue_for_archive(
        query_vector=qvec,
        query_text=query,
        limit=eff_limit,
        date_from=date_from,
        date_to=date_to,
        snapshot_cycle_start=snapshot_cycle_start,
    )
    results = [_hit_to_dialogue_result(h) for h in hits]
    return SearchArchiveResponse(results=results, total_matched=len(results))


def search_all(
    *,
    queries: "AgentScopedQueries",
    embedder: _Embedder,
    query: str,
    limit: int | None,
    date_from: _dt.datetime | None = None,
    date_to: _dt.datetime | None = None,
    snapshot_cycle_start: _dt.datetime | None = None,
    config: SearchArchiveConfig | None = None,
) -> SearchArchiveResponse:
    """Fair-share interleave docs+dialogue. Memorybox-стиль alternating
    merge `[doc_0, dlg_0, doc_1, dlg_1, ...]`, slice `limit`.

    `chunks` НЕ участвует в `all` (D5/D8 в wave-doc'е) — specialized
    scope для citation, смешивание с stitched regions даёт visual
    duplication."""
    cfg = config or SearchArchiveConfig()
    if not query.strip():
        return SearchArchiveResponse()
    eff_limit = _clamp_limit(limit, cfg)

    # ceil так что limit=1 всё равно тянет 1 из каждого канала
    # (interleave даёт docs[0] первым, но dialogue не теряется
    # полностью на следующих итерациях).
    half = math.ceil(eff_limit / 2)

    docs = search_documents(
        queries=queries, embedder=embedder, query=query, limit=half,
        date_from=date_from, date_to=date_to,
        snapshot_cycle_start=snapshot_cycle_start, config=cfg,
    )
    dlgs = search_dialogue(
        queries=queries, embedder=embedder, query=query, limit=half,
        date_from=date_from, date_to=date_to,
        snapshot_cycle_start=snapshot_cycle_start, config=cfg,
    )

    merged: list[SearchArchiveResult] = []
    n = max(len(docs.results), len(dlgs.results))
    for i in range(n):
        if i < len(docs.results):
            merged.append(docs.results[i])
        if i < len(dlgs.results):
            merged.append(dlgs.results[i])
    sliced = merged[:eff_limit]
    return SearchArchiveResponse(results=sliced, total_matched=len(sliced))


__all__ = [
    "SCOPE_DEFAULT",
    "SearchArchiveConfig",
    "SearchArchiveResponse",
    "SearchArchiveResult",
    "search_all",
    "search_chunks",
    "search_dialogue",
    "search_documents",
]
