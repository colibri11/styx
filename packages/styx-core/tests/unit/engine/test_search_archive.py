"""Unit-тесты для engine.search_archive (волна 20).

Orchestrator-уровень: queries и embedder заменены fake'ами,
проверяется shape результатов, fair-share interleave для `all`,
limit clamp и k_candidates расчёт для `documents`.

Integration с реальной БД — `tests/integration/engine/test_search_archive.py`.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from styx.engine.search_archive import (
    SearchArchiveConfig,
    search_all,
    search_chunks,
    search_dialogue,
    search_documents,
)
from styx.storage.queries import ChunkHit, DialogueHit


_DOC_A = uuid.UUID("00000000-0000-0000-0000-00000000000A")
_DOC_B = uuid.UUID("00000000-0000-0000-0000-00000000000B")


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * 768


class _FakeQueries:
    """Имитация AgentScopedQueries для search_archive engine.

    Caller настраивает `chunk_returns` / `dialogue_returns` —
    последовательность ответов, по одному за вызов. Если list short —
    каждый последующий вызов возвращает [].
    """

    def __init__(
        self,
        *,
        chunk_returns: list[list[ChunkHit]] | None = None,
        dialogue_returns: list[list[DialogueHit]] | None = None,
    ) -> None:
        self._chunk_returns = chunk_returns or []
        self._dialogue_returns = dialogue_returns or []
        self.chunk_calls: list[dict] = []
        self.dialogue_calls: list[dict] = []

    def search_chunks_for_archive(self, **kwargs):
        self.chunk_calls.append(kwargs)
        if self.chunk_calls.__len__() <= len(self._chunk_returns):
            return self._chunk_returns[len(self.chunk_calls) - 1]
        return []

    def search_dialogue_for_archive(self, **kwargs):
        self.dialogue_calls.append(kwargs)
        if self.dialogue_calls.__len__() <= len(self._dialogue_returns):
            return self._dialogue_returns[len(self.dialogue_calls) - 1]
        return []


def _ch(doc: uuid.UUID, position: int, content: str, char_start: int, char_end: int, score: float) -> ChunkHit:
    return ChunkHit(
        document_id=doc,
        position=position,
        content=content,
        char_start=char_start,
        char_end=char_end,
        score=score,
    )


def _dlg(role: str, content: str, score: float, *, ts: _dt.datetime | None = None) -> DialogueHit:
    return DialogueHit(
        memory_id=uuid.uuid4(),
        role=role,
        content=content,
        created_at=ts or _dt.datetime(2026, 5, 5, 12, 0, 0),
        score=score,
    )


# ── search_chunks ─────────────────────────────────────────────────


def test_search_chunks_returns_raw_no_stitch() -> None:
    fake_q = _FakeQueries(chunk_returns=[[
        _ch(_DOC_A, 0, "alpha", 0, 5, 0.9),
        _ch(_DOC_A, 5, "beta", 100, 104, 0.6),  # gap, but no stitching
    ]])
    fake_e = _FakeEmbedder()
    resp = search_chunks(
        queries=fake_q, embedder=fake_e, query="hi", limit=5,
    )
    assert resp.total_matched == 2
    assert [r.scope for r in resp.results] == ["chunk", "chunk"]
    assert resp.results[0].chunk_position == 0
    assert resp.results[1].chunk_position == 5
    assert resp.results[0].document_id == str(_DOC_A)
    # snippet = first 300 chars (полный content для коротких).
    assert resp.results[0].snippet == "alpha"
    assert fake_e.calls == ["hi"]


def test_search_chunks_empty_query_no_call() -> None:
    fake_q = _FakeQueries()
    fake_e = _FakeEmbedder()
    resp = search_chunks(
        queries=fake_q, embedder=fake_e, query="   ", limit=5,
    )
    assert resp.results == []
    assert resp.total_matched == 0
    assert fake_e.calls == []
    assert fake_q.chunk_calls == []


def test_search_chunks_limit_clamp_to_max() -> None:
    fake_q = _FakeQueries(chunk_returns=[[]])
    fake_e = _FakeEmbedder()
    cfg = SearchArchiveConfig(default_limit=10, max_limit=15)
    search_chunks(
        queries=fake_q, embedder=fake_e, query="hi", limit=999, config=cfg,
    )
    assert fake_q.chunk_calls[0]["limit"] == 15


def test_search_chunks_limit_none_uses_default() -> None:
    fake_q = _FakeQueries(chunk_returns=[[]])
    fake_e = _FakeEmbedder()
    cfg = SearchArchiveConfig(default_limit=7, max_limit=50)
    search_chunks(
        queries=fake_q, embedder=fake_e, query="hi", limit=None, config=cfg,
    )
    assert fake_q.chunk_calls[0]["limit"] == 7


# ── search_documents ──────────────────────────────────────────────


def test_search_documents_groups_and_stitches() -> None:
    """Два chunk'а одного doc'а с contiguous positions → 1 region;
    ещё один chunk другого doc'а → отдельный region."""
    fake_q = _FakeQueries(chunk_returns=[[
        _ch(_DOC_A, 0, "hello", 0, 5, 0.5),
        _ch(_DOC_A, 1, "world", 5, 10, 0.9),
        _ch(_DOC_B, 7, "isolated", 70, 78, 0.7),
    ]])
    fake_e = _FakeEmbedder()
    resp = search_documents(
        queries=fake_q, embedder=fake_e, query="x y", limit=10,
    )
    assert resp.total_matched == 2
    # Регионы отсортированы по score DESC: doc_A merged (max 0.9) > doc_B (0.7)
    assert resp.results[0].document_id == str(_DOC_A)
    assert resp.results[0].text == "helloworld"
    assert resp.results[0].chunk_positions == (0, 1)
    assert resp.results[0].score == 0.9
    assert resp.results[0].scope == "document"
    assert resp.results[1].document_id == str(_DOC_B)
    assert resp.results[1].chunk_positions == (7,)


def test_search_documents_k_candidates_uses_factor_and_min() -> None:
    """k_candidates = max(limit*factor, k_candidates_min). limit=2, factor=8 → 16
    но clamp на k_candidates_min=80 → 80."""
    fake_q = _FakeQueries(chunk_returns=[[]])
    fake_e = _FakeEmbedder()
    cfg = SearchArchiveConfig(
        default_limit=10, max_limit=50, k_candidates_factor=8, k_candidates_min=80,
    )
    search_documents(
        queries=fake_q, embedder=fake_e, query="x", limit=2, config=cfg,
    )
    assert fake_q.chunk_calls[0]["limit"] == 80


def test_search_documents_k_candidates_factor_dominates_when_limit_high() -> None:
    """limit=20, factor=8 → 160 > min=80 → 160."""
    fake_q = _FakeQueries(chunk_returns=[[]])
    fake_e = _FakeEmbedder()
    cfg = SearchArchiveConfig(
        default_limit=10, max_limit=50, k_candidates_factor=8, k_candidates_min=80,
    )
    search_documents(
        queries=fake_q, embedder=fake_e, query="x", limit=20, config=cfg,
    )
    assert fake_q.chunk_calls[0]["limit"] == 160


# ── search_dialogue ───────────────────────────────────────────────


def test_search_dialogue_returns_dialogue_results() -> None:
    fake_q = _FakeQueries(dialogue_returns=[[
        _dlg("user", "вопрос", 0.6),
        _dlg("assistant", "ответ", 0.8),
    ]])
    fake_e = _FakeEmbedder()
    resp = search_dialogue(
        queries=fake_q, embedder=fake_e, query="x", limit=5,
    )
    assert resp.total_matched == 2
    assert all(r.scope == "dialogue" for r in resp.results)
    assert resp.results[0].role == "user"
    assert resp.results[0].text == "вопрос"
    assert resp.results[0].created_at is not None  # ISO-8601 строка


# ── search_all ────────────────────────────────────────────────────


def test_search_all_interleaves_documents_and_dialogue() -> None:
    """search_documents → 2 regions, search_dialogue → 2 hits.
    Interleave: [doc_0, dlg_0, doc_1, dlg_1]."""
    fake_q = _FakeQueries(
        chunk_returns=[[
            _ch(_DOC_A, 0, "doc-A region", 0, 12, 0.9),
            _ch(_DOC_B, 0, "doc-B region", 0, 12, 0.7),
        ]],
        dialogue_returns=[[
            _dlg("user", "user msg 1", 0.85),
            _dlg("assistant", "assistant msg 1", 0.5),
        ]],
    )
    fake_e = _FakeEmbedder()
    resp = search_all(
        queries=fake_q, embedder=fake_e, query="hi there", limit=4,
    )
    # Interleave: doc_0, dlg_0, doc_1, dlg_1
    assert resp.total_matched == 4
    scopes = [r.scope for r in resp.results]
    assert scopes == ["document", "dialogue", "document", "dialogue"]
    assert resp.results[0].text == "doc-A region"
    assert resp.results[1].text == "user msg 1"
    assert resp.results[2].text == "doc-B region"
    assert resp.results[3].text == "assistant msg 1"


def test_search_all_limit_one_takes_first_doc_only() -> None:
    """limit=1 → half=ceil(1/2)=1; merged = [doc_0, dlg_0]; slice 1 → doc_0."""
    fake_q = _FakeQueries(
        chunk_returns=[[
            _ch(_DOC_A, 0, "doc text", 0, 8, 0.6),
        ]],
        dialogue_returns=[[
            _dlg("user", "dialogue text", 0.9),
        ]],
    )
    fake_e = _FakeEmbedder()
    resp = search_all(
        queries=fake_q, embedder=fake_e, query="x", limit=1,
    )
    assert len(resp.results) == 1
    assert resp.results[0].scope == "document"


def test_search_all_one_channel_empty() -> None:
    """Если один канал пустой — alternating всё равно идёт, остальные
    подтягиваются. dlg empty → результат = [doc_0, doc_1]."""
    fake_q = _FakeQueries(
        chunk_returns=[[
            _ch(_DOC_A, 0, "first", 0, 5, 0.9),
            _ch(_DOC_B, 0, "second", 0, 6, 0.7),
        ]],
        dialogue_returns=[[]],
    )
    fake_e = _FakeEmbedder()
    resp = search_all(
        queries=fake_q, embedder=fake_e, query="x", limit=4,
    )
    assert len(resp.results) == 2
    assert all(r.scope == "document" for r in resp.results)


def test_search_propagates_snapshot_and_dates() -> None:
    """`snapshot_cycle_start`, `date_from`, `date_to` пробрасываются в queries."""
    fake_q = _FakeQueries(chunk_returns=[[]])
    fake_e = _FakeEmbedder()
    cs = _dt.datetime(2026, 5, 4, 12, 0, 0)
    df = _dt.datetime(2026, 5, 1)
    dt = _dt.datetime(2026, 5, 5)
    search_chunks(
        queries=fake_q, embedder=fake_e, query="x", limit=5,
        date_from=df, date_to=dt, snapshot_cycle_start=cs,
    )
    call = fake_q.chunk_calls[0]
    assert call["snapshot_cycle_start"] == cs
    assert call["date_from"] == df
    assert call["date_to"] == dt
