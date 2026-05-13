"""Unit-тесты для engine.stitch (волна 20).

Покрытия:
- empty → []
- single chunk → 1 region (text/range/score = chunk's)
- 3 contiguous chunks → 1 region (склеенный text, max score)
- gap (positions [0, 2]) → 2 regions
- overlap removal: prev.char_end > next.char_start → next prefix дропается
- overlap >= len(next.content) → next ничего не добавляет
- prev.char_end < next.char_start (gap в char-range при contiguous positions
  — теоретически не должно случаться при правильном chunker'е,
  но stitch обрабатывает gracefully — plain concat)
- chunk_positions сохраняет порядок
- region.score = max(chunks.score)
"""

from __future__ import annotations

import uuid

from styx.engine.stitch import StitchChunk, stitch_chunks


def _doc_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


def test_empty_returns_no_regions() -> None:
    assert stitch_chunks(_doc_id(), []) == []


def test_single_chunk_returns_one_region() -> None:
    doc = _doc_id()
    chunks = [StitchChunk(position=0, content="hello", char_start=0, char_end=5, score=0.7)]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    r = regions[0]
    assert r.document_id == doc
    assert r.text == "hello"
    assert r.char_start == 0
    assert r.char_end == 5
    assert r.chunk_positions == (0,)
    assert r.score == 0.7


def test_three_contiguous_chunks_no_overlap_one_region() -> None:
    """Positions [0,1,2], char ranges непересекающиеся — plain concat."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="aaa", char_start=0, char_end=3, score=0.5),
        StitchChunk(position=1, content="bbb", char_start=3, char_end=6, score=0.9),
        StitchChunk(position=2, content="ccc", char_start=6, char_end=9, score=0.4),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    r = regions[0]
    assert r.text == "aaabbbccc"
    assert r.char_start == 0
    assert r.char_end == 9
    assert r.chunk_positions == (0, 1, 2)
    assert r.score == 0.9


def test_three_contiguous_chunks_with_overlap_dedup() -> None:
    """positions [0,1,2], chunks overlap'ятся по 2 chars — overlap удалён.
    chunk0 char[0:5] = 'hello', chunk1 char[3:8] = 'lowor', chunk2
    char[6:11] = 'orld!' — overlap 2 chars между chunk0/chunk1 ('lo'),
    overlap 2 chars между chunk1/chunk2 ('or')."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="hello", char_start=0, char_end=5, score=0.3),
        StitchChunk(position=1, content="lowor", char_start=3, char_end=8, score=0.8),
        StitchChunk(position=2, content="orld!", char_start=6, char_end=11, score=0.5),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    r = regions[0]
    assert r.text == "helloworld!"
    assert r.char_start == 0
    assert r.char_end == 11
    assert r.chunk_positions == (0, 1, 2)
    assert r.score == 0.8


def test_gap_in_positions_two_regions() -> None:
    """positions [0, 2] с missing 1 → две region'ы."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="first", char_start=0, char_end=5, score=0.7),
        StitchChunk(position=2, content="third", char_start=10, char_end=15, score=0.4),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 2
    assert regions[0].text == "first"
    assert regions[0].chunk_positions == (0,)
    assert regions[1].text == "third"
    assert regions[1].chunk_positions == (2,)


def test_overlap_full_dedup() -> None:
    """next chunk целиком внутри prev (overlap >= len(next.content)) —
    second chunk не добавляет текста, но position учитывается."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="abcdef", char_start=0, char_end=6, score=0.6),
        StitchChunk(position=1, content="cd", char_start=2, char_end=4, score=0.9),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    r = regions[0]
    assert r.text == "abcdef"
    assert r.chunk_positions == (0, 1)
    assert r.score == 0.9


def test_no_overlap_at_boundary_concat() -> None:
    """prev.char_end == next.char_start — overlap=0, plain concat."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="abc", char_start=0, char_end=3, score=0.1),
        StitchChunk(position=1, content="def", char_start=3, char_end=6, score=0.2),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    assert regions[0].text == "abcdef"


def test_gap_in_char_range_but_contiguous_positions() -> None:
    """Edge: positions contiguous но char-range gap (chunker аномалия).
    Stitch не делает special handling — просто plain concat (overlap < 0)."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="abc", char_start=0, char_end=3, score=0.5),
        StitchChunk(position=1, content="xyz", char_start=10, char_end=13, score=0.5),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    assert regions[0].text == "abcxyz"
    assert regions[0].char_end == 13


def test_score_max_across_stitched() -> None:
    """region.score = max chunk.score, не sum / mean."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="a", char_start=0, char_end=1, score=0.3),
        StitchChunk(position=1, content="b", char_start=1, char_end=2, score=0.95),
        StitchChunk(position=2, content="c", char_start=2, char_end=3, score=0.6),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 1
    assert regions[0].score == 0.95


def test_two_separate_runs_with_two_gaps() -> None:
    """positions [0,1, gap, 5,6] → 2 region'ы."""
    doc = _doc_id()
    chunks = [
        StitchChunk(position=0, content="aa", char_start=0, char_end=2, score=0.4),
        StitchChunk(position=1, content="bb", char_start=2, char_end=4, score=0.4),
        StitchChunk(position=5, content="ee", char_start=20, char_end=22, score=0.7),
        StitchChunk(position=6, content="ff", char_start=22, char_end=24, score=0.6),
    ]
    regions = stitch_chunks(doc, chunks)
    assert len(regions) == 2
    assert regions[0].text == "aabb"
    assert regions[0].chunk_positions == (0, 1)
    assert regions[1].text == "eeff"
    assert regions[1].chunk_positions == (5, 6)
    assert regions[1].score == 0.7
