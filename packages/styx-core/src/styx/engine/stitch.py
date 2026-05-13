"""Region stitching для search_archive (волна 20).

Port memorybox `recall/stitch.ts` 1:1 с переименованием полей под
Styx-схему (`chunk_index` → `position`, `start_offset`/`end_offset` →
`char_start`/`char_end`).

После search'а по `chunks` каждый hit — индивидуальный chunk; для
scope='documents' нужно склеить соседние chunks одного `document_id`
обратно в один region (overlap, который добавлял chunker, удаляется
по char-offset арифметике). Run of consecutive positions (i, i+1,
i+2, ...) → одна region; gap → две.

Pure function — без I/O, тестируется boundary cases без Postgres'а.

См. `.design/waves/20-search-archive.md` § «D4 Stitching».
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class StitchChunk:
    """Минимальный input-shape для stitching'а.

    `char_start` / `char_end` — UTF-8 byte offsets в `documents.content`
    (миграция 0005, NOT NULL + CHECK char_start ≤ char_end). Fallback
    на plain concat при NULL offset'ах НЕ нужен — schema гарантирует
    их присутствие.

    `score` — per-chunk hybrid score (weighted vector + ts_rank);
    region наследует max score across stitched chunks. Опц. — для
    callers, которым stitch нужен без scoring (например debug-tool
    «fetch full document»), passing 0.0 безопасно.
    """

    position: int
    content: str
    char_start: int
    char_end: int
    score: float = 0.0


@dataclass(frozen=True)
class DocumentRegion:
    """Стычковый результат — последовательный run chunks одного document'а.

    `text` — склеенный content с удалённым overlap'ом.
    `chunk_positions` — позиции исходных chunks (для debug/audit).
    `score` — max score across stitched chunks (region наследует
    лучший hit из своих составляющих).
    """

    document_id: uuid.UUID
    text: str
    char_start: int
    char_end: int
    chunk_positions: tuple[int, ...]
    score: float


def stitch_chunks(
    document_id: uuid.UUID,
    chunks: list[StitchChunk],
) -> list[DocumentRegion]:
    """Pure stitching list'а chunks одного document'а в regions.

    Caller гарантирует что chunks отсортированы по `position` ASC и
    принадлежат одному `document_id`. Эта функция не валидирует.

    Алгоритм (port memorybox stitch.ts:stitchChunks):

    1. Iterate chunks в порядке; current = первый chunk.
    2. Если next.position == prev.position + 1 — append к current'у с
       overlap removal: ``overlap = prev.char_end - next.char_start``;
       если > 0 — content = next.content[overlap:].
    3. Иначе (gap) — закрыть current region, открыть новый.

    Регионы возвращаются в порядке появления (по position).
    """
    if not chunks:
        return []

    regions: list[DocumentRegion] = []
    cur_text = chunks[0].content
    cur_char_start = chunks[0].char_start
    cur_char_end = chunks[0].char_end
    cur_positions: list[int] = [chunks[0].position]
    cur_score = chunks[0].score
    last_position = chunks[0].position
    last_char_end = chunks[0].char_end

    for ch in chunks[1:]:
        contiguous = ch.position == last_position + 1

        if not contiguous:
            regions.append(
                DocumentRegion(
                    document_id=document_id,
                    text=cur_text,
                    char_start=cur_char_start,
                    char_end=cur_char_end,
                    chunk_positions=tuple(cur_positions),
                    score=cur_score,
                )
            )
            cur_text = ch.content
            cur_char_start = ch.char_start
            cur_char_end = ch.char_end
            cur_positions = [ch.position]
            cur_score = ch.score
        else:
            overlap = last_char_end - ch.char_start
            if overlap <= 0:
                cur_text += ch.content
            elif overlap >= len(ch.content):
                pass
            else:
                cur_text += ch.content[overlap:]
            cur_char_end = ch.char_end
            cur_positions.append(ch.position)
            if ch.score > cur_score:
                cur_score = ch.score

        last_position = ch.position
        last_char_end = ch.char_end

    regions.append(
        DocumentRegion(
            document_id=document_id,
            text=cur_text,
            char_start=cur_char_start,
            char_end=cur_char_end,
            chunk_positions=tuple(cur_positions),
            score=cur_score,
        )
    )
    return regions


__all__ = [
    "DocumentRegion",
    "StitchChunk",
    "stitch_chunks",
]
