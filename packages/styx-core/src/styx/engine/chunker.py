"""Hierarchical text chunker для store-routing (волна 19).

Port memorybox `chunker.ts` 1:1 на Python. Делит длинный текст на
overlapping chunks по иерархии разделителей:

1. Paragraph boundary (``\\n\\n``)
2. Sentence boundary (``. `` или ``.\\n``)
3. Hard split на size limit

Затем applies overlap — каждый chunk (кроме первого) расширяется на
``overlap`` chars назад, не выходя за начало предыдущего chunk'а.

Возвращает ``ChunkData[]`` с UTF-8 byte offsets в исходный текст —
для будущего stitching'а (волна 20 search archive) и audit'а.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_SIZE = 1600
"""Дефолт chunk size в chars (~400 tokens на embeddinggemma)."""

DEFAULT_OVERLAP = 320
"""Дефолт overlap в chars (~80 tokens)."""


@dataclass(frozen=True)
class ChunkData:
    """Результат chunker'а: text + UTF-8 byte offsets в оригинал.

    ``char_start`` / ``char_end`` — byte-offsets для совместимости с
    memorybox ``ChunkData.start_offset`` / ``end_offset``. Используются
    при stitching'е (волна 20) — соседние chunks одного document'а
    склеиваются в один region, overlap де-дуплицируется.
    """

    text: str
    char_start: int
    char_end: int


_SENTENCE_BOUNDARY_RE = re.compile(r"\.\s")


def chunk_text(
    text: str,
    *,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkData]:
    """Иерархический split текста на chunks.

    Empty / whitespace-only → ``[]``.

    Алгоритм:

    1. Split по ``\\n\\n`` (`_split_paragraphs`).
    2. Greedy merge параграфов до ``size`` (`_merge_paragraphs`).
    3. Если merged segment > size — sentence split + hard split
       при необходимости (`_split_long_segment`).
    4. Build chunks с overlap'ом (`_build_chunks_with_overlap`).

    Аргументы зеркалят TS-сигнатуру: keyword-only ``size`` и
    ``overlap``. Дефолты — ``DEFAULT_SIZE`` (1600) /
    ``DEFAULT_OVERLAP`` (320).
    """
    if not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    raw_segments = _merge_paragraphs(paragraphs, size)

    segments: list[str] = []
    for seg in raw_segments:
        if len(seg) <= size:
            segments.append(seg)
        else:
            segments.extend(_split_long_segment(seg, size))

    return _build_chunks_with_overlap(text, segments, overlap)


def _split_paragraphs(text: str) -> list[str]:
    """Split по ``\\n\\n`` без сохранения сепаратора.

    Возвращает массив raw-параграфов в исходном порядке. Конкатенация
    через ``\\n\\n`` восстанавливает оригинал (без trailing/leading
    edge-cases).
    """
    return text.split("\n\n")


def _merge_paragraphs(paragraphs: list[str], size_limit: int) -> list[str]:
    """Greedy merge параграфов в segments не больше ``size_limit``.

    Каждый segment склеивается через ``\\n\\n`` (восстанавливая
    разделители). Если параграф один больше ``size_limit`` — он
    становится segment'ом as-is; дальнейший split — задача
    ``_split_long_segment``.
    """
    segments: list[str] = []
    current = ""

    for para in paragraphs:
        if current == "":
            current = para
            continue
        merged = current + "\n\n" + para
        if len(merged) <= size_limit:
            current = merged
        else:
            segments.append(current)
            current = para

    if current != "":
        segments.append(current)

    return segments


def _split_long_segment(text: str, size_limit: int) -> list[str]:
    """Sentence-split → hard-split fallback при превышении size.

    Используется когда merged segment всё ещё > size_limit (один
    параграф длинный). Ищет границы предложений (``. `` / ``.\\n``);
    если предложение само > size_limit — дёрнет ``_hard_split``.
    """
    sentences = _split_sentences(text)
    segments: list[str] = []
    current = ""

    for sent in sentences:
        if len(sent) > size_limit:
            if current != "":
                segments.append(current)
                current = ""
            segments.extend(_hard_split(sent, size_limit))
            continue
        if current == "":
            current = sent
            continue
        merged = current + sent
        if len(merged) <= size_limit:
            current = merged
        else:
            segments.append(current)
            current = sent

    if current != "":
        segments.append(current)

    return segments


def _split_sentences(text: str) -> list[str]:
    """Split на границы предложений ``. `` / ``.\\n``.

    Trailing-сепаратор остаётся в предыдущем sentence (concatenation
    воссоздаёт оригинал byte-for-byte).
    """
    parts: list[str] = []
    remaining = text

    while remaining:
        match = _SENTENCE_BOUNDARY_RE.search(remaining)
        if match is None:
            parts.append(remaining)
            break
        end = match.end()
        parts.append(remaining[:end])
        remaining = remaining[end:]

    return parts


def _hard_split(text: str, size_limit: int) -> list[str]:
    """Грубый split по фиксированному char-limit'у.

    Используется только когда одно предложение > size_limit (длинная
    строка без пробелов / very long word).
    """
    parts: list[str] = []
    i = 0
    while i < len(text):
        parts.append(text[i : i + size_limit])
        i += size_limit
    return parts


def _build_chunks_with_overlap(
    original_text: str,
    segments: list[str],
    overlap: int,
) -> list[ChunkData]:
    """Сопоставляет segments позиции в original_text, applies overlap.

    Каждый segment ищется через ``str.find`` начиная с конца предыдущего
    (segments идут sequentially, без пересечений). Overlap расширяет
    ``char_start`` каждого chunk'а (кроме первого) на ``overlap``
    chars назад — но не дальше начала предыдущего chunk'а.

    UTF-8 byte offsets через ``len(text[:i].encode('utf-8'))`` — для
    cyrillic / emoji bytes != codepoints. Совпадает с
    ``Buffer.byteLength(text.slice(0, i), 'utf-8')`` в TS.
    """
    if not segments:
        return []

    positions: list[tuple[int, int]] = []
    search_from = 0

    for seg in segments:
        idx = original_text.find(seg, search_from)
        if idx == -1:
            start = search_from
            positions.append((start, start + len(seg)))
            search_from = start + len(seg)
        else:
            positions.append((idx, idx + len(seg)))
            search_from = idx + len(seg)

    chunks: list[ChunkData] = []

    for i, (char_start, char_end) in enumerate(positions):
        if i > 0 and overlap > 0:
            prev_start, _ = positions[i - 1]
            char_start = max(char_start - overlap, prev_start)

        chunk_text_value = original_text[char_start:char_end]
        byte_start = len(original_text[:char_start].encode("utf-8"))
        byte_end = byte_start + len(chunk_text_value.encode("utf-8"))

        chunks.append(
            ChunkData(
                text=chunk_text_value,
                char_start=byte_start,
                char_end=byte_end,
            )
        )

    return chunks
