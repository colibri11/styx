"""Unit-тесты для engine.chunker (волна 19).

Покрытия:
- empty / whitespace-only → []
- text ≤ size → 1 chunk без overlap
- size+1 → 2 chunks с overlap'ом
- multi-paragraph (\\n\\n) — split по paragraph boundaries
- multi-sentence (one long paragraph) — split по sentence boundaries
- very long word без пробелов — hard split
- overlap correctness — chunk[i+1].char_start = chunk[i].char_end - overlap
- UTF-8 byte offsets — cyrillic + emoji корректно адресуются
- catastrophic edge: текст ровно size — 1 chunk без overlap'а
"""

from __future__ import annotations

import pytest

from styx.engine.chunker import (
    DEFAULT_OVERLAP,
    DEFAULT_SIZE,
    ChunkData,
    chunk_text,
)


def test_empty_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_whitespace_only_returns_no_chunks() -> None:
    assert chunk_text("   \n\n  \t  ") == []


def test_short_text_one_chunk() -> None:
    text = "Короткая строка ровно один chunk."
    chunks = chunk_text(text, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == text
    assert c.char_start == 0
    assert c.char_end == len(text.encode("utf-8"))


def test_exactly_size_one_chunk() -> None:
    text = "x" * 100
    chunks = chunk_text(text, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_size_plus_one_two_chunks_with_overlap() -> None:
    """Текст 100 + 1 chars без paragraph/sentence breaks → hard split.
    Первый chunk — char[0:100], второй — char[100:101] но с overlap'ом
    на 10 chars назад → char[90:101]."""
    text = "x" * 101
    chunks = chunk_text(text, size=100, overlap=10)
    assert len(chunks) == 2

    assert chunks[0].text == "x" * 100
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 100

    # Второй chunk: extend назад на overlap=10.
    assert chunks[1].text == "x" * 11
    assert chunks[1].char_start == 90
    assert chunks[1].char_end == 101


def test_multi_paragraph_split_on_double_newline() -> None:
    para1 = "Первый параграф со словом стол."
    para2 = "Второй параграф со словом дверь."
    para3 = "Третий параграф длинный, содержит важный смысл."
    text = f"{para1}\n\n{para2}\n\n{para3}"
    # Размер ровно под один параграф — мерж не помещает соседа.
    size = max(len(p) for p in (para1, para2, para3))
    chunks = chunk_text(text, size=size, overlap=0)
    assert len(chunks) == 3
    assert chunks[0].text == para1
    assert chunks[1].text == para2
    assert chunks[2].text == para3


def test_paragraphs_merge_when_fit() -> None:
    """Два коротких параграфа merge'нутся в один chunk если влезают."""
    text = "Параграф один.\n\nПараграф два."
    chunks = chunk_text(text, size=200, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_paragraph_splits_by_sentences() -> None:
    """Один параграф длиннее size → sentence split."""
    sentences = [
        "Первое предложение про что-то одно. ",
        "Второе предложение про что-то другое. ",
        "Третье предложение завершает мысль. ",
    ]
    text = "".join(sentences)
    # size = чуть больше одного предложения, два не помещаются.
    size = len(sentences[0]) + 2
    chunks = chunk_text(text, size=size, overlap=0)
    # Каждое предложение в свой chunk.
    assert len(chunks) == 3
    for ch, s in zip(chunks, sentences):
        assert ch.text == s


def test_very_long_word_hard_splits() -> None:
    """Одно гигантское слово без пробелов → hard split на size_limit."""
    text = "z" * 250
    chunks = chunk_text(text, size=100, overlap=0)
    assert len(chunks) == 3
    assert chunks[0].text == "z" * 100
    assert chunks[1].text == "z" * 100
    assert chunks[2].text == "z" * 50


def test_overlap_extends_each_subsequent_chunk_back() -> None:
    """char_start второго chunk'а = char_start_orig - overlap (но не
    меньше предыдущего char_start)."""
    # Длинная цепочка из A,B,C блоков фиксированной длины.
    a = "a" * 100
    b = "b" * 100
    c = "c" * 100
    text = a + b + c
    chunks = chunk_text(text, size=100, overlap=20)

    assert len(chunks) == 3
    # Chunk 0 — без overlap'а.
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 100
    # Chunk 1 — extend назад на 20.
    assert chunks[1].char_start == 80
    assert chunks[1].char_end == 200
    # Chunk 2 — extend назад на 20.
    assert chunks[2].char_start == 180
    assert chunks[2].char_end == 300


def test_overlap_clamped_to_previous_chunk_start() -> None:
    """overlap=200 при size=100 — extend не выходит за границу
    предыдущего chunk'а."""
    text = "z" * 250
    chunks = chunk_text(text, size=100, overlap=200)
    # 3 chunks: 0..100, 100..200, 200..250.
    # Overlap для chunks[1] = max(100-200, 0) = 0 (start of chunks[0]).
    # Overlap для chunks[2] = max(200-200, 100) = 100 (start of chunks[1]).
    assert chunks[1].char_start == 0
    assert chunks[2].char_start == 100


def test_utf8_byte_offsets_cyrillic() -> None:
    """Cyrillic chars — 2 bytes в UTF-8, char_start/char_end должны
    отражать байты, не codepoint'ы."""
    text = "А" * 300  # 300 codepoints, 600 bytes
    chunks = chunk_text(text, size=100, overlap=0)
    # 100 codepoints на chunk → 200 bytes.
    assert len(chunks) == 3
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 200
    assert chunks[1].char_start == 200
    assert chunks[1].char_end == 400
    assert chunks[2].char_start == 400
    assert chunks[2].char_end == 600


def test_utf8_byte_offsets_emoji() -> None:
    """Emoji (4 bytes UTF-8) — offsets корректны."""
    # 🚀 = 4 bytes UTF-8.
    rocket = "🚀"
    assert len(rocket.encode("utf-8")) == 4
    text = rocket * 60  # 60 codepoints, 240 bytes
    chunks = chunk_text(text, size=20, overlap=0)
    assert len(chunks) == 3
    # 20 codepoints на chunk → 80 bytes.
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 80
    assert chunks[1].char_start == 80
    assert chunks[1].char_end == 160
    assert chunks[2].char_start == 160
    assert chunks[2].char_end == 240


def test_concat_chunks_reproduces_text_for_size_limit_text() -> None:
    """Sanity: для текста ≤ size — text единственного chunk'а ==
    оригинал."""
    text = "Короткое единственное предложение."
    chunks = chunk_text(text, size=200, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_returned_dataclasses_are_frozen() -> None:
    """ChunkData immutable — попытка set атрибута падает."""
    text = "abc"
    chunks = chunk_text(text, size=10, overlap=0)
    with pytest.raises(Exception):
        # frozen dataclass — FrozenInstanceError на attribute set.
        chunks[0].text = "xyz"  # type: ignore[misc]


def test_default_size_and_overlap_constants() -> None:
    assert DEFAULT_SIZE == 1600
    assert DEFAULT_OVERLAP == 320
