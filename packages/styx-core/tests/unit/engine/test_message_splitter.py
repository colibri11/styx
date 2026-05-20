"""Unit-тесты для engine.message_splitter (Defect-fix B).

Покрытия:
- ``split_message``: реплика ≤ лимита → один элемент; длинная режется
  на non-overlapping части ≤ part_chars; round-trip
  (конкатенация = оригинал); абзац/предложение/жёсткий рез как
  иерархия границ.
- ``reassemble_message_groups``: пересборка group-рядов в один блок;
  снятие group-маркеров; идемпотентность; non-group passthrough;
  DESC-порядок частей (recent_messages newest-first).
"""

from __future__ import annotations

from styx.engine.message_splitter import (
    DEFAULT_PART_CHARS,
    needs_split,
    reassemble_message_groups,
    split_message,
)


# ── split_message ─────────────────────────────────────────────────────


def test_short_message_not_split() -> None:
    assert split_message("короткая реплика") == ["короткая реплика"]


def test_empty_message_returns_as_is() -> None:
    assert split_message("") == [""]
    assert split_message("   ") == ["   "]


def test_needs_split_threshold() -> None:
    assert not needs_split("x" * 2000, part_chars=2000)
    assert needs_split("x" * 2001, part_chars=2000)


def test_long_message_split_into_parts_under_limit() -> None:
    text = "Параграф один.\n\nПараграф два с предложением.\n\n" * 200
    parts = split_message(text, part_chars=2000)
    assert len(parts) > 1
    assert all(len(p) <= 2000 for p in parts)


def test_split_round_trip_preserves_content() -> None:
    """Конкатенация частей восстанавливает оригинал byte-for-byte."""
    text = (
        "Первый абзац с двумя предложениями. И вот второе.\n\n"
        "Второй абзац тоже не пустой. Ещё предложение. "
    ) * 150
    parts = split_message(text, part_chars=2000)
    assert "".join(parts) == text


def test_split_by_paragraph_boundary() -> None:
    """Два абзаца, каждый чуть меньше лимита → split по \\n\\n."""
    para_a = "А" * 1500
    para_b = "Б" * 1500
    text = para_a + "\n\n" + para_b
    parts = split_message(text, part_chars=2000)
    assert len(parts) == 2
    assert "".join(parts) == text
    # Первая часть — целый абзац A с сепаратором.
    assert parts[0] == para_a + "\n\n"


def test_split_by_sentence_when_paragraph_too_long() -> None:
    """Один абзац длиннее лимита → sentence-split."""
    sent = "Это предложение. "
    text = sent * 300  # ~5100 chars, один абзац
    parts = split_message(text, part_chars=2000)
    assert len(parts) > 1
    assert all(len(p) <= 2000 for p in parts)
    assert "".join(parts) == text


def test_hard_split_for_no_boundary_text() -> None:
    """Текст без абзацев/точек/пробелов длиннее лимита → жёсткий рез."""
    text = "x" * 5000
    parts = split_message(text, part_chars=2000)
    assert len(parts) == 3
    assert [len(p) for p in parts] == [2000, 2000, 1000]
    assert "".join(parts) == text


def test_default_part_chars_is_2000() -> None:
    assert DEFAULT_PART_CHARS == 2000


# ── reassemble_message_groups ─────────────────────────────────────────


def _grp(content: str, group: str, part: int, parts: int, role: str = "user"):
    return {
        "role": role,
        "content": content,
        "metadata": {"msg_group": group, "part": part, "parts": parts},
    }


def test_reassemble_merges_group_rows() -> None:
    msgs = [
        _grp("Часть один. ", "g1", 0, 2),
        _grp("Часть два.", "g1", 1, 2),
    ]
    out = reassemble_message_groups(msgs)
    assert len(out) == 1
    assert out[0]["content"] == "Часть один. Часть два."
    assert out[0]["role"] == "user"


def test_reassemble_strips_group_markers() -> None:
    out = reassemble_message_groups(
        [_grp("a", "g1", 0, 2), _grp("b", "g1", 1, 2)]
    )
    meta = out[0]["metadata"]
    assert "msg_group" not in meta
    assert "part" not in meta
    assert "parts" not in meta


def test_reassemble_non_group_passthrough() -> None:
    msgs = [
        {"role": "user", "content": "обычная реплика", "metadata": {}},
        {"role": "assistant", "content": "ответ"},
    ]
    out = reassemble_message_groups(msgs)
    assert out == msgs


def test_reassemble_mixed_group_and_plain() -> None:
    msgs = [
        {"role": "user", "content": "вопрос", "metadata": {}},
        _grp("A", "g1", 0, 2, role="assistant"),
        _grp("B", "g1", 1, 2, role="assistant"),
        {"role": "user", "content": "ещё вопрос", "metadata": {}},
    ]
    out = reassemble_message_groups(msgs)
    assert len(out) == 3
    assert out[0]["content"] == "вопрос"
    assert out[1]["content"] == "AB"
    assert out[1]["role"] == "assistant"
    assert out[2]["content"] == "ещё вопрос"


def test_reassemble_desc_order_parts() -> None:
    """recent_messages отдаёт ряды newest-first → части группы идут
    по убыванию part; пересборка всё равно склеивает в правильном
    порядке (сортировка по part)."""
    msgs = [
        _grp("вторая. ", "g1", 1, 2),
        _grp("первая. ", "g1", 0, 2),
    ]
    out = reassemble_message_groups(msgs)
    assert len(out) == 1
    assert out[0]["content"] == "первая. вторая. "


def test_reassemble_idempotent() -> None:
    once = reassemble_message_groups([_grp("a", "g1", 0, 2), _grp("b", "g1", 1, 2)])
    twice = reassemble_message_groups(once)
    assert once == twice


def test_reassemble_two_distinct_groups() -> None:
    msgs = [
        _grp("X1", "g1", 0, 2),
        _grp("X2", "g1", 1, 2),
        _grp("Y1", "g2", 0, 2),
        _grp("Y2", "g2", 1, 2),
    ]
    out = reassemble_message_groups(msgs)
    assert len(out) == 2
    assert out[0]["content"] == "X1X2"
    assert out[1]["content"] == "Y1Y2"


def test_reassemble_ignores_malformed_group_metadata() -> None:
    """metadata с msg_group но без валидных part/parts — passthrough."""
    msgs = [
        {"role": "user", "content": "a", "metadata": {"msg_group": "g1"}},
        {"role": "user", "content": "b", "metadata": {"msg_group": "g1", "part": "x"}},
    ]
    out = reassemble_message_groups(msgs)
    assert out == msgs


def test_reassemble_warns_on_incomplete_group(caplog) -> None:
    """Q4: собрано частей меньше заявленного parts → log.warning с id
    группы и счётчиками. Поведение не меняется — возвращаем что есть."""
    import logging

    # Группа g1 заявляет parts=3, но в окне только 2 части (0 и 1).
    msgs = [
        _grp("первая. ", "g1", 0, 3),
        _grp("вторая.", "g1", 1, 3),
    ]
    with caplog.at_level(logging.WARNING, logger="styx.engine.message_splitter"):
        out = reassemble_message_groups(msgs)
    # Поведение не меняется: пересобрали что есть в один блок.
    assert len(out) == 1
    assert out[0]["content"] == "первая. вторая."
    # Диагностика выдана.
    assert any(
        "g1" in rec.message and "неполная" in rec.message
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    )


def test_reassemble_no_warning_on_complete_group(caplog) -> None:
    """Полная группа (собрано = parts) — warning не выдаётся."""
    import logging

    msgs = [_grp("a", "g1", 0, 2), _grp("b", "g1", 1, 2)]
    with caplog.at_level(logging.WARNING, logger="styx.engine.message_splitter"):
        reassemble_message_groups(msgs)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
