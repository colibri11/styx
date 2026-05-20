"""Unit-тесты styx.config — валидация параметров при загрузке.

Не требуют Postgres: ``config.load`` собирает StyxConfig из env/json
и валидирует инварианты до любого обращения к БД.
"""

from __future__ import annotations

import pytest

from styx import config


def test_load_rejects_split_part_chars_over_content_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m2: message_split_part_chars >= MEMORIES_CONTENT_LIMIT — fail-fast.

    Сплиттер режет реплику на части ≤ part_chars; если part_chars не
    меньше CHECK-лимита memories.content — каждый длинный turn упадёт
    с ContentTooLongError. load() должен отказать на старте, не
    молчаливо clamp'ить.
    """
    from styx.storage.queries import MEMORIES_CONTENT_LIMIT

    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.setenv(
        "STYX_MESSAGE_SPLIT_PART_CHARS", str(MEMORIES_CONTENT_LIMIT)
    )
    with pytest.raises(ValueError, match="message_split_part_chars"):
        config.load()


def test_load_rejects_split_part_chars_above_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Значение строго больше лимита — также отвергается."""
    from styx.storage.queries import MEMORIES_CONTENT_LIMIT

    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.setenv(
        "STYX_MESSAGE_SPLIT_PART_CHARS", str(MEMORIES_CONTENT_LIMIT + 500)
    )
    with pytest.raises(ValueError, match="message_split_part_chars"):
        config.load()


def test_load_accepts_split_part_chars_below_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Значение строго меньше лимита проходит — конфиг собирается."""
    from styx.storage.queries import MEMORIES_CONTENT_LIMIT

    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.setenv(
        "STYX_MESSAGE_SPLIT_PART_CHARS", str(MEMORIES_CONTENT_LIMIT - 1)
    )
    cfg = config.load()
    assert cfg.message_split_part_chars == MEMORIES_CONTENT_LIMIT - 1


def test_load_accepts_default_split_part_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Дефолт (2000) валиден — load без override не падает."""
    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.delenv("STYX_MESSAGE_SPLIT_PART_CHARS", raising=False)
    cfg = config.load()
    assert cfg.message_split_part_chars == 2000
