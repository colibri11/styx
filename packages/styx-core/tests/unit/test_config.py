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


def test_load_reads_causal_affect_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.setenv("STYX_AFFECTIVE_TRANSITION_ENABLED", "false")
    monkeypatch.setenv("STYX_AFFECTIVE_TRANSITION_TIMEOUT_S", "3.25")

    cfg = config.load()

    assert cfg.affective_transition_enabled is False
    assert cfg.affective_transition_timeout_s == pytest.approx(3.25)


def test_load_reads_bounded_cognition_reduction_and_residue_retry_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.setenv("STYX_COGNITION_REDUCTION_WAIT_S", "0.6")
    monkeypatch.setenv("STYX_ACT_RESIDUE_RETRY_TICK_S", "12.5")
    monkeypatch.setenv("STYX_ACT_RESIDUE_MAX_ATTEMPTS", "4")

    cfg = config.load()

    assert cfg.cognition_reduction_wait_s == pytest.approx(0.6)
    assert cfg.act_residue_retry_tick_s == pytest.approx(12.5)
    assert cfg.act_residue_max_attempts == 4


def test_load_clamps_cognition_reduction_wait_and_retry_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://u:p@h:5432/styx")
    monkeypatch.setenv("STYX_COGNITION_REDUCTION_WAIT_S", "99")
    monkeypatch.setenv("STYX_ACT_RESIDUE_RETRY_TICK_S", "0")
    monkeypatch.setenv("STYX_ACT_RESIDUE_MAX_ATTEMPTS", "99")

    cfg = config.load()

    assert cfg.cognition_reduction_wait_s == pytest.approx(5.0)
    assert cfg.act_residue_retry_tick_s == pytest.approx(1.0)
    assert cfg.act_residue_max_attempts == 20
