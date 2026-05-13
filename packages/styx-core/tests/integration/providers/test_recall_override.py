"""Прохождение recall-override из StyxConfig до StyxMemoryCore._recall_config.

Цепочка: ENV → ``styx.config.load`` → ``_build_recall_config`` →
``provider._recall_config``. Без живой БД — тестируем только что
override корректно резолвится в RecallConfig.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from styx.config import StyxConfig, load as load_config
from styx.providers.memory import StyxMemoryCore, _build_recall_config
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


# -------- StyxConfig: ENV / json --------------------------------------


def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Минимально валидный env для load() — DSN."""
    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://x/y")


def test_no_override_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_env(monkeypatch)
    monkeypatch.delenv("STYX_RECALL_MIN_SCORE", raising=False)
    monkeypatch.delenv("STYX_RECALL_DIALOGUE_MIN_SCORE", raising=False)
    cfg = load_config()
    assert cfg.recall_min_score is None
    assert cfg.recall_dialogue_min_score is None


def test_env_recall_min_score_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_env(monkeypatch)
    monkeypatch.setenv("STYX_RECALL_MIN_SCORE", "0.42")
    cfg = load_config()
    assert cfg.recall_min_score == pytest.approx(0.42)


def test_env_recall_dialogue_min_score_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_env(monkeypatch)
    monkeypatch.setenv("STYX_RECALL_DIALOGUE_MIN_SCORE", "0.55")
    cfg = load_config()
    assert cfg.recall_dialogue_min_score == pytest.approx(0.55)


def test_json_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_env(monkeypatch)
    monkeypatch.delenv("STYX_RECALL_MIN_SCORE", raising=False)
    (tmp_path / "styx.json").write_text(
        json.dumps({"database_url": "postgresql://x/y", "recall_min_score": 0.33}),
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.recall_min_score == pytest.approx(0.33)


def test_env_overrides_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ENV выигрывает у styx.json (общий приоритет load())."""
    _stub_env(monkeypatch)
    monkeypatch.setenv("STYX_RECALL_MIN_SCORE", "0.99")
    (tmp_path / "styx.json").write_text(
        json.dumps({"database_url": "postgresql://x/y", "recall_min_score": 0.33}),
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.recall_min_score == pytest.approx(0.99)


# -------- _build_recall_config ---------------------------------------


def _config(**overrides) -> StyxConfig:
    """Минимальный StyxConfig с возможностью точечного override."""
    base = {
        "database_url": "postgresql://x/y",
    }
    base.update(overrides)
    return StyxConfig(**base)


def test_build_recall_no_override_returns_defaults() -> None:
    cfg = _build_recall_config(_config())
    assert cfg is DEFAULT_RECALL_CONFIG


def test_build_recall_full_override() -> None:
    cfg = _build_recall_config(_config(recall_min_score=0.4))
    assert cfg.full.min_score == pytest.approx(0.4)
    # Остальное — дефолт.
    assert cfg.full.memory_limit == DEFAULT_RECALL_CONFIG.full.memory_limit
    assert cfg.companion == DEFAULT_RECALL_CONFIG.companion


def test_build_recall_dialogue_override() -> None:
    cfg = _build_recall_config(_config(recall_dialogue_min_score=0.55))
    assert cfg.companion.dialogue.min_score == pytest.approx(0.55)
    # full не тронут.
    assert cfg.full == DEFAULT_RECALL_CONFIG.full


def test_build_recall_both_overrides() -> None:
    cfg = _build_recall_config(
        _config(recall_min_score=0.4, recall_dialogue_min_score=0.55)
    )
    assert cfg.full.min_score == pytest.approx(0.4)
    assert cfg.companion.dialogue.min_score == pytest.approx(0.55)


# -------- provider — без БД, проверяем только pre-initialize state ----


def test_provider_default_recall_config_before_initialize() -> None:
    """До initialize get_tool_schemas работает на дефолте."""
    p = StyxMemoryCore()
    assert p._recall_config is DEFAULT_RECALL_CONFIG
    schemas = p.get_tool_schemas()
    recall = next(s for s in schemas if s["name"] == "styx_recall")
    description = recall["parameters"]["properties"]["limit"]["description"]
    assert str(DEFAULT_RECALL_CONFIG.full.memory_limit) in description
