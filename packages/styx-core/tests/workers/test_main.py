"""Unit-тесты для ``styx.workers.main`` (entry-point worker'а)."""

from __future__ import annotations

import os

import pytest

from styx.config import StyxConfig
from styx.workers.main import _redact_dsn, build_worker
from styx.workers.handlers.importance import IMPORTANCE_TASK_TYPE


def _config(dsn: str = "postgresql://u:p@h:5432/db") -> StyxConfig:
    return StyxConfig(database_url=dsn)


def test_build_worker_registers_importance_handler() -> None:
    w = build_worker(_config())
    # Внутренний реестр — приватный, проверяем через попытку дёрнуть
    # (handler существует, не None).
    assert IMPORTANCE_TASK_TYPE in w._handlers


def test_build_worker_registers_batch_consolidation_handler() -> None:
    """Волна 14: dialogue_batch_consolidation handler зарегистрирован."""
    from styx.workers.handlers.dialogue_batch_consolidation import (
        DIALOGUE_BATCH_TASK_TYPE,
    )
    w = build_worker(_config())
    assert DIALOGUE_BATCH_TASK_TYPE in w._handlers


def test_build_worker_registers_batch_consolidation_scheduler() -> None:
    """Волна 14: periodic-task scheduler зарегистрирован."""
    w = build_worker(_config())
    periodic_names = {p.name for p in w._periodic}
    assert "batch_consolidation_scheduler" in periodic_names


def test_build_worker_uses_config_llm_settings() -> None:
    cfg = StyxConfig(
        database_url="postgresql://x@h/d",
        llm_url="http://qwen:11434",
        llm_model="some-other-model",
        llm_rate_limit_capacity=2,
        llm_rate_limit_refill_per_s=0.5,
    )
    w = build_worker(cfg)
    assert w._llm.model == "some-other-model"
    assert w._rate_limit._capacity == 2.0
    assert w._rate_limit._refill == 0.5


def test_redact_dsn_hides_password() -> None:
    assert _redact_dsn("postgresql://user:secret@h:5432/db") == "postgresql://user:***@h:5432/db"


def test_redact_dsn_passthrough_when_no_credentials() -> None:
    assert _redact_dsn("postgresql://h:5432/db") == "postgresql://h:5432/db"


def test_redact_dsn_passthrough_for_unknown_format() -> None:
    assert _redact_dsn("just-a-string") == "just-a-string"


# ── End-to-end CLI ─────────────────────────────────────────────────────


def test_worker_run_once_empty_queue(migrated_db: str) -> None:
    """``workers.main.run(once=True)`` на пустой очереди завершается 0.

    После Phase A ``styx`` CLI не имеет ``worker`` подкоманды (поглощено
    в ``daemon run``). Тест переписан на прямой вызов ``run(once=True)``,
    чтобы проверить тот же путь — drain очереди и exit 0.
    """
    import os

    os.environ["STYX_DATABASE_URL"] = migrated_db
    from styx.workers.main import run

    rc = run(once=True)
    assert rc == 0
