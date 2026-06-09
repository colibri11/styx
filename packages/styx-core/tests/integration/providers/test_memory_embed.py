"""Тесты embed-after-commit hook в StyxMemoryCore.sync_turn."""

from __future__ import annotations

import logging
import uuid

import psycopg
import pytest

from styx.embedding import EmbeddingError, FakeEmbeddingClient
from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    return migrated_db


def _provider_with_fake_embed(
    monkeypatch: pytest.MonkeyPatch,
    embed_client: object,
) -> StyxMemoryCore:
    """Создаёт provider, подменяя factory чтобы не лезть в реальный Ollama."""
    monkeypatch.setattr(
        "styx.providers.memory.make_embedding_client",
        lambda **_: embed_client,
    )
    return StyxMemoryCore()


def test_sync_turn_writes_embedding_after_commit(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    embed = FakeEmbeddingClient()
    p = _provider_with_fake_embed(monkeypatch, embed)
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        p.sync_turn("привет, как дела?", "хорошо, спасибо!", session_id=sid)

        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content, embedding IS NOT NULL "
                    "FROM memories WHERE agent_id = 'alpha' ORDER BY seq"
                )
                rows = cur.fetchall()

        assert len(rows) == 2
        for role, content, has_emb in rows:
            assert has_emb, f"{role}: embedding не записан"
    finally:
        p.shutdown()


def test_sync_turn_continues_on_embed_error(
    provider_env: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ollama down → memory записан с embedding=NULL, лог error, не падаем."""

    class _BrokenEmbed:
        @property
        def dim(self) -> int:
            return 768

        def embed(self, text: str) -> list[float]:
            raise EmbeddingError("ollama unreachable")

    p = _provider_with_fake_embed(monkeypatch, _BrokenEmbed())
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="beta")
    try:
        with caplog.at_level(logging.WARNING):
            p.sync_turn("user msg", "asst msg", session_id=sid)

        # Memory всё равно записан.
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*), count(embedding) "
                    "FROM memories WHERE agent_id = 'beta'"
                )
                total, with_emb = cur.fetchone()

        assert total == 2
        assert with_emb == 0  # embedding NULL у обоих
        assert any("embed-after-commit" in r.message for r in caplog.records)
    finally:
        p.shutdown()


def test_dialogue_save_continues_on_embed_error(
    provider_env: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ollama down → dialogue_save сохраняет реплику с embedding=NULL.

    Регресс волны 34: embed-after-commit в dialogue_save — best-effort
    (сообщение durably закоммичено в 1-м блоке). EmbeddingError не должен
    приводить к 500; реплика остаётся в memories, embedding NULL до reembed.
    """

    class _BrokenEmbed:
        @property
        def dim(self) -> int:
            return 768

        def embed(self, text: str) -> list[float]:
            raise EmbeddingError("ollama unreachable")

    p = _provider_with_fake_embed(monkeypatch, _BrokenEmbed())
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="delta")
    try:
        with caplog.at_level(logging.WARNING):
            memory_id = p.dialogue_save(
                role="user", content="реплика без embedding", session_id=sid
            )

        assert memory_id is not None

        # Реплика реально записана, embedding NULL.
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, embedding IS NULL "
                    "FROM memories WHERE id = %s AND agent_id = 'delta'",
                    (memory_id,),
                )
                row = cur.fetchone()

        assert row is not None
        assert row[0] == "реплика без embedding"
        assert row[1] is True  # embedding IS NULL
    finally:
        p.shutdown()


def test_sync_turn_partial_failure_still_writes_some(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если упало на втором сообщении — первое всё равно с вектором."""

    calls = {"n": 0}

    class _FailSecond:
        @property
        def dim(self) -> int:
            return 768

        def embed(self, text: str) -> list[float]:
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeEmbeddingClient().embed(text)
            raise EmbeddingError("simulated mid-turn failure")

    p = _provider_with_fake_embed(monkeypatch, _FailSecond())
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="gamma")
    try:
        p.sync_turn("first", "second", session_id=sid)

        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, embedding IS NOT NULL FROM memories "
                    "WHERE agent_id = 'gamma' ORDER BY seq"
                )
                rows = cur.fetchall()

        # User-message получил embedding, assistant — нет (Ollama упал).
        assert rows == [("user", True), ("assistant", False)]
    finally:
        p.shutdown()
