"""Unit-тесты для styx.commands.reembed."""

from __future__ import annotations

import time
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.commands.reembed import (
    REEMBED_MODE_ALL,
    REEMBED_MODE_NULL_ONLY,
    run_reembed,
)
from styx.embedding import EmbeddingError, FakeEmbeddingClient


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'reembed-test-%'")
    conn.commit()
    conn.close()


def _insert_memory(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    content: str,
    with_embedding: bool = False,
) -> uuid.UUID:
    embed = FakeEmbeddingClient(dim=768) if with_embedding else None
    if embed:
        vec = embed.embed(content)
        vec_lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, embedding) "
                "VALUES (%s, 'user', %s, %s::vector) RETURNING id",
                (agent_id, content, vec_lit),
            )
            row = cur.fetchone()
    else:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content) "
                "VALUES (%s, 'user', %s) RETURNING id",
                (agent_id, content),
            )
            row = cur.fetchone()
    conn.commit()
    return row[0]


def _has_embedding(conn: psycopg.Connection, mid: uuid.UUID) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding IS NOT NULL FROM memories WHERE id = %s", (mid,)
        )
        return bool(cur.fetchone()[0])


# ── Backfill (--null-only) ────────────────────────────────────────────


def test_reembed_backfill_null_only(db) -> None:
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    null_ids = [
        _insert_memory(db, agent_id=agent, content=f"null-{i}", with_embedding=False)
        for i in range(3)
    ]
    has_id = _insert_memory(
        db, agent_id=agent, content="already-has", with_embedding=True
    )

    embed = FakeEmbeddingClient(dim=768)
    result = run_reembed(
        conn=db, embed_client=embed, mode=REEMBED_MODE_NULL_ONLY, rate_per_second=100.0
    )

    assert result.processed == 3
    assert result.failed == 0
    for mid in null_ids:
        assert _has_embedding(db, mid)
    assert _has_embedding(db, has_id)  # был и остался


# ── Re-embed (--all) ───────────────────────────────────────────────────


def test_reembed_all_mode_overwrites(db) -> None:
    """`--all` пересчитывает все, даже те у которых уже есть вектор."""
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    has_id = _insert_memory(
        db, agent_id=agent, content="already-has", with_embedding=True
    )
    null_id = _insert_memory(
        db, agent_id=agent, content="null-one", with_embedding=False
    )

    embed = FakeEmbeddingClient(dim=768)
    result = run_reembed(
        conn=db, embed_client=embed, mode=REEMBED_MODE_ALL, rate_per_second=100.0
    )
    assert result.processed == 2
    assert _has_embedding(db, has_id)
    assert _has_embedding(db, null_id)


# ── Agent filter ───────────────────────────────────────────────────────


def test_reembed_filters_by_agent_id(db) -> None:
    agent_a = f"reembed-test-A-{uuid.uuid4().hex[:6]}"
    agent_b = f"reembed-test-B-{uuid.uuid4().hex[:6]}"
    a_ids = [
        _insert_memory(db, agent_id=agent_a, content=f"a-{i}") for i in range(2)
    ]
    b_ids = [
        _insert_memory(db, agent_id=agent_b, content=f"b-{i}") for i in range(3)
    ]

    embed = FakeEmbeddingClient(dim=768)
    result = run_reembed(
        conn=db, embed_client=embed, agent_id=agent_a, rate_per_second=100.0
    )

    assert result.processed == 2
    for mid in a_ids:
        assert _has_embedding(db, mid)
    for mid in b_ids:
        assert not _has_embedding(db, mid)  # B не задет


# ── Limit ─────────────────────────────────────────────────────────────


def test_reembed_respects_limit(db) -> None:
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    ids = [_insert_memory(db, agent_id=agent, content=f"m-{i}") for i in range(5)]

    embed = FakeEmbeddingClient(dim=768)
    result = run_reembed(
        conn=db, embed_client=embed, agent_id=agent, limit=3, rate_per_second=100.0
    )
    assert result.processed == 3

    embedded_count = sum(1 for mid in ids if _has_embedding(db, mid))
    assert embedded_count == 3


# ── Dry-run ────────────────────────────────────────────────────────────


def test_reembed_dry_run_no_writes(db) -> None:
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    ids = [_insert_memory(db, agent_id=agent, content=f"m-{i}") for i in range(4)]

    embed = FakeEmbeddingClient(dim=768)
    result = run_reembed(
        conn=db, embed_client=embed, agent_id=agent, dry_run=True
    )
    assert result.dry_run is True
    assert result.would_process == 4
    assert result.processed == 0
    # Никакая memory не embedded.
    for mid in ids:
        assert not _has_embedding(db, mid)


def test_reembed_dry_run_respects_limit(db) -> None:
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    for i in range(10):
        _insert_memory(db, agent_id=agent, content=f"m-{i}")

    embed = FakeEmbeddingClient(dim=768)
    result = run_reembed(
        conn=db, embed_client=embed, agent_id=agent, dry_run=True, limit=4
    )
    assert result.would_process == 4


# ── Embed errors ──────────────────────────────────────────────────────


class _FlakyFakeEmbed(FakeEmbeddingClient):
    """Бросает на N-м вызове."""

    def __init__(self, *, fail_on_index: int) -> None:
        super().__init__(dim=768)
        self._fail_on = fail_on_index
        self._n = 0

    def embed(self, text: str) -> list[float]:
        idx = self._n
        self._n += 1
        if idx == self._fail_on:
            raise EmbeddingError("simulated embed failure")
        return super().embed(text)


def test_reembed_continues_on_embed_error(db) -> None:
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    ids = [_insert_memory(db, agent_id=agent, content=f"m-{i}") for i in range(5)]

    embed = _FlakyFakeEmbed(fail_on_index=2)  # 3-й embed упадёт
    result = run_reembed(
        conn=db, embed_client=embed, agent_id=agent, rate_per_second=100.0
    )
    assert result.processed == 4
    assert result.failed == 1


# ── Rate-limit ────────────────────────────────────────────────────────


def test_reembed_rate_limited(db) -> None:
    """rate_per_second=10, 30 памятей → не быстрее ~ (30-capacity)/10s."""
    agent = f"reembed-test-{uuid.uuid4().hex[:6]}"
    for i in range(30):
        _insert_memory(db, agent_id=agent, content=f"m-{i}")

    embed = FakeEmbeddingClient(dim=768)
    started = time.monotonic()
    result = run_reembed(
        conn=db, embed_client=embed, agent_id=agent, rate_per_second=10.0
    )
    elapsed = time.monotonic() - started
    assert result.processed == 30
    # capacity = max(1, int(10)) = 10 burst, остальные 20 по 10/s = 2s.
    # Берём с запасом — главное что > 1 секунды.
    assert elapsed >= 1.0


# ── Invalid args ──────────────────────────────────────────────────────


def test_reembed_invalid_mode(db) -> None:
    with pytest.raises(ValueError, match="mode"):
        run_reembed(
            conn=db, embed_client=FakeEmbeddingClient(), mode="weird"
        )


def test_reembed_invalid_batch(db) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        run_reembed(
            conn=db, embed_client=FakeEmbeddingClient(), batch_size=0
        )


def test_reembed_invalid_rate(db) -> None:
    with pytest.raises(ValueError, match="rate_per_second"):
        run_reembed(
            conn=db, embed_client=FakeEmbeddingClient(), rate_per_second=0.0
        )
