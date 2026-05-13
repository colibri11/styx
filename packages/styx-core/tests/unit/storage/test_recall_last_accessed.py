"""recall_full обновляет last_accessed_at у возвращённых memories."""

from __future__ import annotations

import datetime as _dt
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.embedding import FakeEmbeddingClient
from styx.storage.queries import AgentScopedQueries
from styx.storage.recall import recall_full
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'recall-touch-%'")
    conn.commit()
    conn.close()


def test_recall_full_updates_last_accessed_at(db) -> None:
    """recall_full → возвращённые memories получают свежий last_accessed_at."""
    agent = f"recall-touch-{uuid.uuid4().hex[:6]}"
    embed = FakeEmbeddingClient(dim=768)
    queries = AgentScopedQueries(db, agent)

    # Insert memory старого возраста с last_accessed_at в прошлом.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories "
            "  (agent_id, role, content, embedding, last_accessed_at) "
            "VALUES (%s, 'user', %s, %s::vector, "
            "        now() - interval '7 days') RETURNING id",
            (
                agent,
                "Старая память про qwen3:4b-local на ollama",
                _vec_lit(embed.embed("Старая память про qwen3:4b-local на ollama")),
            ),
        )
        memory_id = cur.fetchone()[0]
    db.commit()

    # До recall.
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT last_accessed_at FROM memories WHERE id = %s", (memory_id,)
        )
        before = cur.fetchone()["last_accessed_at"]

    # FakeEmbedding hash'ит full text — для надёжного match'а
    # используем одинаковый текст. min_score опускаем до 0 чтобы не
    # зависеть от точной схожести.
    from dataclasses import replace as _replace

    full_cfg = _replace(DEFAULT_RECALL_CONFIG.full, min_score=0.0)

    result = recall_full(
        queries=queries,
        embed_client=embed,
        query="Старая память про qwen3:4b-local на ollama",
        full_config=full_cfg,
        record_events=False,  # не нужно для этого теста
    )
    db.commit()
    assert len(result.memories) >= 1
    assert any(h.id == memory_id for h in result.memories)

    # После recall — last_accessed_at должен подняться (минимум на дни).
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT last_accessed_at FROM memories WHERE id = %s", (memory_id,)
        )
        after = cur.fetchone()["last_accessed_at"]
    assert after > before


def test_update_last_accessed_at_only_owns_memories(db) -> None:
    """Чужой agent не задевает наши memories."""
    agent_a = f"recall-touch-A-{uuid.uuid4().hex[:6]}"
    agent_b = f"recall-touch-B-{uuid.uuid4().hex[:6]}"
    embed = FakeEmbeddingClient(dim=768)

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories "
            "  (agent_id, role, content, embedding, last_accessed_at) "
            "VALUES (%s, 'user', 'foreign', %s::vector, "
            "        now() - interval '5 days') RETURNING id",
            (agent_b, _vec_lit(embed.embed("foreign"))),
        )
        foreign_id = cur.fetchone()[0]
    db.commit()

    queries_a = AgentScopedQueries(db, agent_a)
    rc = queries_a.update_last_accessed_at([foreign_id])
    db.commit()
    assert rc == 0  # чужая memory не задета


def _vec_lit(v: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"
