"""Тесты AgentScopedQueries методов для auto-link (волна 18).

Главное: cross-agent SELECT (без agent_id фильтра) и идемпотентность
INSERT'а через UNIQUE constraint.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.engine.auto_link import AutoLinkNeighbor
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    base[1] = (1.0 - seed * seed) ** 0.5
    return base


def _embed_with_offset(offset: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = (1.0 - offset * offset) ** 0.5
    base[1] = offset
    return base


# ── find_auto_link_candidates ────────────────────────────────────────


def test_find_candidates_empty_db(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    cands = q.find_auto_link_candidates(
        _embed(1.0),
        max_distance=0.25, max_links=3, exclude_id=uuid.uuid4(),
    )
    assert cands == []


def test_find_candidates_returns_within_distance(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    near = q.insert_memory(
        role="summary", content="близкая запись",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    far = q.insert_memory(
        role="summary", content="далёкая запись",
        kind="note", kind_src="subjective",
        embedding=_embed(0.0),  # ортогональный
    )
    conn.commit()

    cands = q.find_auto_link_candidates(
        _embed_with_offset(0.0),
        max_distance=0.10, max_links=3, exclude_id=uuid.uuid4(),
    )
    ids = {c.id for c in cands}
    assert near in ids
    assert far not in ids


def test_find_candidates_excludes_self(conn: psycopg.Connection) -> None:
    """exclude_id обязателен (новый ряд в той же транзакции виден SELECT'ом).
    """
    q = AgentScopedQueries(conn, agent_id="alpha")
    self_id = q.insert_memory(
        role="summary", content="self",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    other = q.insert_memory(
        role="summary", content="other",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    conn.commit()

    cands = q.find_auto_link_candidates(
        _embed(1.0),
        max_distance=0.25, max_links=3, exclude_id=self_id,
    )
    ids = {c.id for c in cands}
    assert self_id not in ids
    assert other in ids


def test_find_candidates_skips_superseded_and_null_embedding(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    superseded = q.insert_memory(
        role="summary", content="старый",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    new = q.insert_memory(
        role="summary", content="новый",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by = %s WHERE id = %s",
            (new, superseded),
        )
    q.insert_memory(
        role="summary", content="без вектора",
        kind="note", kind_src="subjective",
    )
    conn.commit()

    cands = q.find_auto_link_candidates(
        _embed_with_offset(0.0),
        max_distance=0.25, max_links=3, exclude_id=uuid.uuid4(),
    )
    ids = {c.id for c in cands}
    assert superseded not in ids
    assert new in ids


def test_find_candidates_cross_agent(conn: psycopg.Connection) -> None:
    """Главное свойство волны 18: SELECT не фильтрует по agent_id."""
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")

    foreign_close = b.insert_memory(
        role="summary", content="вещь от beta",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    own = a.insert_memory(
        role="summary", content="вещь от alpha",
        kind="note", kind_src="subjective",
        embedding=_embed(0.0),  # ортогональный obeим
    )
    conn.commit()

    # alpha ищет соседей и должна найти beta'у — ребро cross-agent.
    cands = a.find_auto_link_candidates(
        _embed_with_offset(0.05),
        max_distance=0.25, max_links=3, exclude_id=own,
    )
    ids = {c.id for c in cands}
    assert foreign_close in ids


def test_find_candidates_top_k_limit(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    self_id = uuid.uuid4()
    for _ in range(8):
        q.insert_memory(
            role="summary", content="row",
            kind="note", kind_src="subjective",
            embedding=_embed_with_offset(0.05),
        )
    conn.commit()
    cands = q.find_auto_link_candidates(
        _embed_with_offset(0.0),
        max_distance=0.5, max_links=3, exclude_id=self_id,
    )
    assert len(cands) == 3


def test_find_candidates_tie_break_by_created_at(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    older = q.insert_memory(
        role="summary", content="older",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    newer = q.insert_memory(
        role="summary", content="newer",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    conn.commit()
    cands = q.find_auto_link_candidates(
        _embed(1.0),
        max_distance=0.5, max_links=2, exclude_id=uuid.uuid4(),
    )
    assert [c.id for c in cands] == [older, newer]


# ── insert_auto_link_relations ────────────────────────────────────────


def test_insert_relations_creates_rows(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src = uuid.uuid4()
    n1 = AutoLinkNeighbor(id=uuid.uuid4(), cosine_distance=0.05)
    n2 = AutoLinkNeighbor(id=uuid.uuid4(), cosine_distance=0.10)
    q.insert_auto_link_relations(src, [n1, n2])
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_type='memory' AND source_id=%s "
            "   AND relation='related_to'",
            (src,),
        )
        assert cur.fetchone()[0] == 2


def test_insert_relations_idempotent_on_conflict(
    conn: psycopg.Connection,
) -> None:
    """Двойной insert одной (source, target, relation) тройки = 1 ряд."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    src = uuid.uuid4()
    n = AutoLinkNeighbor(id=uuid.uuid4(), cosine_distance=0.05)
    q.insert_auto_link_relations(src, [n])
    q.insert_auto_link_relations(src, [n])
    q.insert_auto_link_relations(src, [n, n, n])
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_type='memory' AND source_id=%s "
            "   AND target_type='memory' AND target_id=%s "
            "   AND relation='related_to'",
            (src, n.id),
        )
        assert cur.fetchone()[0] == 1


def test_insert_relations_empty_no_op(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_auto_link_relations(uuid.uuid4(), [])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relations")
        assert cur.fetchone()[0] == 0
