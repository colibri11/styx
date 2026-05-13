"""Тесты AgentScopedQueries методов для relations API (волна 21).

Тесты cross-agent traversal, idempotency UPSERT'а Hebbian, и
recursive CTE с relation_filter в каждой ветке.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


# ── upsert_co_retrieved_pair ─────────────────────────────────────────


def test_upsert_inserts_new_relation_with_initial_weight(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src, tgt = uuid.uuid4(), uuid.uuid4()
    q.upsert_co_retrieved_pair(
        source_id=src, target_id=tgt,
        initial_weight=1.1, weight_bump=0.1, weight_max=2.0,
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT weight, metadata->>'last_reinforced' FROM relations "
            " WHERE source_id=%s AND target_id=%s AND relation='co_retrieved'",
            (src, tgt),
        )
        row = cur.fetchone()
    assert row is not None
    assert abs(row[0] - 1.1) < 1e-9
    assert row[1] is not None  # last_reinforced timestamp


def test_upsert_bumps_existing_relation(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src, tgt = uuid.uuid4(), uuid.uuid4()
    q.upsert_co_retrieved_pair(
        source_id=src, target_id=tgt,
        initial_weight=1.1, weight_bump=0.1, weight_max=2.0,
    )
    q.upsert_co_retrieved_pair(
        source_id=src, target_id=tgt,
        initial_weight=1.1, weight_bump=0.1, weight_max=2.0,
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT weight FROM relations "
            " WHERE source_id=%s AND target_id=%s AND relation='co_retrieved'",
            (src, tgt),
        )
        weight = cur.fetchone()[0]
    assert abs(weight - 1.2) < 1e-9


def test_upsert_caps_at_weight_max(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src, tgt = uuid.uuid4(), uuid.uuid4()
    # 1.1 + 10 × 0.1 = 2.1 → capped at 2.0
    for _ in range(15):
        q.upsert_co_retrieved_pair(
            source_id=src, target_id=tgt,
            initial_weight=1.1, weight_bump=0.1, weight_max=2.0,
        )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT weight FROM relations "
            " WHERE source_id=%s AND target_id=%s",
            (src, tgt),
        )
        weight = cur.fetchone()[0]
    assert abs(weight - 2.0) < 1e-9


# ── query_relations ──────────────────────────────────────────────────


def test_query_relations_filter_by_source(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src = uuid.uuid4()
    tgt1, tgt2 = uuid.uuid4(), uuid.uuid4()
    other = uuid.uuid4()
    q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt1, relation="related_to",
    )
    q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt2, relation="related_to",
    )
    q.insert_link(
        source_type="memory", source_id=other,
        target_type="memory", target_id=tgt1, relation="related_to",
    )
    conn.commit()
    rows = q.query_relations(source_id=src)
    assert len(rows) == 2
    assert all(r["source_id"] == src for r in rows)


def test_query_relations_filter_by_relation(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src, tgt = uuid.uuid4(), uuid.uuid4()
    q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt, relation="related_to",
    )
    q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt, relation="supersedes",
    )
    conn.commit()
    rows = q.query_relations(source_id=src, relation="related_to")
    assert len(rows) == 1
    assert rows[0]["relation"] == "related_to"


# ── traverse_graph ───────────────────────────────────────────────────


def _seed_memory(conn, agent: str, content: str) -> uuid.UUID:
    q = AgentScopedQueries(conn, agent)
    return q.insert_memory(
        role="summary", content=content,
        kind="note", kind_src="subjective",
    )


def test_traverse_depth_1_outgoing_and_incoming(conn: psycopg.Connection) -> None:
    """Depth 1: видим прямых соседей через outgoing и incoming рёбра."""
    a = _seed_memory(conn, "alpha", "node A")
    b = _seed_memory(conn, "alpha", "node B")
    c = _seed_memory(conn, "alpha", "node C")
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_link(
        source_type="memory", source_id=a,
        target_type="memory", target_id=b, relation="related_to",
    )
    q.insert_link(
        source_type="memory", source_id=c,
        target_type="memory", target_id=a, relation="related_to",
    )
    conn.commit()
    nodes = q.traverse_graph(root_id=a, depth=1)
    ids = {n["id"] for n in nodes}
    assert ids == {b, c}


def test_traverse_depth_2_reaches_grandchildren(
    conn: psycopg.Connection,
) -> None:
    a = _seed_memory(conn, "alpha", "A")
    b = _seed_memory(conn, "alpha", "B")
    c = _seed_memory(conn, "alpha", "C")
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_link(
        source_type="memory", source_id=a, target_type="memory",
        target_id=b, relation="related_to",
    )
    q.insert_link(
        source_type="memory", source_id=b, target_type="memory",
        target_id=c, relation="related_to",
    )
    conn.commit()
    nodes = q.traverse_graph(root_id=a, depth=2)
    ids_to_depth = {n["id"]: n["depth"] for n in nodes}
    assert b in ids_to_depth and ids_to_depth[b] == 1
    assert c in ids_to_depth and ids_to_depth[c] == 2


def test_traverse_depth_capped_at_3(conn: psycopg.Connection) -> None:
    """depth=10 → capped at 3."""
    nodes_chain: list[uuid.UUID] = [
        _seed_memory(conn, "alpha", f"n{i}") for i in range(6)
    ]
    q = AgentScopedQueries(conn, agent_id="alpha")
    for i in range(5):
        q.insert_link(
            source_type="memory", source_id=nodes_chain[i],
            target_type="memory", target_id=nodes_chain[i + 1],
            relation="related_to",
        )
    conn.commit()
    nodes = q.traverse_graph(root_id=nodes_chain[0], depth=10)
    max_depth = max(n["depth"] for n in nodes)
    assert max_depth == 3


def test_traverse_relation_filter_in_recursive(
    conn: psycopg.Connection,
) -> None:
    """relation_filter применяется в каждой ветке CTE — recursive step
    не должен пройти через другой relation тип."""
    a = _seed_memory(conn, "alpha", "A")
    b = _seed_memory(conn, "alpha", "B")
    c = _seed_memory(conn, "alpha", "C")
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_link(
        source_type="memory", source_id=a, target_type="memory",
        target_id=b, relation="related_to",
    )
    q.insert_link(
        source_type="memory", source_id=b, target_type="memory",
        target_id=c, relation="supersedes",
    )
    conn.commit()
    # filter = related_to → достигаем b, но дальше через supersedes — не идём
    nodes = q.traverse_graph(
        root_id=a, depth=3, relation_filter="related_to",
    )
    ids = {n["id"] for n in nodes}
    assert b in ids
    assert c not in ids


def test_traverse_cross_agent(conn: psycopg.Connection) -> None:
    """Cross-agent: ряды разных агентов соединены auto-link'ом, traversal
    видит все (без agent_id фильтра)."""
    a = _seed_memory(conn, "alpha", "A")
    b_foreign = _seed_memory(conn, "beta", "B (cross-agent)")
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_link(
        source_type="memory", source_id=a, target_type="memory",
        target_id=b_foreign, relation="related_to",
    )
    conn.commit()
    nodes = q.traverse_graph(root_id=a, depth=1)
    ids = {n["id"] for n in nodes}
    assert b_foreign in ids


# ── insert_link ──────────────────────────────────────────────────────


def test_insert_link_returns_true_when_new(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    src, tgt = uuid.uuid4(), uuid.uuid4()
    created = q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt, relation="custom",
    )
    conn.commit()
    assert created is True


def test_insert_link_returns_false_when_duplicate(
    conn: psycopg.Connection,
) -> None:
    """ON CONFLICT DO NOTHING — повторный INSERT возвращает False."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    src, tgt = uuid.uuid4(), uuid.uuid4()
    q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt, relation="custom",
    )
    second = q.insert_link(
        source_type="memory", source_id=src,
        target_type="memory", target_id=tgt, relation="custom",
    )
    conn.commit()
    assert second is False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_id=%s AND target_id=%s AND relation='custom'",
            (src, tgt),
        )
        assert cur.fetchone()[0] == 1
