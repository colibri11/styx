"""Тесты AgentScopedQueries методов для dialogue tools (волна 24).

Покрытия:
- ``dialogue_search`` — hybrid (FTS+vector) и pure-vector mode;
  фильтры session_id / after / before; agent_id isolation.
- ``dialogue_recent`` — DESC by seq; фильтры session_id / before;
  role IN ('user','assistant') фильтр.
- ``dialogue_list_sessions`` — GROUP BY с counts + first/last;
  реплики без session_id игнорируются.
- ``dialogue_prepare_summary`` — ASC by created_at+seq; пустая
  session возвращает пустой список.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import datetime as _dt
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


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    return base


def _insert_session(conn: psycopg.Connection, agent_id: str) -> uuid.UUID:
    sid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, agent_id) VALUES (%s, %s)",
            (sid, agent_id),
        )
    conn.commit()
    return sid


def _insert_dialogue_row(
    q: AgentScopedQueries,
    *,
    role: str,
    content: str,
    session_id: uuid.UUID | None,
    embedding: list[float] | None,
    metadata: dict | None = None,
) -> uuid.UUID:
    return q.insert_message(
        role=role,
        content=content,
        session_id=session_id,
        embedding=embedding,
        metadata=metadata,
    )


def test_dialogue_search_pure_vector_returns_top_match(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="hello world",
        session_id=sid, embedding=_embed(0.9),
    )
    _insert_dialogue_row(
        q, role="assistant", content="goodbye",
        session_id=sid, embedding=_embed(0.1),
    )
    conn.commit()

    hits = q.dialogue_search(
        query_vector=_embed(0.9),
        query_text=None,
        limit=5,
    )
    assert len(hits) == 2
    assert hits[0].content == "hello world"
    assert 0.0 <= hits[0].score <= 1.0
    assert hits[0].score > hits[1].score


def test_dialogue_search_hybrid_with_text_query(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="postgres performance tuning",
        session_id=sid, embedding=_embed(0.5),
    )
    _insert_dialogue_row(
        q, role="assistant", content="how is the weather today",
        session_id=sid, embedding=_embed(0.5),
    )
    conn.commit()

    hits = q.dialogue_search(
        query_vector=_embed(0.5),
        query_text="postgres",
        limit=5,
    )
    assert len(hits) == 2
    assert hits[0].content == "postgres performance tuning"


def test_dialogue_search_session_filter(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    s1 = _insert_session(conn, "alpha")
    s2 = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="from session one",
        session_id=s1, embedding=_embed(0.7),
    )
    _insert_dialogue_row(
        q, role="user", content="from session two",
        session_id=s2, embedding=_embed(0.7),
    )
    conn.commit()

    hits = q.dialogue_search(
        query_vector=_embed(0.7),
        session_id=s1,
        limit=10,
    )
    assert len(hits) == 1
    assert hits[0].content == "from session one"


def test_dialogue_search_agent_isolation(conn: psycopg.Connection) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    _insert_dialogue_row(
        qa, role="user", content="alpha message",
        session_id=None, embedding=_embed(0.5),
    )
    _insert_dialogue_row(
        qb, role="user", content="beta message",
        session_id=None, embedding=_embed(0.5),
    )
    conn.commit()

    hits_a = qa.dialogue_search(query_vector=_embed(0.5), limit=10)
    hits_b = qb.dialogue_search(query_vector=_embed(0.5), limit=10)
    assert {h.content for h in hits_a} == {"alpha message"}
    assert {h.content for h in hits_b} == {"beta message"}


def test_dialogue_search_skips_no_embedding(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    _insert_dialogue_row(
        q, role="user", content="with embed",
        session_id=None, embedding=_embed(0.4),
    )
    _insert_dialogue_row(
        q, role="user", content="without embed",
        session_id=None, embedding=None,
    )
    conn.commit()

    hits = q.dialogue_search(query_vector=_embed(0.4), limit=10)
    assert {h.content for h in hits} == {"with embed"}


def test_dialogue_recent_orders_chronological_after_reverse(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="first", session_id=sid, embedding=None,
    )
    _insert_dialogue_row(
        q, role="assistant", content="second", session_id=sid, embedding=None,
    )
    _insert_dialogue_row(
        q, role="user", content="third", session_id=sid, embedding=None,
    )
    conn.commit()

    rows = q.dialogue_recent(limit=10)
    # DESC by seq → "third", "second", "first".
    assert [r.content for r in rows] == ["third", "second", "first"]
    # Caller reverse → chronological.
    assert [r.content for r in reversed(rows)] == ["first", "second", "third"]


def test_dialogue_recent_filters_role(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    _insert_dialogue_row(
        q, role="user", content="dialog row",
        session_id=None, embedding=None,
    )
    _insert_dialogue_row(
        q, role="system", content="system row",
        session_id=None, embedding=None,
    )
    _insert_dialogue_row(
        q, role="tool", content="tool row",
        session_id=None, embedding=None,
    )
    conn.commit()

    rows = q.dialogue_recent(limit=10)
    assert [r.content for r in rows] == ["dialog row"]


def test_dialogue_recent_session_filter(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    s1 = _insert_session(conn, "alpha")
    s2 = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="s1-msg", session_id=s1, embedding=None,
    )
    _insert_dialogue_row(
        q, role="user", content="s2-msg", session_id=s2, embedding=None,
    )
    conn.commit()

    rows = q.dialogue_recent(limit=10, session_id=s1)
    assert [r.content for r in rows] == ["s1-msg"]


def test_dialogue_list_sessions_groups_with_counts(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    s1 = _insert_session(conn, "alpha")
    s2 = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="s1-1", session_id=s1, embedding=None,
    )
    _insert_dialogue_row(
        q, role="assistant", content="s1-2", session_id=s1, embedding=None,
    )
    _insert_dialogue_row(
        q, role="user", content="s2-1", session_id=s2, embedding=None,
    )
    conn.commit()

    sessions = q.dialogue_list_sessions(limit=10)
    counts = {s.session_id: s.message_count for s in sessions}
    assert counts == {s1: 2, s2: 1}


def test_dialogue_list_sessions_skips_null_session_id(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    s1 = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="with session", session_id=s1, embedding=None,
    )
    _insert_dialogue_row(
        q, role="user", content="without session", session_id=None, embedding=None,
    )
    conn.commit()

    sessions = q.dialogue_list_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0].session_id == s1
    assert sessions[0].message_count == 1


def test_dialogue_list_sessions_orders_by_last_at_desc(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    s1 = _insert_session(conn, "alpha")
    s2 = _insert_session(conn, "alpha")
    # s1 — старая.
    _insert_dialogue_row(
        q, role="user", content="old", session_id=s1, embedding=None,
    )
    # s2 — новая (вставляется позже, выше seq, выше created_at в clock_timestamp()).
    _insert_dialogue_row(
        q, role="user", content="new", session_id=s2, embedding=None,
    )
    conn.commit()

    sessions = q.dialogue_list_sessions(limit=10)
    assert sessions[0].session_id == s2
    assert sessions[1].session_id == s1


def test_dialogue_prepare_summary_chronological(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="первая", session_id=sid, embedding=None,
    )
    _insert_dialogue_row(
        q, role="assistant", content="вторая", session_id=sid, embedding=None,
    )
    _insert_dialogue_row(
        q, role="user", content="третья", session_id=sid, embedding=None,
    )
    conn.commit()

    rows = q.dialogue_prepare_summary(session_id=sid, limit=200)
    assert [r.content for r in rows] == ["первая", "вторая", "третья"]


def test_dialogue_prepare_summary_filters_non_dialogue_roles(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="user msg", session_id=sid, embedding=None,
    )
    _insert_dialogue_row(
        q, role="system", content="system msg", session_id=sid, embedding=None,
    )
    _insert_dialogue_row(
        q, role="assistant", content="assistant msg",
        session_id=sid, embedding=None,
    )
    conn.commit()

    rows = q.dialogue_prepare_summary(session_id=sid)
    assert [r.content for r in rows] == ["user msg", "assistant msg"]


def test_dialogue_prepare_summary_empty_session_returns_empty(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    conn.commit()
    rows = q.dialogue_prepare_summary(session_id=sid)
    assert rows == []


def test_dialogue_search_after_before_filters(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = _insert_session(conn, "alpha")
    _insert_dialogue_row(
        q, role="user", content="early", session_id=sid, embedding=_embed(0.3),
    )
    _insert_dialogue_row(
        q, role="user", content="middle", session_id=sid, embedding=_embed(0.3),
    )
    _insert_dialogue_row(
        q, role="user", content="late", session_id=sid, embedding=_embed(0.3),
    )
    conn.commit()

    # Возьмём timestamp middle через прямой SELECT.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT created_at FROM memories "
            "WHERE agent_id = 'alpha' AND content = 'middle'"
        )
        middle_ts = cur.fetchone()[0]

    # after = middle_ts — only middle и late.
    hits = q.dialogue_search(
        query_vector=_embed(0.3),
        after=middle_ts,
        limit=10,
    )
    contents = {h.content for h in hits}
    assert contents == {"middle", "late"}

    # before = middle_ts — only early и middle.
    hits = q.dialogue_search(
        query_vector=_embed(0.3),
        before=middle_ts,
        limit=10,
    )
    contents = {h.content for h in hits}
    assert contents == {"early", "middle"}
