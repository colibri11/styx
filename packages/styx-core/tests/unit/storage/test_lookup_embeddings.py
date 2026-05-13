"""Тест batch lookup embedding'ов по content (волна 12).

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


def _embed(seed: float) -> list[float]:
    base = [0.0] * 768
    base[0] = seed
    base[1] = 1.0 - abs(seed)
    return base


def test_lookup_returns_embeddings_for_known_contents(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(role="user", content="hello", session_id=sid, embedding=_embed(0.1))
    q.insert_message(role="user", content="world", session_id=sid, embedding=_embed(0.2))

    out = q.lookup_embeddings_by_content(["hello", "world"])
    assert set(out.keys()) == {"hello", "world"}
    assert len(out["hello"]) == 768
    assert abs(out["hello"][0] - 0.1) < 1e-6


def test_lookup_skips_null_embeddings(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(role="user", content="no embed", session_id=sid)

    out = q.lookup_embeddings_by_content(["no embed"])
    assert out == {}


def test_lookup_skips_unknown_content(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(role="user", content="known", session_id=sid, embedding=_embed(0.5))

    out = q.lookup_embeddings_by_content(["known", "missing"])
    assert "known" in out
    assert "missing" not in out


def test_lookup_isolates_by_agent_id(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    sid = uuid.uuid4()
    a.upsert_session(sid)
    b.upsert_session(sid)
    a.insert_message(role="user", content="text", session_id=sid, embedding=_embed(0.1))

    assert "text" in a.lookup_embeddings_by_content(["text"])
    assert b.lookup_embeddings_by_content(["text"]) == {}


def test_lookup_empty_input_returns_empty_dict(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    assert q.lookup_embeddings_by_content([]) == {}


def test_lookup_handles_duplicate_contents(conn: psycopg.Connection) -> None:
    """Несколько ряды с одинаковым content — DISTINCT ON выберет один."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(role="user", content="dup", session_id=sid, embedding=_embed(0.1))
    q.insert_message(role="assistant", content="dup", session_id=sid, embedding=_embed(0.2))

    out = q.lookup_embeddings_by_content(["dup"])
    assert "dup" in out
    assert len(out["dup"]) == 768
