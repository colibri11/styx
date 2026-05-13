"""Юнит-тесты AgentScopedQueries — изоляция по agent_id, базовый CRUD."""

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


def test_agent_id_is_required():
    with pytest.raises(ValueError):
        AgentScopedQueries(conn=None, agent_id="")  # type: ignore[arg-type]


def test_upsert_and_insert_basic(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.upsert_session(sid)  # повторный — не падает

    mid = q.insert_message(role="user", content="hi", session_id=sid)
    assert isinstance(mid, uuid.UUID)
    assert q.count_messages() == 1
    assert q.count_messages(session_id=sid) == 1


def test_scope_isolation_between_agents(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")

    sid = uuid.uuid4()
    a.upsert_session(sid)
    a.insert_message(role="user", content="alpha-only", session_id=sid)

    assert a.count_messages() == 1
    assert b.count_messages() == 0

    recent_b = b.recent_messages(limit=10)
    assert recent_b == []


def test_recent_messages_orders_by_created_at_desc(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(role="user", content="first", session_id=sid)
    q.insert_message(role="assistant", content="second", session_id=sid)
    q.insert_message(role="user", content="third", session_id=sid)

    recent = q.recent_messages(limit=10, session_id=sid)
    assert [m.content for m in recent] == ["third", "second", "first"]


def test_embedding_writes_as_vector(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    embedding = [0.0] * 768
    embedding[0] = 1.0
    mid = q.insert_message(
        role="user",
        content="vector check",
        session_id=sid,
        embedding=embedding,
    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding IS NOT NULL FROM memories WHERE id = %s",
            (mid,),
        )
        assert cur.fetchone()[0] is True


def test_metadata_round_trip(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(
        role="user",
        content="x",
        session_id=sid,
        metadata={"source": "telegram", "thread_id": 17},
    )
    [msg] = q.recent_messages(limit=1, session_id=sid)
    assert msg.metadata == {"source": "telegram", "thread_id": 17}


def test_no_raw_sql_to_memories_outside_wrapper() -> None:
    """Guard: storage/queries.py — единственное место с SQL на memories.

    Если в src/styx/ появится прямой ``cur.execute("... memories ...")``
    или ``... sessions ...`` вне queries.py — тест ловит регрессию.
    Regex покрывает multiline SQL и варианты с переменными.
    """
    import re
    from pathlib import Path

    PAT = re.compile(
        r'\b(FROM|INTO|UPDATE)\s+memories\b|\b(FROM|INTO|UPDATE)\s+sessions\b',
        re.IGNORECASE,
    )

    src_root = Path(__file__).resolve().parents[2] / "src" / "styx"
    # Cross-agent worker-код (волны 7a-d) не имеет agent-context'а —
    # admin-tier SQL без agent_id фильтра. Exemption для workers/sweep/
    # и workers/handlers/ (см. ADR § 18 в decisions.md).
    exempt_paths = {
        ("workers", "sweep", "lifecycle.py"),
        ("workers", "sweep", "runner.py"),
        # emotional/state.py имеет admin-tier SELECT DISTINCT agent_id
        # в list_active_agent_ids — для emotional_tick'а в worker'е.
        ("emotional", "state.py"),
    }
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        if path.name == "queries.py" or path.parts[-2:] == ("storage", "schema"):
            continue
        rel_parts = path.relative_to(src_root).parts
        if rel_parts in exempt_paths:
            continue
        text = path.read_text(encoding="utf-8")
        if PAT.search(text):
            offenders.append(str(path.relative_to(src_root)))

    assert not offenders, f"raw SQL вне queries.py: {offenders}"
