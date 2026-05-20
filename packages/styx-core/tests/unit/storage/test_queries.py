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


# ── insert_message content-length invariant (Defect-fix 2 / core-инвариант) ──


def test_insert_message_rejects_content_over_limit(
    conn: psycopg.Connection,
) -> None:
    """insert_message бросает ContentTooLongError при content > 2400 —
    страховка нижнего уровня, до CheckViolation Postgres'а."""
    from styx.storage.queries import ContentTooLongError, MEMORIES_CONTENT_LIMIT

    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    too_long = "x" * (MEMORIES_CONTENT_LIMIT + 1)
    with pytest.raises(ContentTooLongError):
        q.insert_message(role="user", content=too_long, session_id=sid)


def test_insert_message_accepts_content_at_limit(
    conn: psycopg.Connection,
) -> None:
    from styx.storage.queries import MEMORIES_CONTENT_LIMIT

    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    at_limit = "y" * MEMORIES_CONTENT_LIMIT
    mid = q.insert_message(role="user", content=at_limit, session_id=sid)
    assert isinstance(mid, uuid.UUID)


def test_insert_memory_rejects_content_over_limit(
    conn: psycopg.Connection,
) -> None:
    """insert_memory бросает ContentTooLongError при content > 2400 —
    симметричная страховка к insert_message (m3, defense-in-depth)."""
    from styx.storage.queries import ContentTooLongError, MEMORIES_CONTENT_LIMIT

    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    too_long = "z" * (MEMORIES_CONTENT_LIMIT + 1)
    with pytest.raises(ContentTooLongError):
        q.insert_memory(
            role="user", content=too_long, kind="episode",
            kind_src="subjective", session_id=sid,
        )


def test_insert_memory_accepts_content_at_limit(
    conn: psycopg.Connection,
) -> None:
    """Легальный subjective-write (≤ лимита) guard не задевает."""
    from styx.storage.queries import MEMORIES_CONTENT_LIMIT

    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    at_limit = "w" * MEMORIES_CONTENT_LIMIT
    mid = q.insert_memory(
        role="user", content=at_limit, kind="episode",
        kind_src="subjective", session_id=sid,
    )
    assert isinstance(mid, uuid.UUID)


# ── recent_messages group-aware behaviour (Defect-fix B) ──────────────


def _seed_group(
    q: AgentScopedQueries, sid: uuid.UUID, group: str, parts: list[str],
    role: str = "user",
) -> None:
    for idx, text in enumerate(parts):
        q.insert_message(
            role=role, content=text, session_id=sid,
            metadata={"msg_group": group, "part": idx, "parts": len(parts)},
        )


def test_recent_messages_reassembles_group(conn: psycopg.Connection) -> None:
    """Ряды одной группы склеиваются обратно в один блок."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    _seed_group(q, sid, "g1", ["Часть А. ", "Часть Б."])
    conn.commit()

    recent = q.recent_messages(limit=10, session_id=sid)
    assert len(recent) == 1
    assert recent[0].content == "Часть А. Часть Б."
    # group-маркеры сняты.
    assert "msg_group" not in recent[0].metadata


def test_recent_messages_no_reassemble_returns_raw_parts(
    conn: psycopg.Connection,
) -> None:
    """reassemble_groups=False — сырые ряды с group-маркерами."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    _seed_group(q, sid, "g1", ["Часть А. ", "Часть Б."])
    conn.commit()

    recent = q.recent_messages(
        limit=10, session_id=sid, reassemble_groups=False,
    )
    assert len(recent) == 2
    assert all("msg_group" in m.metadata for m in recent)


def test_recent_messages_limit_does_not_cut_group(
    conn: psycopg.Connection,
) -> None:
    """LIMIT-boundary: limit обрывает группу — добираем недостающие
    части, чтобы свежая реплика не въехала в окно половиной."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    # Сначала старая реплика, потом группа из 3 рядов.
    q.insert_message(role="user", content="старое", session_id=sid)
    _seed_group(q, sid, "g1", ["p0. ", "p1. ", "p2."])
    conn.commit()

    # limit=2 обрезал бы группу (попали бы только p2 и p1).
    recent = q.recent_messages(limit=2, session_id=sid)
    # Группа добрана целиком → собрана в один блок.
    contents = [m.content for m in recent]
    assert "p0. p1. p2." in contents
    # Свежая группа не обрезана — все 3 части склеены.
    group_block = next(c for c in contents if "p0." in c)
    assert group_block == "p0. p1. p2."


def test_recent_messages_plain_message_unaffected(
    conn: psycopg.Connection,
) -> None:
    """Обычные реплики (без msg_group) ведут себя как раньше — DESC."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    sid = uuid.uuid4()
    q.upsert_session(sid)
    q.insert_message(role="user", content="один", session_id=sid)
    q.insert_message(role="assistant", content="два", session_id=sid)
    conn.commit()
    recent = q.recent_messages(limit=10, session_id=sid)
    assert [m.content for m in recent] == ["два", "один"]
