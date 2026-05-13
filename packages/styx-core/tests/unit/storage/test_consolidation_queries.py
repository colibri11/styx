"""Тесты AgentScopedQueries методов для memory consolidation (волна 22).

Postgres-skip: на host без БД скипается. Запускается в Docker
integration suite.
"""

from __future__ import annotations

import datetime as _dt
import uuid

import psycopg
import pytest

from styx.storage import migrate
from styx.storage.queries import (
    AgentScopedQueries,
    enqueue_llm_task,
    get_memory_daily_state,
    parse_vector,
    set_memory_daily_state,
)


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    base[1] = (1.0 - seed * seed) ** 0.5 if seed * seed <= 1.0 else 0.0
    return base


# ── select_consolidation_window ──────────────────────────────────────


def test_window_returns_nothing_on_empty_db(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    rows = q.select_consolidation_window(
        window_from=now - _dt.timedelta(days=7),
        window_to=now - _dt.timedelta(hours=24),
    )
    assert rows == []


def test_window_excludes_consolidation_daily(conn: psycopg.Connection) -> None:
    """kind_src='dialogue_consolidation_daily' отсекается (рекурсия).

    Расчётный возрастной фильтр требует чтобы memory была старше 24h —
    в тесте сдвигаем `created_at` через UPDATE.
    """
    q = AgentScopedQueries(conn, agent_id="alpha")
    keep = q.insert_memory(
        role="summary", content="оставляем",
        kind="note", kind_src="dialogue_batch_consolidation",
        embedding=_embed(0.1),
    )
    drop = q.insert_memory(
        role="summary", content="отбрасываем",
        kind="note", kind_src="dialogue_consolidation_daily",
        embedding=_embed(0.1),
    )
    # Сдвинем created_at в прошлое чтобы попало в окно.
    long_ago = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=2)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET created_at = %s WHERE id IN (%s, %s)",
            (long_ago, keep, drop),
        )
    conn.commit()

    now = _dt.datetime.now(tz=_dt.timezone.utc)
    rows = q.select_consolidation_window(
        window_from=now - _dt.timedelta(days=7),
        window_to=now - _dt.timedelta(hours=24),
    )
    ids = {r["id"] for r in rows}
    assert keep in ids
    assert drop not in ids


def test_window_excludes_superseded_and_null_embedding(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    superseded = q.insert_memory(
        role="summary", content="superseded",
        kind="note", kind_src="subjective",
        embedding=_embed(0.1),
    )
    new = q.insert_memory(
        role="summary", content="new",
        kind="note", kind_src="subjective",
        embedding=_embed(0.2),
    )
    null_emb = q.insert_memory(
        role="summary", content="без вектора",
        kind="note", kind_src="subjective",
    )
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by = %s WHERE id = %s",
            (new, superseded),
        )
        # Сдвинем все в прошлое чтобы попасть в окно.
        long_ago = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=2)
        cur.execute(
            "UPDATE memories SET created_at = %s "
            "WHERE id IN (%s, %s, %s)",
            (long_ago, superseded, new, null_emb),
        )
    conn.commit()
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    rows = q.select_consolidation_window(
        window_from=now - _dt.timedelta(days=7),
        window_to=now - _dt.timedelta(hours=24),
    )
    ids = {r["id"] for r in rows}
    assert superseded not in ids
    assert null_emb not in ids
    assert new in ids


def test_window_filters_by_agent(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    own = a.insert_memory(
        role="summary", content="alpha", kind="note", kind_src="subjective",
        embedding=_embed(0.1),
    )
    foreign = b.insert_memory(
        role="summary", content="beta", kind="note", kind_src="subjective",
        embedding=_embed(0.1),
    )
    long_ago = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=2)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET created_at = %s WHERE id IN (%s, %s)",
            (long_ago, own, foreign),
        )
    conn.commit()
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    rows = a.select_consolidation_window(
        window_from=now - _dt.timedelta(days=7),
        window_to=now - _dt.timedelta(hours=24),
    )
    ids = {r["id"] for r in rows}
    assert own in ids
    assert foreign not in ids


# ── insert_memory_consolidation_application ─────────────────────────


def test_insert_application_validates_min_2_sources(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    task_id = enqueue_llm_task(
        conn, task_type="memory_daily_consolidation", payload={},
    )
    with pytest.raises(ValueError):
        q.insert_memory_consolidation_application(
            task_id=task_id, source_ids=[uuid.uuid4()],
        )


def test_insert_application_returns_id_and_loads(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sources = [
        q.insert_memory(
            role="summary", content=f"src{i}", kind="note",
            kind_src="subjective", embedding=_embed(0.1),
        )
        for i in range(3)
    ]
    task_id = enqueue_llm_task(
        conn, task_type="memory_daily_consolidation", payload={},
    )
    app_id = q.insert_memory_consolidation_application(
        task_id=task_id, source_ids=sources,
    )
    conn.commit()
    rows = q.load_pending_consolidation_applications()
    assert len(rows) == 1
    assert rows[0]["application_id"] == app_id
    assert len(rows[0]["source_ids"]) == 3


# ── load_memories_for_consolidation ─────────────────────────────────


def test_load_memories_preserves_order(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    ids = [
        q.insert_memory(
            role="summary", content=f"src{i}", kind="note",
            kind_src="subjective", embedding=_embed(0.1),
        )
        for i in range(3)
    ]
    conn.commit()
    # Запросим в обратном порядке — load должен вернуть в порядке payload'а.
    reversed_ids = list(reversed(ids))
    rows = q.load_memories_for_consolidation(reversed_ids)
    assert [r["id"] for r in rows] == reversed_ids


def test_load_memories_filters_by_agent(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    own = a.insert_memory(
        role="summary", content="alpha", kind="note", kind_src="subjective",
        embedding=_embed(0.1),
    )
    foreign = b.insert_memory(
        role="summary", content="beta", kind="note", kind_src="subjective",
        embedding=_embed(0.1),
    )
    conn.commit()
    rows = a.load_memories_for_consolidation([own, foreign])
    assert len(rows) == 1
    assert rows[0]["id"] == own


# ── insert_consolidated_memory + supersede ──────────────────────────


def test_insert_consolidated_memory_kind_src_and_metadata(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sources = [
        q.insert_memory(
            role="summary", content=f"src{i}", kind="note",
            kind_src="subjective", embedding=_embed(0.1),
        )
        for i in range(3)
    ]
    new_id = q.insert_consolidated_memory(
        content="merged", embedding=_embed(0.5),
        kind="note", visibility="shared",
        source_ids=sources, application_id=42,
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind_src, content, kind, visibility, "
            "       importance_provisional, metadata "
            "FROM memories WHERE id = %s",
            (new_id,),
        )
        row = cur.fetchone()
    assert row[0] == "dialogue_consolidation_daily"
    assert row[1] == "merged"
    assert row[2] == "note"
    assert row[3] == "shared"
    assert float(row[4]) == 0.7
    cons_meta = row[5]["consolidation"]
    assert cons_meta["source_count"] == 3
    assert cons_meta["llm_task_application_id"] == 42
    assert sorted(cons_meta["source_ids"]) == sorted(str(s) for s in sources)


def test_supersede_sources_idempotent_with_null_filter(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sources = [
        q.insert_memory(
            role="summary", content=f"src{i}", kind="note",
            kind_src="subjective", embedding=_embed(0.1),
        )
        for i in range(3)
    ]
    new1 = q.insert_consolidated_memory(
        content="m1", embedding=_embed(0.5), kind="note",
        visibility="shared", source_ids=sources, application_id=1,
    )
    rc1 = q.mark_consolidation_sources_superseded(
        new_memory_id=new1, source_ids=sources,
    )
    assert rc1 == 3
    # Повторный вызов с другим new_memory_id должен пропустить уже
    # superseded ряды.
    new2 = q.insert_consolidated_memory(
        content="m2", embedding=_embed(0.5), kind="note",
        visibility="shared", source_ids=sources, application_id=2,
    )
    rc2 = q.mark_consolidation_sources_superseded(
        new_memory_id=new2, source_ids=sources,
    )
    assert rc2 == 0
    conn.commit()


def test_mark_consolidation_applied_transitions(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    sources = [
        q.insert_memory(
            role="summary", content=f"src{i}", kind="note",
            kind_src="subjective", embedding=_embed(0.1),
        )
        for i in range(2)
    ]
    task_id = enqueue_llm_task(
        conn, task_type="memory_daily_consolidation", payload={},
    )
    app_id = q.insert_memory_consolidation_application(
        task_id=task_id, source_ids=sources,
    )
    new_id = q.insert_consolidated_memory(
        content="m", embedding=_embed(0.5), kind="note",
        visibility="shared", source_ids=sources, application_id=app_id,
    )
    q.mark_consolidation_applied(application_id=app_id, new_memory_id=new_id)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, new_memory_id FROM "
            "memory_consolidation_applications WHERE id = %s",
            (app_id,),
        )
        row = cur.fetchone()
    assert row[0] == "applied"
    assert row[1] == new_id


# ── KV state helpers ─────────────────────────────────────────────────


def test_memory_daily_state_upsert_and_read(
    conn: psycopg.Connection,
) -> None:
    assert get_memory_daily_state(conn, "alpha") is None
    state = {
        "last_run_at": "2026-05-05T12:00:00+00:00",
        "last_window_to": "2026-05-05T12:00:00+00:00",
        "last_enqueued": 3,
    }
    set_memory_daily_state(conn, "alpha", state)
    conn.commit()
    got = get_memory_daily_state(conn, "alpha")
    assert got == state

    # Update — ON CONFLICT.
    state2 = {**state, "last_enqueued": 5}
    set_memory_daily_state(conn, "alpha", state2)
    conn.commit()
    assert get_memory_daily_state(conn, "alpha") == state2


def test_memory_daily_state_isolation_per_agent(
    conn: psycopg.Connection,
) -> None:
    set_memory_daily_state(conn, "alpha", {"last_enqueued": 1})
    set_memory_daily_state(conn, "beta", {"last_enqueued": 2})
    conn.commit()
    assert get_memory_daily_state(conn, "alpha") == {"last_enqueued": 1}
    assert get_memory_daily_state(conn, "beta") == {"last_enqueued": 2}
