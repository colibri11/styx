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


def test_batch_memory_inherits_latest_source_affective_snapshot(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, valence, arousal, dominance, confidence, "
            " causal_context, computation_version, at) "
            "VALUES ('alpha', 0.4, 0.2, -0.1, 0.75, "
            " '[{\"evidence_id\": 7, \"source_ref\": \"turn-source\", "
            "\"cause_class\": \"execution_risk\", \"status\": \"active\"}]'::jsonb, "
            " 'test-v1', '2026-01-01T00:00:00Z') "
            "RETURNING id",
        )
        state_id = cur.fetchone()[0]
    source_id = q.insert_message(role="assistant", content="source turn")

    # Worker-time state must never replace the source window's snapshot.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, valence, arousal, dominance, confidence, "
            " computation_version, at) "
            "VALUES ('alpha', -0.9, 0.9, 0.9, 1.0, 'worker', "
            " '2026-01-02T00:00:00Z')",
        )

    memory_id = q.insert_batch_memory(
        content="summary", archive_ref={}, source_ids=[source_id],
    )
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT emotional_context_state_id, emotional_context_valence, "
            "       emotional_context_confidence, emotional_context_causes "
            "FROM memories WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()

    assert row["emotional_context_state_id"] == state_id
    assert float(row["emotional_context_valence"]) == pytest.approx(0.4)
    assert float(row["emotional_context_confidence"]) == pytest.approx(0.75)
    assert row["emotional_context_causes"][0]["source_ref"] == "turn-source"


def test_batch_memory_keeps_unknown_legacy_source_affect_null(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    source_id = q.insert_message(role="user", content="legacy source")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, valence, arousal, dominance, confidence, "
            " computation_version) "
            "VALUES ('alpha', 0.9, 0.9, 0.9, 1.0, 'worker')",
        )
    memory_id = q.insert_batch_memory(
        content="derived", archive_ref={}, source_ids=[source_id],
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT emotional_context_valence, emotional_context_state_id "
            "FROM memories WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
    assert row == (None, None)


def test_daily_consolidation_inherits_latest_source_affective_snapshot(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    source_ids: list[uuid.UUID] = []
    for index, valence in enumerate((-0.6, 0.7), start=1):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emotional_state "
                "(agent_id, at, valence, arousal, dominance, confidence, "
                " causal_context, computation_version) "
                "VALUES ('alpha', %s, %s, 0.1, 0.2, 0.8, %s, 'test-v1')",
                (
                    _dt.datetime(2026, 1, index, tzinfo=_dt.timezone.utc),
                    valence,
                    psycopg.types.json.Jsonb([{"source": index}]),
                ),
            )
        source_ids.append(
            q.insert_memory(
                role="summary",
                content=f"source-{index}",
                kind="note",
                kind_src="subjective",
                embedding=_embed(0.1),
            )
        )

    # Состояние worker'а не должно подменять контекст исходных memories.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, at, valence, arousal, dominance, confidence, "
            " computation_version) "
            "VALUES ('alpha', '2026-01-03T00:00:00Z', -0.9, 0, 0, 1, "
            " 'worker-state')",
        )

    new_id = q.insert_consolidated_memory(
        content="merged",
        embedding=_embed(0.5),
        kind="note",
        visibility="shared",
        source_ids=source_ids,
        application_id=7,
    )
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT emotional_context_valence, emotional_context_at, "
            "       emotional_context_causes FROM memories WHERE id = %s",
            (new_id,),
        )
        row = cur.fetchone()

    assert float(row["emotional_context_valence"]) == pytest.approx(0.7)
    assert row["emotional_context_at"] == _dt.datetime(
        2026, 1, 2, tzinfo=_dt.timezone.utc
    )
    assert row["emotional_context_causes"] == [{"source": 2}]


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
