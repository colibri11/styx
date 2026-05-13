"""Тесты enqueue_classification + classifier_run_at idempotency."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'enqueue-test-%'")
        cur.execute(
            "DELETE FROM llm_tasks WHERE task_type = 'usage_classification' "
            "  AND payload->>'agent_id' LIKE 'enqueue-test-%'"
        )
    conn.commit()
    conn.close()


def _seed_recall_event(
    conn: psycopg.Connection, agent: str, content: str = "x"
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content) "
            "VALUES (%s, 'user', %s) RETURNING id",
            (agent, content),
        )
        memory_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO recall_events (memory_id, query_hash, match_score) "
            "VALUES (%s, %s, %s) RETURNING id",
            (memory_id, b"\x00" * 32, 0.5),
        )
        recall_id = cur.fetchone()[0]
    conn.commit()
    return recall_id


def test_enqueue_classification_inserts_pending_task(db) -> None:
    agent = f"enqueue-test-{uuid.uuid4().hex[:6]}"
    r1 = _seed_recall_event(db, agent)
    r2 = _seed_recall_event(db, agent)

    q = AgentScopedQueries(db, agent)
    q.enqueue_classification(
        recall_event_ids=[r1, r2], llm_output_text="ответ агента"
    )
    db.commit()

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT task_type, status, payload "
            "  FROM llm_tasks "
            " WHERE task_type = 'usage_classification' "
            "   AND payload->>'agent_id' = %s",
            (agent,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row["status"] == "pending"
    assert row["payload"]["recall_event_ids"] == [r1, r2]
    assert row["payload"]["agent_id"] == agent
    assert row["payload"]["llm_output_text"] == "ответ агента"


def test_enqueue_classification_sets_classifier_run_at(db) -> None:
    agent = f"enqueue-test-{uuid.uuid4().hex[:6]}"
    r1 = _seed_recall_event(db, agent)

    q = AgentScopedQueries(db, agent)
    q.enqueue_classification(recall_event_ids=[r1], llm_output_text="x")
    db.commit()

    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT classifier_run_at FROM recall_events WHERE id = %s", (r1,)
        )
        row = cur.fetchone()
    assert row["classifier_run_at"] is not None


def test_enqueue_classification_idempotent_via_run_at_guard(db) -> None:
    """Повторный enqueue для уже classified ids — UPDATE no-op'ит,
    но INSERT в llm_tasks происходит (handler разберётся в reconcile)."""
    agent = f"enqueue-test-{uuid.uuid4().hex[:6]}"
    r1 = _seed_recall_event(db, agent)

    q = AgentScopedQueries(db, agent)
    q.enqueue_classification(recall_event_ids=[r1], llm_output_text="первый")
    db.commit()
    q.enqueue_classification(recall_event_ids=[r1], llm_output_text="второй")
    db.commit()

    # Один classifier_run_at у recall_event (UPDATE guard'ит IS NULL,
    # второй раз UPDATE не сработает).
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT classifier_run_at FROM recall_events WHERE id = %s", (r1,))
        row = cur.fetchone()
    assert row["classifier_run_at"] is not None

    # Но в llm_tasks два task'а — это ОК, handler reconcile'ит.
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM llm_tasks "
            " WHERE task_type = 'usage_classification' "
            "   AND payload->>'agent_id' = %s",
            (agent,),
        )
        n = cur.fetchone()[0]
    assert n == 2


def test_enqueue_empty_list_noop(db) -> None:
    agent = f"enqueue-test-{uuid.uuid4().hex[:6]}"
    q = AgentScopedQueries(db, agent)
    q.enqueue_classification(recall_event_ids=[], llm_output_text="x")
    db.commit()
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM llm_tasks "
            " WHERE task_type = 'usage_classification' "
            "   AND payload->>'agent_id' = %s",
            (agent,),
        )
        assert cur.fetchone()[0] == 0


def test_record_recall_event_returns_id(db) -> None:
    agent = f"enqueue-test-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content) "
            "VALUES (%s, 'user', 'x') RETURNING id",
            (agent,),
        )
        memory_id = cur.fetchone()[0]
    db.commit()

    q = AgentScopedQueries(db, agent)
    rec_id = q.record_recall_event(
        memory_id=memory_id, query_hash=b"\x01" * 32, match_score=0.7
    )
    db.commit()
    assert isinstance(rec_id, int)
    assert rec_id > 0

    # Повторный вызов с тем же query_hash — ON CONFLICT DO UPDATE
    # тот же id (UNIQUE по memory_id+query_hash).
    rec_id2 = q.record_recall_event(
        memory_id=memory_id, query_hash=b"\x01" * 32, match_score=0.9
    )
    db.commit()
    assert rec_id2 == rec_id
