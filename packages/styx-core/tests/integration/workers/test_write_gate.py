"""Write-gate integration: apply-sweeper'ы держат gate (волна 22).

Реальный Postgres + turn_state.observe в активном turn'е →
ни reinterpret apply, ни memory consolidation apply не применяются;
после close — оба применяются.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from styx import turn_state
from styx.engine.memory_consolidation import MemoryConsolidationConfig
from styx.storage import migrate
from styx.storage.queries import (
    AgentScopedQueries,
    enqueue_llm_task,
)
from styx.workers.sweep.memory_consolidation import (
    run_memory_consolidation_apply_sweep,
)
from styx.workers.sweep.reinterpret_apply import run_reinterpret_apply_sweep


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


@pytest.fixture
def db(clean_db: str):
    migrate.run(clean_db)
    conn = psycopg.connect(clean_db)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def reset_turn_state():
    turn_state.reset()
    yield
    turn_state.reset()


def _seed_memory(conn: psycopg.Connection, agent_id: str = "alpha") -> uuid.UUID:
    q = AgentScopedQueries(conn, agent_id)
    mid = q.insert_memory(
        role="summary", content="старый текст", kind="note",
        kind_src="subjective",
        embedding=[1.0, 0.0] + [0.0] * 766,
    )
    conn.commit()
    return mid


def _enqueue_reinterpret_done(
    conn: psycopg.Connection, agent_id: str, mid: uuid.UUID,
) -> tuple[uuid.UUID, int]:
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge",
        payload={"agent_id": agent_id, "new_understanding_text": "x"},
        memory_id=mid,
    )
    merged = {
        "skip": False, "skip_reason": None,
        "merged_text": "новый объединённый текст",
        "merged_embedding": [0.7, 0.7] + [0.0] * 766,
        "previous_text": "старый текст",
        "previous_embedding": [1.0, 0.0] + [0.0] * 766,
        "new_understanding_text": "x", "weight_applied": 0.5,
        "agent_id": agent_id,
    }
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE llm_tasks SET status='done', result=%s WHERE id=%s",
            (Jsonb(merged), task_id),
        )
    q = AgentScopedQueries(conn, agent_id)
    app_id = q.insert_reinterpret_application(task_id=task_id, memory_id=mid)
    conn.commit()
    return task_id, app_id


def test_reinterpret_apply_blocked_then_unblocked(db) -> None:
    mid = _seed_memory(db)
    _enqueue_reinterpret_done(db, "alpha", mid)

    # Open active turn — apply blocked.
    turn_state.observe("alpha")
    summary1 = run_reinterpret_apply_sweep(db)
    assert summary1.applied == 0
    with db.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id=%s", (mid,))
        assert cur.fetchone()[0] == "старый текст"

    # Close turn — apply succeeds on next sweep.
    turn_state.close("alpha")
    summary2 = run_reinterpret_apply_sweep(db)
    assert summary2.applied == 1
    with db.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id=%s", (mid,))
        assert cur.fetchone()[0] == "новый объединённый текст"


def test_memory_consolidation_apply_blocked_then_unblocked(db) -> None:
    sources = []
    for i in range(3):
        sources.append(_seed_memory(db))
    task_id = enqueue_llm_task(
        db, task_type="memory_daily_consolidation",
        payload={
            "agent_id": "alpha",
            "memory_ids": [str(s) for s in sources],
        },
    )
    merged = {
        "skip": False, "skip_reason": None,
        "consolidated_text": "общий смысл",
        "consolidated_embedding": [0.5] * 768,
        "agent_id": "alpha",
        "source_ids": [str(s) for s in sources],
        "source_kinds": ["note"] * 3,
        "source_visibility": ["shared"] * 3,
    }
    with db.cursor() as cur:
        cur.execute(
            "UPDATE llm_tasks SET status='done', result=%s WHERE id=%s",
            (Jsonb(merged), task_id),
        )
    q = AgentScopedQueries(db, "alpha")
    q.insert_memory_consolidation_application(
        task_id=task_id, source_ids=sources,
    )
    db.commit()

    # Open active turn — apply blocked.
    turn_state.observe("alpha")
    summary1 = run_memory_consolidation_apply_sweep(db)
    assert summary1.applied == 0
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE kind_src='dialogue_consolidation_daily'",
        )
        assert cur.fetchone()[0] == 0

    # Close turn — apply succeeds.
    turn_state.close("alpha")
    summary2 = run_memory_consolidation_apply_sweep(db)
    assert summary2.applied == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE kind_src='dialogue_consolidation_daily'",
        )
        assert cur.fetchone()[0] == 1
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE id=ANY(%s) AND superseded_by IS NOT NULL",
            (sources,),
        )
        assert cur.fetchone()[0] == 3


def test_apply_idle_after_ttl_expiry(db) -> None:
    """observe → wait TTL → next observe opens new cycle, but apply
    using prev observe still sees turn closed (TTL → is_active=False).
    """
    import datetime as _dt
    mid = _seed_memory(db)
    _enqueue_reinterpret_done(db, "alpha", mid)

    now = _dt.datetime.now(tz=_dt.timezone.utc)
    turn_state.observe("alpha", now=now)

    # Через TTL+1 — is_active возвращает False.
    later = now + _dt.timedelta(seconds=120)
    summary = run_reinterpret_apply_sweep(db, now=later)
    assert summary.applied == 1
