"""Тесты AgentScopedQueries методов для reinterpret (волна 22).

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
    worker_load_memory_for_reinterpret,
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


def _make_memory(
    queries: AgentScopedQueries, *, content: str = "test"
) -> uuid.UUID:
    return queries.insert_memory(
        role="summary", content=content,
        kind="note", kind_src="subjective",
        embedding=_embed(0.1),
    )


def _set_causal_provenance(
    conn: psycopg.Connection,
    memory_id: uuid.UUID,
    provenance: str,
) -> None:
    if provenance == "validated_act_residue":
        act_id = uuid.uuid4()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT agent_id FROM memories WHERE id=%s", (memory_id,)
            )
            agent_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO cognitive_acts "
                "(id,agent_id,host_key,status,completed_at) "
                "VALUES (%s,%s,%s,'completed',clock_timestamp())",
                (act_id, agent_id, f"test-{act_id}"),
            )
            cur.execute(
                "UPDATE memories SET line_provenance=%s,cognitive_act_id=%s,"
                " residue_ordinal=0,residue_reducer_version='test_v1',"
                " residue_input_hash=%s,residue_causal_role='choice',"
                " residue_confidence=0.8,"
                " residue_evidence='[{\"source\":\"channel_output\","
                "\"key\":\"assistant_response\"}]'::jsonb,"
                " residue_predecessors='[]'::jsonb,residue_line_root_hash=%s,"
                " causal_node_hash=%s,causal_node_kind='act_residue',"
                " causal_payload_version='causal_node_v1',line_status='active' "
                "WHERE id=%s",
                (
                    provenance, act_id, "b" * 64, "a" * 64,
                    "c" * 64, memory_id,
                ),
            )
        return
    with conn.cursor() as cur:
        operation_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO causal_operations ("
            "id,agent_id,operation_key,operation_kind,input_line_version,"
            "input_root_hash,request_hash,algorithm_name,algorithm_version,status) "
            "VALUES (%s,'alpha',%s,'reinterpret',0,%s,%s,'test','v1','running')",
            (operation_id, f"test:{memory_id}", "0" * 64, "d" * 64),
        )
        cur.execute(
            "UPDATE memories SET line_provenance=%s,causal_node_hash=%s,"
            "causal_node_kind='reinterpretation',"
            "causal_payload_version='causal_node_v1',causal_operation_id=%s,"
            "line_status='active' WHERE id=%s",
            (provenance, "e" * 64, operation_id, memory_id),
        )


# ── find_pending_reinterpret_application ─────────────────────────────


def test_find_pending_returns_none_for_fresh_memory(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    conn.commit()
    assert q.find_pending_reinterpret_application(mid) is None


def test_find_pending_returns_id_when_pending(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge",
        payload={"agent_id": "alpha", "new_understanding_text": "x"},
    )
    app_id = q.insert_reinterpret_application(task_id=task_id, memory_id=mid)
    conn.commit()
    assert q.find_pending_reinterpret_application(mid) == app_id


def test_find_pending_filters_by_agent(conn: psycopg.Connection) -> None:
    """Cross-agent isolation: agent A не видит pending агента B."""
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    mid_a = _make_memory(a)
    mid_b = _make_memory(b, content="beta memory")
    task_a = enqueue_llm_task(
        conn, task_type="reinterpret_merge", payload={"x": 1},
    )
    a.insert_reinterpret_application(task_id=task_a, memory_id=mid_a)
    task_b = enqueue_llm_task(
        conn, task_type="reinterpret_merge", payload={"x": 1},
    )
    b.insert_reinterpret_application(task_id=task_b, memory_id=mid_b)
    conn.commit()
    # alpha видит только свою.
    assert a.find_pending_reinterpret_application(mid_a) is not None
    assert a.find_pending_reinterpret_application(mid_b) is None


# ── latest_reinterpretation_at ───────────────────────────────────────


def test_latest_reinterpretation_at_none_for_fresh_memory(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    conn.commit()
    assert q.latest_reinterpretation_at(mid) is None


def test_latest_reinterpretation_at_returns_most_recent(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    rid1 = q.insert_memory_reinterpretation(
        memory_id=mid, previous_text="a", new_understanding_text="b",
        merged_text="c",
        previous_embedding=_embed(0.1), merged_embedding=_embed(0.2),
        weight_applied=0.5,
    )
    rid2 = q.insert_memory_reinterpretation(
        memory_id=mid, previous_text="c", new_understanding_text="d",
        merged_text="e",
        previous_embedding=_embed(0.2), merged_embedding=_embed(0.3),
        weight_applied=0.5,
    )
    conn.commit()
    assert rid1 != rid2
    last = q.latest_reinterpretation_at(mid)
    assert last is not None
    assert isinstance(last, _dt.datetime)


# ── memory_exists ────────────────────────────────────────────────────


def test_memory_exists_true_for_own(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    conn.commit()
    assert q.memory_exists(mid) is True


def test_memory_exists_false_for_other_agent(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    mid_b = _make_memory(b, content="beta")
    conn.commit()
    assert a.memory_exists(mid_b) is False


def test_memory_exists_false_for_unknown(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    assert q.memory_exists(uuid.uuid4()) is False


@pytest.mark.parametrize(
    "provenance", ["validated_act_residue", "validated_transform"]
)
def test_reinterpret_routes_active_causal_rows_without_allowing_legacy_mutation(
    conn: psycopg.Connection, provenance: str
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q, content="canonical")
    _set_causal_provenance(conn, mid, provenance)
    conn.commit()

    assert q.memory_exists(mid) is True
    assert worker_load_memory_for_reinterpret(conn, mid) is not None
    assert q.apply_reinterpret_update(
        memory_id=mid,
        merged_text="mutated",
        merged_embedding=_embed(0.5),
    ) == 0
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge", payload={}
    )
    assert q.insert_reinterpret_application(task_id=task_id, memory_id=mid) > 0
    with conn.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id=%s", (mid,))
        assert cur.fetchone()[0] == "canonical"


# ── insert_reinterpret_application + load_pending ───────────────────


def test_insert_application_returns_id_and_loads(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge", payload={"x": 1},
    )
    app_id = q.insert_reinterpret_application(task_id=task_id, memory_id=mid)
    conn.commit()
    rows = q.load_pending_reinterpret_applications()
    assert len(rows) == 1
    row = rows[0]
    assert row["application_id"] == app_id
    assert row["memory_id"] == str(mid)
    assert row["task_status"] == "pending"


def test_load_pending_filters_by_agent(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    mid_a = _make_memory(a)
    mid_b = _make_memory(b, content="beta")
    ta = enqueue_llm_task(conn, task_type="reinterpret_merge", payload={})
    tb = enqueue_llm_task(conn, task_type="reinterpret_merge", payload={})
    a.insert_reinterpret_application(task_id=ta, memory_id=mid_a)
    b.insert_reinterpret_application(task_id=tb, memory_id=mid_b)
    conn.commit()
    rows_a = a.load_pending_reinterpret_applications()
    rows_b = b.load_pending_reinterpret_applications()
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0]["memory_id"] == str(mid_a)
    assert rows_b[0]["memory_id"] == str(mid_b)


# ── apply_reinterpret_update + insert_memory_reinterpretation ────────


def test_apply_reinterpret_update_changes_content_and_embedding(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q, content="старое")
    new_emb = _embed(0.5)
    rc = q.apply_reinterpret_update(
        memory_id=mid, merged_text="новое", merged_embedding=new_emb,
    )
    conn.commit()
    assert rc == 1
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content, embedding FROM memories WHERE id = %s",
            (mid,),
        )
        row = cur.fetchone()
    assert row[0] == "новое"
    # embedding в pgvector — string-литерал; проверим длину парсингом.
    from styx.storage.queries import parse_vector
    parsed = parse_vector(row[1])
    assert parsed is not None
    assert len(parsed) == 768


def test_apply_reinterpret_update_other_agent_no_op(
    conn: psycopg.Connection,
) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    mid_b = _make_memory(b, content="beta memory")
    conn.commit()
    rc = a.apply_reinterpret_update(
        memory_id=mid_b, merged_text="hijack",
        merged_embedding=_embed(0.5),
    )
    assert rc == 0


def test_insert_memory_reinterpretation_validates_embeddings(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    conn.commit()
    with pytest.raises(ValueError):
        q.insert_memory_reinterpretation(
            memory_id=mid, previous_text="a", new_understanding_text="b",
            merged_text="c",
            previous_embedding=[], merged_embedding=_embed(0.1),
            weight_applied=0.5,
        )
    with pytest.raises(ValueError):
        q.insert_memory_reinterpretation(
            memory_id=mid, previous_text="a", new_understanding_text="b",
            merged_text="c",
            previous_embedding=[1.0, 2.0], merged_embedding=_embed(0.1),
            weight_applied=0.5,
        )


# ── mark_*_skipped / mark_*_applied ──────────────────────────────────


def test_mark_applied_transitions_status(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge", payload={},
    )
    app_id = q.insert_reinterpret_application(task_id=task_id, memory_id=mid)
    q.mark_reinterpret_applied(app_id)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM reinterpret_applications WHERE id = %s",
            (app_id,),
        )
        assert cur.fetchone()[0] == "applied"


def test_mark_skipped_idempotent_for_already_applied(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _make_memory(q)
    task_id = enqueue_llm_task(
        conn, task_type="reinterpret_merge", payload={},
    )
    app_id = q.insert_reinterpret_application(task_id=task_id, memory_id=mid)
    q.mark_reinterpret_applied(app_id)
    conn.commit()
    # mark_skipped после applied — WHERE status=pending_sleep отфильтрует.
    q.mark_reinterpret_skipped(app_id)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM reinterpret_applications WHERE id = %s",
            (app_id,),
        )
        assert cur.fetchone()[0] == "applied"
