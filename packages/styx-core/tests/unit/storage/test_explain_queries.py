"""Тесты AgentScopedQueries методов для explain/analytics/confirm_usage
(волна 25).

Покрытия:
- ``factor_select_columns`` — генерирует SELECT-фрагмент со всеми
  ожидаемыми aliases.
- ``explain_decompose_target`` / ``explain_decompose_rank`` — happy
  path + agent_id isolation.
- ``explain_lifetime_main`` / ``_recall_history`` / ``_co_retrieval``.
- ``explain_topk`` — items ordered by score; total ≥ items.
- ``analytics_for_agent`` — counts, agent isolation.
- ``confirm_usage_update`` — UPDATE recall_events; cross-agent guard;
  idempotent.

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import datetime as _dt
import uuid

import psycopg
import pytest

from styx.storage import migrate
from styx.storage.queries import (
    AgentScopedQueries,
    factor_select_columns,
)
from styx.storage.scoring import (
    BuildFactorExprsOptions,
    build_factor_exprs,
)


# ── factor_select_columns: чистый string-test (PG не нужен) ─────────


def test_factor_select_columns_contains_all_aliases() -> None:
    factors = build_factor_exprs(
        {"text_query": "hi"},
        BuildFactorExprsOptions(
            text_query_param_index=2,
            usage_norm_p75=0.0,
            emotional_baseline=None,
        ),
    )
    out = factor_select_columns(factors)
    expected_aliases = [
        "AS vector_sim",
        "AS bm25_rank",
        "AS base_match",
        "AS relevance_factor",
        "AS recency_factor",
        "AS frequency_factor",
        "AS lifecycle_factor",
        "AS feedback_factor",
        "AS importance_factor",
        "AS importance_effective",
        "AS diversity_factor",
        "AS decay_factor",
        "AS lambda_base",
        "AS effective_lambda",
        "AS usage_count_30d",
        "AS usage_factor",
        "AS emotional_resonance_factor",
        "AS age_days_sql",
        "AS final_score",
    ]
    for alias in expected_aliases:
        assert alias in out, f"missing alias: {alias}"


def test_factor_select_columns_pure_vector_bm25_null() -> None:
    """Без text_query → bm25 column становится NULL литералом."""
    factors = build_factor_exprs(
        {},
        BuildFactorExprsOptions(
            text_query_param_index=None,
            usage_norm_p75=0.0,
            emotional_baseline=None,
        ),
    )
    out = factor_select_columns(factors)
    assert "NULL::real AS bm25_rank" in out


# ── PG-зависимые тесты (skip при отсутствии STYX_TEST_DATABASE_URL) ─


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _embed(seed: float, dim: int = 768) -> list[float]:
    base = [0.0] * dim
    base[0] = seed
    return base


def _insert_subjective(
    q: AgentScopedQueries,
    *,
    content: str,
    embedding: list[float] | None,
    kind: str = "fact",
    importance_provisional: float | None = None,
) -> uuid.UUID:
    # Subjective writes → role='summary' (CHECK memories_role_check
    # разрешает {user, assistant, tool, system, summary}).
    return q.insert_memory(
        role="summary",
        content=content,
        kind=kind,
        kind_src="subjective",
        embedding=embedding,
        importance_provisional=importance_provisional,
    )


# ── explain_decompose ──────────────────────────────────────────────


def test_explain_decompose_target_happy(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _insert_subjective(q, content="hello", embedding=_embed(0.7))
    conn.commit()

    row = q.explain_decompose_target(
        memory_id=mid,
        query_vector=_embed(0.7),
        query_text="hello",
    )
    assert row is not None
    assert row["id"] == mid
    assert row["kind"] == "fact"
    assert row["agent_id"] == "alpha"
    # все factor-колонки заполнены
    assert row["vector_sim"] is not None
    assert row["base_match"] is not None
    assert row["recency_factor"] is not None
    assert row["decay_factor"] is not None
    assert row["final_score"] is not None
    assert row["age_days_sql"] is not None


def test_explain_decompose_target_agent_isolation(
    conn: psycopg.Connection,
) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    mid = _insert_subjective(qa, content="alpha-only", embedding=_embed(0.5))
    conn.commit()

    # alpha видит свою memory
    row = qa.explain_decompose_target(
        memory_id=mid, query_vector=_embed(0.5), query_text="alpha-only",
    )
    assert row is not None

    # beta — не видит
    row_b = qb.explain_decompose_target(
        memory_id=mid, query_vector=_embed(0.5), query_text="alpha-only",
    )
    assert row_b is None


def test_explain_decompose_rank_returns_int(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    _insert_subjective(q, content="aaa", embedding=_embed(0.9))
    _insert_subjective(q, content="bbb", embedding=_embed(0.5))
    _insert_subjective(q, content="ccc", embedding=_embed(0.1))
    conn.commit()

    rank = q.explain_decompose_rank(
        target_score=0.0,  # все имеют score > 0 → rank=4 (3+1)
        query_vector=_embed(0.9),
        query_text="aaa",
    )
    assert isinstance(rank, int)
    assert rank >= 1


# ── explain_lifetime ───────────────────────────────────────────────


def test_explain_lifetime_main_happy(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _insert_subjective(q, content="lifetime memory", embedding=None)
    conn.commit()

    row = q.explain_lifetime_main(memory_id=mid)
    assert row is not None
    assert row["id"] == mid
    assert row["agent_id"] == "alpha"
    assert row["kind"] == "fact"
    assert row["age_days"] is not None
    # пока не было recall — нули
    assert int(row["total_recall_events"] or 0) == 0


def test_explain_lifetime_main_other_agent_returns_none(
    conn: psycopg.Connection,
) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    mid = _insert_subjective(qa, content="alpha-only", embedding=None)
    conn.commit()

    assert qb.explain_lifetime_main(memory_id=mid) is None


def test_explain_lifetime_recall_history_orders_desc(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _insert_subjective(q, content="recalled", embedding=None)
    conn.commit()

    qhash1 = b"\x01" * 32
    qhash2 = b"\x02" * 32
    q.record_recall_event(
        memory_id=mid, query_hash=qhash1, match_score=0.5,
    )
    q.record_recall_event(
        memory_id=mid, query_hash=qhash2, match_score=0.7,
    )
    conn.commit()

    history = q.explain_lifetime_recall_history(memory_id=mid, limit=10)
    assert len(history) == 2
    # ORDER BY matched_at DESC — последний record_recall_event вверху
    assert history[0]["match_score"] == pytest.approx(0.7, rel=1e-3)


def test_explain_lifetime_co_retrieval_orders_by_weight(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    a = _insert_subjective(q, content="a", embedding=None)
    b = _insert_subjective(q, content="b-target-preview", embedding=None)
    c = _insert_subjective(q, content="c-target-preview", embedding=None)
    conn.commit()

    # a↔b weight=2.0; a↔c weight=1.1
    q.upsert_co_retrieved_pair(
        source_id=a, target_id=b, initial_weight=2.0,
        weight_bump=0.0, weight_max=10.0,
    )
    q.upsert_co_retrieved_pair(
        source_id=a, target_id=c, initial_weight=1.1,
        weight_bump=0.0, weight_max=10.0,
    )
    conn.commit()

    links = q.explain_lifetime_co_retrieval(memory_id=a, limit=10)
    assert len(links) == 2
    weights = [float(link["weight"]) for link in links]
    assert weights == sorted(weights, reverse=True)
    assert links[0]["target_id"] == b
    assert links[0]["target_content"] == "b-target-preview"


# ── explain_topk ───────────────────────────────────────────────────


def test_explain_topk_orders_by_score_desc(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    _insert_subjective(q, content="strong", embedding=_embed(0.95))
    _insert_subjective(q, content="medium", embedding=_embed(0.5))
    _insert_subjective(q, content="weak",   embedding=_embed(0.05))
    conn.commit()

    rows, total = q.explain_topk(
        query_vector=_embed(0.95),
        query_text="strong",
        limit=3,
    )
    assert total == 3
    contents = [r["content"] for r in rows]
    assert contents[0] == "strong"
    # все factor columns заполнены
    assert all(r["final_score"] is not None for r in rows)


def test_explain_topk_agent_isolation(conn: psycopg.Connection) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    _insert_subjective(qa, content="alpha-only", embedding=_embed(0.5))
    _insert_subjective(qb, content="beta-only",  embedding=_embed(0.5))
    conn.commit()

    rows, total = qa.explain_topk(
        query_vector=_embed(0.5), query_text="x", limit=10,
    )
    assert total == 1
    assert rows[0]["content"] == "alpha-only"


def test_explain_topk_kinds_filter(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    _insert_subjective(
        q, content="fact-row", embedding=_embed(0.5), kind="fact",
    )
    _insert_subjective(
        q, content="note-row", embedding=_embed(0.5), kind="note",
    )
    conn.commit()

    rows, total = q.explain_topk(
        query_vector=_embed(0.5), query_text="x",
        kinds=["fact"], limit=10,
    )
    assert total == 1
    assert rows[0]["content"] == "fact-row"


def test_explain_topk_llm_tasks_returns_dict(
    conn: psycopg.Connection,
) -> None:
    """llm_tasks batch fetch — пустой результат пока нет задач (worker
    отключён в тестах). Метод должен вернуть пустой dict без падения."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _insert_subjective(q, content="x", embedding=None)
    conn.commit()

    out = q.explain_topk_llm_tasks(memory_ids=[mid])
    assert isinstance(out, dict)
    # Триггер `enqueue_importance_scoring` мог положить задачу — проверяем
    # только тип, не содержимое.


def test_explain_topk_llm_tasks_empty_input(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    out = q.explain_topk_llm_tasks(memory_ids=[])
    assert out == {}


# ── analytics_for_agent ────────────────────────────────────────────


def test_analytics_for_agent_counts(conn: psycopg.Connection) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    _insert_subjective(qa, content="a1", embedding=None, kind="fact")
    _insert_subjective(qa, content="a2", embedding=None, kind="note")
    _insert_subjective(qb, content="b1", embedding=None, kind="fact")
    conn.commit()

    out = qa.analytics_for_agent()
    agent = out["agents"][0]
    assert agent["agent_id"] == "alpha"
    assert agent["display_name"] is None
    assert agent["memories_count"] == 2
    assert agent["memories_by_kind"] == {"fact": 1, "note": 1}
    assert agent["dialogue_messages_count"] == 0  # нет user/assistant rows
    # global содержит totals только по этому агенту
    assert out["global"]["total_memories"] == 2


def test_analytics_for_agent_dialogue_count(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    q.insert_message(role="user", content="hello", embedding=None)
    q.insert_message(role="assistant", content="hi", embedding=None)
    q.insert_message(role="tool", content="x", embedding=None)  # не считается
    conn.commit()

    out = q.analytics_for_agent()
    assert out["agents"][0]["dialogue_messages_count"] == 2
    assert out["global"]["total_dialogue_messages"] == 2


def test_analytics_pending_indexing_counts_null_embedding(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    _insert_subjective(q, content="with embed", embedding=_embed(0.3))
    _insert_subjective(q, content="without embed", embedding=None)
    conn.commit()

    out = q.analytics_for_agent()
    pending = out["pending_indexing"]
    # pending_memories — глобально (без agent filter), берёт ≥1 row
    assert pending["memories"] >= 1


# ── confirm_usage_update ───────────────────────────────────────────


def test_confirm_usage_update_flips_used_in_output(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _insert_subjective(q, content="recall me", embedding=None)
    conn.commit()

    qhash = b"\x42" * 32
    re_id = q.record_recall_event(
        memory_id=mid, query_hash=qhash, match_score=0.5,
    )
    conn.commit()

    matched = q.confirm_usage_update(memory_ids=[mid])
    conn.commit()
    assert matched == {mid}

    with conn.cursor() as cur:
        cur.execute(
            "SELECT used_in_output FROM recall_events WHERE id = %s",
            (re_id,),
        )
        flag = cur.fetchone()[0]
    assert flag is True


def test_confirm_usage_cross_agent_guard(conn: psycopg.Connection) -> None:
    qa = AgentScopedQueries(conn, agent_id="alpha")
    qb = AgentScopedQueries(conn, agent_id="beta")
    mid = _insert_subjective(qa, content="alpha-mem", embedding=None)
    conn.commit()

    qa.record_recall_event(
        memory_id=mid, query_hash=b"\x33" * 32, match_score=0.5,
    )
    conn.commit()

    matched = qb.confirm_usage_update(memory_ids=[mid])
    conn.commit()
    assert matched == set()  # beta не имеет доступа к alpha-mem


def test_confirm_usage_idempotent(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = _insert_subjective(q, content="x", embedding=None)
    conn.commit()
    q.record_recall_event(
        memory_id=mid, query_hash=b"\x44" * 32, match_score=0.5,
    )
    conn.commit()

    matched1 = q.confirm_usage_update(memory_ids=[mid])
    conn.commit()
    matched2 = q.confirm_usage_update(memory_ids=[mid])
    conn.commit()
    assert matched1 == matched2 == {mid}


def test_confirm_usage_empty_input(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    assert q.confirm_usage_update(memory_ids=[]) == set()


def test_confirm_usage_missing_memory_returns_empty(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    fake = uuid.uuid4()
    matched = q.confirm_usage_update(memory_ids=[fake])
    conn.commit()
    assert matched == set()
