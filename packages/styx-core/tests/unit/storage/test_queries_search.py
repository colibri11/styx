"""Тесты search_similar / record_recall_event / update_embedding /
compute_agent_usage_p75 (фаза D)."""

from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries, MemoryHit


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _seed(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    role: str,
    content: str,
    embedding: list[float],
    kind: str = "episode",
) -> uuid.UUID:
    sid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, agent_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (sid, agent_id),
        )
        cur.execute(
            "INSERT INTO memories "
            "(agent_id, session_id, role, content, kind, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (
                agent_id,
                sid,
                role,
                content,
                kind,
                "[" + ",".join(repr(x) for x in embedding) + "]",
            ),
        )
        return cur.fetchone()[0]


def _unit_vec(*nonzero: tuple[int, float], dim: int = 768) -> list[float]:
    """Удобный конструктор: задаёт несколько компонент, остальные 0."""
    v = [0.0] * dim
    for i, x in nonzero:
        v[i] = x
    return v


# ── search_similar ────────────────────────────────────────────────────


def test_search_similar_returns_top_k_by_similarity(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")

    a = _seed(conn, agent_id="alpha", role="user", content="apples and pears",
              embedding=_unit_vec((0, 1.0)))
    b = _seed(conn, agent_id="alpha", role="user", content="orthogonal topic",
              embedding=_unit_vec((1, 1.0)))
    c = _seed(conn, agent_id="alpha", role="user", content="opposite vector",
              embedding=_unit_vec((0, -1.0)))
    conn.commit()

    hits = q.search_similar(query_vector=_unit_vec((0, 1.0)), limit=5)
    ids = [h.id for h in hits]
    # a (sim=1) > b (sim=0) > c (sim=-1).
    assert ids[0] == a
    assert hits[0].score > hits[1].score > hits[2].score
    # base_match = 1 - cosine_distance — для a равно 1.0.
    assert abs(hits[0].match_score - 1.0) < 1e-6


def test_search_similar_excludes_other_agents(conn: psycopg.Connection) -> None:
    """Application-level WHERE по agent_id — соблюдение N1 (decisions §17.1)."""
    _seed(conn, agent_id="alpha", role="user", content="alpha-text",
          embedding=_unit_vec((0, 1.0)))
    _seed(conn, agent_id="beta", role="user", content="beta-text",
          embedding=_unit_vec((0, 1.0)))
    conn.commit()

    q_alpha = AgentScopedQueries(conn, agent_id="alpha")
    hits = q_alpha.search_similar(query_vector=_unit_vec((0, 1.0)), limit=5)
    contents = [h.content for h in hits]
    assert "alpha-text" in contents
    assert "beta-text" not in contents


def test_search_similar_excludes_null_embedding(conn: psycopg.Connection) -> None:
    """Memories без embedding в выборку не попадают."""
    q = AgentScopedQueries(conn, agent_id="gamma")
    sid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, agent_id) VALUES (%s, %s)",
            (sid, "gamma"),
        )
        cur.execute(
            "INSERT INTO memories (agent_id, session_id, role, content) "
            "VALUES (%s, %s, %s, %s)",
            ("gamma", sid, "user", "no embedding"),
        )
    conn.commit()

    hits = q.search_similar(query_vector=_unit_vec((0, 1.0)), limit=5)
    assert hits == []


def test_search_similar_excludes_superseded(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="delta")
    a = _seed(conn, agent_id="delta", role="user", content="superseded",
              embedding=_unit_vec((0, 1.0)))
    b = _seed(conn, agent_id="delta", role="user", content="current",
              embedding=_unit_vec((0, 1.0)))
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by = %s WHERE id = %s",
            (b, a),
        )
    conn.commit()

    hits = q.search_similar(query_vector=_unit_vec((0, 1.0)), limit=5)
    contents = [h.content for h in hits]
    assert contents == ["current"]


def test_search_similar_hybrid_with_query_text(conn: psycopg.Connection) -> None:
    """С query_text — base_match включает BM25 ветку через content_tsv."""
    q = AgentScopedQueries(conn, agent_id="epsilon")
    _seed(conn, agent_id="epsilon", role="user", content="apples pears bananas",
          embedding=_unit_vec((0, 1.0)))
    _seed(conn, agent_id="epsilon", role="user", content="completely orthogonal text",
          embedding=_unit_vec((1, 1.0)))
    conn.commit()

    hits = q.search_similar(
        query_vector=_unit_vec((0, 1.0)),
        query_text="apples bananas",
        limit=5,
    )
    # Hybrid score должен ставить apples-вариант выше.
    assert hits[0].content == "apples pears bananas"


def test_search_similar_include_embedding(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="zeta")
    _seed(conn, agent_id="zeta", role="user", content="x",
          embedding=_unit_vec((0, 1.0), (5, 0.5)))
    conn.commit()

    hits = q.search_similar(
        query_vector=_unit_vec((0, 1.0)),
        limit=5,
        include_embedding=True,
    )
    assert hits[0].embedding is not None
    assert len(hits[0].embedding) == 768


def test_search_similar_no_embedding_default(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="eta")
    _seed(conn, agent_id="eta", role="user", content="x",
          embedding=_unit_vec((0, 1.0)))
    conn.commit()

    hits = q.search_similar(query_vector=_unit_vec((0, 1.0)), limit=5)
    assert hits[0].embedding is None


def test_search_similar_returns_metadata(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="theta")
    _seed(conn, agent_id="theta", role="assistant", content="x",
          embedding=_unit_vec((0, 1.0)), kind="decision")
    conn.commit()

    hits = q.search_similar(query_vector=_unit_vec((0, 1.0)), limit=5)
    h = hits[0]
    assert h.kind == "decision"
    assert h.role == "assistant"
    assert isinstance(h.metadata, dict)


# ── record_recall_event ───────────────────────────────────────────────


def test_record_recall_event_basic(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="iota")
    mid = _seed(conn, agent_id="iota", role="user", content="x",
                embedding=_unit_vec((0, 1.0)))
    conn.commit()

    qhash = hashlib.sha256(b"query about x").digest()
    q.record_recall_event(memory_id=mid, query_hash=qhash, match_score=0.85)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT match_score, query_hash FROM recall_events "
            "WHERE memory_id = %s",
            (mid,),
        )
        rows = cur.fetchall()

    assert len(rows) == 1
    score, stored_hash = rows[0]
    assert abs(score - 0.85) < 1e-5
    assert bytes(stored_hash) == qhash


def test_record_recall_event_upserts_on_duplicate(conn: psycopg.Connection) -> None:
    """Повтор с тем же (memory_id, query_hash) → matched_at и match_score
    обновляются, новой строки нет."""
    q = AgentScopedQueries(conn, agent_id="kappa")
    mid = _seed(conn, agent_id="kappa", role="user", content="x",
                embedding=_unit_vec((0, 1.0)))
    qhash = hashlib.sha256(b"abc").digest()

    q.record_recall_event(memory_id=mid, query_hash=qhash, match_score=0.5)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT matched_at FROM recall_events WHERE memory_id = %s",
            (mid,),
        )
        first_at = cur.fetchone()[0]

    q.record_recall_event(memory_id=mid, query_hash=qhash, match_score=0.9)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), max(matched_at), max(match_score) "
            "FROM recall_events WHERE memory_id = %s",
            (mid,),
        )
        cnt, second_at, score = cur.fetchone()

    assert cnt == 1
    assert second_at >= first_at
    assert abs(score - 0.9) < 1e-5


def test_record_recall_event_different_hashes_make_different_rows(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="lambda")
    mid = _seed(conn, agent_id="lambda", role="user", content="x",
                embedding=_unit_vec((0, 1.0)))

    q.record_recall_event(
        memory_id=mid, query_hash=hashlib.sha256(b"q1").digest(), match_score=0.5
    )
    q.record_recall_event(
        memory_id=mid, query_hash=hashlib.sha256(b"q2").digest(), match_score=0.7
    )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM recall_events WHERE memory_id = %s", (mid,)
        )
        assert cur.fetchone()[0] == 2


# ── update_embedding ──────────────────────────────────────────────────


def test_update_embedding_writes_vector(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="mu")
    mid = q.insert_message(role="user", content="no vec yet")
    conn.commit()

    new_vec = _unit_vec((0, 1.0), (10, 0.5))
    q.update_embedding(mid, new_vec)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding IS NOT NULL FROM memories WHERE id = %s", (mid,)
        )
        assert cur.fetchone()[0] is True


def test_update_embedding_respects_agent_isolation(conn: psycopg.Connection) -> None:
    """Чужой agent через свой Q не может обновить чужую memory."""
    q_a = AgentScopedQueries(conn, agent_id="agent-a")
    mid = q_a.insert_message(role="user", content="alpha owns this")
    conn.commit()

    q_b = AgentScopedQueries(conn, agent_id="agent-b")
    q_b.update_embedding(mid, _unit_vec((0, 1.0)))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM memories WHERE id = %s", (mid,))
        emb = cur.fetchone()[0]
    # Apply via agent-b не должен пройти — agent_id mismatch в WHERE.
    assert emb is None


# ── compute_agent_usage_p75 ───────────────────────────────────────────


def test_usage_p75_zero_when_no_used_in_output(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="nu")
    mid = _seed(conn, agent_id="nu", role="user", content="x",
                embedding=_unit_vec((0, 1.0)))
    qhash = hashlib.sha256(b"q").digest()
    q.record_recall_event(memory_id=mid, query_hash=qhash, match_score=0.5)
    conn.commit()
    # used_in_output дефолтно false → percentile_cont(0.75) на пустом
    # GROUP BY не вернёт ничего → 0.0.

    p75 = q.compute_agent_usage_p75()
    assert p75 == 0.0


def test_usage_p75_returns_p75_when_used_in_output_set(
    conn: psycopg.Connection,
) -> None:
    """Когда volna 7c (классификатор) выставит used_in_output, должна
    давать осмысленные значения. Проверим вручную."""
    q = AgentScopedQueries(conn, agent_id="xi")
    mids = []
    for i in range(4):
        mid = _seed(conn, agent_id="xi", role="user", content=f"x{i}",
                    embedding=_unit_vec((0, 1.0)))
        mids.append(mid)

    counts = [1, 3, 5, 7]  # за 4 memories разные used-in-output
    for mid, n in zip(mids, counts):
        for k in range(n):
            qhash = hashlib.sha256(f"{mid}-{k}".encode()).digest()
            q.record_recall_event(memory_id=mid, query_hash=qhash, match_score=0.5)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE recall_events SET used_in_output = true "
                "WHERE memory_id = %s",
                (mid,),
            )
    conn.commit()

    p75 = q.compute_agent_usage_p75()
    # P75 от [1,3,5,7] = 5.5 (linear interpolation).
    assert 4.5 < p75 < 6.5
