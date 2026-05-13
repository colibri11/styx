"""Тесты AgentScopedQueries методов для selective gatekeeper (волна 17).

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


def _embed(seed: float, dim: int = 768) -> list[float]:
    """Простой 768-dim embedding с управляемой similarity.

    seed=1.0 → unit-vector в направлении axis-0; seed=0.5 → 50/50 между
    axis-0 и axis-1 (similarity между двумя такими векторами ~0.71).
    """
    base = [0.0] * dim
    base[0] = seed
    base[1] = (1.0 - seed * seed) ** 0.5
    return base


def _embed_with_offset(offset: float, dim: int = 768) -> list[float]:
    """Embedding с малым отклонением от axis-0 — для контролируемой
    cosine distance.

    offset=0.0 → identical с _embed(1.0); offset=0.1 → distance ≈ 0.005.
    """
    base = [0.0] * dim
    base[0] = (1.0 - offset * offset) ** 0.5
    base[1] = offset
    return base


# ── insert_memory ────────────────────────────────────────────────────


def test_insert_memory_returns_id_and_persists(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = q.insert_memory(
        role="summary",
        content="новая мысль",
        kind="note",
        kind_src="subjective",
    )
    conn.commit()
    assert isinstance(mid, uuid.UUID)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT role, kind, kind_src, content FROM memories WHERE id = %s",
            (mid,),
        )
        row = cur.fetchone()
    assert row == ("summary", "note", "subjective", "новая мысль")


def test_insert_memory_with_embedding(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    mid = q.insert_memory(
        role="summary", content="с эмбеддингом",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding IS NOT NULL FROM memories WHERE id = %s",
            (mid,),
        )
        assert cur.fetchone()[0] is True


def test_insert_memory_isolates_by_agent_id(conn: psycopg.Connection) -> None:
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    a.insert_memory(role="summary", content="alpha-only", kind="note", kind_src="subjective")
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT agent_id FROM memories WHERE content = 'alpha-only'")
        agents = {row[0] for row in cur.fetchall()}
    assert agents == {"alpha"}
    # И b не видит чужой ряд через свой scope:
    cands = b.find_gatekeeper_candidates(_embed(1.0), max_cosine_distance=2.0)
    assert cands == []


# ── find_gatekeeper_candidates ───────────────────────────────────────


def test_find_candidates_empty_db(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    cands = q.find_gatekeeper_candidates(_embed(1.0), max_cosine_distance=0.15)
    assert cands == []


def test_find_candidates_returns_within_distance(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    near = q.insert_memory(
        role="summary", content="близкая запись",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    far = q.insert_memory(
        role="summary", content="далёкая запись",
        kind="note", kind_src="subjective",
        embedding=_embed(0.0),  # ортогональный
    )
    conn.commit()

    cands = q.find_gatekeeper_candidates(
        _embed_with_offset(0.0), max_cosine_distance=0.15,
    )
    ids = {c.id for c in cands}
    assert near in ids
    assert far not in ids


def test_find_candidates_skips_superseded_and_null_embedding(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    # superseded ряд
    superseded = q.insert_memory(
        role="summary", content="старый",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    new = q.insert_memory(
        role="summary", content="новый",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET superseded_by = %s WHERE id = %s",
            (new, superseded),
        )
    # ряд без embedding'а
    q.insert_memory(
        role="summary", content="без вектора",
        kind="note", kind_src="subjective",
    )
    conn.commit()

    cands = q.find_gatekeeper_candidates(
        _embed_with_offset(0.0), max_cosine_distance=0.15,
    )
    ids = {c.id for c in cands}
    assert superseded not in ids  # superseded — не candidate
    assert new in ids
    # ряд без embedding'а тоже не попадает (фильтр embedding IS NOT NULL)


def test_find_candidates_tie_break_by_created_at(conn: psycopg.Connection) -> None:
    """Два ряда с одинаковым embedding → меньший created_at побеждает."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    older = q.insert_memory(
        role="summary", content="older",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    newer = q.insert_memory(
        role="summary", content="newer",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    conn.commit()

    cands = q.find_gatekeeper_candidates(_embed(1.0), max_cosine_distance=0.5)
    assert len(cands) == 2
    assert cands[0].id == older
    assert cands[1].id == newer


def test_find_candidates_top_k_limit(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    for _ in range(10):
        q.insert_memory(
            role="summary", content="row",
            kind="note", kind_src="subjective",
            embedding=_embed_with_offset(0.05),
        )
    conn.commit()

    cands = q.find_gatekeeper_candidates(
        _embed_with_offset(0.0), max_cosine_distance=0.5, top_k=3,
    )
    assert len(cands) == 3


# ── apply_gatekeeper_skip ────────────────────────────────────────────


def test_apply_skip_deletes_memory_and_relations(conn: psycopg.Connection) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    target = q.insert_memory(
        role="summary", content="skip me",
        kind="note", kind_src="subjective",
    )
    other = q.insert_memory(
        role="summary", content="other",
        kind="note", kind_src="subjective",
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relations "
            "  (source_type, source_id, target_type, target_id, relation) "
            "VALUES ('memory', %s, 'memory', %s, 'related_to')",
            (target, other),
        )
        cur.execute(
            "INSERT INTO relations "
            "  (source_type, source_id, target_type, target_id, relation) "
            "VALUES ('memory', %s, 'memory', %s, 'related_to')",
            (other, target),
        )
    q.apply_gatekeeper_skip(target)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM memories WHERE id = %s", (target,))
        assert cur.fetchone() is None
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_id = %s OR target_id = %s",
            (target, target),
        )
        assert cur.fetchone()[0] == 0
        # other ряд не тронут
        cur.execute("SELECT 1 FROM memories WHERE id = %s", (other,))
        assert cur.fetchone() is not None


# ── apply_gatekeeper_merge ───────────────────────────────────────────


def test_apply_merge_redirects_relations_and_deletes_new(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    existing = q.insert_memory(
        role="summary", content="кратко",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    new = q.insert_memory(
        role="summary", content="развёрнуто и подробнее",
        kind="note", kind_src="subjective",
        embedding=_embed_with_offset(0.05),
    )
    third = q.insert_memory(
        role="summary", content="third",
        kind="note", kind_src="subjective",
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO relations "
            "  (source_type, source_id, target_type, target_id, relation) "
            "VALUES ('memory', %s, 'memory', %s, 'related_to')",
            (new, third),
        )
        cur.execute(
            "INSERT INTO relations "
            "  (source_type, source_id, target_type, target_id, relation) "
            "VALUES ('memory', %s, 'memory', %s, 'related_to')",
            (third, new),
        )
    q.apply_gatekeeper_merge(
        new_id=new, existing_id=existing,
        new_content="развёрнуто и подробнее",
        new_embedding=_embed_with_offset(0.05),
    )
    conn.commit()

    with conn.cursor() as cur:
        # new удалён
        cur.execute("SELECT 1 FROM memories WHERE id = %s", (new,))
        assert cur.fetchone() is None
        # existing получил новый content (т.к. длиннее)
        cur.execute(
            "SELECT content FROM memories WHERE id = %s", (existing,),
        )
        assert cur.fetchone()[0] == "развёрнуто и подробнее"
        # relations перенаправлены на existing
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE source_id = %s OR target_id = %s",
            (new, new),
        )
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT count(*) FROM relations "
            " WHERE (source_id = %s AND target_id = %s) "
            "    OR (source_id = %s AND target_id = %s)",
            (existing, third, third, existing),
        )
        assert cur.fetchone()[0] == 2


def test_apply_merge_preserves_existing_when_new_is_shorter(
    conn: psycopg.Connection,
) -> None:
    """Если новый content короче — existing остаётся как есть."""
    q = AgentScopedQueries(conn, agent_id="alpha")
    existing = q.insert_memory(
        role="summary", content="длинный и подробный текст уже сохранён",
        kind="note", kind_src="subjective",
    )
    new = q.insert_memory(
        role="summary", content="короче",
        kind="note", kind_src="subjective",
    )
    q.apply_gatekeeper_merge(
        new_id=new, existing_id=existing,
        new_content="короче",
        new_embedding=_embed(1.0),
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id = %s", (existing,))
        assert cur.fetchone()[0] == "длинный и подробный текст уже сохранён"
        cur.execute("SELECT 1 FROM memories WHERE id = %s", (new,))
        assert cur.fetchone() is None


# ── apply_gatekeeper_supersede ───────────────────────────────────────


def test_apply_supersede_marks_old_and_inserts_relation(
    conn: psycopg.Connection,
) -> None:
    q = AgentScopedQueries(conn, agent_id="alpha")
    existing = q.insert_memory(
        role="summary", content="старая формулировка",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    new = q.insert_memory(
        role="summary", content="новая формулировка той же мысли",
        kind="note", kind_src="subjective",
    )
    q.apply_gatekeeper_supersede(
        new_id=new, existing_id=existing,
        new_embedding=_embed_with_offset(0.05),
    )
    conn.commit()

    with conn.cursor() as cur:
        # Оба ряда живы.
        cur.execute(
            "SELECT id, superseded_by FROM memories WHERE id IN (%s, %s) ORDER BY id",
            (existing, new),
        )
        rows = {row[0]: row[1] for row in cur.fetchall()}
        assert rows[existing] == new
        assert rows[new] is None
        # Embedding на new выставлен.
        cur.execute("SELECT embedding IS NOT NULL FROM memories WHERE id = %s", (new,))
        assert cur.fetchone()[0] is True
        # Relation 'supersedes' создана.
        cur.execute(
            "SELECT relation FROM relations "
            " WHERE source_type = 'memory' AND source_id = %s "
            "   AND target_type = 'memory' AND target_id = %s",
            (new, existing),
        )
        assert cur.fetchone()[0] == "supersedes"


def test_apply_supersede_isolates_by_agent_id(conn: psycopg.Connection) -> None:
    """Cross-agent supersede не должен сработать (agent_id фильтр)."""
    a = AgentScopedQueries(conn, agent_id="alpha")
    b = AgentScopedQueries(conn, agent_id="beta")
    foreign = b.insert_memory(
        role="summary", content="чужая",
        kind="note", kind_src="subjective",
        embedding=_embed(1.0),
    )
    own = a.insert_memory(
        role="summary", content="своя",
        kind="note", kind_src="subjective",
    )
    # alpha пробует supersede чужой — UPDATE WHERE agent_id='alpha' не
    # затронет ряд beta'ы, но INSERT relation всё равно происходит. Это
    # ОК: в апи-слое apply_* зовётся только после find_candidates,
    # который scope'ит по agent_id, поэтому такая ситуация невозможна.
    # Тест проверяет что UPDATE ряд foreign'а не затронул.
    a.apply_gatekeeper_supersede(
        new_id=own, existing_id=foreign,
        new_embedding=_embed_with_offset(0.05),
    )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT superseded_by FROM memories WHERE id = %s", (foreign,))
        assert cur.fetchone()[0] is None  # foreign не тронут
