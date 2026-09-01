"""Тесты recall_full pipeline."""

from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

from styx.embedding import EmbeddingError, FakeEmbeddingClient
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries
from styx.storage.recall import format_recall_text, recall_full
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


def _seed(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    content: str,
    embed_client: FakeEmbeddingClient,
    role: str = "user",
    kind: str = "episode",
) -> uuid.UUID:
    sid = uuid.uuid4()
    vec = embed_client.embed(content)
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
                "[" + ",".join(repr(x) for x in vec) + "]",
            ),
        )
        return cur.fetchone()[0]


def test_recall_full_returns_relevant_memories(conn: psycopg.Connection) -> None:
    """Same FakeEmbeddingClient → одинаковый seed → query 'apples' максимально
    близко к memory 'apples and pears'."""
    embed = FakeEmbeddingClient()
    agent = "alpha"
    target = _seed(conn, agent_id=agent, content="apples and pears", embed_client=embed)
    _seed(conn, agent_id=agent, content="completely different topic", embed_client=embed)
    _seed(conn, agent_id=agent, content="another unrelated thing", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id=agent)
    result = recall_full(
        queries=q, embed_client=embed, query="apples and pears",
    )
    conn.commit()

    assert len(result.memories) >= 1
    # Топ-1 — точное совпадение query == content (FakeEmbedding детерминирован).
    assert result.memories[0].id == target
    assert result.memories[0].score > 0.32  # min_score дефолт (волна 8: 0.6 → 0.32)


def test_recall_full_filters_by_min_score(conn: psycopg.Connection) -> None:
    """Если все score ниже min_score — пустой результат."""
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="beta", content="x", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="beta")
    # Высокий min_score → ничего не пройдёт (ортогональный query).
    from dataclasses import replace
    cfg = replace(DEFAULT_RECALL_CONFIG.full, min_score=0.99)

    result = recall_full(
        queries=q, embed_client=embed, query="totally different query",
        full_config=cfg,
    )
    assert result.memories == []
    assert result.queried_count >= 0


def test_recall_full_records_recall_events(conn: psycopg.Connection) -> None:
    embed = FakeEmbeddingClient()
    target = _seed(conn, agent_id="gamma", content="record me", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="gamma")
    recall_full(queries=q, embed_client=embed, query="record me")
    conn.commit()

    qhash = hashlib.sha256(b"record me").digest()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT match_score, query_hash FROM recall_events "
            "WHERE memory_id = %s",
            (target,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    score, stored_hash = rows[0]
    assert bytes(stored_hash) == qhash
    assert score > 0


def test_recall_full_skips_recall_events_when_disabled(
    conn: psycopg.Connection,
) -> None:
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="delta", content="x", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="delta")
    recall_full(
        queries=q, embed_client=embed, query="x", record_events=False,
    )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM recall_events")
        assert cur.fetchone()[0] == 0


def test_recall_full_handles_embed_error_gracefully() -> None:
    """EmbeddingError при embed query → пустой результат, без падения."""

    class _BadEmbed:
        @property
        def dim(self) -> int:
            return 768

        def embed(self, text: str) -> list[float]:
            raise EmbeddingError("ollama unavailable")

    # Без conn здесь не запустить search_similar — но мы вылетаем раньше
    # через embed_client.embed(query).
    class _StubQueries:
        agent_id = "x"

        def search_similar(self, **kwargs):
            raise AssertionError("не должно быть вызвано")

        def record_recall_event(self, **kwargs):
            raise AssertionError("не должно быть вызвано")

    result = recall_full(
        queries=_StubQueries(),  # type: ignore[arg-type]
        embed_client=_BadEmbed(),
        query="anything",
    )
    assert result.memories == []
    assert result.queried_count == 0


def test_recall_full_internal_dedup_runs(conn: psycopg.Connection) -> None:
    """Два почти-идентичных текста → одна победа."""
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="epsilon", content="apples are great", embed_client=embed)
    # Идентичный embedding (FakeEmbedding детерминирован) → cluster collapse.
    a2 = _seed(conn, agent_id="epsilon", content="apples are great", embed_client=embed)
    _seed(conn, agent_id="epsilon", content="absolutely different topic", embed_client=embed)
    conn.commit()
    # Заметка: оба ряда «apples are great» имеют одинаковые embedding и
    # почти-одинаковые scores. internal_dedup склеит их в один.
    _ = a2

    q = AgentScopedQueries(conn, agent_id="epsilon")
    result = recall_full(
        queries=q, embed_client=embed, query="apples are great",
    )
    conn.commit()

    # Дубликат должен быть отсечён.
    assert result.internal_duplicates_removed >= 1
    contents = [m.content for m in result.memories]
    assert contents.count("apples are great") == 1


def test_format_recall_text_empty() -> None:
    from styx.storage.recall import RecallResult
    r = RecallResult(memories=[], queried_count=0, internal_duplicates_removed=0)
    assert format_recall_text(r) == "<no memories matched>"


def test_format_recall_text_renders_memories(conn: psycopg.Connection) -> None:
    embed = FakeEmbeddingClient()
    _seed(conn, agent_id="zeta", content="hello world", embed_client=embed)
    conn.commit()

    q = AgentScopedQueries(conn, agent_id="zeta")
    result = recall_full(queries=q, embed_client=embed, query="hello world")
    conn.commit()

    text = format_recall_text(result)
    assert "hello world" in text
    assert "score=" in text
    assert "role=user" in text


def test_recall_exposes_only_structured_bounded_affective_provenance(
    conn: psycopg.Connection,
) -> None:
    embed = FakeEmbeddingClient()
    agent = "affect-provenance"
    context_at = "2026-05-03T12:00:00Z"
    causes = [
        {
            "evidence_id": index + 1,
            "source_ref": f"turn-{index + 1}",
            "cause_class": "execution_risk",
            "cause_subject": "tool_outcome",
            "status": "active",
            "intensity": 0.6,
            "confidence": 0.7,
            "observed_at": "2026-05-03T11:59:00Z",
            "lease_expires_at": "2026-05-03T12:14:00Z",
            "cause_summary": "PRIVATE FREE PROSE",
            "style_instruction": "sound sad",
        }
        for index in range(10)
    ]
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, at, valence, arousal, dominance, confidence, "
            " computation_version) VALUES (%s, %s, -0.2, 0.4, 0.1, 0.75, "
            " 'test') RETURNING id",
            (agent, context_at),
        )
        state_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO memories "
            "(agent_id, role, content, embedding, emotional_context_valence, "
            " emotional_context_arousal, emotional_context_dominance, "
            " emotional_context_state_id, emotional_context_at, "
            " emotional_context_confidence, emotional_context_causes) "
            "VALUES (%s, 'assistant', 'structured evidence memory', %s, "
            " -0.2, 0.4, 0.1, %s, %s, 0.75, %s)",
            (
                agent,
                "[" + ",".join(repr(x) for x in embed.embed(
                    "structured evidence memory"
                )) + "]",
                state_id,
                context_at,
                psycopg.types.json.Jsonb(causes),
            ),
        )
    conn.commit()

    result = recall_full(
        queries=AgentScopedQueries(conn, agent_id=agent),
        embed_client=embed,
        query="structured evidence memory",
        record_events=False,
    )
    assert result.memories
    provenance = result.memories[0].affective_provenance
    assert provenance is not None
    assert provenance.state_id == state_id
    assert provenance.vad.valence == pytest.approx(-0.2)
    assert provenance.confidence == pytest.approx(0.75)
    assert len(provenance.causal_refs) == 8
    first_ref = provenance.causal_refs[0]
    assert first_ref.cause_subject == "tool_outcome"
    assert first_ref.status_at_capture == "active"
    assert first_ref.current_status is None
    assert first_ref.intensity == pytest.approx(0.6)
    assert first_ref.confidence == pytest.approx(0.7)
    assert first_ref.observed_at.isoformat().startswith("2026-05-03T11:59:00")

    text = format_recall_text(result)
    assert "affect_evidence=" in text
    assert '"state_id":' in text
    assert '"vad":' in text
    assert '"source_ref":"turn-1"' in text
    assert '"status_at_capture":"active"' in text
    assert "PRIVATE FREE PROSE" not in text
    assert "sound sad" not in text


def test_recall_legacy_null_affect_has_no_evidence_marker(
    conn: psycopg.Connection,
) -> None:
    embed = FakeEmbeddingClient()
    _seed(
        conn,
        agent_id="legacy-affect-null",
        content="legacy no affect",
        embed_client=embed,
    )
    conn.commit()
    result = recall_full(
        queries=AgentScopedQueries(conn, agent_id="legacy-affect-null"),
        embed_client=embed,
        query="legacy no affect",
        record_events=False,
    )
    assert result.memories[0].affective_provenance is None
    assert "affect_evidence=" not in format_recall_text(result)


def _seed_affective_pair(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    embed: FakeEmbeddingClient,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Две семантически равные памяти, отличающиеся только state snapshot."""
    vec = "[" + ",".join(repr(x) for x in embed.embed("same query")) + "]"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories "
            "(agent_id, role, content, embedding, "
            " emotional_context_valence, emotional_context_arousal, "
            " emotional_context_dominance) VALUES "
            "(%s, 'user', 'positive-context', %s, 0.8, 0.4, 0.3), "
            "(%s, 'user', 'negative-context', %s, -0.8, -0.4, -0.3) "
            "RETURNING id",
            (agent_id, vec, agent_id, vec),
        )
        first, second = cur.fetchall()
    return first[0], second[0]


@pytest.mark.parametrize(
    "agent,state_vad,expected",
    [
        ("affect-positive", (0.8, 0.4, 0.3), "positive-context"),
        ("affect-negative", (-0.8, -0.4, -0.3), "negative-context"),
    ],
)
def test_current_affect_changes_recall_top1_before_generation(
    conn: psycopg.Connection,
    agent: str,
    state_vad: tuple[float, float, float],
    expected: str,
) -> None:
    """Одинаковая семантика + разные state дают разный top-1."""
    embed = FakeEmbeddingClient()
    _seed_affective_pair(conn, agent_id=agent, embed=embed)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, valence, arousal, dominance, confidence, "
            " computation_version) VALUES (%s, %s, %s, %s, 1.0, "
            " 'test-causal-v1')",
            (agent, *state_vad),
        )
    conn.commit()

    result = recall_full(
        queries=AgentScopedQueries(conn, agent_id=agent),
        embed_client=embed,
        query="same query",
        record_events=False,
    )
    assert result.memories
    assert result.memories[0].content == expected
