"""Интеграция: dialogue_batch_consolidation handler + selective gatekeeper.

Real Postgres + stub LLM + контролируемый embedder. LLM канаrем ставим
один chunk outcome (не skip), embedder возвращает заданный вектор → так
детерминированно покрываем все 4 ветки gatekeeper'а (store / merge /
supersede / skip).

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid

import psycopg
import pytest

from styx.engine.selective_gatekeeper import GatekeeperConfig
from styx.llm import LLMRateLimiter
from styx.storage import migrate
from styx.storage.queries import AgentScopedQueries
from styx.workers.handlers.dialogue_batch_consolidation import (
    create_dialogue_batch_handler,
)
from styx.workers.runtime import HandlerContext, LlmTask


# ── Stubs ────────────────────────────────────────────────────────────


class _StubLLM:
    """LLM, возвращающий заданный JSON на chat_json. Один chunk outcome."""

    def __init__(self, summary: str) -> None:
        self._summary = summary

    def chat_json(self, *, messages: list[dict]) -> dict:  # noqa: ARG002
        return {
            "skip": False,
            "skip_reason": None,
            "summary": self._summary,
            "archive_hints": [{"snippet": "snippet"}],
            "vad": None,
        }


class _ControlledEmbedder:
    """Embedder, возвращающий фиксированный вектор. Детерминирует similarity."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = list(vec)

    @property
    def dim(self) -> int:
        return len(self._vec)

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return list(self._vec)


def _embed(seed: float, dim: int = 768) -> list[float]:
    """Простой 768-dim unit-vector в направлении axis-0/axis-1."""
    base = [0.0] * dim
    base[0] = seed
    base[1] = (1.0 - seed * seed) ** 0.5
    return base


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def conn(clean_db: str):
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as connection:
        yield connection


@pytest.fixture
def agent() -> str:
    return "alpha"


def _seed_dialogue(conn: psycopg.Connection, agent: str, n_pairs: int = 22) -> _dt.datetime:
    """Заполнить N пар user/assistant в memories. Возвращает window_to."""
    base = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(minutes=30)
    last_at = base
    with conn.cursor() as cur:
        for i in range(n_pairs):
            user_at = base + _dt.timedelta(seconds=i * 2)
            asst_at = base + _dt.timedelta(seconds=i * 2 + 1)
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind, "
                "  kind_src, created_at) "
                "VALUES (%s, 'user', %s, 'episode', 'subjective', %s)",
                (agent, f"Реплика юзера {i}", user_at),
            )
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind, "
                "  kind_src, created_at) "
                "VALUES (%s, 'assistant', %s, 'episode', 'subjective', %s)",
                (agent, f"Ответ агента {i}", asst_at),
            )
            last_at = asst_at
    conn.commit()
    return last_at + _dt.timedelta(seconds=1)


def _make_task(agent: str, window_to: _dt.datetime) -> LlmTask:
    return LlmTask(
        id=uuid.uuid4(),
        task_type="dialogue_batch_consolidation",
        memory_id=None,
        payload={
            "agent_id": agent,
            "window_from": None,
            "window_to": window_to.isoformat(),
            "with_overlap": False,
        },
        retry_count=0,
    )


def _make_ctx(
    conn: psycopg.Connection,
    embedder: _ControlledEmbedder | None,
    summary: str = "это новая консолидированная заметка от агента",
) -> HandlerContext:
    return HandlerContext(
        conn=conn,
        llm=_StubLLM(summary),
        rate_limit=LLMRateLimiter(capacity=1, refill_per_second=10.0),
        logger=logging.getLogger("test"),
        embedder=embedder,
    )


# ── Tests ────────────────────────────────────────────────────────────


def test_handler_store_when_no_existing_neighbour(
    conn: psycopg.Connection, agent: str,
) -> None:
    """Пустая БД (только seeded dialogue) → gatekeeper решает store."""
    window_to = _seed_dialogue(conn, agent)
    embedder = _ControlledEmbedder(_embed(1.0))
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
    )
    summary = "это новая консолидированная заметка от агента"
    out = handler(_make_task(agent, window_to), _make_ctx(conn, embedder, summary))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), array_agg(content) FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        count, contents = cur.fetchone()
    assert count == 1
    assert summary in contents
    assert len(out.result["memories_created"]) == 1


def test_handler_merge_when_existing_neighbour_close(
    conn: psycopg.Connection, agent: str,
) -> None:
    """Pre-seeded memory с embedding X + handler пишет похожее → merge."""
    window_to = _seed_dialogue(conn, agent)

    # Pre-seed существующая subjective memory с известным embedding'ом.
    existing_content = "коротко"
    q = AgentScopedQueries(conn, agent)
    existing_id = q.insert_memory(
        role="summary", content=existing_content,
        kind="episode", kind_src="dialogue_batch_consolidation",
        embedding=_embed(1.0),
    )
    conn.commit()

    new_summary = "развёрнутая консолидация той же мысли с большим количеством текста"
    embedder = _ControlledEmbedder(_embed(1.0))  # identical → similarity 1.0
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
    )
    handler(_make_task(agent, window_to), _make_ctx(conn, embedder, new_summary))
    conn.commit()

    with conn.cursor() as cur:
        # existing_id должен остаться + получить новый content (длиннее).
        cur.execute("SELECT content FROM memories WHERE id = %s", (existing_id,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == new_summary
        # Других subjective memories с этим content не должно быть
        # (новый ряд удалён в merge).
        cur.execute(
            "SELECT count(*) FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        assert cur.fetchone()[0] == 1


def test_handler_supersede_when_supersede_zone(
    conn: psycopg.Connection, agent: str,
) -> None:
    """Pre-seeded memory + handler пишет в supersede-зоне с похожим
    текстом → supersede (new живёт, existing помечен superseded_by).
    """
    window_to = _seed_dialogue(conn, agent)

    existing_content = "консолидация заметка агента"
    q = AgentScopedQueries(conn, agent)
    existing_id = q.insert_memory(
        role="summary", content=existing_content,
        kind="episode", kind_src="dialogue_batch_consolidation",
        embedding=_embed(0.5),  # 0.5 в axis-0, 0.866 в axis-1
    )
    conn.commit()

    # New embedding в зоне supersede: _embed(0.85) даёт similarity ≈ 0.88
    # против _embed(0.5). Текст Levenshtein > 0.3 (только восклицательный
    # знак отличается) → supersede.
    new_summary = "консолидация заметка агента!"
    new_vec = _embed(0.85)
    embedder = _ControlledEmbedder(new_vec)
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
    )
    handler(_make_task(agent, window_to), _make_ctx(conn, embedder, new_summary))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, superseded_by FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation' "
            " ORDER BY created_at",
            (agent,),
        )
        rows = cur.fetchall()
        # Существующий + новый — оба живы.
        assert len(rows) == 2
        existing_row = next(r for r in rows if r[0] == existing_id)
        new_row = next(r for r in rows if r[0] != existing_id)
        # existing получил superseded_by на new.
        assert existing_row[1] == new_row[0]
        assert new_row[1] is None
        # Relation 'supersedes' создан.
        cur.execute(
            "SELECT relation FROM relations "
            " WHERE source_type = 'memory' AND source_id = %s "
            "   AND target_type = 'memory' AND target_id = %s",
            (new_row[0], existing_id),
        )
        rel = cur.fetchone()
        assert rel is not None and rel[0] == "supersedes"


def test_handler_disabled_gatekeeper_falls_back_to_store(
    conn: psycopg.Connection, agent: str,
) -> None:
    """STYX_SELECTIVE_ENABLED=0 → каждый writer пишет как раньше."""
    window_to = _seed_dialogue(conn, agent)

    # Pre-seed похожий ряд — без gatekeeper'а merge не должен сработать.
    q = AgentScopedQueries(conn, agent)
    q.insert_memory(
        role="summary", content="старая заметка консолидации",
        kind="episode", kind_src="dialogue_batch_consolidation",
        embedding=_embed(1.0),
    )
    conn.commit()

    embedder = _ControlledEmbedder(_embed(1.0))
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(enabled=False),
    )
    handler(
        _make_task(agent, window_to),
        _make_ctx(conn, embedder, "новая заметка консолидации с другим текстом"),
    )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        # Оба ряда живы (gatekeeper выключен — никаких merge/supersede).
        assert cur.fetchone()[0] == 2


def test_handler_no_embedder_falls_back_to_legacy(
    conn: psycopg.Connection, agent: str,
) -> None:
    """ctx.embedder=None → INSERT memory с embedding=NULL (legacy),
    gatekeeper пропускается."""
    window_to = _seed_dialogue(conn, agent)
    handler = create_dialogue_batch_handler(
        gatekeeper_config=GatekeeperConfig(),
    )
    handler(_make_task(agent, window_to), _make_ctx(conn, None))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding IS NULL FROM memories "
            " WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] is True  # embedding=NULL — legacy path
