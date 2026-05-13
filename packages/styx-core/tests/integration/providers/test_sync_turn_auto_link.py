"""Integration: StyxMemoryCore.sync_turn + auto-link (волна 18).

После embed-after-commit для каждого user/assistant ряда зовётся
auto-link. Cross-agent (D2 в waves/18 wave-doc): dialogue ряд alpha
связывается с pre-seeded subjective memory beta'ы по embedding-сходству.

Используем `FakeEmbeddingClient`: одинаковый текст → одинаковый вектор
→ similarity 1.0 → попадает в auto-link окно (max_distance=0.25).
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.embedding import FakeEmbeddingClient
from styx.providers.memory import StyxMemoryCore
from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str) -> str:
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    return migrated_db


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> StyxMemoryCore:
    embed = FakeEmbeddingClient(dim=768)
    monkeypatch.setattr(
        "styx.providers.memory.make_embedding_client", lambda **_: embed
    )
    # Sentiment отключаем чтобы не дёргать LLM-stub.
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "0")
    return StyxMemoryCore()


def _seed_foreign_subjective(
    dsn: str, foreign_agent: str, content: str,
) -> uuid.UUID:
    """Pre-seed subjective memory чужого агента с тем же embedding'ом
    что FakeEmbeddingClient вернёт на ``content``.
    """
    embed = FakeEmbeddingClient(dim=768)
    vec = embed.embed(content)
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, foreign_agent)
        mid = q.insert_memory(
            role="summary", content=content,
            kind="note", kind_src="subjective",
            embedding=vec,
        )
        conn.commit()
    return mid


def test_sync_turn_auto_links_dialogue_to_cross_agent_memory(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Алёна пишет реплику, тематически близкую к памяти Веры — auto-link
    создаёт ребро dialogue (memories.id alpha) → memory (id beta).
    """
    content = "напоминание про релиз пятницы команды mobile"
    foreign_id = _seed_foreign_subjective(
        provider_env, foreign_agent="beta", content=content,
    )

    p = _make_provider(monkeypatch)
    agent = f"alpha-{uuid.uuid4().hex[:6]}"
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn(
            user_content=content,
            assistant_content="Понял, напомню.",
            session_id=sid,
        )

        # alpha-ряд (user) появился, embedded, и auto-link создал ребро
        # к foreign_id (beta).
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM memories "
                    " WHERE agent_id = %s AND role = 'user' "
                    " ORDER BY seq DESC LIMIT 1",
                    (agent,),
                )
                user_mid = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(*) FROM relations "
                    " WHERE source_type='memory' AND source_id=%s "
                    "   AND target_type='memory' AND target_id=%s "
                    "   AND relation='related_to'",
                    (user_mid, foreign_id),
                )
                assert cur.fetchone()[0] == 1
    finally:
        p.shutdown()


def test_sync_turn_no_auto_link_when_disabled(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STYX_AUTO_LINK_ENABLED=0 → нет ни одного related_to ребра."""
    content = "тестовый контент с одинаковым текстом для дедупликации"
    foreign_id = _seed_foreign_subjective(
        provider_env, foreign_agent="beta", content=content,
    )
    monkeypatch.setenv("STYX_AUTO_LINK_ENABLED", "0")

    p = _make_provider(monkeypatch)
    agent = f"alpha-{uuid.uuid4().hex[:6]}"
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn(
            user_content=content,
            assistant_content="ok",
            session_id=sid,
        )
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM relations "
                    " WHERE relation='related_to' AND target_id=%s",
                    (foreign_id,),
                )
                assert cur.fetchone()[0] == 0
    finally:
        p.shutdown()


def test_sync_turn_auto_link_idempotent_on_repeat(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Та же реплика дважды → одно ребро (UNIQUE constraint)."""
    content = "повторяющаяся реплика про деплой ночью"
    foreign_id = _seed_foreign_subjective(
        provider_env, foreign_agent="beta", content=content,
    )

    p = _make_provider(monkeypatch)
    agent = f"alpha-{uuid.uuid4().hex[:6]}"
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn(content, "ok", session_id=sid)
        p.sync_turn(content, "ok2", session_id=sid)
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                # Каждый sync_turn создал свой user-ряд (history capture).
                cur.execute(
                    "SELECT count(*) FROM memories "
                    " WHERE agent_id = %s AND role='user'",
                    (agent,),
                )
                assert cur.fetchone()[0] == 2
                # Каждый ряд имеет своё ребро на foreign_id.
                cur.execute(
                    "SELECT count(*) FROM relations "
                    " WHERE relation='related_to' AND target_id=%s",
                    (foreign_id,),
                )
                # 2 разных source_id'а (два user-ряда), один target —
                # 2 ребра. UNIQUE constraint допускает разные source.
                assert cur.fetchone()[0] == 2
    finally:
        p.shutdown()
