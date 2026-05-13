"""End-to-end hot-tier (волна 11) в реальном Hermes-Docker.

Проверяем что:
1. ``StyxMemoryCore.initialize()`` configure'ит hot_tier.
2. После compress'а с recall'ом hot заполняется memory_id'ами возвращённых
   items (put-on-success).
3. ``STYX_HOT_TIER_ENABLED=0`` полностью отключает: hot_tier.get_state("alpha") — None,
   ничего не складывается.

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration/test_hot_tier_e2e.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

from styx.engine.context import StyxComposer as _StyxComposer

def StyxContextEngine(**kw):
    """Test alias: Composer привязанный к agent_id='alpha'."""
    return _StyxComposer("alpha", **kw)


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


@pytest.fixture
def styx_stack():
    """Чистый StyxMemoryCore + миграция + cleanup."""
    import psycopg

    from styx.engine import focus_tracker, hot_tier, salient_bridge
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    salient_bridge.reset_all()
    focus_tracker.reset_all()
    hot_tier.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)

    yield p, sid, agent

    p.shutdown()
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    hot_tier.reset_all()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_initialize_configures_hot_tier(styx_stack) -> None:
    from styx.engine import hot_tier

    p, _, _ = styx_stack
    s = hot_tier.get_state("alpha")
    assert s is not None
    assert s.entries == {}
    assert s.ttl_s == 300.0
    assert s.lru_bound == 100


def test_recall_populates_hot_tier(styx_stack) -> None:
    """После compress'а с recall'ом hot содержит returned memory_ids."""
    from styx.engine import hot_tier
    from styx.engine.salient import SALIENT_MARKER

    p, sid, _ = styx_stack
    p.sync_turn(
        "Расскажи про embedding-модели Ollama для Styx.",
        "Используем embeddinggemma:300m-qat-q8_0, dim=768, multilingual.",
        session_id=sid,
    )
    p.sync_turn(
        "А по миграциям что?",
        "Мигрировали схему с pg18 + pgvector, миграция 0002 — port из memorybox.",
        session_id=sid,
    )

    engine = StyxContextEngine(
        context_length=100_000, protect_first_n=3, protect_last_n=6,
    )
    msgs = [
        {"role": "system", "content": "you are styx assistant"},
        {"role": "user", "content": "продолжаем сессию"},
        {"role": "assistant", "content": "ok готов"},
        {
            "role": "user",
            "content": "какую embedding модель использует Styx",
        },
    ]
    out = engine.compress(msgs, current_tokens=None)
    has_salient = any(
        m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))
        for m in out
    )

    s = hot_tier.get_state("alpha")
    assert s is not None
    if has_salient:
        # Recall сработал → put_many должен был положить items в hot.
        assert len(s.entries) > 0, (
            "salient inject состоялся, но hot пуст — put_many не сработал"
        )


def test_hot_tier_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """STYX_HOT_TIER_ENABLED=0 → state остаётся None после initialize."""
    import psycopg

    from styx.engine import focus_tracker, hot_tier, salient_bridge
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    monkeypatch.setenv("STYX_HOT_TIER_ENABLED", "0")
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    hot_tier.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        # hot_tier disabled — state None.
        assert hot_tier.get_state("alpha") is None

        # Recall'ы должны продолжать работать (fallback на чистый БД-pipeline).
        p.sync_turn(
            "Сообщение про Styx.",
            "Принял запись.",
            session_id=sid,
        )
        # После sync_turn никакого put в hot быть не должно.
        assert hot_tier.get_state("alpha") is None
    finally:
        p.shutdown()
        salient_bridge.reset_all()
        focus_tracker.reset_all()
        hot_tier.reset_all()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
            conn.commit()
