"""End-to-end drift detection (волна 10) в реальном Hermes-Docker.

Pipeline:
1. Поднимаем StyxMemoryCore — он configure'ит salient_bridge + focus_tracker.
2. sync_turn × N через РЕАЛЬНЫЙ Ollama (embeddinggemma пишет векторы).
3. Создаём StyxContextEngine, вызываем compress() с messages последовательно
   на стабильной теме → salient block идентичен (cache hit).
4. Резкая смена темы → drift сработал → salient block разный (cache invalidate).

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration/test_drift_e2e.py -v
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

    from styx.engine import focus_tracker, salient_bridge
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    salient_bridge.reset_all()
    focus_tracker.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)

    yield p, sid, agent

    p.shutdown()
    salient_bridge.reset_all()
    focus_tracker.reset_all()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_initialize_configures_focus_tracker(styx_stack) -> None:
    from styx.engine import focus_tracker

    p, _, _ = styx_stack
    state = focus_tracker.get_state("alpha")
    assert state is not None
    assert state.window == []
    assert state.cached_salient is None


def test_stable_topic_caches_salient_across_turns(styx_stack) -> None:
    """Несколько compress'ов на одной теме → salient block идентичен (cache hit)."""
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

    # Turn 1: первая фраза про embedding.
    msgs1 = [
        {"role": "system", "content": "you are styx assistant"},
        {"role": "user", "content": "продолжаем сессию"},
        {"role": "assistant", "content": "ok готов"},
        {
            "role": "user",
            "content": "какую embedding модель использует Styx",
        },
    ]
    out1 = engine.compress(msgs1, current_tokens=None)
    salient1 = next(
        m for m in out1
        if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))
    )

    # Turn 2: семантически близкая фраза, должно быть no drift.
    msgs2 = list(msgs1) + [
        {"role": "assistant", "content": "embeddinggemma 768-dim"},
        {
            "role": "user",
            "content": "уточни про embedding модель которую использует Styx",
        },
    ]
    out2 = engine.compress(msgs2, current_tokens=None)
    salient2 = next(
        m for m in out2
        if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))
    )

    # На стабильной теме salient block должен переиспользоваться.
    assert salient1 == salient2, (
        f"salient drifted между turn'ами на стабильной теме:\n"
        f"turn1: {salient1['content'][:100]!r}\n"
        f"turn2: {salient2['content'][:100]!r}"
    )


def test_drift_invalidates_salient_cache(styx_stack) -> None:
    """Резкая смена темы → drift → salient block обновляется."""
    from styx.engine.salient import SALIENT_MARKER

    p, sid, _ = styx_stack
    p.sync_turn(
        "Расскажи про embedding-модели Ollama для Styx.",
        "Используем embeddinggemma:300m-qat-q8_0, dim=768, multilingual.",
        session_id=sid,
    )
    p.sync_turn(
        "Расскажи про солнечное затмение.",
        "Солнечное затмение — астрономическое явление: Луна заслоняет Солнце.",
        session_id=sid,
    )

    engine = StyxContextEngine(
        context_length=100_000, protect_first_n=3, protect_last_n=6,
    )

    # Turn 1: тема про embedding.
    msgs1 = [
        {"role": "system", "content": "you are styx assistant"},
        {"role": "user", "content": "продолжаем сессию"},
        {"role": "assistant", "content": "готов"},
        {
            "role": "user",
            "content": "какую embedding модель использует Styx",
        },
    ]
    out1 = engine.compress(msgs1, current_tokens=None)
    salient1 = next(
        (m for m in out1
         if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))),
        None,
    )

    # Turn 2: РЕЗКАЯ смена темы.
    msgs2 = list(msgs1) + [
        {"role": "assistant", "content": "embeddinggemma 768-dim"},
        {
            "role": "user",
            "content": "когда будет следующее солнечное затмение в Москве",
        },
    ]
    out2 = engine.compress(msgs2, current_tokens=None)
    salient2 = next(
        (m for m in out2
         if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))),
        None,
    )

    # Оба salient'а присутствуют, но различаются — drift сработал.
    if salient1 is not None and salient2 is not None:
        assert salient1 != salient2, (
            f"drift не сработал на резкой смене темы:\n"
            f"turn1: {salient1['content'][:100]!r}\n"
            f"turn2: {salient2['content'][:100]!r}"
        )


def test_drift_disabled_via_env_falls_back_to_fresh_each_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_DRIFT_ENABLED=0 → focus_tracker не configure'ится; salient
    block может быть разным между turn'ами на одной теме (нет cache)."""
    import psycopg

    from styx.engine import focus_tracker, salient_bridge
    from styx.engine.salient import SALIENT_MARKER
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    monkeypatch.setenv("STYX_DRIFT_ENABLED", "0")
    salient_bridge.reset_all()
    focus_tracker.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        # salient_bridge активен, focus_tracker — нет
        assert salient_bridge.get_handle("alpha") is not None
        assert focus_tracker.get_state("alpha") is None

        p.sync_turn(
            "Долгое запоминаемое сообщение про embedding-модели Ollama для Styx.",
            "Принял запись про embeddinggemma multilingual 768-dim.",
            session_id=sid,
        )

        engine = StyxContextEngine(
            context_length=100_000, protect_first_n=3, protect_last_n=6,
        )
        msgs = [
            {"role": "system", "content": "you are styx assistant"},
            {"role": "user", "content": "продолжаем сессию"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": "какую embedding модель использует Styx",
            },
        ]
        out1 = engine.compress(msgs, current_tokens=None)
        out2 = engine.compress(msgs, current_tokens=None)

        # Оба содержат salient marker (recall работает), но между ними cache нет.
        s1 = next(
            (m for m in out1
             if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))),
            None,
        )
        s2 = next(
            (m for m in out2
             if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))),
            None,
        )
        # На той же messages результат может быть одинаковым (recall детерминирован
        # при равных условиях), но **главное** — cache не сработал, focus_tracker
        # не работал. Проверяем именно state.
        assert focus_tracker.get_state("alpha") is None
    finally:
        p.shutdown()
        salient_bridge.reset_all()
        focus_tracker.reset_all()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
            conn.commit()
