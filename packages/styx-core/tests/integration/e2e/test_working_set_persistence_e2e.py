"""End-to-end persistence working set'а (волна 13) в реальном Hermes-Docker.

Проверяем три сценария:
1. Restart simulation: provider 1 заполняет focus_tracker.window и hot_tier
   через sync_turn + styx_recall; shutdown (final flush). Provider 2
   (новый initialize, тот же agent_id) — focus + hot восстановлены.
2. STYX_WORKING_SET_PERSISTENCE_ENABLED=0 на provider 2 → state холодный
   несмотря на сохранённую строку.
3. Stale state — past TTL → cold start. Эмулируем UPDATE updated_at в БД.

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest \\
        /opt/styx/tests/integration/test_working_set_persistence_e2e.py -v
"""

from __future__ import annotations

import datetime as _dt
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


def _reset_all_state() -> None:
    from styx.engine import (
        eviction_relevance_bridge,
        focus_tracker,
        hot_tier,
        pre_llm_inject,
        salient_bridge,
        working_set_persistence,
        transport as transport_mod,
    )

    working_set_persistence.stop_all()
    eviction_relevance_bridge.reset_all()
    hot_tier.reset_all()
    focus_tracker.reset_all()
    salient_bridge.reset_all()
    pre_llm_inject.reset_all()
    transport_mod._reset_for_test()


def _cleanup_agent(dsn: str, agent: str) -> None:
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_restart_restores_window_and_cached_salient() -> None:
    """Provider 1 заполняет state через sync_turn → shutdown. Provider 2 —
    state восстановлен."""
    from styx.engine import focus_tracker
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)
    _reset_all_state()

    agent = "alpha"

    # Provider 1 — заполняем state через compress (build_salient_block →
    # focus_tracker.observe + recall_full → hot_tier.put_many).
    p1 = StyxMemoryCore()
    p1.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    try:
        sid1 = str(uuid.uuid4())
        p1.sync_turn(
            "Опиши embedding-модель в Styx подробнее.",
            "В Styx используем embeddinggemma:300m-qat-q8_0, dim=768, multilingual.",
            session_id=sid1,
        )
        p1.sync_turn(
            "А что про миграции?",
            "Volna 7 — port memorybox-схемы 24 миграции в одну 0002.",
            session_id=sid1,
        )

        # Compress на третьей user-реплике — build_salient_block observe'нет
        # focus_tracker (заполнит window), recall_full → put_many в hot_tier.
        engine = StyxContextEngine(
            context_length=100_000, protect_first_n=3, protect_last_n=6,
        )
        msgs = [
            {"role": "system", "content": "you are styx assistant"},
            {"role": "user", "content": "продолжаем"},
            {"role": "assistant", "content": "готов"},
            {"role": "user", "content": "какие embedding-модели в Styx?"},
        ]
        engine.compress(msgs, current_tokens=None)

        state_before = focus_tracker.get_state("alpha")
        assert state_before is not None
        assert len(state_before.window) >= 1, (
            "compress должен был observe'нуть user-embed в focus_tracker"
        )
    finally:
        p1.shutdown()  # final flush

    _reset_all_state()  # симулируем рестарт процесса

    # Provider 2 — тот же agent_id, новый session.
    p2 = StyxMemoryCore()
    p2.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    try:
        state_after = focus_tracker.get_state("alpha")
        assert state_after is not None
        assert len(state_after.window) >= 1, (
            "focus_tracker.window должен быть восстановлен после restart'а"
        )
    finally:
        p2.shutdown()
        _reset_all_state()
        _cleanup_agent(dsn, agent)


def test_persistence_disabled_keeps_cold_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_WORKING_SET_PERSISTENCE_ENABLED=0 на втором provider'е → state не
    восстанавливается, несмотря на сохранённую строку."""
    from styx.engine import focus_tracker
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)
    _reset_all_state()

    agent = "alpha"

    # Provider 1 — записывает state.
    p1 = StyxMemoryCore()
    p1.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    try:
        state = focus_tracker.get_state("alpha")
        assert state is not None
        state.window.append([0.5] * 768)
        state.cached_salient = {"role": "user", "content": "marker"}
    finally:
        p1.shutdown()

    _reset_all_state()
    monkeypatch.setenv("STYX_WORKING_SET_PERSISTENCE_ENABLED", "0")

    # Provider 2 — persistence отключён.
    p2 = StyxMemoryCore()
    p2.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    try:
        state = focus_tracker.get_state("alpha")
        # focus_tracker.configure() вызвался (drift_enabled=True по дефолту),
        # но restore не зовётся → window пуст, cached_salient None.
        assert state is not None
        assert state.window == []
        assert state.cached_salient is None
    finally:
        p2.shutdown()
        _reset_all_state()
        _cleanup_agent(dsn, agent)


def test_stale_state_past_ttl_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """State старше STYX_WORKING_SET_TTL_S → cold start."""
    import psycopg

    from styx.engine import focus_tracker
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)
    _reset_all_state()

    agent = "alpha"

    p1 = StyxMemoryCore()
    p1.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    try:
        state = focus_tracker.get_state("alpha")
        assert state is not None
        state.cached_salient = {"role": "user", "content": "old marker"}
    finally:
        p1.shutdown()

    # Подкручиваем updated_at в прошлое, превышая default TTL (24h).
    backdate = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=2)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE working_set SET updated_at = %s WHERE agent_id = %s",
                (backdate, agent),
            )
        conn.commit()

    _reset_all_state()

    p2 = StyxMemoryCore()
    p2.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    try:
        state = focus_tracker.get_state("alpha")
        assert state is not None
        # Past TTL → cold; cached_salient не восстановлен.
        assert state.cached_salient is None
    finally:
        p2.shutdown()
        _reset_all_state()
        _cleanup_agent(dsn, agent)
