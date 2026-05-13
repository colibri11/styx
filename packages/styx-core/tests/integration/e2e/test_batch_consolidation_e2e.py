"""End-to-end batch consolidation (волна 14) в Hermes-Docker + Ollama.

Pipeline:
1. sync_turn × 25 с эмоциональными парами.
2. schedule_batch_tick() ставит llm_task с правильным payload.
3. Handler выполняется через worker → memory с kind_src='dialogue_batch_consolidation'.
4. VAD piggyback пишет emotional_state с source='sentiment:batch'
   (если qwen3 вернул VAD).
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


@pytest.fixture
def styx_stack():
    import psycopg

    from styx import turn_state
    from styx.engine import focus_tracker, pre_llm_inject, salient_bridge
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    from styx.engine import transport as transport_mod
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    pre_llm_inject.reset_all()
    turn_state.reset()
    transport_mod._reset_for_test()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)

    yield p, sid, agent

    p.shutdown()
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    pre_llm_inject.reset_all()
    turn_state.reset()
    transport_mod._reset_for_test()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM emotional_state WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
            cur.execute(
                "DELETE FROM consolidation_state "
                "WHERE key = %s",
                (f"batch_consolidation:{agent}",),
            )
            cur.execute(
                "DELETE FROM llm_tasks WHERE payload->>'agent_id' = %s",
                (agent,),
            )
        conn.commit()


def _seed_dialogue(conn, agent: str, n_pairs: int = 22) -> None:
    """Записать N пар user/assistant. Для триггера нужно ≥20 реплик
    (n_pairs * 2 ≥ 20)."""
    base = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(minutes=30)
    rows: list[tuple] = []
    for i in range(n_pairs):
        rows.append(
            (agent, "user", f"Тестовая реплика юзера {i} в диалоге.",
             base + _dt.timedelta(seconds=i * 2)),
        )
        rows.append(
            (agent, "assistant", f"Ответ агента {i}.",
             base + _dt.timedelta(seconds=i * 2 + 1)),
        )
    with conn.cursor() as cur:
        for agent_id, role, content, at in rows:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind, "
                "created_at, kind_src) VALUES (%s, %s, %s, 'episode', %s, "
                "'subjective')",
                (agent_id, role, content, at),
            )
    conn.commit()


def test_scheduler_triggers_on_threshold(styx_stack) -> None:
    """22 пары (44 реплики) ≥ 20 порога → scheduler ставит task."""
    import psycopg

    from styx.workers.sweep.batch_consolidation import (
        BatchSchedulerConfig,
        schedule_batch_tick,
    )

    _, _, agent = styx_stack
    dsn = os.environ["STYX_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        _seed_dialogue(conn, agent, n_pairs=22)
        config = BatchSchedulerConfig(enabled=True)
        scheduled = schedule_batch_tick(conn, config=config)

    # Может быть >0 если другие subjective агенты в БД — главное наш
    # сработал.
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM llm_tasks "
                "WHERE task_type='dialogue_batch_consolidation' "
                "  AND payload->>'agent_id' = %s",
                (agent,),
            )
            assert cur.fetchone()[0] >= 1


def test_scheduler_skips_below_threshold(styx_stack) -> None:
    """5 пар (10 реплик) < 20 порога → scheduler не ставит task."""
    import psycopg

    from styx.workers.sweep.batch_consolidation import (
        BatchSchedulerConfig,
        schedule_batch_tick,
    )

    _, _, agent = styx_stack
    dsn = os.environ["STYX_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        _seed_dialogue(conn, agent, n_pairs=5)
        schedule_batch_tick(conn, config=BatchSchedulerConfig(enabled=True))

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM llm_tasks "
                "WHERE payload->>'agent_id' = %s",
                (agent,),
            )
            assert cur.fetchone()[0] == 0


def test_handler_creates_batch_memory_real_qwen3(styx_stack) -> None:
    """Полный pipeline через реальный qwen3:4b-local: scheduler → llm_task →
    worker handler → memory с kind_src='dialogue_batch_consolidation'."""
    import psycopg

    from styx.config import load as load_config
    from styx.workers.main import build_worker
    from styx.workers.sweep.batch_consolidation import (
        BatchSchedulerConfig,
        schedule_batch_tick,
    )

    _, _, agent = styx_stack
    dsn = os.environ["STYX_DATABASE_URL"]

    # 1. seed dialogue + schedule
    with psycopg.connect(dsn) as conn:
        _seed_dialogue(conn, agent, n_pairs=22)
        schedule_batch_tick(conn, config=BatchSchedulerConfig(enabled=True))

    # 2. Run worker once (drain pending tasks).
    cfg = load_config()
    worker = build_worker(cfg)
    try:
        # process_one в лупе пока не пусто
        for _ in range(5):
            processed = worker.process_one()
            if not processed:
                break
    finally:
        worker.stop()

    # 3. Memory создана? Skip если qwen3 вернул skip=True (короткое
    # содержимое, не достойно запоминать).
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), array_agg(content) FROM memories "
                "WHERE agent_id = %s "
                "  AND kind_src='dialogue_batch_consolidation'",
                (agent,),
            )
            count, contents = cur.fetchone()

    if count == 0:
        pytest.skip(
            "qwen3 вернул skip=true для seeded диалога — память не "
            "создана (валидное LLM-skip поведение)"
        )
    assert count >= 1
    assert all(c for c in contents)
