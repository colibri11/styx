"""End-to-end pre_llm_inject (волна 15) в реальном Hermes-Docker.

Pipeline:
1. Поднимаем StyxMemoryCore — он configure'ит pre_llm_inject framework.
2. sync_turn с эмоциональной peer-репликой — qwen3:4b-local через РЕАЛЬНЫЙ
   Ollama извлекает VAD, sync_turn пишет в emotional_state с
   metadata={"hot_vad": [v, a, d]}.
3. Вызываем on_pre_llm_call — channel peer_vad читает запись, формирует
   текст «Peer прозвучал: <phrase>.».
4. Если VAD не извлёкся (qwen3 redirected, time-out) — тест skipped с
   объяснением (так же как test_sentiment_e2e на pure-VAD'е).

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration/test_pre_llm_e2e.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


@pytest.fixture
def styx_stack():
    """Чистый StyxMemoryCore + миграция + cleanup."""
    import psycopg

    from styx.engine import focus_tracker, pre_llm_inject, salient_bridge
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    salient_bridge.reset_all()
    focus_tracker.reset_all()
    pre_llm_inject.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)

    yield p, sid, agent

    p.shutdown()
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    pre_llm_inject.reset_all()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM emotional_state WHERE agent_id = %s", (agent,)
            )
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
            cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_initialize_configures_pre_llm_inject(styx_stack) -> None:
    from styx.engine import pre_llm_inject

    p, _, _ = styx_stack
    handle = pre_llm_inject.get_handle("alpha")
    assert handle is not None
    assert handle.queries is p.queries
    assert pre_llm_inject.is_enabled("alpha") is True


def test_on_pre_llm_call_returns_none_without_sync_turn(styx_stack) -> None:
    """Без записанной hot_sentiment записи — channel skip, hook → None."""
    from styx.engine import pre_llm_inject

    _, sid, _ = styx_stack
    out = pre_llm_inject.on_pre_llm_call("alpha", 
        session_id=sid,
        user_message="hello",
        conversation_history=[],
        is_first_turn=True,
        model="x",
        platform="",
        sender_id="",
    )
    assert out is None


def test_sync_turn_writes_hot_vad_metadata(styx_stack) -> None:
    """sync_turn → emotional_state row с metadata.hot_vad от реального qwen3."""
    import psycopg

    p, sid, agent = styx_stack
    p.sync_turn(
        user_content="Заебало вообще всё, три часа хрень какую-то делаю.",
        assistant_content="Понимаю, тяжело.",
        session_id=sid,
    )

    dsn = os.environ["STYX_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT metadata, valence, arousal, dominance "
                "FROM emotional_state "
                "WHERE agent_id = %s AND source = 'hot_sentiment'",
                (agent,),
            )
            rows = cur.fetchall()

    if not rows:
        pytest.skip(
            "qwen3 не вернул VAD ни на одну реплику — sentiment fail-open"
        )

    metadata, *_ = rows[0]
    assert isinstance(metadata, dict), f"metadata должен быть dict: {metadata!r}"
    assert "hot_vad" in metadata, f"metadata не содержит hot_vad: {metadata!r}"
    hot_vad = metadata["hot_vad"]
    assert isinstance(hot_vad, list) and len(hot_vad) == 3
    for v in hot_vad:
        assert isinstance(v, (int, float))
        assert -1.0 <= v <= 1.0


def test_pre_llm_call_inject_with_seeded_vad(styx_stack) -> None:
    """Детерминированный e2e: пишем VAD напрямую в emotional_state, минуя
    sentiment hot-path. Проверяем что pipeline pre_llm_inject → channel
    peer_vad возвращает корректный текст с phrase'ом из правильного октанта."""
    from styx.emotional.state import EmotionalVector, append_emotional_state
    from styx.engine import pre_llm_inject

    p, sid, agent = styx_stack
    # Записываем известную VAD (positive valence, positive arousal,
    # positive dominance → октант "pospospos" → "оживлённо и уверенно").
    vad = EmotionalVector(valence=0.8, arousal=0.6, dominance=0.5)
    # append_emotional_state применяет clamp; мы передаём как delta поверх
    # neutral base — итог в БД равен этому VAD'у.
    with p._write_lock:
        append_emotional_state(
            p._conn, agent, vad,
            source="hot_sentiment",
            metadata={"hot_vad": [vad.valence, vad.arousal, vad.dominance]},
        )
        p._conn.commit()

    out = pre_llm_inject.on_pre_llm_call("alpha", 
        session_id=sid,
        user_message="follow-up",
        conversation_history=[],
        is_first_turn=False,
        model="x",
        platform="",
        sender_id="",
    )
    assert out is not None
    assert "context" in out
    assert "Peer прозвучал:" in out["context"]
    assert "оживлённо и уверенно" in out["context"]


def test_pre_llm_call_skips_below_min_norm(styx_stack) -> None:
    """VAD с нормой < 0.2 → channel skip → on_pre_llm_call → None."""
    from styx.emotional.state import EmotionalVector, append_emotional_state
    from styx.engine import pre_llm_inject

    p, sid, agent = styx_stack
    vad = EmotionalVector(valence=0.05, arousal=0.05, dominance=0.05)
    with p._write_lock:
        append_emotional_state(
            p._conn, agent, vad,
            source="hot_sentiment",
            metadata={"hot_vad": [vad.valence, vad.arousal, vad.dominance]},
        )
        p._conn.commit()

    out = pre_llm_inject.on_pre_llm_call("alpha", 
        session_id=sid,
        user_message="follow-up",
        conversation_history=[],
        is_first_turn=False,
        model="x",
        platform="",
        sender_id="",
    )
    assert out is None


def test_full_pipeline_inject_through_pre_llm_call(styx_stack) -> None:
    """sync_turn → emotional_state → channel peer_vad → on_pre_llm_call inject."""
    import psycopg

    from styx.engine import pre_llm_inject
    from styx.engine.pre_llm_channels.peer_vad import OCTANTS

    p, sid, agent = styx_stack
    p.sync_turn(
        user_content="Заебало вообще всё, три часа хрень какую-то делаю.",
        assistant_content="Понимаю, тяжело.",
        session_id=sid,
    )

    # Skip если sentiment не сработал (qwen3 weak на эмоциях)
    dsn = os.environ["STYX_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM emotional_state "
                "WHERE agent_id = %s AND source = 'hot_sentiment' "
                "AND metadata->'hot_vad' IS NOT NULL",
                (agent,),
            )
            count = cur.fetchone()[0]
    if count == 0:
        pytest.skip("sentiment не вернул VAD — sentiment fail-open")

    out = pre_llm_inject.on_pre_llm_call("alpha", 
        session_id=sid,
        user_message="follow-up message",
        conversation_history=[],
        is_first_turn=False,
        model="x",
        platform="",
        sender_id="",
    )

    # Если VAD близок к нейтральному (norm < 0.2) — channel skipnет.
    # В этом случае out = None. Мы не строго требуем inject — sentiment
    # на конкретной реплике может выйти близко к нулю.
    if out is None:
        pytest.skip(
            "VAD близок к нейтральному (norm < 0.2) — channel peer_vad "
            "skip'нул как design intends"
        )

    assert "context" in out
    assert out["context"].startswith("Peer прозвучал:")
    # Phrase должна быть из словаря OCTANTS
    phrase_found = any(p in out["context"] for p in OCTANTS.values())
    assert phrase_found, (
        f"phrase не из словаря OCTANTS: {out['context']!r}"
    )
