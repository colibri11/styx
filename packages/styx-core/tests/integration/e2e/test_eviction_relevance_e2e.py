"""End-to-end eviction relevance-aware (волна 12) в реальном Hermes-Docker.

Pipeline:
1. StyxMemoryCore initialize → eviction_relevance_bridge configured.
2. sync_turn × N через РЕАЛЬНЫЙ Ollama — embedding'и body messages
   попадают в memories.
3. compress() с переполнением окна:
   - С volna 12 enabled и centroid'ом, близким к anchor'у — anchor message
     keep'ится между head и tail (несмотря на recency-eviction).
   - С STYX_EVICTION_RELEVANCE_ENABLED=0 — anchor message отсутствует
     в выходе compress'а (kроме salient block'а, который имеет свой
     маркер и не равен anchor по content).

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration/test_eviction_relevance_e2e.py -v
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


ANCHOR_USER = (
    "Какую embedding-модель использует Styx и сколько у неё измерений?"
)
ANCHOR_ASSISTANT = (
    "Используем embeddinggemma:300m-qat-q8_0, размерность 768."
)


@pytest.fixture
def styx_stack():
    """Чистый StyxMemoryCore + миграция + cleanup."""
    import psycopg

    from styx.engine import (
        eviction_relevance_bridge,
        focus_tracker,
        hot_tier,
        salient_bridge,
    )
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    eviction_relevance_bridge.reset_all()
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    hot_tier.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)

    yield p, sid, agent

    p.shutdown()
    eviction_relevance_bridge.reset_all()
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    hot_tier.reset_all()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_initialize_configures_handle(styx_stack) -> None:
    from styx.engine import eviction_relevance_bridge

    p, _, agent = styx_stack
    h = eviction_relevance_bridge.get_handle("alpha")
    assert h is not None
    assert h.agent_id == agent
    assert h.keep_k == 2
    assert h.threshold == 0.4


def _build_oversized_messages() -> list[dict]:
    """3 head + middle с anchor + 6 tail = 15 messages.

    Middle содержит 6 messages: noise + ANCHOR_USER + noise + ANCHOR_ASSISTANT
    + noise. Тail заканчивается на user-query про embedding (focus → centroid
    близкий к anchor).
    """
    return (
        [
            {"role": "system", "content": "you are styx assistant"},
            {"role": "user", "content": "продолжаем сессию"},
            {"role": "assistant", "content": "ok готов"},
        ]
        + [
            {"role": "user", "content": "Расскажи о погоде в Москве."},
            {"role": "assistant", "content": "Сегодня солнечно, +18°C."},
            {"role": "user", "content": ANCHOR_USER},
            {"role": "assistant", "content": ANCHOR_ASSISTANT},
            {"role": "user", "content": "Какие планы на выходные?"},
            {"role": "assistant", "content": "Не знаю, пока не решил."},
        ]
        + [
            {"role": "user", "content": "А что насчёт фильмов?"},
            {"role": "assistant", "content": "Можно посмотреть Дюну."},
            {"role": "user", "content": "Хорошо."},
            {"role": "assistant", "content": "Что-то ещё?"},
            {"role": "user", "content": "Расскажи про embedding-модели для Styx."},
        ]
    )


def _seed_anchor_embeddings(p, sid: str) -> None:
    """sync_turn для anchor пары — гарантирует что embedding'и в БД."""
    p.sync_turn(ANCHOR_USER, ANCHOR_ASSISTANT, session_id=sid)
    p.sync_turn(
        "Расскажи про embedding-модели для Styx.",
        "Они работают на Ollama-эндпоинте.",
        session_id=sid,
    )
    # Также зашлём пару нерелевантных, чтобы middle-эмбеды для прочих
    # сообщений тоже нашлись.
    p.sync_turn(
        "Расскажи о погоде в Москве.",
        "Сегодня солнечно, +18°C.",
        session_id=sid,
    )
    p.sync_turn(
        "Какие планы на выходные?",
        "Не знаю, пока не решил.",
        session_id=sid,
    )
    p.sync_turn(
        "А что насчёт фильмов?",
        "Можно посмотреть Дюну.",
        session_id=sid,
    )
    p.sync_turn("Хорошо.", "Что-то ещё?", session_id=sid)


def _force_eviction_engine():

    e = StyxContextEngine(
        context_length=100, threshold_percent=0.5,
        protect_first_n=3, protect_last_n=6,
    )
    # forсим переполнение бюджета — current_tokens > threshold_tokens
    e.threshold_tokens = 1
    return e


def test_anchor_kept_in_middle_when_enabled(styx_stack) -> None:
    """С волной 12 enabled — anchor присутствует в финальном body как
    обычный user-message (не salient block)."""
    from styx.engine.salient import SALIENT_MARKER

    p, sid, _ = styx_stack
    _seed_anchor_embeddings(p, sid)

    engine = _force_eviction_engine()
    msgs = _build_oversized_messages()
    out = engine.compress(msgs, current_tokens=10_000)

    # Anchor user text ≠ salient block (у того маркер).
    anchor_present = any(
        m.get("role") == "user"
        and m.get("content") == ANCHOR_USER
        and SALIENT_MARKER not in str(m.get("content", ""))
        for m in out
    )
    assert anchor_present, (
        "ANCHOR_USER message должен присутствовать в compress output'е "
        "(keep'нут eviction'ом по relevance к focus centroid'у)"
    )


def test_anchor_dropped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """С STYX_EVICTION_RELEVANCE_ENABLED=0 — anchor не keep'ится через
    relevance путь (recency-only eviction)."""
    import psycopg

    from styx.engine import (
        eviction_relevance_bridge,
        focus_tracker,
        hot_tier,
        salient_bridge,
    )
    from styx.engine.salient import SALIENT_MARKER
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    monkeypatch.setenv("STYX_EVICTION_RELEVANCE_ENABLED", "0")

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    eviction_relevance_bridge.reset_all()
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    hot_tier.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        _seed_anchor_embeddings(p, sid)
        assert eviction_relevance_bridge.get_handle("alpha") is None

        engine = _force_eviction_engine()
        msgs = _build_oversized_messages()
        out = engine.compress(msgs, current_tokens=10_000)

        # Anchor содержится только в salient block (если recall его
        # достал) — но не как точный user-message с тем же content'ом.
        anchor_as_plain_user = any(
            m.get("role") == "user"
            and m.get("content") == ANCHOR_USER
            and SALIENT_MARKER not in str(m.get("content", ""))
            for m in out
        )
        assert not anchor_as_plain_user, (
            "Без волны 12 anchor user-message не должен keep'иться в "
            "middle'е через relevance путь"
        )
    finally:
        p.shutdown()
        eviction_relevance_bridge.reset_all()
        salient_bridge.reset_all()
        focus_tracker.reset_all()
        hot_tier.reset_all()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
            conn.commit()
