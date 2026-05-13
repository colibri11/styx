"""End-to-end salient memories injection (волна 9) в реальном Hermes-Docker.

Pipeline:
1. Поднимаем StyxMemoryCore — он configure'ит salient_bridge.
2. sync_turn × N через РЕАЛЬНЫЙ Ollama (embeddinggemma пишет векторы).
3. Создаём StyxContextEngine, вызываем compress() с messages
   содержащим тематически близкий last user.
4. Проверяем: salient message с маркером в output; в content'е
   упомянута одна из записанных memories.

Параллельно прогоняется регрессия test_e2e_smoke (digest на head=3
байт-стабилен между turn'ами).

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration/test_salient_e2e.py -v
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

    from styx.engine import salient_bridge
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    salient_bridge.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)

    yield p, sid, agent

    p.shutdown()
    salient_bridge.reset_all()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_initialize_configures_salient_bridge(styx_stack) -> None:
    """initialize() → handle is not None; shutdown() сбросит."""
    from styx.engine import salient_bridge

    p, _, _ = styx_stack
    handle = salient_bridge.get_handle("alpha")
    assert handle is not None
    assert handle.queries is p.queries


def test_compress_injects_recalled_memory_into_active_suffix(
    styx_stack,
) -> None:
    """Главный сценарий волны 9: compress() сам инжектит relevant memory.

    Записываем 3 разных темы через sync_turn, затем compress'им messages
    у которых last user — про одну из тем. Salient block должен:
    - присутствовать в output (один user-role message с SALIENT_MARKER);
    - содержать что-то из записанной memory (или хотя бы вернуть
      непустой результат — recall pipeline отработал).
    """
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
    p.sync_turn(
        "Как тестируется retrieval?",
        "Через FakeEmbeddingClient на хосте и реальный Ollama в Docker.",
        session_id=sid,
    )

    engine = StyxContextEngine(
        context_length=100_000, protect_first_n=3, protect_last_n=6
    )
    messages = [
        {"role": "system", "content": "you are styx assistant"},
        {"role": "user", "content": "продолжаем сессию"},
        {"role": "assistant", "content": "ок, готов"},
        {
            "role": "user",
            "content": "какую embedding модель использует Styx",
        },
    ]
    out = engine.compress(messages, current_tokens=None)

    salient_msgs = [
        m for m in out
        if m.get("role") == "user" and SALIENT_MARKER in str(m.get("content", ""))
    ]
    assert len(salient_msgs) == 1, (
        f"ожидаем ровно один salient блок, нашли {len(salient_msgs)}: {out!r}"
    )
    salient_text = salient_msgs[0]["content"].lower()
    # Recall сработал — в блоке упомянута одна из записанных тем.
    assert (
        "embeddinggemma" in salient_text
        or "embedding" in salient_text
        or "ollama" in salient_text
        or "styx" in salient_text
    ), f"salient блок не содержит relevant content: {salient_text!r}"


def test_compress_first_n_byte_stable_across_turns(styx_stack) -> None:
    """Регрессия для test_e2e_smoke pipeline'а: digest от head=3
    байт-стабилен между двумя последовательными compress'ами, даже
    когда last user разный (и значит salient разный)."""
    from styx.engine.transport import compute_prefix_digest

    p, sid, _ = styx_stack
    p.sync_turn(
        "Стабильный prefix важен для prompt cache.",
        "Принято, держим первые 3 message неизменными.",
        session_id=sid,
    )

    engine = StyxContextEngine(
        context_length=100_000, protect_first_n=3, protect_last_n=6
    )

    base = [
        {"role": "system", "content": "you are styx assistant"},
        {"role": "user", "content": "продолжаем сессию"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "первый запрос про prompt cache"},
    ]
    out1 = engine.compress(list(base), current_tokens=None)

    base.append({"role": "assistant", "content": "обработал"})
    base.append({"role": "user", "content": "второй запрос про другую тему"})
    out2 = engine.compress(list(base), current_tokens=None)

    digest1 = compute_prefix_digest(out1, head_count=3)
    digest2 = compute_prefix_digest(out2, head_count=3)
    assert digest1 == digest2, (
        f"head digest не стабилен между turn'ами: {digest1!r} vs {digest2!r}"
    )


def test_salient_disabled_via_env_skips_inject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STYX_SALIENT_ENABLED=0 → bridge не configure'ится → compress
    возвращает старый head+tail."""
    import psycopg

    from styx.engine import salient_bridge
    from styx.engine.salient import SALIENT_MARKER
    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    monkeypatch.setenv("STYX_SALIENT_ENABLED", "0")
    salient_bridge.reset_all()

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    try:
        p.sync_turn(
            "тестовое сообщение для записи в long-tier",
            "принял запись",
            session_id=sid,
        )
        # bridge не сконфигурирован — handle is None
        assert salient_bridge.get_handle("alpha") is None

        engine = StyxContextEngine(
            context_length=100_000, protect_first_n=3, protect_last_n=6
        )
        messages = [
            {"role": "system", "content": "you are styx assistant"},
            {"role": "user", "content": "продолжаем сессию"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "релевантный длинный запрос про тему"},
        ]
        out = engine.compress(messages, current_tokens=None)
        # Salient не должен появиться
        assert not any(
            SALIENT_MARKER in str(m.get("content", "")) for m in out
        )
    finally:
        p.shutdown()
        salient_bridge.reset_all()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
            conn.commit()
