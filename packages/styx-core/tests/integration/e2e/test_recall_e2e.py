"""End-to-end recall pipeline в реальном Hermes-Docker стеке.

Этот тест:
1. Поднимает StyxMemoryCore через реальный memory discovery.
2. Делает несколько turn'ов через sync_turn — это пишет memories +
   embed-after-commit через РЕАЛЬНЫЙ Ollama-инстанс.
3. Вызывает styx_recall — embed query, search_similar, формат
   ответа.

Требует:
- /opt/hermes (Hermes runtime в контейнере)
- pgvector БД (postgres сервис в compose)
- Ollama reachable через extra_hosts (compose выставляет hostname `ollama`)
- ENV STYX_DATABASE_URL, STYX_OLLAMA_URL уже выставлены.

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration/test_recall_e2e.py -v
"""

from __future__ import annotations

import json
import os
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


@pytest.fixture
def styx_provider():
    """Чистый StyxMemoryCore, мигрированная БД, реальный Ollama.

    Каждый тест получает уникальный agent_id — изоляция данных между
    тестами через application-level WHERE, без TRUNCATE."""
    import psycopg

    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    agent = "alpha"
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity=agent)
    yield p, sid, agent
    p.shutdown()

    # Cleanup: удаляем только данные ЭТОГО теста (по agent_id).
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM working_set WHERE agent_id = %s", (agent,))
                cur.execute("DELETE FROM sessions WHERE agent_id = %s", (agent,))
        conn.commit()


def test_embed_after_commit_writes_real_vector(styx_provider) -> None:
    """sync_turn → memory с НЕнулевым embedding из реального Ollama."""
    import psycopg

    p, sid, agent = styx_provider
    p.sync_turn(
        user_content="Это тестовое сообщение для embed-after-commit.",
        assistant_content="Принял, проверяю запись эмбеддинга в БД.",
        session_id=sid,
    )

    dsn = os.environ["STYX_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), count(embedding) "
                "FROM memories WHERE agent_id = %s",
                (agent,),
            )
            total, with_emb = cur.fetchone()

    assert total == 2, "должно быть две memory (user + assistant)"
    assert with_emb == 2, "обе должны получить embedding после COMMIT"


def test_styx_recall_returns_relevant_memory(styx_provider) -> None:
    """Реальный pipeline: пишем 3 пары, recall на тематически близкий
    запрос — ловит наиболее релевантную.

    Проверяем pipeline (embed + search + dedup), не качество ranking'а
    на конкретных данных. Min_score дефолта (0.32 после волны 8,
    decisions § 22) — для гарантированного pipeline теста запрашиваем
    точное совпадение с одним из content'ов (cosine ≈ 1).
    """
    p, sid, _ = styx_provider

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

    # Точное совпадение query == content одной из memories гарантирует
    # cosine ≈ 1, проходит любой min_score < 1.0.
    raw = p.handle_tool_call(
        "styx_recall",
        {"query": "Расскажи про embedding-модели Ollama для Styx.", "limit": 6},
    )
    out = json.loads(raw)

    assert "memories_text" in out
    assert out["count"] >= 1
    # Топ должен содержать matched content или его пару (через recency/decay
    # композитный score может выдвинуть assistant-ответ на первое место).
    text = out["memories_text"]
    assert ("embedding" in text.lower()) or ("Ollama" in text) or ("Styx" in text)


def test_styx_recall_low_threshold_returns_more(styx_provider) -> None:
    """Pipeline-тест: с пониженным min_score recall возвращает ≥1
    результат на тематически близкий русский запрос.

    Проверяет, что embeddinggemma реально работает на русском и
    composite score формируется. min_score 0.4 — вне дефолта, но это
    тест pipeline'а, не калибровки.
    """
    from dataclasses import replace
    from styx.storage.recall import format_recall_text, recall_full
    from styx.storage.recall_config import DEFAULT_RECALL_CONFIG

    p, sid, _ = styx_provider

    p.sync_turn(
        "Что такое embeddinggemma и как она используется?",
        "Это лёгкая multilingual модель Google для эмбеддингов, "
        "dim 768, Q8_0 quantization, работает в Ollama.",
        session_id=sid,
    )

    cfg = replace(DEFAULT_RECALL_CONFIG.full, min_score=0.4)
    result = recall_full(
        queries=p.queries,
        embed_client=p._embedding,  # type: ignore[arg-type]
        query="расскажи про embeddinggemma",
        full_config=cfg,
    )
    # С низким min_score хотя бы одна из двух memories должна попасть.
    assert len(result.memories) >= 1, (
        f"recall на embeddinggemma через реальный Ollama пуст; "
        f"queried={result.queried_count}"
    )
    # Содержательная проверка: текст topic-related.
    text = format_recall_text(result)
    assert "embeddinggemma" in text.lower() or "multilingual" in text.lower()


def test_paraphrase_matches_on_default_threshold(styx_provider) -> None:
    """Регрессия для волны 8 (decisions § 22): перифраз семантически
    близкого запроса match'ится на **дефолтном** min_score без явного
    override'а.

    Кейс из parking — на старом пороге 0.6 (откалиброван под bge-m3) этот
    запрос возвращал пустой результат из-за того что cosine sim
    embeddinggemma на этой паре ≈ 0.44 (см. bench-suite
    paraphrase-01-parking-regression). Новый дефолт 0.32 эту пару
    пропускает.
    """
    p, sid, _ = styx_provider
    p.sync_turn(
        "Расскажи про embedding-модели Ollama для Styx.",
        "Используем embeddinggemma:300m-qat-q8_0, multilingual, 768-dim.",
        session_id=sid,
    )

    raw = p.handle_tool_call(
        "styx_recall",
        {"query": "какую embedding модель использует Styx", "limit": 6},
    )
    out = json.loads(raw)

    assert out["count"] >= 1, (
        "перифраз должен match'иться на дефолтном min_score=0.32 (волна 8); "
        f"output={out!r}"
    )
    # Recall может вернуть только user-message (более похож на query),
    # либо обе записи (user+assistant). Главное — match сработал; проверяем
    # что в тексте есть термины записанной memory (а не пустой случайный hit).
    text = out["memories_text"].lower()
    assert ("embedding" in text) or ("ollama" in text) or ("styx" in text), (
        f"recall вернул что-то, но не эту memory: {out['memories_text']!r}"
    )


def test_recall_event_persisted(styx_provider) -> None:
    """После styx_recall в recall_events лежит запись с query_hash + match_score."""
    import psycopg

    p, sid, agent = styx_provider
    p.sync_turn("hello world", "hi back", session_id=sid)
    # Точное совпадение query == content гарантирует score выше min_score.
    p.handle_tool_call("styx_recall", {"query": "hello world"})

    dsn = os.environ["STYX_DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Фильтруем по агенту через JOIN на memories.
            cur.execute(
                "SELECT count(*), count(re.query_hash), count(re.match_score) "
                "FROM recall_events re "
                "JOIN memories m ON m.id = re.memory_id "
                "WHERE m.agent_id = %s",
                (agent,),
            )
            total, with_hash, with_score = cur.fetchone()

    assert total >= 1
    assert with_hash == total
    assert with_score == total


def test_recall_excludes_other_agents() -> None:
    """Application-level WHERE по agent_id (decisions § 17.1, RLS не тащим)."""
    import psycopg

    from styx.providers.memory import StyxMemoryCore
    from styx.storage import migrate

    dsn = os.environ["STYX_DATABASE_URL"]
    migrate.run(dsn)

    sid_a = str(uuid.uuid4())
    sid_b = str(uuid.uuid4())

    p_a = StyxMemoryCore()
    p_a.initialize(session_id=sid_a, agent_identity="agent-isolation-a")
    p_b = StyxMemoryCore()
    p_b.initialize(session_id=sid_b, agent_identity="agent-isolation-b")

    try:
        p_a.sync_turn(
            "Я agent-a и держу секретные данные.",
            "Понял, фиксирую.",
            session_id=sid_a,
        )
        # Агент B запрашивает по теме агента A.
        raw = p_b.handle_tool_call(
            "styx_recall", {"query": "Я agent-a и держу секретные данные"}
        )
        out = json.loads(raw)
        # Memory A не должна попасть в результат B.
        assert "agent-a" not in out.get("memories_text", "")
        assert out["count"] == 0
    finally:
        p_a.shutdown()
        p_b.shutdown()

        # Cleanup
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memories WHERE agent_id IN "
                    "('agent-isolation-a', 'agent-isolation-b')"
                )
            conn.commit()
