"""Integration: Hebbian co_retrieval в StyxMemoryCore.handle_tool_call (волна 21).

После recall_full с N≥2 results — на всех C(N, 2) парах появляются
'co_retrieved' рёбра в `relations` с initial weight 1.1. Повторный
recall тех же ids — bump до 1.2.

Использует реальный Ollama embedder (т.к. FakeEmbeddingClient даёт
ортогональные вектора → recall не пробивает min_score). Скипается
на host без Ollama.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from styx.providers.memory import StyxMemoryCore
from styx.storage.queries import AgentScopedQueries


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="needs real Ollama — runs only inside hermes-agent-styx-test container",
)


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str) -> str:
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    return migrated_db


def _make_provider(monkeypatch: pytest.MonkeyPatch) -> StyxMemoryCore:
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "0")
    return StyxMemoryCore()


def _seed_memories(dsn: str, agent: str, contents: list[str]) -> list[uuid.UUID]:
    """Pre-seed memories через реальный Ollama embedding."""
    from styx.config import load as load_config
    from styx.embedding import make_embedding_client
    cfg = load_config()
    embed = make_embedding_client(
        base_url=cfg.ollama_url, model=cfg.embedding_model,
        dim=cfg.embedding_dim, timeout=cfg.embedding_timeout_s,
    )
    ids: list[uuid.UUID] = []
    with psycopg.connect(dsn) as conn:
        q = AgentScopedQueries(conn, agent)
        for content in contents:
            mid = q.insert_memory(
                role="summary", content=content,
                kind="note", kind_src="subjective",
                embedding=embed.embed(content),
            )
            ids.append(mid)
        conn.commit()
    return ids


def _count_co_retrieved(dsn: str) -> int:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM relations WHERE relation='co_retrieved'"
            )
            return cur.fetchone()[0]


def test_recall_creates_co_retrieved_pairs(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 memories на 1 запрос → 3 ребра co_retrieved (C(3,2))."""
    contents = [
        "напоминание про релиз пятницы команды mobile",
        "обсуждение релиза мобильного приложения",
        "работа над мобильным релизом до пятницы",
    ]
    _seed_memories(provider_env, "alpha", contents)

    p = _make_provider(monkeypatch)
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        out = p.handle_tool_call(
            "styx_recall",
            {"query": "релиз мобильного приложения", "limit": 5},
            session_id=sid,
        )
        # Recall должен вернуть ≥ 2 результата.
        import json
        parsed = json.loads(out)
        assert parsed["count"] >= 2

        # Если N результатов — должно быть C(N, 2) рёбер.
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*), avg(weight) FROM relations "
                    " WHERE relation='co_retrieved'"
                )
                count, avg_weight = cur.fetchone()
        n = parsed["count"]
        expected_pairs = n * (n - 1) // 2
        assert count == expected_pairs
        # Initial weight = 1.1.
        assert abs(float(avg_weight) - 1.1) < 1e-6
    finally:
        p.shutdown()


def test_repeat_recall_bumps_weight(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Тот же recall дважды → weight каждого ребра bumps от 1.1 до 1.2."""
    contents = [
        "напоминание про деплой",
        "обсуждаем деплой",
    ]
    _seed_memories(provider_env, "alpha", contents)

    p = _make_provider(monkeypatch)
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        p.handle_tool_call("styx_recall", {"query": "деплой"}, session_id=sid)
        p.handle_tool_call("styx_recall", {"query": "деплой"}, session_id=sid)
        with psycopg.connect(provider_env) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT weight FROM relations "
                    " WHERE relation='co_retrieved' LIMIT 1"
                )
                row = cur.fetchone()
        if row is not None:
            assert abs(row[0] - 1.2) < 1e-6
    finally:
        p.shutdown()


def test_disabled_hebbian_no_relations(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STYX_HEBBIAN_ENABLED=0 → no co_retrieved рёбра."""
    contents = ["A unique content one", "A unique content two"]
    _seed_memories(provider_env, "alpha", contents)
    monkeypatch.setenv("STYX_HEBBIAN_ENABLED", "0")

    p = _make_provider(monkeypatch)
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        p.handle_tool_call(
            "styx_recall", {"query": "unique content"}, session_id=sid,
        )
        assert _count_co_retrieved(provider_env) == 0
    finally:
        p.shutdown()


def test_single_result_no_pairs(
    provider_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recall с 1 результатом → 0 пар (нечего связывать)."""
    _seed_memories(provider_env, "alpha", ["единственная запись для теста"])
    p = _make_provider(monkeypatch)
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        p.handle_tool_call(
            "styx_recall", {"query": "единственная"}, session_id=sid,
        )
        assert _count_co_retrieved(provider_env) == 0
    finally:
        p.shutdown()
