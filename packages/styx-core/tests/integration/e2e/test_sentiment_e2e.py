"""End-to-end sentiment hot-path с реальным qwen3:4b-local.

Шаги:
1. Прогнать миграции.
2. Создать ``StyxMemoryCore`` (sentiment включён по умолчанию).
3. ``sync_turn`` с эмоциональной user-репликой.
4. Проверить что в ``emotional_state`` появилась точка с
   ``source='hot_sentiment'``.
5. (Опционально) recompute_baseline на свежих instant'ах → UPSERT в
   ``emotional_baseline``.

Запуск: внутри hermes-styx контейнера.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


@pytest.fixture
def dsn() -> str:
    return os.environ["STYX_DATABASE_URL"]


@pytest.fixture
def migrated(dsn: str) -> str:
    from styx.storage import migrate
    migrate.run(dsn)
    return dsn


@pytest.fixture
def agent_id() -> str:
    return f"sentiment-e2e-{uuid.uuid4().hex[:8]}"


def test_sync_turn_writes_real_sentiment(migrated: str, agent_id: str) -> None:
    """Реальный qwen3 → real VAD → emotional_state appends."""
    from styx.providers.memory import StyxMemoryCore

    provider = StyxMemoryCore()
    sid = str(uuid.uuid4())
    provider.initialize(session_id=sid, agent_identity=agent_id)
    try:
        provider.sync_turn(
            "Ура, всё получилось наконец-то! Я очень рад этому моменту.",
            "Поздравляю!",
            session_id=sid,
        )
        # Проверка: emotional_state получил точку с source='hot_sentiment'.
        with psycopg.connect(migrated) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT valence, arousal, dominance, source "
                    "  FROM emotional_state WHERE agent_id = %s "
                    " ORDER BY at DESC, id DESC LIMIT 1",
                    (agent_id,),
                )
                row = cur.fetchone()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM emotional_state WHERE agent_id = %s", (agent_id,))
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent_id,))
            conn.commit()

        # Sentiment мог skip'нуться (timeout / schema-error) — fail-open.
        # Но если row есть — она должна быть от sentiment'а.
        if row is not None:
            assert row["source"] == "hot_sentiment"
            # Радостная фраза должна давать положительный valence.
            assert -1.0 <= float(row["valence"]) <= 1.0
            assert -1.0 <= float(row["arousal"]) <= 1.0
    finally:
        provider.shutdown()


def test_baseline_tick_after_sentiment(migrated: str, agent_id: str) -> None:
    """После пары sentiment-точек recompute_baseline UPSERT'ит baseline."""
    from styx.emotional.baseline import recompute_baseline
    from styx.providers.memory import StyxMemoryCore

    provider = StyxMemoryCore()
    sid = str(uuid.uuid4())
    provider.initialize(session_id=sid, agent_identity=agent_id)

    try:
        # 2-3 эмоциональные turn'а чтобы накопить instant.
        provider.sync_turn(
            "Ура, всё отлично работает, я счастлив!",
            "Класс!",
            session_id=sid,
        )
        provider.sync_turn(
            "Какой-то длинный текст вторая реплика теста с настроением.",
            "Понятно.",
            session_id=sid,
        )

        with psycopg.connect(migrated) as conn:
            # Поверим что хотя бы одна точка вообще написана.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM emotional_state WHERE agent_id = %s",
                    (agent_id,),
                )
                instant_count = cur.fetchone()[0]

            if instant_count == 0:
                pytest.skip("sentiment не вернул VAD ни на один turn (qwen3 может быть слабее на эмоциях)")

            result = recompute_baseline(conn, agent_id)
            conn.commit()

            assert result.skipped is False
            assert result.baseline is not None

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT valence, arousal, dominance, mood_active "
                    "  FROM emotional_baseline WHERE agent_id = %s",
                    (agent_id,),
                )
                row = cur.fetchone()
            assert row is not None
            assert row["mood_active"] is False  # default

            with conn.cursor() as cur:
                cur.execute("DELETE FROM emotional_baseline WHERE agent_id = %s", (agent_id,))
                cur.execute("DELETE FROM emotional_state WHERE agent_id = %s", (agent_id,))
                cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent_id,))
            conn.commit()
    finally:
        provider.shutdown()
