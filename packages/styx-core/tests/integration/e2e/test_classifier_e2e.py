"""End-to-end classifier с реальным qwen3:4b-local.

Шаги:
1. INSERT memory + recall_event фикстурой.
2. ``enqueue_classification`` → pending llm_task.
3. ``styx worker run --once`` → handler вызывает qwen3, классифицирует,
   flip'ает used_in_output.
4. Проверить итог в БД.
"""

from __future__ import annotations

import os
import time
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
    return f"classifier-e2e-{uuid.uuid4().hex[:8]}"


def test_classifier_real_qwen3_flips_used_in_output(
    migrated: str, agent_id: str
) -> None:
    """Полный пайплайн: enqueue → worker run --once → флип used_in_output."""
    from styx.workers.main import run as worker_run

    # Drain любые leftover'ы от предыдущих тестов чтобы наш task был
    # единственным влияющим на assertion.
    with psycopg.connect(migrated) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM llm_tasks WHERE status = 'pending'"
            )
        conn.commit()

    with psycopg.connect(migrated) as conn:
        # INSERT memory + recall_event.
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind) "
                "VALUES (%s, 'user', %s, 'fact') RETURNING id",
                (
                    agent_id,
                    "пользователь предпочитает qwen3:4b-local из-за большого контекстного окна",
                ),
            )
            memory_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO recall_events "
                "  (memory_id, query_hash, match_score, used_in_output) "
                "VALUES (%s, %s, %s, false) RETURNING id",
                (memory_id, b"\x42" * 32, 0.8),
            )
            recall_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO llm_tasks (task_type, payload) VALUES "
                "  ('usage_classification', %s::jsonb)",
                (
                    f'{{"recall_event_ids": [{recall_id}], '
                    f'"llm_output_text": "Для этой задачи подойдёт qwen3:4b-local — у неё большое окно контекста, '
                    f'что у нас является приоритетом.", '
                    f'"agent_id": "{agent_id}"}}',
                ),
            )
        conn.commit()

    # worker_run(once=True) для случая когда sidecar styx-worker не
    # запущен. Если sidecar в compose'е активен — он уже мог
    # claim'нуть task через FOR UPDATE SKIP LOCKED; once-pass увидит
    # пустую очередь и вернёт 0. В любом случае дальше polling до
    # терминального статуса (по аналогии с
    # test_importance_worker_e2e._wait_for_task_completion).
    rc = worker_run(once=True)
    assert rc == 0

    deadline = time.monotonic() + 30.0
    task = None
    row = None
    while time.monotonic() < deadline:
        with psycopg.connect(migrated) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT status, result, error FROM llm_tasks "
                    " WHERE task_type = 'usage_classification' "
                    "   AND payload->>'agent_id' = %s "
                    " ORDER BY created_at DESC LIMIT 1",
                    (agent_id,),
                )
                task = cur.fetchone()
        if task and task["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert task is not None, "task не найдена в llm_tasks"
    assert task["status"] in ("done", "failed"), (
        f"task застряла в status={task['status']!r} (sidecar или once-worker не "
        f"довёл до терминального состояния за 30s)"
    )

    with psycopg.connect(migrated) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT used_in_output FROM recall_events WHERE id = %s", (recall_id,)
            )
            row = cur.fetchone()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent_id,))
        conn.commit()

    assert task["status"] == "done", (
        f"task failed: error={task.get('error')!r} result={task.get('result')!r}"
    )
    result = task["result"]
    if result.get("skip") is True:
        # Реальный qwen3 мог skip'нуть — fail-open.
        assert row["used_in_output"] is False
    else:
        # Содержательный ответ опирается на memory → used=true → flipped.
        assert row["used_in_output"] is True
        assert result["flipped"] >= 1
