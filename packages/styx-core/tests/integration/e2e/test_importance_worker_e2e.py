"""End-to-end test для importance worker'а с реальным qwen3:4b-local.

Шаги:
1. Прогнать миграции (волна 7 уже завела триггер enqueue_importance_scoring).
2. INSERT memory через провайдер (sync_turn) → триггер ставит task в llm_tasks.
3. Запустить ``styx worker run --once`` в текущем процессе → claim'ит task,
   вызывает qwen3:4b-local на настроенной Ollama, валидирует JSON.
4. Проверить:
   - task.status в (``done``, ``failed``);
   - если done и не skip — ``importance_final`` ∈ [0, 1];
   - если done и skip — ``importance_final`` остаётся NULL, result содержит skip_reason.

Требует:
- /opt/hermes (запуск только в контейнере).
- pgvector БД (postgres из docker-compose).
- Ollama достижим через alias ``ollama:11434`` (extra_hosts в compose).
- qwen3:4b-local скачан на этой Ollama.
- ENV ``STYX_DATABASE_URL``, ``STYX_LLM_URL``, ``STYX_LLM_MODEL`` выставлены.

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d --build
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest \\
        /opt/styx/tests/integration/test_importance_worker_e2e.py -v
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
    return f"importance-e2e-{uuid.uuid4().hex[:8]}"


def _insert_memory(conn: psycopg.Connection, agent_id: str, content: str) -> uuid.UUID:
    """INSERT memory; триггер автоматически ставит pending task в llm_tasks."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content, kind) "
            " VALUES (%s, %s, %s, %s) RETURNING id",
            (agent_id, "user", content, "fact"),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def _wait_for_task_completion(
    conn: psycopg.Connection, memory_id: uuid.UUID, timeout_s: float = 60.0
) -> dict:
    """Ждать пока llm_task для memory_id перейдёт в done/failed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status, result, error FROM llm_tasks "
                " WHERE memory_id = %s "
                " ORDER BY created_at DESC LIMIT 1",
                (memory_id,),
            )
            row = cur.fetchone()
        if row and row["status"] in ("done", "failed"):
            return row
        time.sleep(1)
    raise AssertionError(f"task для memory {memory_id} не завершилась за {timeout_s}s")


def test_importance_worker_real_qwen3(migrated: str, agent_id: str) -> None:
    """Полный пайплайн: INSERT → trigger → worker → qwen3 → importance_final."""
    from styx.workers.main import run as worker_run

    content = (
        "пользователь предпочитает qwen3:4b-local из-за большого контекстного окна, "
        "несмотря на меньшее число параметров. Это устойчивое техническое "
        "предпочтение для structured JSON задач."
    )

    with psycopg.connect(migrated) as conn:
        memory_id = _insert_memory(conn, agent_id, content)

        # Проверим что триггер сработал — должна быть pending row.
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status FROM llm_tasks WHERE memory_id = %s "
                " ORDER BY created_at DESC LIMIT 1",
                (memory_id,),
            )
            row = cur.fetchone()
        assert row is not None, "триггер не поставил task в llm_tasks"
        assert row["status"] == "pending"

    # Прогнать worker в single-shot режиме.
    rc = worker_run(once=True)
    assert rc == 0

    # Проверить итог.
    with psycopg.connect(migrated) as conn:
        task = _wait_for_task_completion(conn, memory_id, timeout_s=5.0)

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT importance_final, importance_provisional FROM memories "
                " WHERE id = %s",
                (memory_id,),
            )
            mem = cur.fetchone()

        # Cleanup для повторных прогонов.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent_id,))
        conn.commit()

    assert task["status"] == "done", (
        f"task failed: error={task.get('error')!r} result={task.get('result')!r}"
    )
    result = task["result"]
    assert result is not None
    assert "skip" in result

    if result["skip"] is False:
        score = result["importance_score"]
        assert 0.0 <= float(score) <= 1.0
        assert mem["importance_final"] is not None
        assert abs(float(mem["importance_final"]) - float(score)) < 1e-6
    else:
        # Skip-path: модель решила что оценивать нечего.
        assert mem["importance_final"] is None
        assert isinstance(result["skip_reason"], str)
        assert len(result["skip_reason"]) > 0


def test_worker_run_once_empty_queue_no_op(migrated: str) -> None:
    """``styx worker run --once`` на пустой очереди: rc=0, никакой работы."""
    from styx.workers.main import run as worker_run

    rc = worker_run(once=True)
    assert rc == 0
