"""End-to-end test для lifecycle sweep'а в Hermes-Docker стеке.

Шаги:
1. Прогнать миграции.
2. INSERT 50 фикстурных memories разных возрастов.
3. ``styx worker sweep`` через CLI один раз.
4. Проверить что memories распределились по lifecycle bucket'ам в
   соответствии с порогом, и в ``sweep_runs`` появилась строка status='success'.

Без LLM — sweep чистый SQL.
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
    return f"sweep-e2e-{uuid.uuid4().hex[:8]}"


def _insert_memory(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    content: str = "fixture",
    age_days: float = 0,
    lifecycle: str = "fresh",
) -> uuid.UUID:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories "
            "  (agent_id, role, content, lifecycle, created_at) "
            "VALUES (%s, 'user', %s, %s, now() - make_interval(secs => %s)) "
            "RETURNING id",
            (agent_id, content, lifecycle, age_days * 86400),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def test_sweep_moves_old_memories_through_lifecycle(migrated: str, agent_id: str) -> None:
    """Старые fresh memories каскадом fresh → settled → dormant в одном sweep'е.

    Cold-start пороги: fresh→settled=1d, settled→dormant=7d. Memory без
    last_accessed_at и age=10d сначала попадает в settled (age>1d), затем
    в dormant (idle=10d>7d, idle = COALESCE(last_accessed_at, created_at)).
    Это intentional cascade — соответствует memorybox.
    """
    from styx.workers.sweep.runner import run_sweep

    with psycopg.connect(migrated) as conn:
        # 5 свежих (age=0.1d), 5 на settled-границе (age=2d, idle уже сегодня),
        # 5 старых (age=10d).
        young_ids = [_insert_memory(conn, agent_id=agent_id, age_days=0.1) for _ in range(5)]
        # Memory с age=2d но last_accessed_at = сегодня (idle=0) — settled.
        with conn.cursor() as cur:
            recent_settled_ids: list[uuid.UUID] = []
            for _ in range(5):
                cur.execute(
                    "INSERT INTO memories "
                    "  (agent_id, role, content, lifecycle, created_at, last_accessed_at) "
                    "VALUES (%s, 'user', 'recent-touched', 'fresh', "
                    "        now() - interval '2 days', now()) "
                    "RETURNING id",
                    (agent_id,),
                )
                recent_settled_ids.append(cur.fetchone()[0])
        conn.commit()
        old_ids = [_insert_memory(conn, agent_id=agent_id, age_days=10) for _ in range(5)]

    result = run_sweep(migrated)
    assert result.status == "success"
    assert result.skipped is False
    assert "lifecycle_refresh" in result.summary

    with psycopg.connect(migrated) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, lifecycle FROM memories WHERE agent_id = %s",
                (agent_id,),
            )
            rows = {r["id"]: r["lifecycle"] for r in cur.fetchall()}
        # Cleanup.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent_id,))
        conn.commit()

    for yid in young_ids:
        assert rows[yid] == "fresh", f"young {yid} expected fresh"
    for sid in recent_settled_ids:
        # age=2d>1d → settled; idle=0d<7d → НЕ переходит дальше → settled.
        assert rows[sid] == "settled", f"recent_touched {sid} expected settled"
    for oid in old_ids:
        # age=10d>1d → settled; idle=10d>7d → каскадом в dormant.
        assert rows[oid] == "dormant", f"old {oid} expected dormant"


def test_sweep_writes_runs_row(migrated: str) -> None:
    """sweep_runs пополняется строкой status='success'."""
    from styx.workers.sweep.runner import run_sweep

    result = run_sweep(migrated)
    assert result.sweep_id is not None
    with psycopg.connect(migrated) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status, finished_at FROM sweep_runs WHERE id = %s",
                (result.sweep_id,),
            )
            row = cur.fetchone()
    assert row["status"] == "success"
    assert row["finished_at"] is not None
