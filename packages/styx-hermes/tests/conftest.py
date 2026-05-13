"""Top-level conftest для styx-hermes tests.

Подключает Hermes-checkout к sys.path до сбора тестов (нужен для импорта
ABC из ``agent.*``). Также включает Postgres-фикстуры для integration/e2e
тестов которые требуют реальную БД.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from styx_hermes import _hermes_path

_hermes_path.ensure_on_path()


DSN_ENV = "STYX_TEST_DATABASE_URL"

_STYX_DROP_STMTS = [
    "DROP TABLE IF EXISTS working_set CASCADE",
    "DROP TABLE IF EXISTS memory_consolidation_applications CASCADE",
    "DROP TABLE IF EXISTS reinterpret_applications CASCADE",
    "DROP TABLE IF EXISTS memory_reinterpretations CASCADE",
    "DROP TABLE IF EXISTS emotional_baseline CASCADE",
    "DROP TABLE IF EXISTS emotional_state CASCADE",
    "DROP TABLE IF EXISTS sweep_runs CASCADE",
    "DROP TABLE IF EXISTS consolidation_state CASCADE",
    "DROP TABLE IF EXISTS llm_tasks CASCADE",
    "DROP TABLE IF EXISTS relations CASCADE",
    "DROP FUNCTION IF EXISTS enqueue_importance_scoring() CASCADE",
    "DROP TABLE IF EXISTS recall_events CASCADE",
    "DROP TABLE IF EXISTS memories CASCADE",
    "DROP TABLE IF EXISTS sessions CASCADE",
    "DROP TABLE IF EXISTS _styx_migrations CASCADE",
]


@pytest.fixture(scope="session")
def dsn() -> str:
    value = os.environ.get(DSN_ENV)
    if not value:
        pytest.skip(f"{DSN_ENV} не задан — интеграционные тесты пропущены")
    return value


@pytest.fixture
def clean_db(dsn: str) -> str:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for stmt in _STYX_DROP_STMTS:
                cur.execute(stmt)
        conn.commit()
    return dsn


@pytest.fixture
def migrated_db(clean_db: str) -> str:
    from styx.storage import migrate
    migrate.run(clean_db)
    return clean_db
