"""Top-level conftest для styx-core tests.

Не подключает Hermes-checkout к sys.path — core импортируется без
Hermes ABC. Только Postgres-фикстуры для integration / storage тестов.
"""

from __future__ import annotations

import os

import psycopg
import pytest


DSN_ENV = "STYX_TEST_DATABASE_URL"

_STYX_DROP_STMTS = [
    # Working set state persistence (волна 13) — независимая таблица.
    "DROP TABLE IF EXISTS working_set CASCADE",
    # Memorybox-port таблицы (волна 7) — порядок матчит FK зависимости.
    "DROP TABLE IF EXISTS memory_consolidation_applications CASCADE",
    "DROP TABLE IF EXISTS reinterpret_applications CASCADE",
    "DROP TABLE IF EXISTS memory_reinterpretations CASCADE",
    "DROP TABLE IF EXISTS emotional_baseline CASCADE",
    "DROP TABLE IF EXISTS emotional_state CASCADE",
    "DROP TABLE IF EXISTS sweep_runs CASCADE",
    "DROP TABLE IF EXISTS consolidation_state CASCADE",
    "DROP TABLE IF EXISTS llm_tasks CASCADE",
    "DROP TABLE IF EXISTS relations CASCADE",
    # Documents + chunks (волна 19, миграция 0005). chunks через FK
    # снесётся CASCADE'ом, но прописываем явно для порядка.
    "DROP TABLE IF EXISTS chunks CASCADE",
    "DROP TABLE IF EXISTS documents CASCADE",
    # Триггер на memories ссылается на enqueue_importance_scoring;
    # CASCADE на DROP TABLE memories снесёт триггер, а функцию очистим
    # явно.
    "DROP FUNCTION IF EXISTS enqueue_importance_scoring() CASCADE",
    # Базовые Styx-таблицы.
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
    """DSN после полного сброса styx-объектов в этой БД."""
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for stmt in _STYX_DROP_STMTS:
                cur.execute(stmt)
        conn.commit()
    return dsn


@pytest.fixture
def migrated_db(clean_db: str) -> str:
    """DSN после применения миграций. Доступен во всех тестах."""
    from styx.storage import migrate
    migrate.run(clean_db)
    return clean_db
