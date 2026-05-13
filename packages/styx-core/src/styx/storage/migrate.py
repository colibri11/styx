"""Идемпотентный мигратор Styx storage.

Применяет .sql файлы из styx/storage/schema/ в лексикографическом порядке.
Применённые файлы трекаются в таблице ``_styx_migrations``; повторный
прогон — no-op.

CLI:
    python -m styx.storage.migrate [DATABASE_URL]

Без аргумента читает DSN из ``$STYX_DATABASE_URL`` либо ``$DATABASE_URL``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import psycopg

log = logging.getLogger(__name__)

SCHEMA_PACKAGE = "styx.storage.schema"
MIGRATIONS_TABLE = "_styx_migrations"


@dataclass(frozen=True)
class Migration:
    name: str
    sql: str


def _bootstrap_sql() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
        name        text PRIMARY KEY,
        applied_at  timestamptz NOT NULL DEFAULT now()
    );
    """


def discover_migrations() -> list[Migration]:
    """Читает .sql из пакета schema/ в порядке имени."""
    files = resources.files(SCHEMA_PACKAGE)
    out: list[Migration] = []
    for entry in sorted(files.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".sql"):
            continue
        out.append(Migration(name=entry.name, sql=entry.read_text(encoding="utf-8")))
    return out


def applied_names(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT name FROM {MIGRATIONS_TABLE}")
        return {row[0] for row in cur.fetchall()}


def apply(conn: psycopg.Connection, migrations: list[Migration]) -> list[str]:
    """Применяет недостающие миграции. Возвращает список имён применённых.

    Каждая миграция применяется в собственной транзакции; ошибка в одной
    не влияет на уже применённые.
    """
    with conn.cursor() as cur:
        cur.execute(_bootstrap_sql())
    conn.commit()

    already = applied_names(conn)
    applied: list[str] = []
    for migration in migrations:
        if migration.name in already:
            log.debug("skip %s — already applied", migration.name)
            continue
        log.info("apply %s", migration.name)
        with conn.cursor() as cur:
            cur.execute(migration.sql)
            cur.execute(
                f"INSERT INTO {MIGRATIONS_TABLE} (name) VALUES (%s)",
                (migration.name,),
            )
        conn.commit()
        applied.append(migration.name)
    return applied


def run(dsn: str) -> list[str]:
    migrations = discover_migrations()
    with psycopg.connect(dsn) as conn:
        return apply(conn, migrations)


def _resolve_dsn(argv: list[str]) -> str:
    if len(argv) > 1:
        return argv[1]
    for var in ("STYX_DATABASE_URL", "DATABASE_URL"):
        value = os.environ.get(var)
        if value:
            return value
    raise SystemExit(
        "DSN не задан. Передай первым аргументом или экспортируй STYX_DATABASE_URL."
    )


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("STYX_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    dsn = _resolve_dsn(sys.argv)
    applied = run(dsn)
    if applied:
        print(f"applied: {', '.join(applied)}")
    else:
        print("no migrations to apply")


if __name__ == "__main__":
    main()
