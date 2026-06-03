"""Styx CLI — ``styx <subcommand>``.

- ``styx migrate [DSN]`` — применить миграции схемы Postgres.
- ``styx reembed`` — backfill ``memories.embedding`` (NULL-fill или
  full re-embed после смены модели).
- ``styx rename-agent <old> <new>`` — переименовать ``agent_id`` по
  всем agent-scoped таблицам (волна 32; админская миграция).
- ``styx daemon run`` — поднять HTTP API + worker pool в одном процессе.
- ``styx daemon healthcheck [--url ...]`` — стук в /healthz remote daemon.

Команда ``setup`` (установка shim в HERMES_HOME) переехала в styx-hermes
как отдельная утилита ``styx-hermes-setup``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger(__name__)


def cmd_migrate(args: argparse.Namespace) -> int:
    from styx.storage.migrate import _resolve_dsn, run

    dsn = _resolve_dsn(["styx", args.dsn] if args.dsn else ["styx"])
    applied = run(dsn)
    if applied:
        print(f"applied: {', '.join(applied)}")
    else:
        print("no migrations to apply")
    return 0


def cmd_reembed(args: argparse.Namespace) -> int:
    """``styx reembed`` — backfill эмбеддингов."""
    import psycopg

    from styx.commands.reembed import (
        REEMBED_MODE_ALL,
        REEMBED_MODE_NULL_ONLY,
        run_reembed,
    )
    from styx.config import load as load_config
    from styx.embedding import make_embedding_client

    config = load_config()
    embed = make_embedding_client(
        base_url=config.ollama_url,
        model=config.embedding_model,
        dim=config.embedding_dim,
        timeout=config.embedding_timeout_s,
    )
    mode = REEMBED_MODE_ALL if args.all else REEMBED_MODE_NULL_ONLY

    with psycopg.connect(config.database_url) as conn:
        result = run_reembed(
            conn=conn,
            embed_client=embed,
            mode=mode,
            agent_id=args.agent_id,
            limit=args.limit,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            rate_per_second=args.rate_per_second,
        )

    if args.dry_run:
        print(f"would process {result.would_process} memories")
    else:
        print(f"processed={result.processed} failed={result.failed}")
    return 0


def cmd_rename_agent(args: argparse.Namespace) -> int:
    """``styx rename-agent <old> <new>`` — переименование agent_id.

    Открывает ``psycopg.connect``; ``with conn:`` коммитит на чистом
    выходе / rollback на исключении — это и даёт **одну транзакцию** на
    все UPDATE'ы (атомарность «всё-или-ничего»). Частичный rename =
    расщеплённое `я`, хуже отказа.
    """
    import psycopg

    from styx.commands.rename_agent import run_rename_agent
    from styx.config import load as load_config

    config = load_config()

    # Интерактивный confirm — только если не --yes, не --dry-run и tty
    # доступен. В неинтерактивной среде (docker exec без -t) confirm
    # пропускается (как будто --yes), чтобы не зависнуть на input().
    if not args.dry_run and not args.yes and sys.stdin.isatty():
        prompt = (
            f"Переименовать agent_id '{args.old}' → '{args.new}' "
            "по всем agent-scoped таблицам? Это прямая запись в БД "
            "(daemon должен быть остановлен / агент неактивен). [y/N]: "
        )
        answer = input(prompt).strip().lower()
        if answer not in ("y", "yes"):
            print("отменено")
            return 1

    try:
        with psycopg.connect(config.database_url) as conn:
            result = run_rename_agent(
                conn=conn,
                old=args.old,
                new=args.new,
                dry_run=args.dry_run,
            )
    except ValueError as exc:
        # Штатные операторские отказы (collision / нет агента / валидация)
        # — чистый stderr + return 1, без traceback (migration-команда,
        # collision/«нет агента» — ожидаемые юзер-кейсы, не баг).
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    verb = "would rename" if result.dry_run else "renamed"
    for table, count in result.tables:
        if count:
            print(f"  {table}: {count}")
    print(
        f"{verb} {result.total_rows} rows across {len(result.tables)} tables "
        f"({result.old!r} → {result.new!r})"
    )
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """``styx daemon <subcommand>`` — run / healthcheck."""
    from styx.http.server import healthcheck, run_daemon

    if args.daemon_cmd == "run":
        run_daemon(bind=args.bind, port=args.port)
        return 0
    if args.daemon_cmd == "healthcheck":
        return healthcheck(args.url)
    raise SystemExit(f"unknown daemon subcommand: {args.daemon_cmd!r}")


def main(argv: list[str] | None = None) -> int:
    from styx.observability.logging import setup_logging

    setup_logging(
        format=os.environ.get("STYX_LOG_FORMAT", "text"),
        level=os.environ.get("STYX_LOG_LEVEL", "INFO"),
    )

    parser = argparse.ArgumentParser(prog="styx", description="Styx CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    migrate = subparsers.add_parser("migrate", help="Применить миграции БД")
    migrate.add_argument("dsn", nargs="?", help="DSN; иначе из STYX_DATABASE_URL")
    migrate.set_defaults(func=cmd_migrate)

    reembed = subparsers.add_parser(
        "reembed",
        help="Backfill memories.embedding (NULL-fill или full re-embed)"
    )
    mode_group = reembed.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--null-only",
        action="store_true",
        default=True,
        help="Только memories с NULL embedding'ом (default)"
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Полный re-embed (после смены модели)"
    )
    reembed.add_argument("--agent-id", help="Фильтр по agent_id")
    reembed.add_argument("--limit", type=int, help="Максимум памятей за прогон")
    reembed.add_argument(
        "--dry-run",
        action="store_true",
        help="SELECT count и выход; UPDATE'ы не идут"
    )
    reembed.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Размер выборки за итерацию (default 50)"
    )
    reembed.add_argument(
        "--rate-per-second",
        type=float,
        default=5.0,
        help="Лимит embed-вызовов в секунду (default 5.0)"
    )
    reembed.set_defaults(func=cmd_reembed)

    rename = subparsers.add_parser(
        "rename-agent",
        help="Переименовать agent_id по всем agent-scoped таблицам",
        description=(
            "Переименовать agent_id <old> → <new> во всех таблицах со "
            "столбцом agent_id (schema-driven). Одна транзакция, UUID не "
            "трогаются. ОПЕРАЦИОННАЯ ОГОВОРКА: это прямая запись в БД; если "
            "daemon в этот момент держит in-memory state агента <old> "
            "(focus_tracker / hot_tier / working_set), он осиротеет. "
            "Выполняй на НЕАКТИВНОМ агенте либо рестартни daemon после."
        ),
    )
    rename.add_argument("old", help="Текущий agent_id")
    rename.add_argument("new", help="Новый agent_id (не должен существовать)")
    rename.add_argument(
        "--dry-run",
        action="store_true",
        help="Только counts по таблицам для <old>; UPDATE'ы не идут",
    )
    rename.add_argument(
        "--yes",
        action="store_true",
        help="Без интерактивного подтверждения (для скриптов / docker exec)",
    )
    rename.set_defaults(func=cmd_rename_agent)

    daemon = subparsers.add_parser(
        "daemon",
        help="Run / probe styx-core HTTP daemon (см. Phase C)",
    )
    daemon_subs = daemon.add_subparsers(dest="daemon_cmd", required=True)
    daemon_run = daemon_subs.add_parser(
        "run", help="Поднять HTTP API + worker pool в одном процессе"
    )
    daemon_run.add_argument("--bind", default="127.0.0.1", help="HTTP bind address")
    daemon_run.add_argument("--port", type=int, default=8788, help="HTTP port")
    daemon_run.set_defaults(func=cmd_daemon)
    daemon_check = daemon_subs.add_parser(
        "healthcheck", help="Стук в /healthz remote daemon"
    )
    daemon_check.add_argument(
        "--url", default="http://127.0.0.1:8788", help="Daemon base URL"
    )
    daemon_check.set_defaults(func=cmd_daemon)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
