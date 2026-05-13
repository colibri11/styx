"""``styx-hermes-setup`` CLI — установка styx-memory shim в HERMES_HOME.

General plugin (styx) регистрируется через entry-point pip'а
автоматически, его shim в HERMES_HOME не копируется (избегаем
double-registration).

Эта команда живёт в styx-hermes пакете отдельно от ``styx`` CLI core'а —
core daemon ничего не знает про Hermes-shim.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

PLUGIN_DIRS = ("styx-memory",)

# Path-traversal guard: куда разрешено ставить shim.
# STYX_ALLOW_HERMES_HOME=1 снимает ограничение для опытных пользователей.
_ALLOWED_ROOTS = (
    Path.home(),
    Path("/tmp"),
    Path("/var/folders"),  # macOS tmpdir (pytest tmp_path)
    Path("/var/tmp"),
    Path("/opt"),  # Docker installations
)


def _resolve_hermes_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes"


def _check_hermes_home_safe(target: Path) -> None:
    if os.environ.get("STYX_ALLOW_HERMES_HOME"):
        return
    resolved = target.resolve()
    for allowed in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(allowed.resolve())
            return
        except ValueError:
            continue
    raise SystemExit(
        f"hermes_home должен быть под $HOME, /opt или временной директорией.\n"
        f"  Запрошено: {target}\n"
        f"  Разрешено: $HOME, /tmp, /var/folders, /var/tmp, /opt\n"
        f"  Для обхода: STYX_ALLOW_HERMES_HOME=1"
    )


def cmd_setup(args: argparse.Namespace) -> int:
    hermes_home = _resolve_hermes_home(args.hermes_home).resolve()
    _check_hermes_home_safe(hermes_home)

    target_root = hermes_home / "plugins"
    target_root.mkdir(parents=True, mode=0o700, exist_ok=True)

    source_root = resources.files("styx_hermes.resources.plugins")

    written: list[str] = []
    for name in PLUGIN_DIRS:
        src = source_root / name
        dst = target_root / name
        if dst.exists() or dst.is_symlink():
            if not args.force:
                print(f"skip {dst} (уже существует, --force чтобы перезаписать)")
                continue
            if dst.is_symlink():
                raise SystemExit(
                    f"Отказано: {dst} является симлинком. "
                    "Удалите вручную перед --force."
                )
            shutil.rmtree(dst)
        dst.mkdir(parents=True, mode=0o700)
        for entry in src.iterdir():
            if entry.name == "__pycache__":
                continue
            (dst / entry.name).write_text(
                entry.read_text(encoding="utf-8"), encoding="utf-8"
            )
        written.append(name)

    if written:
        print(f"installed: {', '.join(written)} → {target_root}")
    else:
        print(f"nothing installed (target: {target_root})")

    print()
    print("Дальше:")
    print("  1. config.yaml: memory.provider: styx-memory")
    print("  2. config.yaml: plugins.enabled += ['styx']")
    print("  3. STYX_DAEMON_URL=http://127.0.0.1:8788 (или styx.json в HERMES_HOME)")
    print("  4. Запустить styx-core daemon: styx daemon run")
    print("  5. Запустить Hermes — general plugin (styx) подхватится")
    print("     через entry-point, memory provider — через directory shim")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="styx-hermes-setup",
        description=(
            "Установить styx-memory shim в $HERMES_HOME/plugins/. "
            "General plugin (styx) подхватывается через entry-point автоматически."
        ),
    )
    parser.add_argument(
        "--hermes-home",
        help="Путь к HERMES_HOME (по умолчанию ~/.hermes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующий shim",
    )
    args = parser.parse_args(argv)
    return cmd_setup(args)


if __name__ == "__main__":
    sys.exit(main())
