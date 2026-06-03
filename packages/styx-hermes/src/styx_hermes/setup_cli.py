"""``styx-hermes-setup`` CLI — установка styx-memory shim в HERMES_HOME.

General plugin (styx) регистрируется через entry-point pip'а
автоматически, его shim в HERMES_HOME не копируется (избегаем
double-registration).

Эта команда живёт в styx-hermes пакете отдельно от ``styx`` CLI core'а —
core daemon ничего не знает про Hermes-shim.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path

import yaml

PLUGIN_DIRS = ("styx-memory",)

# Класс допустимых имён профилей Hermes: строчные буквы/цифры, дефис,
# подчёркивание; первый символ — буква/цифра. Совпадает с правилами
# Hermes для имён профилей. 'default' отклоняется отдельно — база
# подключается БЕЗ --profile.
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

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


def _validate_profile_name(name: str) -> None:
    """Проверить имя --profile по классу Hermes. Английская ошибка + nonzero."""
    if name == "default":
        raise SystemExit(
            "Invalid --profile 'default': the base profile is attached without "
            "--profile. Run --attach with no --profile to attach the base."
        )
    if not _PROFILE_NAME_RE.match(name):
        raise SystemExit(
            f"Invalid --profile {name!r}: profile names must be lowercase "
            "alphanumeric with '-'/'_' (pattern: ^[a-z0-9][a-z0-9_-]*$)."
        )


def _resolve_config_path(hermes_home: Path, profile: str | None) -> Path:
    """Путь к target config.yaml. base → <home>/config.yaml;
    named → <home>/profiles/<name>/config.yaml."""
    if profile is None:
        return hermes_home / "config.yaml"
    return hermes_home / "profiles" / profile / "config.yaml"


def _compute_patched(data: dict) -> tuple[dict, bool]:
    """Чистая функция attach-патча. Возвращает (новый dict, changed?).

    Семантика идентична прежнему cont-init-патчу config.yaml:
      - memory.provider = "styx-memory"
      - plugins.enabled — гарантировать list, добавить "styx" без дублей
      - context.engine = "styx"
    Исходный dict не мутируется (работаем по deep-copy), чтобы caller мог
    сравнить before/after.
    """
    patched = copy.deepcopy(data)

    patched.setdefault("memory", {})["provider"] = "styx-memory"

    plugins = patched.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])
    if not isinstance(enabled, list):
        enabled = []
    if "styx" not in enabled:
        enabled.append("styx")
    plugins["enabled"] = enabled

    patched.setdefault("context", {})["engine"] = "styx"

    changed = patched != data
    return patched, changed


@dataclass
class AttachResult:
    changed: bool
    backup_path: Path | None
    config_path: Path


def _attach(config_path: Path) -> AttachResult:
    """Файловая операция attach. config_path обязан существовать (caller
    проверяет и печатает English-ошибку). Бэкап делается только когда патч
    реально меняет файл; идемпотентный повтор — no-op без бэкапа."""
    raw = config_path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8")) or {}

    patched, changed = _compute_patched(data)
    if not changed:
        return AttachResult(changed=False, backup_path=None, config_path=config_path)

    # СНАЧАЛА бэкап исходных байт дословно, ПОТОМ запись патча.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = config_path.with_name(f"{config_path.name}.bak.{ts}")
    backup_path.write_bytes(raw)

    config_path.write_text(
        yaml.safe_dump(patched, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return AttachResult(changed=True, backup_path=backup_path, config_path=config_path)


def cmd_attach(args: argparse.Namespace) -> int:
    profile = args.profile
    if profile is not None:
        _validate_profile_name(profile)

    hermes_home = _resolve_hermes_home(args.hermes_home).resolve()
    _check_hermes_home_safe(hermes_home)

    config_path = _resolve_config_path(hermes_home, profile)
    label = profile if profile is not None else "base"

    if not config_path.exists():
        if profile is not None:
            raise SystemExit(
                f"Config not found for profile {profile!r} at {config_path}. "
                "The profile has no config.yaml yet — initialize it first "
                "(run the profile once), then re-run --attach."
            )
        raise SystemExit(f"Config not found at {config_path}.")

    result = _attach(config_path)

    if not result.changed:
        print(f"'{label}' already attached to Styx; no changes")
        return 0

    print(f"attached '{label}' to Styx:")
    print("  memory.provider = styx-memory")
    print("  plugins.enabled += styx")
    print("  context.engine = styx")
    print(f"  config: {config_path}")
    print(f"  backup: {result.backup_path}")
    print(
        "Restart the profile gateway / container to apply "
        "(a running gateway reverts config edits)."
    )
    return 0


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
    print("  3. config.yaml: context.engine: styx")
    print("     (иначе Hermes использует built-in compressor — Styx-движок")
    print("      зарегистрирован, но не выбран)")
    print("  4. STYX_DAEMON_URL=http://127.0.0.1:8788 (или styx.json в HERMES_HOME)")
    print("  5. Запустить styx-core daemon: styx daemon run")
    print("  6. Запустить Hermes — general plugin (styx) подхватится")
    print("     через entry-point, memory provider — через directory shim")
    print()
    print("  Либо подключить профиль к Styx идемпотентным attach (правит")
    print("  config.yaml: memory.provider/plugins.enabled/context.engine):")
    print("    styx-hermes-setup --attach                  # база")
    print("    styx-hermes-setup --attach --profile <name> # именованный профиль")
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
    parser.add_argument(
        "--attach",
        action="store_true",
        help=(
            "Вместо установки shim'а — идемпотентно подключить профиль к Styx "
            "(патч config.yaml: memory.provider/plugins.enabled/context.engine)"
        ),
    )
    parser.add_argument(
        "--profile",
        help=(
            "Имя именованного профиля для --attach "
            "(<hermes_home>/profiles/<name>/config.yaml). "
            "Без --profile подключается база (<hermes_home>/config.yaml)."
        ),
    )
    args = parser.parse_args(argv)
    if args.attach:
        return cmd_attach(args)
    if args.profile is not None:
        parser.error("--profile применяется только вместе с --attach")
    return cmd_setup(args)


if __name__ == "__main__":
    sys.exit(main())
