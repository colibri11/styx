"""Unit-тесты attach-команды styx-hermes-setup (волна 33).

attach подключает любой Hermes-профиль (base или именованный) к Styx
идемпотентным патчем его config.yaml: memory.provider/plugins.enabled.
context.engine attach НЕ выставляет (компрессию всего окна ведёт сам
Hermes; Styx — memory-provider) и снимает legacy context.engine: styx
при ре-attach. Чистая установка оставляет Styx сиротой — attach это
явный opt-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from styx_hermes import setup_cli


def _write_config(path: Path, data: dict) -> bytes:
    """Записать config.yaml, вернуть записанные байты (для сравнения с бэкапом)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path.read_bytes()


def _backups(config_path: Path) -> list[Path]:
    return sorted(config_path.parent.glob(f"{config_path.name}.bak.*"))


# --- attach base ----------------------------------------------------------


def test_attach_base(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    config = home / "config.yaml"
    original = _write_config(config, {"model": {"name": "x"}})

    rc = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc == 0

    data = yaml.safe_load(config.read_text())
    assert data["memory"]["provider"] == "styx-memory"
    assert data["plugins"]["enabled"] == ["styx"]
    # context.engine attach НЕ выставляет — компрессию ведёт сам Hermes.
    assert "context" not in data
    # Существующие ключи сохранены.
    assert data["model"]["name"] == "x"

    # Бэкап создан и байт-в-байт равен исходнику.
    backups = _backups(config)
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


# --- attach named ---------------------------------------------------------


def test_attach_named(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    config = home / "profiles" / "agent-a" / "config.yaml"
    _write_config(config, {"model": {"name": "y"}})

    rc = setup_cli.main(["--attach", "--profile", "agent-a", "--hermes-home", str(home)])
    assert rc == 0

    data = yaml.safe_load(config.read_text())
    assert data["memory"]["provider"] == "styx-memory"
    assert data["plugins"]["enabled"] == ["styx"]
    assert "context" not in data
    assert data["model"]["name"] == "y"

    assert len(_backups(config)) == 1


# --- no config → English error + nonzero ---------------------------------


def test_attach_named_missing_config(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    expected_path = home / "profiles" / "missing" / "config.yaml"

    with pytest.raises(SystemExit) as exc:
        setup_cli.main(
            ["--attach", "--profile", "missing", "--hermes-home", str(home)]
        )
    # SystemExit с message-строкой → nonzero (truthy code).
    assert exc.value.code != 0
    msg = str(exc.value.code)
    assert "Config not found" in msg
    assert "missing" in msg
    assert str(expected_path) in msg
    # config.yaml НЕ создан.
    assert not expected_path.exists()


def test_attach_base_missing_config(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True)
    expected_path = home / "config.yaml"

    with pytest.raises(SystemExit) as exc:
        setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert exc.value.code != 0
    msg = str(exc.value.code)
    assert "Config not found" in msg
    assert str(expected_path) in msg
    assert not expected_path.exists()


# --- merge plugins.enabled (сохранить существующие, без дублей) -----------


def test_attach_merges_existing_plugins(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    config = home / "config.yaml"
    _write_config(config, {"plugins": {"enabled": ["foo"]}})

    rc = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc == 0

    data = yaml.safe_load(config.read_text())
    assert data["plugins"]["enabled"] == ["foo", "styx"]


def test_attach_no_duplicate_styx_in_plugins(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    config = home / "config.yaml"
    _write_config(config, {"plugins": {"enabled": ["foo", "styx"]}})

    # styx уже есть, memory отсутствует → патч ещё меняет (memory).
    rc = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc == 0
    data = yaml.safe_load(config.read_text())
    assert data["plugins"]["enabled"] == ["foo", "styx"]  # без дубля


# --- ре-attach снимает legacy context.engine: styx ------------------------


def test_attach_removes_legacy_context_engine(tmp_path: Path) -> None:
    """Конфиг, привязанный прежней версией (context.engine: styx), при
    ре-attach должен потерять engine — Hermes возвращается к своему
    штатному компрессору. context без других ключей убирается целиком."""
    home = tmp_path / "hermes"
    config = home / "config.yaml"
    _write_config(
        config,
        {
            "memory": {"provider": "styx-memory"},
            "plugins": {"enabled": ["styx"]},
            "context": {"engine": "styx"},
        },
    )

    rc = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc == 0

    data = yaml.safe_load(config.read_text())
    assert "context" not in data
    assert data["memory"]["provider"] == "styx-memory"
    assert data["plugins"]["enabled"] == ["styx"]


def test_attach_removes_engine_keeps_other_context_keys(tmp_path: Path) -> None:
    """Снимается только engine: styx; прочие ключи context — сохраняются."""
    home = tmp_path / "hermes"
    config = home / "config.yaml"
    _write_config(
        config,
        {"context": {"engine": "styx", "max_tokens": 4096}},
    )

    rc = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc == 0

    data = yaml.safe_load(config.read_text())
    assert "engine" not in data["context"]
    assert data["context"]["max_tokens"] == 4096


# --- идемпотентный повтор: no change, no new backup, exit 0 ---------------


def test_attach_idempotent_rerun(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    home = tmp_path / "hermes"
    config = home / "config.yaml"
    _write_config(config, {"model": {"name": "x"}})

    rc1 = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc1 == 0
    capsys.readouterr()
    backups_after_first = _backups(config)
    assert len(backups_after_first) == 1
    bytes_after_first = config.read_bytes()

    # Второй прогон — уже подключён → no-op.
    rc2 = setup_cli.main(["--attach", "--hermes-home", str(home)])
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "already attached" in out

    # НОВОГО бэкапа не появилось, файл не изменился.
    assert _backups(config) == backups_after_first
    assert config.read_bytes() == bytes_after_first


# --- bad profile name → error, nonzero -----------------------------------


@pytest.mark.parametrize("bad", ["Bad Name", "default", "UPPER", "with/slash", ""])
def test_attach_bad_profile_name(tmp_path: Path, bad: str) -> None:
    home = tmp_path / "hermes"
    with pytest.raises(SystemExit) as exc:
        setup_cli.main(["--attach", "--profile", bad, "--hermes-home", str(home)])
    assert exc.value.code != 0
    msg = str(exc.value.code)
    assert "Invalid --profile" in msg


# --- --profile без --attach → ошибка argparse ----------------------------


def test_profile_requires_attach(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    with pytest.raises(SystemExit) as exc:
        setup_cli.main(["--profile", "agent-a", "--hermes-home", str(home)])
    # argparse error → exit code 2.
    assert exc.value.code != 0


# --- pure-функция патча --------------------------------------------------


def test_compute_patched_idempotent() -> None:
    base = {"model": {"name": "x"}}
    patched, changed = setup_cli._compute_patched(base)
    assert changed is True
    # Исходник не мутирован.
    assert base == {"model": {"name": "x"}}

    patched2, changed2 = setup_cli._compute_patched(patched)
    assert changed2 is False
    assert patched2 == patched


def test_compute_patched_non_list_enabled() -> None:
    # plugins.enabled не-list → пересоздаётся как [styx].
    base = {"plugins": {"enabled": "oops"}}
    patched, changed = setup_cli._compute_patched(base)
    assert changed is True
    assert patched["plugins"]["enabled"] == ["styx"]
