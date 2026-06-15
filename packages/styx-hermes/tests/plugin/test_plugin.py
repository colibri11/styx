"""Юнит-тесты styx_hermes.plugin / styx_hermes.memory_plugin / styx.cli setup."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

import pytest

from styx_hermes import memory_plugin, plugin
from styx_hermes.setup_cli import cmd_setup, main as cli_main
from styx_hermes.engine import transport as transport_mod
from styx_hermes.providers.memory import StyxMemoryProvider


# -- ctx-фейки ------------------------------------------------------------


class FakePluginContext:
    """Имитирует hermes_cli.plugins.PluginContext (только использованные методы)."""

    def __init__(self) -> None:
        self.context_engine = None
        self.tools: list[dict] = []
        self.hooks: list[tuple[str, object]] = []

    def register_context_engine(self, engine) -> None:
        if self.context_engine is not None:
            raise RuntimeError("уже зарегистрирован")
        self.context_engine = engine

    def register_tool(self, *a, **kw) -> None:
        self.tools.append({"args": a, "kwargs": kw})

    def register_hook(self, name, callback) -> None:
        self.hooks.append((name, callback))


class FakeMemoryCollector:
    """Имитирует _ProviderCollector из plugins/memory/__init__.py."""

    def __init__(self) -> None:
        self.provider = None

    def register_memory_provider(self, provider) -> None:
        self.provider = provider


# -- styx_hermes.plugin (general) -----------------------------------------------


@pytest.fixture(autouse=True)
def _restore_transport_registry():
    """Восстанавливает дефолтные транспорты + сбрасывает core state."""
    yield
    from agent.transports import register_transport
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport
    register_transport("chat_completions", ChatCompletionsTransport)
    register_transport("codex_responses", ResponsesApiTransport)
    from styx.engine import transport as core_transport
    core_transport.reset_all()
    from styx_hermes import _agent_session
    _agent_session.clear_session()


def test_plugin_register_transport_and_hook_no_engine() -> None:
    from agent.transports import get_transport

    ctx = FakePluginContext()
    plugin.register(ctx)

    # Styx в Hermes — memory-provider, НЕ context engine: компрессию всего
    # окна ведёт сам Hermes. Плагин не должен подменять собой компрессор.
    assert ctx.context_engine is None

    inst = get_transport("chat_completions")
    assert isinstance(inst, transport_mod.StyxOpenAITransport)

    # Волна 15: pre_llm_call hook зарегистрирован.
    hook_names = {name for name, _cb in ctx.hooks}
    assert "pre_llm_call" in hook_names


def test_plugin_register_does_not_mention_register_memory_provider() -> None:
    """Эвристика Hermes ищет строку 'register_memory_provider' в init.py.

    styx-shim делегирует в styx_hermes.plugin — его исходник НЕ должен содержать
    эту строку, иначе general PluginManager пометит styx-плагин exclusive.
    """
    import styx_hermes.plugin as mod

    src = Path(mod.__file__).read_text("utf-8")
    assert "register_memory_provider" not in src
    assert "MemoryProvider" not in src


# -- styx_hermes.memory_plugin ---------------------------------------------------


def test_memory_plugin_registers_provider() -> None:
    collector = FakeMemoryCollector()
    memory_plugin.register(collector)
    assert isinstance(collector.provider, StyxMemoryProvider)


def test_styx_memory_shim_mentions_memory_marker() -> None:
    """Эвристика Hermes должна найти маркер для маршрутизации в memory discovery."""
    from importlib import resources

    src = (resources.files("styx_hermes.resources.plugins")
           / "styx-memory" / "__init__.py").read_text(encoding="utf-8")
    # shim импортирует styx_hermes.memory_plugin → строка попадает в источник
    assert "memory_plugin" in src

    # А сам styx_hermes.memory_plugin содержит маркер для эвристики
    import styx_hermes.memory_plugin as mp_mod
    mp_src = Path(mp_mod.__file__).read_text("utf-8")
    assert "register_memory_provider" in mp_src


# -- transport.configure синхронизация из MemoryProvider.initialize ------


def test_initialize_configures_transport_agent_id(
    monkeypatch: pytest.MonkeyPatch, migrated_db: str
) -> None:
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)

    p = StyxMemoryProvider()
    sid = str(uuid.uuid4())
    try:
        p.initialize(session_id=sid, agent_identity="alpha-agent")
        tr = transport_mod.StyxOpenAITransport()
        kwargs = tr.build_kwargs("gpt-x", [{"role": "user", "content": "hi"}])
        assert kwargs["prompt_cache_key"] == "alpha-agent"
    finally:
        p.shutdown()


# -- CLI setup ------------------------------------------------------------


def test_cli_setup_writes_memory_shim(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """`hermes-styx setup` копирует только styx-memory shim.

    General plugin (`styx`) подхватывается через entry-point
    `hermes_agent.plugins`, его shim в `$HERMES_HOME/plugins/styx/` не
    нужен — иначе double-registration.
    """
    rc = cli_main(["--hermes-home", str(tmp_path)])
    assert rc == 0

    plugins_root = tmp_path / "plugins"
    assert (plugins_root / "styx-memory" / "__init__.py").exists()
    assert (plugins_root / "styx-memory" / "plugin.yaml").exists()
    # General plugin shim НЕ должен копироваться.
    assert not (plugins_root / "styx" / "__init__.py").exists()

    out = capsys.readouterr().out
    assert "installed: styx-memory" in out
    assert "memory.provider: styx-memory" in out


def test_cli_setup_skips_existing_without_force(tmp_path: Path) -> None:
    cli_main(["--hermes-home", str(tmp_path)])
    # повторный — без --force не должен трогать
    rc = cli_main(["--hermes-home", str(tmp_path)])
    assert rc == 0


def test_cli_setup_force_overwrites(tmp_path: Path) -> None:
    cli_main(["--hermes-home", str(tmp_path)])
    target = tmp_path / "plugins" / "styx-memory" / "__init__.py"
    target.write_text("# corrupted\n", encoding="utf-8")
    cli_main(["--hermes-home", str(tmp_path), "--force"])
    assert "from styx_hermes.memory_plugin import register" in target.read_text(
        encoding="utf-8"
    )


def test_cli_setup_uses_env_hermes_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cli_main([])
    assert (tmp_path / "plugins" / "styx-memory" / "__init__.py").exists()
