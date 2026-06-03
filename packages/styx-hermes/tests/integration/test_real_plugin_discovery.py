"""Integration tests против реального Hermes runtime.

Эти тесты запускаются ВНУТРИ контейнера hermes-agent-styx-test
(см. docker/docker-compose.test.yml). Снаружи их запускать нельзя —
им нужен установленный Hermes на python path.

С bundled-модели (волна 33) styx-memory shim лежит в образе как
bundled-каталог `/opt/hermes/plugins/memory/styx-memory/` — он не
копируется скриптом в HERMES_HOME, его кладёт образ. Активация
профиля под Styx — отдельный явный шаг `styx-hermes-setup --attach`
(патчит config.yaml активного профиля). Чистая установка оставляет
Styx «сиротой»: без attach ни `plugins.enabled: [styx]`, ни
`memory.provider: styx-memory` в config нет, и general-плагин не
загружается (Hermes opt-in: `_get_enabled_plugins()` → None).

Discovery-тесты ниже сами дог-фудят attach на активном config
(через `styx_hermes.setup_cli._attach`) и восстанавливают исходный
config в teardown — config шарится между тестами в контейнере.

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration -v
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest


# Skip all tests if not running inside the Hermes container.
# Маркером служит наличие /opt/hermes — base image с Hermes.
pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


def _active_config_path() -> Path:
    """Путь к активному config.yaml — тем же способом, что Hermes.

    `PluginManager`/`_get_enabled_plugins` читают `load_config()`, а тот —
    `get_config_path()` = `get_hermes_home() / "config.yaml"` (в контейнере
    HERMES_HOME=/opt/data → /opt/data/config.yaml). Резолвим через тот же
    хелпер; fallback /opt/data/config.yaml если импорт недоступен.
    """
    try:
        from hermes_cli.config import get_config_path

        return Path(get_config_path())
    except Exception:
        return Path("/opt/data/config.yaml")


@contextmanager
def _attached_active_config():
    """Дог-фуд реального attach на активном config + restore в teardown.

    Сохраняем исходные байты до патча, в finally возвращаем их дословно и
    подчищаем созданный `_attach`'ем `config.yaml.bak.<ts>`. Тест не должен
    оставлять активный config привязанным к styx — он шарится между тестами
    в контейнере.
    """
    from styx_hermes.setup_cli import _attach

    config_path = _active_config_path()
    original = config_path.read_bytes()
    result = _attach(config_path)
    try:
        yield
    finally:
        config_path.write_bytes(original)
        if result.backup_path is not None and result.backup_path.exists():
            result.backup_path.unlink()


# -- Plugin discovery -----------------------------------------------------


def test_styx_memory_picked_up_by_memory_discovery() -> None:
    """Hermes memory discovery находит bundled styx-memory shim.

    Проверяет:
    - Эвристика `register_memory_provider` в shim __init__.py срабатывает
    - load_memory_provider("styx-memory") возвращает StyxMemoryCore
    - _ProviderCollector корректно собирает provider из register(ctx)
    """
    from plugins.memory import load_memory_provider
    from styx_hermes.providers.memory import StyxMemoryProvider

    provider = load_memory_provider("styx-memory")
    assert provider is not None, (
        "styx-memory shim не подхвачен memory discovery — проверь, что "
        "bundled-каталог /opt/hermes/plugins/memory/styx-memory/ есть в "
        "образе (volna 33: shim bundled, не копируется скриптом)"
    )
    assert isinstance(provider, StyxMemoryProvider)
    assert provider.name == "styx-memory"


def test_styx_general_picked_up_by_plugin_manager() -> None:
    """General PluginManager Hermes находит styx shim и загружает его.

    Не помечает kind="exclusive" (нет memory-маркеров в shim) →
    plugin реально загружается → register(ctx) вызван →
    StyxContextEngine зарегистрирован → StyxOpenAITransport+
    StyxCodexTransport перетёрли _REGISTRY.
    """
    from agent.transports import get_transport
    from hermes_cli.plugins import PluginManager
    from styx_hermes.engine.context import StyxContextEngine
    from styx_hermes.engine.transport import (
        StyxCodexTransport,
        StyxOpenAITransport,
    )

    # Hermes opt-in: без plugins.enabled:[styx] в активном config general
    # плагин помечается enabled=False и не грузится. Дог-фудим реальный
    # attach, restore в teardown.
    with _attached_active_config():
        pm = PluginManager()
        pm.discover_and_load(force=True)

        # ContextEngine зарегистрирован
        engine = pm._context_engine
        assert engine is not None, (
            "PluginManager не зарегистрировал ContextEngine — possibly styx "
            "shim не загрузился (проверь attach в config: plugins.enabled "
            "содержит 'styx', kind!=exclusive в plugin.yaml)"
        )
        assert isinstance(engine, StyxContextEngine)
        assert engine.name == "styx"

        # Transports перетёрли дефолты
        cc = get_transport("chat_completions")
        cr = get_transport("codex_responses")
        assert isinstance(cc, StyxOpenAITransport), (
            f"chat_completions всё ещё дефолтный: {type(cc).__name__}"
        )
        assert isinstance(cr, StyxCodexTransport), (
            f"codex_responses всё ещё дефолтный: {type(cr).__name__}"
        )


def test_styx_plugin_listed_as_loaded() -> None:
    """Styx-plugin реально загружен (enabled) в PluginManager после attach.

    Не тавтология «присутствует в _plugins»: после attach плагин должен
    быть enabled=True (Hermes opt-in грузит только из plugins.enabled),
    иначе LoadedPlugin остаётся enabled=False и register(ctx) не вызван.
    """
    from hermes_cli.plugins import PluginManager

    with _attached_active_config():
        pm = PluginManager()
        pm.discover_and_load(force=True)

        # Плагин должен быть в загруженных под ключом 'styx' (или '*/styx')
        plugin_keys = set(pm._plugins.keys())
        styx_keys = [p for p in plugin_keys if p == "styx" or p.endswith("/styx")]
        assert styx_keys, (
            f"styx plugin не виден в PluginManager._plugins: {sorted(plugin_keys)}"
        )

        # …и реально enabled (а не просто зарегистрирован как disabled)
        loaded = pm._plugins[styx_keys[0]]
        assert loaded.enabled is True, (
            f"styx plugin присутствует, но enabled={loaded.enabled} "
            f"(error={loaded.error!r}) — attach не включил plugins.enabled?"
        )


# -- Provider lifecycle через настоящий Hermes ABC ------------------------


def test_real_provider_lifecycle_writes_to_postgres() -> None:
    """Полный lifecycle MemoryProvider end-to-end на реальной БД.

    Через настоящий Hermes-style вызов: load_memory_provider →
    initialize → sync_turn → проверить запись в Postgres.
    """
    from plugins.memory import load_memory_provider

    provider = load_memory_provider("styx-memory")
    assert provider is not None

    session_id = str(uuid.uuid4())
    provider.initialize(
        session_id=session_id,
        agent_identity="integration-agent",
        hermes_home="/opt/data",
        platform="cli",
    )
    try:
        provider.sync_turn(
            user_content="ping",
            assistant_content="pong",
            session_id=session_id,
        )

        # После split provider — HTTP wrapper, доступа к queries нет.
        # Проверяем напрямую через Postgres что обе записи появились.
        import os
        import psycopg

        dsn = os.environ["STYX_TEST_DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM memories "
                    "WHERE agent_id = %s AND session_id = %s "
                    "ORDER BY seq",
                    ("integration-agent", uuid.UUID(session_id)),
                )
                rows = cur.fetchall()
        assert len(rows) == 2
        roles = {r[0] for r in rows}
        contents = {r[1] for r in rows}
        assert roles == {"user", "assistant"}
        assert contents == {"ping", "pong"}
    finally:
        provider.shutdown()


def test_initialize_sync_to_transport_global() -> None:
    """После initialize provider'а — transport знает agent_id.

    Инвариант: prompt_cache_key = agent_identity синхронизирован через
    ``styx_hermes._agent_session`` из MemoryProvider.initialize до первого
    Transport.build_kwargs.
    """
    from plugins.memory import load_memory_provider
    from styx_hermes import _agent_session
    from styx_hermes.engine.transport import StyxOpenAITransport

    _agent_session.clear_session()

    provider = load_memory_provider("styx-memory")
    sid = str(uuid.uuid4())
    provider.initialize(
        session_id=sid,
        agent_identity="cache-key-check",
        hermes_home="/opt/data",
        platform="cli",
    )
    try:
        tr = StyxOpenAITransport()
        kwargs = tr.build_kwargs(
            "gpt-x", [{"role": "user", "content": "x"}]
        )
        assert kwargs["prompt_cache_key"] == "cache-key-check"
    finally:
        provider.shutdown()
        _agent_session.clear_session()
