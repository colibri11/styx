"""Integration tests против реального Hermes runtime.

Эти тесты запускаются ВНУТРИ контейнера hermes-agent-styx-test
(см. docker/docker-compose.test.yml). Снаружи их запускать нельзя —
им нужен установленный Hermes на python path и shim-плагины,
скопированные styx-bootstrap.sh в HERMES_HOME/plugins/.

Запуск:
    docker compose -f docker/docker-compose.test.yml up -d
    docker compose -f docker/docker-compose.test.yml exec hermes-styx \\
        /opt/hermes/.venv/bin/pytest /opt/styx/tests/integration -v
"""

from __future__ import annotations

import os
import uuid

import pytest


# Skip all tests if not running inside the Hermes container.
# Маркером служит наличие /opt/hermes — base image с Hermes.
pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


# -- Plugin discovery -----------------------------------------------------


def test_styx_memory_picked_up_by_memory_discovery() -> None:
    """Hermes memory discovery находит styx-memory shim в HERMES_HOME/plugins/.

    Проверяет:
    - Эвристика `register_memory_provider` в shim __init__.py срабатывает
    - load_memory_provider("styx-memory") возвращает StyxMemoryCore
    - _ProviderCollector корректно собирает provider из register(ctx)
    """
    from plugins.memory import load_memory_provider
    from styx_hermes.providers.memory import StyxMemoryProvider

    provider = load_memory_provider("styx-memory")
    assert provider is not None, (
        "styx-memory shim не подхвачен memory discovery — проверь "
        "что styx-bootstrap.sh выполнил setup и shim есть в "
        "$HERMES_HOME/plugins/styx-memory/"
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

    pm = PluginManager()
    pm.discover_and_load(force=True)

    # ContextEngine зарегистрирован
    engine = pm._context_engine
    assert engine is not None, (
        "PluginManager не зарегистрировал ContextEngine — возможно styx "
        "shim не загрузился (проверь kind!=exclusive в plugin.yaml и "
        "отсутствие memory-маркеров в shim __init__.py)"
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
    """Styx-plugin виден в PluginManager._plugins после discovery."""
    from hermes_cli.plugins import PluginManager

    pm = PluginManager()
    pm.discover_and_load(force=True)

    # Плагин должен быть в загруженных
    plugin_keys = set(pm._plugins.keys())
    styx_loaded = any("styx" == p or p.endswith("/styx") for p in plugin_keys)
    assert styx_loaded, (
        f"styx plugin не виден в PluginManager._plugins: {sorted(plugin_keys)}"
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
