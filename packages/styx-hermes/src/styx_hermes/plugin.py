"""Hermes general-plugin entry для Styx.

Регистрирует ContextEngine и Transport через PluginContext. Долгая
память регистрируется отдельным shim'ом (см. ``styx_hermes.memory_plugin``).
Установка memory shim'а в ``$HERMES_HOME/plugins/styx-memory/`` —
``styx-hermes-setup`` CLI.

ВАЖНО: исходник этого модуля НЕ должен упоминать имя метода для
регистрации memory provider'а — общий PluginManager Hermes эвристически
ищет такую строку в плагин ``__init__.py`` и помечает плагин
``kind="exclusive"``, после чего отказывается его грузить (см.
``hermes_cli/plugins.py:781-803``).
"""

from __future__ import annotations

import logging

from styx_hermes import _hermes_path

_hermes_path.ensure_on_path()

from styx_hermes.engine.context import StyxContextEngine  # noqa: E402
from styx_hermes.engine.pre_llm_hook import on_pre_llm_call  # noqa: E402
from styx_hermes.engine.transport import register_with_hermes  # noqa: E402

log = logging.getLogger(__name__)


def register(ctx) -> None:
    """Hermes plugin entry-point — engine, transport и pre_llm hook."""
    register_with_hermes()  # подменяет ChatCompletionsTransport / Responses в _REGISTRY
    log.info(
        "StyxOpenAITransport+StyxCodexTransport зарегистрированы "
        "(api_mode='chat_completions'+'codex_responses')"
    )

    ctx.register_context_engine(StyxContextEngine())
    log.info("StyxContextEngine зарегистрирован как context engine")

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    log.info("Styx pre_llm_call hook зарегистрирован")
