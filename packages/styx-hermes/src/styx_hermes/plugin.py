"""Hermes general-plugin entry для Styx.

Styx в Hermes — memory-provider + transport + pre_llm hook. Память
подмешивается каналами provider'а (``prefetch`` / ``system_prompt_block``
per-turn + tools + ``on_pre_compress``), а компрессию всего окна ведёт
сам Hermes своим штатным компрессором — Styx context engine'ом НЕ
подменяется. Этот entry-point регистрирует только Transport (через
``register_with_hermes``) и pre_llm hook. Долгая память регистрируется
отдельным shim'ом (см. ``styx_hermes.memory_plugin``); установка memory
shim'а в ``$HERMES_HOME/plugins/styx-memory/`` — ``styx-hermes-setup`` CLI.

ВАЖНО: исходник этого модуля НЕ должен упоминать имя метода для
регистрации memory provider'а — общий PluginManager Hermes эвристически
ищет такую строку в плагин ``__init__.py`` и помечает плагин
``kind="exclusive"``, после чего отказывается его грузить (см.
``hermes_cli/plugins.py:1549-1571`` (маркер ``:1561``, discovery-skip
``:1355``), Hermes v0.18.0/v2026.7.1 — было ``:1314-1336`` на
v0.17.0/v2026.6.19).
"""

from __future__ import annotations

import logging

from styx_hermes import _hermes_path

_hermes_path.ensure_on_path()

from styx_hermes.engine.pre_llm_hook import on_pre_llm_call  # noqa: E402
from styx_hermes.engine.transport import register_with_hermes  # noqa: E402

log = logging.getLogger(__name__)


def register(ctx) -> None:
    """Hermes plugin entry-point — transport и pre_llm hook.

    Context engine НЕ регистрируется: компрессию всего окна ведёт сам
    Hermes, Styx подмешивает только память (provider-каналами).
    """
    register_with_hermes()  # подменяет ChatCompletionsTransport / Responses в _REGISTRY
    log.info(
        "StyxOpenAITransport+StyxCodexTransport зарегистрированы "
        "(api_mode='chat_completions'+'codex_responses')"
    )

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    log.info("Styx pre_llm_call hook зарегистрирован")
