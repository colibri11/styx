"""Hermes memory-discovery shim для Styx.

Этот файл копируется CLI-командой ``styx-hermes-setup`` в
``$HERMES_HOME/plugins/styx-memory/__init__.py``. Hermes находит его
через эвристику ``register_memory_provider`` в исходнике и направляет
в memory discovery, минуя общий PluginManager.
"""

from styx_hermes.memory_plugin import register  # noqa: F401
