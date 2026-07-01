"""Hermes memory-discovery entry для Styx.

Hermes ищет memory provider'ов в ``$HERMES_HOME/plugins/<name>/`` через
эвристику: если в ``__init__.py`` плагина встречается строка
``register_memory_provider`` или ``MemoryProvider``, плагин считается
memory provider'ом и попадает в memory discovery (см.
``hermes_cli/plugins.py:1314-1336`` (Hermes v0.17.0/v2026.6.19 — было
``:781-803`` на v0.16.0) и ``plugins/memory/__init__.py``).

memory discovery вызывает ``register(ctx)`` плагина, передавая
``_ProviderCollector`` — упрощённый ctx с единственным методом
``register_memory_provider``. Остальные регистрации (transport,
pre_llm hook) в этой ветке discovery не работают — для них существует
``styx_hermes.plugin``.
"""

from __future__ import annotations

from styx_hermes.providers.memory import StyxMemoryProvider


def register(ctx) -> None:
    """Memory discovery entry-point.

    ``ctx`` здесь — ``_ProviderCollector`` из
    ``plugins/memory/__init__.py``.
    """
    ctx.register_memory_provider(StyxMemoryProvider())
