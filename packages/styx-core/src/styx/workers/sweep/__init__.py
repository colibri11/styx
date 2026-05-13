"""Consolidation sweep — periodic SQL-задачи в worker-runtime.

В волне 7b — единственная задача ``lifecycle_refresh`` (fresh → settled
→ dormant). Архитектура расширяема под будущие задачи:
- ``expired_cleanup`` (если когда-нибудь введём TTL на memories).
- ``dead_prune`` (после волны 9 reinterpret).
- ``relation_decay`` (после первого писателя в `relations`).
- ``reinterpret_apply`` / ``consolidation_apply`` (волны 9 / 10).
"""
