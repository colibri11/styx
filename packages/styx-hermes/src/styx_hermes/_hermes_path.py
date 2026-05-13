"""Подключение Hermes к sys.path для импорта Hermes ABC.

Styx — плагин Hermes Agent. В проде Hermes уже установлен в окружение
(``pip install hermes-agent`` или drop-in внутри Hermes), и
``from agent.memory_provider import MemoryProvider`` работает без
дополнительной настройки.

В dev/tests Hermes может быть склонирован отдельно. Тогда переменная
``HERMES_PATH`` добавляется в ``sys.path``. Идемпотентно.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def ensure_on_path() -> Path | None:
    """Добавляет Hermes-checkout в sys.path если задан и ещё не там.

    Смотрит ТОЛЬКО переменную окружения ``HERMES_PATH``.
    Если не задана — возвращает None (Hermes должен быть доступен через pip).
    Если задана, но путь невалиден — возвращает None + log.warning.
    Не падает: если Hermes уже доступен (pip install) — просто no-op.
    """
    raw = os.environ.get("HERMES_PATH")
    if not raw:
        return None

    candidate = Path(raw)

    if not candidate.exists():
        log.warning(
            "HERMES_PATH is set but path does not exist: %s", candidate
        )
        return None
    if not (candidate / "agent" / "memory_provider.py").exists():
        log.warning(
            "HERMES_PATH points to a directory without agent/memory_provider.py: %s",
            candidate,
        )
        return None

    candidate_str = str(candidate)
    if candidate_str in sys.path:
        return candidate
    sys.path.insert(0, candidate_str)
    return candidate
