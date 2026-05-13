"""Observability — structured logging.

- ``logging`` — ``setup_logging(format, level)`` + ``JsonFormatter`` +
  ``log_event(logger, event, **fields)`` helper. Единая точка
  инициализации логирования для CLI и daemon; формат переключается
  через ``STYX_LOG_FORMAT=json|text`` (default text).

Healthz endpoint переехал в FastAPI routes
(``packages/styx-core/src/styx/http/routes/healthz.py``, Phase C).
"""

from __future__ import annotations

from styx.observability.logging import JsonFormatter, log_event, setup_logging

__all__ = [
    "JsonFormatter",
    "log_event",
    "setup_logging",
]
