"""Structured logging для Styx (волна 16).

Два формата:

- ``text`` (default) — стандартный человекочитаемый формат
  ``%(asctime)s %(levelname)s %(name)s %(message)s``. Для dev / host
  тестов — читать глазами привычнее.
- ``json`` — JSON-line на строку: ``{"ts", "level", "logger", "event",
  ...fields}``. Для production deployment в Hermes; парсится внешним
  сборщиком (vector / fluentbit) без regex'ов.

Активация через ENV ``STYX_LOG_FORMAT`` либо параметр
``setup_logging(format=...)``.

``log_event(logger, event, **fields)`` — единая точка вызова для
critical-path событий (compress / recall / sweep_cycle / drift_detected
/ working_set_save / ollama_call). В json-режиме поля попадают в
JSON-line как top-level keys; в text-режиме — как часть сообщения.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


# Стандартные атрибуты LogRecord — их не выводим как extra-поля.
_RECORD_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",  # py3.12+
    }
)


class JsonFormatter(logging.Formatter):
    """JSON-line formatter.

    Каждый LogRecord → одна строка JSON. Если у record'а есть атрибут
    ``event`` (через ``extra={"event": ..., ...}``) — он попадает в
    ключ ``event``, остальные extra-поля — top-level. Если ``event``
    нет — пишется ``"event": "log"`` + ``"msg"`` с отрендеренным
    сообщением.

    Exception info (``logger.exception(...)`` или ``exc_info=True``)
    рендерится в поле ``exc``.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
        }

        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = str(event)
        else:
            payload["event"] = "log"
            payload["msg"] = record.getMessage()

        # Extra-поля — всё что не стандартный атрибут LogRecord и не уже
        # записано в payload.
        for key, value in record.__dict__.items():
            if key in _RECORD_RESERVED or key in payload or key == "event":
                continue
            if key.startswith("_"):
                continue
            payload[key] = _sanitize(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        elif record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, ensure_ascii=False, default=str)


def _sanitize(value: Any) -> Any:
    """Сделать значение JSON-serializable. Простые типы — пропускаем,
    остальное — через str().
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    return str(value)


_MANAGED_ATTR = "_styx_managed"


def setup_logging(*, format: str = "text", level: str = "INFO") -> None:
    """Идемпотентная инициализация root logger'а.

    Удаляет только собственные handler'ы (с атрибутом ``_styx_managed``)
    и добавляет новый StreamHandler с выбранным formatter'ом. Чужие
    handler'ы (pytest caplog, другие frameworks) оставляются —
    setup_logging не должен ломать тестовое перехватчики.
    """
    fmt_normalized = (format or "text").lower().strip()
    level_normalized = (level or "INFO").upper().strip()

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED_ATTR, False):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass

    handler = logging.StreamHandler(stream=sys.stderr)
    setattr(handler, _MANAGED_ATTR, True)
    if fmt_normalized == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
    root.addHandler(handler)

    try:
        root.setLevel(level_normalized)
    except (TypeError, ValueError):
        root.setLevel(logging.INFO)


def _format_value(value: Any) -> str:
    """Render для text-режима лога — компактнее чем repr."""
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "None"
    if isinstance(value, str):
        return value if " " not in value else f'"{value}"'
    return str(value)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Записать structured-event в logger.

    В JSON-режиме — JSON-line с top-level полями. В text-режиме —
    обычное info-сообщение с event как msg и fields как часть текста.

    Используется на critical-path точках (compress, recall, sweep_cycle,
    drift_detected, working_set_save, ollama_call).
    """
    if not fields:
        logger.info(event, extra={"event": event})
        return
    extra: dict[str, Any] = {"event": event}
    extra.update(fields)
    text_msg = event + " " + " ".join(
        f"{k}={_format_value(v)}" for k, v in fields.items()
    )
    logger.info(text_msg, extra=extra)
