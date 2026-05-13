"""Тесты structured logging (волна 16)."""

from __future__ import annotations

import io
import json
import logging

import pytest

from styx.observability.logging import (
    JsonFormatter,
    log_event,
    setup_logging,
)


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def _make_record(
    *,
    level: int = logging.INFO,
    msg: str = "hello",
    name: str = "styx.test",
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


# ── JsonFormatter ────────────────────────────────────────────────────────


def test_json_formatter_basic_record_no_event():
    """Plain log call — event="log", msg=отрендеренное сообщение."""
    record = _make_record(msg="hello world")
    payload = _format(record)
    assert payload["event"] == "log"
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "styx.test"
    assert "ts" in payload
    assert payload["ts"].endswith("Z")


def test_json_formatter_event_record_extra_fields_top_level():
    """log_event-style record — event как top-level, extra-поля рядом."""
    record = _make_record(
        msg="compress evicted=4",
        extra={"event": "compress", "agent_id": "a-1", "evicted": 4},
    )
    payload = _format(record)
    assert payload["event"] == "compress"
    assert payload["agent_id"] == "a-1"
    assert payload["evicted"] == 4
    # msg не дублируется в JSON-режиме когда event есть
    assert "msg" not in payload


def test_json_formatter_handles_complex_values():
    """Несериализуемые типы пропускаются через _sanitize."""
    record = _make_record(
        msg="x",
        extra={"event": "x", "obj": object(), "lst": [1, 2, 3]},
    )
    payload = _format(record)
    assert payload["event"] == "x"
    assert isinstance(payload["obj"], str)
    assert payload["lst"] == [1, 2, 3]


def test_json_formatter_renders_exception_info():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        record = logging.LogRecord(
            name="styx.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="failure",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = _format(record)
    assert payload["level"] == "ERROR"
    assert "exc" in payload
    assert "RuntimeError" in payload["exc"]


def test_json_formatter_unicode_no_escape():
    record = _make_record(msg="привет")
    payload_str = JsonFormatter().format(record)
    assert "привет" in payload_str  # ensure_ascii=False


# ── setup_logging ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_logging():
    """Изолируем тесты — удаляем только managed handler'ы между ними."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_styx_managed", False):
            root.removeHandler(h)


def _managed_handlers():
    root = logging.getLogger()
    return [h for h in root.handlers if getattr(h, "_styx_managed", False)]


def test_setup_logging_text_format():
    setup_logging(format="text", level="INFO")
    handlers = _managed_handlers()
    assert len(handlers) == 1
    assert not isinstance(handlers[0].formatter, JsonFormatter)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_json_format():
    setup_logging(format="json", level="DEBUG")
    handlers = _managed_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0].formatter, JsonFormatter)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_idempotent():
    """Повторные вызовы — один managed handler в итоге."""
    setup_logging(format="text")
    setup_logging(format="json")
    setup_logging(format="text")
    assert len(_managed_handlers()) == 1


def test_setup_logging_preserves_external_handlers():
    """Foreign handler (например pytest caplog) не трогается setup_logging."""
    foreign = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        setup_logging(format="json")
        assert foreign in root.handlers
        assert len(_managed_handlers()) == 1
    finally:
        root.removeHandler(foreign)


def test_setup_logging_unknown_format_falls_back_to_text():
    setup_logging(format="bogus")
    handler = _managed_handlers()[0]
    assert not isinstance(handler.formatter, JsonFormatter)


def test_setup_logging_invalid_level_falls_back_to_info():
    setup_logging(format="text", level="NOTALEVEL")
    assert logging.getLogger().level == logging.INFO


# ── log_event ────────────────────────────────────────────────────────────


def test_log_event_emits_text_message_with_fields():
    setup_logging(format="text", level="INFO")
    buf = io.StringIO()
    text_handler = logging.StreamHandler(buf)
    text_handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("styx.test.event")
    logger.addHandler(text_handler)
    logger.setLevel(logging.INFO)
    try:
        log_event(logger, "compress", evicted=4, agent_id="a-1")
    finally:
        logger.removeHandler(text_handler)
    line = buf.getvalue().strip()
    assert "compress" in line
    assert "evicted=4" in line
    assert "agent_id=a-1" in line


def test_log_event_emits_json_with_top_level_fields():
    setup_logging(format="json", level="INFO")
    buf = io.StringIO()
    json_handler = logging.StreamHandler(buf)
    json_handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("styx.test.event_json")
    logger.addHandler(json_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        log_event(logger, "recall", results_count=5, elapsed_ms=12)
    finally:
        logger.removeHandler(json_handler)
        logger.propagate = True
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "recall"
    assert payload["results_count"] == 5
    assert payload["elapsed_ms"] == 12
    assert payload["logger"] == "styx.test.event_json"


def test_log_event_no_fields():
    """Без extra-полей — event как msg."""
    logger = logging.getLogger("styx.test.event_empty")
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        log_event(logger, "ping")
    finally:
        logger.removeHandler(h)
        logger.propagate = True
    payload = json.loads(buf.getvalue().strip())
    assert payload["event"] == "ping"
