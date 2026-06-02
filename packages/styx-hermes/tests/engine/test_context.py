"""Regression-покрытие compat-фиксов StyxContextEngine (host=Hermes v0.15.2).

Защищает три HARD-BREAK поверхности, обнаруженные при сверке с
v0.15.2 conversation_loop / agent_init / conversation_compression:

1. ``update_model(..., api_mode=...)`` — host зовёт БЕЗ try/except
   (agent_init.py:1441, run_agent.py:673). Если сигнатура не примет
   ``api_mode`` — TypeError рвёт turn.
2. Будущие неизвестные kwargs host'а — поглощаются ``**kwargs`` без
   разлома сигнатуры в minor-bump.
3. ``compress(..., focus_topic=..., force=...)`` — host
   (conversation_compression.py:316) при TypeError ретраит БЕЗ
   focus_topic (deflate fallback, focus_topic теряется). ``**kwargs``
   поглощает ``force`` и держит реальное тело на основном пути.

Adapter-класс — из ``styx_hermes.engine.context`` (НЕ core
``styx.engine.context``). Эталон стиля — ``test_transport.py``.
"""

from __future__ import annotations

import pytest

from styx_hermes import _agent_session
from styx_hermes.engine.context import StyxContextEngine


@pytest.fixture(autouse=True)
def _reset_session():
    """Очистить per-process session до/после теста.

    compress() читает ``_agent_session.get_session()``; без активной
    session фикс должен давать pass-through. Чистим с обеих сторон, чтобы
    ни предыдущий тест, ни этот не оставили session для соседей.
    """
    _agent_session.clear_session()
    yield
    _agent_session.clear_session()


# -- #1 HARD-BREAK: update_model(api_mode=...) -----------------------------


def test_update_model_accepts_api_mode_keyword() -> None:
    """api_mode как KEYWORD не даёт TypeError и обновляет state."""
    e = StyxContextEngine(context_length=1000)
    e.update_model(
        model="m",
        context_length=10000,
        base_url="",
        api_key="",
        provider="",
        api_mode="chat_completions",
    )
    assert e.context_length == 10000
    assert e.threshold_tokens == int(10000 * e.threshold_percent)


# -- **kwargs поглощает будущие unknown host-kwargs ------------------------


def test_update_model_absorbs_future_kwargs() -> None:
    """Неизвестный kwarg не рвёт сигнатуру; threshold пересчитан."""
    e = StyxContextEngine(context_length=1000)
    e.update_model("m", 2000, future_kwarg=1)
    assert e.context_length == 2000
    assert e.threshold_tokens == int(2000 * e.threshold_percent)


# -- #3 compress с host-добавленными kwargs (focus_topic + force) ----------


def test_compress_reaches_body_with_host_kwargs() -> None:
    """compress(..., focus_topic=, force=) не TypeError'ит.

    Без активной session — pass-through (len неизменен). Это и есть
    основное тело: host НЕ должен ловить TypeError и уходить в deflate
    fallback без focus_topic.
    """
    e = StyxContextEngine(context_length=1000)
    msgs = [{"role": "user", "content": "x"}]
    out = e.compress(msgs, current_tokens=10, focus_topic="auth", force=True)
    assert len(out) == 1


# -- дешёвые ассерты на should_compress / on_session_reset -----------------


def test_should_compress_always_true() -> None:
    """Styx владеет каждым turn'ом — should_compress всегда True."""
    e = StyxContextEngine(context_length=1000)
    assert e.should_compress(prompt_tokens=999999) is True


def test_on_session_reset_zeroes_counters() -> None:
    """on_session_reset() без аргументов обнуляет token-счётчики."""
    e = StyxContextEngine(context_length=1000)
    e.last_total_tokens = 123
    e.compression_count = 7
    e.on_session_reset()
    assert e.last_total_tokens == 0
    assert e.compression_count == 0
