"""Opt-in LLM-wrap infrastructure для HTTP routes (волна 30, Phase B).

Default response каждого LLM-facing route остаётся «raw» —
программный shape с типизированным Pydantic-объектом, удобный для
программных caller'ов (CLI debugging, тесты, host-side провайдеры).
Plugin'ы (OpenClaw, Hermes), вызывающие эти routes ради передачи
результата в LLM, могут опт-ин'нуться в обёрнутое представление через
``?wrap_for_llm=1`` query-param ИЛИ header ``X-Wrap-For-LLM: 1``.
В этом случае response получает дополнительное поле ``llm_text`` —
preformatted строка с маркером таксономии волны 30, которую LLM
видит как «вот результат от styx_<channel>», а не как часть текущего
сообщения собеседника.

Канал = суффикс styx-тега (без префикса ``styx-``):
- ``salient`` — automatic recall block (используется в
  ``StyxComposer``, не в HTTP routes);
- ``recall`` — `/recall`, `styx_recall` tool;
- ``archive`` — `/search_archive`, `styx_search_archive` tool;
- ``dialogue`` — `/dialogue/*` (5 endpoints);
- ``relations`` — `/relations/{query,graph_traverse}`;
- ``explain`` — `/explain` (только LLM-выпуски, не observability);
- ``working-set`` — зарезервирован под future inject channel.

D4 wave-doc'а: Variant A (core wrap'ит, opt-in), а не Variant B
(plugin переводит). Variant A гарантирует одно место правды и
симметрию для всех hosts. Default raw — для не-LLM caller'ов.
"""

from __future__ import annotations

import json
from typing import Any, Final

from fastapi import Header, Query

from styx.engine.context import wrap_text_with_styx_tag

WRAP_CHANNELS: Final[frozenset[str]] = frozenset(
    {
        "salient",
        "recall",
        "archive",
        "dialogue",
        "relations",
        "explain",
        "working-set",
    }
)


_TRUE_HEADER_VALUES: Final[frozenset[str]] = frozenset(
    {"1", "true", "yes", "on"}
)


def should_wrap_for_llm(
    wrap_for_llm: int | None = Query(default=None, ge=0, le=1),
    x_wrap_for_llm: str | None = Header(default=None, alias="X-Wrap-For-LLM"),
) -> bool:
    """FastAPI dependency: вернуть True если caller хочет LLM-wrap.

    Поддерживает два эквивалентных способа:

    - ``?wrap_for_llm=1`` query-param (значения 0/1, integer);
    - ``X-Wrap-For-LLM: 1`` header (case-insensitive значения
      ``1`` / ``true`` / ``yes`` / ``on``).

    Хотя бы один из них установлен в truthy → return True. Plugin'ы
    обычно ставят header (один раз в HTTP клиенте); CLI/curl —
    query-param.
    """
    if wrap_for_llm is not None and wrap_for_llm == 1:
        return True
    if x_wrap_for_llm is not None:
        if x_wrap_for_llm.strip().lower() in _TRUE_HEADER_VALUES:
            return True
    return False


def wrap_for_llm(payload: Any, channel: str) -> str:
    """Сериализует payload как pretty-JSON и оборачивает в styx-тег.

    Tag — ``<styx-{channel}>...</styx-{channel}>``. Каналы — из
    ``WRAP_CHANNELS``; неизвестный — ``ValueError``.

    Payload может быть Pydantic BaseModel (вызовем ``.model_dump()``),
    dict, list или примитив. JSON serialize с ``ensure_ascii=False``
    (русский остаётся читаемым), ``indent=2`` (LLM лучше парсит
    structured), ``default=str`` для datetime/UUID/Decimal.
    """
    if channel not in WRAP_CHANNELS:
        raise ValueError(
            f"unknown wrap channel {channel!r}; "
            f"valid: {sorted(WRAP_CHANNELS)}"
        )
    if hasattr(payload, "model_dump"):
        # Pydantic BaseModel — берём serialize-ready dict (с alias'ами).
        # `exclude={"llm_text"}` — защита от self-reference: response model
        # содержит поле `llm_text`, которое мы как раз и формируем; без
        # exclude получим бесконечную вложенность (None в первый ход,
        # потом строка → re-wrap → ещё одна вложенность, и т.д.).
        payload = payload.model_dump(
            by_alias=True, mode="json", exclude={"llm_text"}
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return wrap_text_with_styx_tag(text, f"styx-{channel}")


def populate_llm_text(
    response: Any,
    channel: str,
    *,
    wrap: bool,
) -> Any:
    """Convenience helper для endpoint'ов: если ``wrap`` — заполняет
    ``response.llm_text`` обёрнутой строкой; иначе — no-op.

    Возвращает тот же объект (для chainability). Используется в каждом
    из 11 LLM-facing routes для устранения boilerplate (Phase C).
    Response model должен наследовать ``_LlmWrappableResponse``
    (поле ``llm_text``); если не наследует — AttributeError.
    """
    if wrap:
        response.llm_text = wrap_for_llm(response, channel)
    return response
