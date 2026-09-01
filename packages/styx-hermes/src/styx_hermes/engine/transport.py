"""Styx Hermes transports — наследники ChatCompletionsTransport / Responses.

Pure logic: build_kwargs локально (не HTTP вызов в core daemon — это hot
path, дублировать round-trip не нужно). Импортируют чистые helper'ы из
``styx.engine.transport`` (core): ``compute_prefix_digest``,
``log_prefix_slice``, ``styx_cache_key_override``.

agent_id для логического ``cache_scope_id`` берётся из ``_agent_session``
(set'ится в ``MemoryProvider.initialize``). Физический ``session_id`` при
этом остаётся у Hermes: он нужен для изоляции transcript и не должен
подменяться identity агента. Точный ``prompt_cache_key`` задаётся только
явным per-call/per-agent override; его bounding и provider-gates остаются
за upstream transport.
"""

from __future__ import annotations

import logging
from typing import Any

from styx_hermes import _agent_session, _hermes_path

_hermes_path.ensure_on_path()
from agent.transports.anthropic import AnthropicTransport  # noqa: E402
from agent.transports.chat_completions import ChatCompletionsTransport  # noqa: E402
from agent.transports.codex import ResponsesApiTransport  # noqa: E402

from styx.engine import transport as _core_transport  # noqa: E402
from styx.engine.transport import (  # noqa: E402
    log_prefix_slice as _log_prefix_slice_core,
    styx_cache_key_override as _override_core,
)

log = logging.getLogger(__name__)


def _configured_agent_cache_key(agent_id: str) -> str | None:
    """Вернуть именно configured key, не смешивая его с agent_id fallback.

    Выпущенный styx-core 1.0.10 не имеет отдельного public accessor для
    ``TransportState.prompt_cache_key``: его resolver намеренно возвращает
    ``agent_id`` и для fallback, и для exact override с тем же значением.
    Поэтому сначала читаем существующий state registry через узкий
    compatibility shim. Если конкретная версия core скрыла registry,
    сохраняем прежнюю совместимость для всех различимых override через
    публичный resolver — новая версия/API core не становится обязательной.
    """
    states = getattr(_core_transport, "_STATES", None)
    lock = getattr(_core_transport, "_LOCK", None)

    def _read_state() -> str | None:
        if not isinstance(states, dict):
            return None
        state = states.get(agent_id)
        value = getattr(state, "prompt_cache_key", None)
        return str(value) if value else None

    if lock is not None:
        try:
            with lock:
                configured = _read_state()
        except (AttributeError, TypeError):
            configured = _read_state()
    else:
        configured = _read_state()
    if configured is not None:
        return configured

    resolved = _override_core(agent_id, {})
    if resolved and resolved != agent_id:
        return str(resolved)
    return None


def _prepare_cache_params(params: dict[str, Any]) -> dict[str, Any]:
    """Подготовить cache-параметры для Hermes 0.21+.

    Default Styx identity — это логическая область кеша, а не готовый wire
    key. Hermes сам строит content-addressed/bounded ``prompt_cache_key`` и
    сохраняет физический ``session_id`` в transport headers. Явный Styx
    override остаётся точным key и передаётся штатным ``request_overrides``.
    """
    prepared = dict(params)
    session = _agent_session.get_session()
    agent_id = session[0] if session is not None else ""
    if agent_id:
        # Scope участвует в content addressing даже при exact override:
        # transport headers и прочая lineage-логика не должны откатываться
        # к физическому session_id только из-за явного wire key.
        prepared.setdefault("cache_scope_id", agent_id)

    request_overrides = prepared.get("request_overrides")
    if request_overrides is not None and not isinstance(request_overrides, dict):
        # Не маскируем невалидный upstream input своей нормализацией.
        return prepared
    overrides = dict(request_overrides or {})
    original_extra_body = overrides.get("extra_body")
    extra_body = (
        dict(original_extra_body)
        if isinstance(original_extra_body, dict)
        else original_extra_body
    )

    # Одна строгая цепочка precedence для всех transport pathways:
    # direct per-call > native top-level > native nested > per-agent.
    candidates = [
        params.get("prompt_cache_key"),
        overrides.get("prompt_cache_key"),
        extra_body.get("prompt_cache_key")
        if isinstance(extra_body, dict)
        else None,
        _configured_agent_cache_key(agent_id) if agent_id else None,
    ]
    exact_key = next((str(value) for value in candidates if value), None)

    # Direct Styx param — только input для resolution, не upstream param.
    prepared.pop("prompt_cache_key", None)

    # Сначала удаляем все конкурирующие wire locations, сохраняя остальные
    # request_overrides/extra_body поля. Затем кладём effective key ровно в
    # одно provider-native место.
    overrides.pop("prompt_cache_key", None)
    if isinstance(extra_body, dict):
        extra_body.pop("prompt_cache_key", None)
        if extra_body or isinstance(original_extra_body, dict):
            overrides["extra_body"] = extra_body

    if exact_key is not None:
        if params.get("is_xai_responses") is True:
            if not isinstance(extra_body, dict):
                extra_body = {}
            extra_body["prompt_cache_key"] = exact_key
            overrides["extra_body"] = extra_body
        else:
            overrides["prompt_cache_key"] = exact_key

    if request_overrides is not None or overrides:
        prepared["request_overrides"] = overrides
    return prepared


def _canonicalize_cache_key(
    api_kwargs: dict[str, Any], *, nested: bool
) -> str | None:
    """Оставить effective key в единственной wire location.

    Upstream уже применил bounding. Здесь только устраняем неоднозначность
    SDK merge: top-level имеет precedence над ``extra_body``; xAI Responses
    получает итог в ``extra_body``, остальные pathways — top-level.
    """
    extra_body_value = api_kwargs.get("extra_body")
    extra_body = (
        dict(extra_body_value) if isinstance(extra_body_value, dict) else None
    )
    top_key = api_kwargs.get("prompt_cache_key")
    nested_key = extra_body.get("prompt_cache_key") if extra_body else None
    effective = top_key or nested_key

    api_kwargs.pop("prompt_cache_key", None)
    if extra_body is not None:
        extra_body.pop("prompt_cache_key", None)

    if effective:
        effective = str(effective)
        if nested:
            if extra_body is None:
                extra_body = {}
            extra_body["prompt_cache_key"] = effective
        else:
            api_kwargs["prompt_cache_key"] = effective

    if extra_body is not None:
        if extra_body or isinstance(extra_body_value, dict):
            api_kwargs["extra_body"] = extra_body
        else:
            api_kwargs.pop("extra_body", None)
    return effective


def _wire_log(api_kwargs: dict[str, Any]) -> None:
    """sha256-digest payload головы (через core helper)."""
    session = _agent_session.get_session()
    agent_id = session[0] if session else ""
    _log_prefix_slice_core(agent_id, api_kwargs)


# ── chat_completions ──────────────────────────────────────────────────────


class StyxOpenAITransport(ChatCompletionsTransport):
    """Transport для OpenAI / OpenAI-compatible через chat_completions."""

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        cache_params = _prepare_cache_params(params)
        api_kwargs = super().build_kwargs(
            model, messages, tools=tools, **cache_params
        )
        _canonicalize_cache_key(api_kwargs, nested=False)

        _wire_log(api_kwargs)
        return api_kwargs


# ── codex_responses ───────────────────────────────────────────────────────


class StyxCodexTransport(ResponsesApiTransport):
    """Transport для OpenAI через ChatGPT Plus + Codex OAuth (Responses API)."""

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        cache_params = _prepare_cache_params(params)
        api_kwargs = super().build_kwargs(
            model, messages, tools=tools, **cache_params
        )
        effective_key = _canonicalize_cache_key(
            api_kwargs, nested=params.get("is_xai_responses") is True
        )
        if params.get("is_codex_backend") is True and effective_key:
            headers = api_kwargs.get("extra_headers")
            merged_headers = dict(headers) if isinstance(headers, dict) else {}
            merged_headers["x-client-request-id"] = effective_key
            api_kwargs["extra_headers"] = merged_headers

        _wire_log(api_kwargs)
        return api_kwargs


# ── anthropic_messages (волна 29 Phase E) ────────────────────────────────


class StyxAnthropicTransport(AnthropicTransport):
    """Transport для нативного Anthropic SDK (api_mode=anthropic_messages).

    Default ``AnthropicTransport`` Hermes уже умеет cache_control
    разметку через актуальный upstream planner (стабильная system-boundary,
    затем rolling tail с безопасным fallback). С волной 29 мы
    переиспользуем дефолт — он совместим с тем что Styx инжектит
    salient через ``MemoryProvider.prefetch()`` (system prompt
    addition), а не в messages.

    Override:
    - ``extract_cache_stats`` — после default extraction шлём stats
      в core daemon через POST /agent/cache_stats. Фоновый
      fire-and-forget HTTP call (тротлится только тем что Hermes
      вызывает stats после каждого turn'а — обычно 1-10/min).

    Future iteration: если cache hit rate <80% на длинных сессиях,
    потребуется явная manipulation cache_control marker placement
    (snять с rolling tail, поставить на стабильный prefix point) —
    это будет отдельная мини-волна по результатам production
    metrics.
    """

    def extract_cache_stats(self, response: Any) -> dict[str, int] | None:
        stats = super().extract_cache_stats(response)
        # Push в styx-core независимо от того, есть ли non-zero stats
        # — нулевые тоже информативны (cache miss every turn = бага).
        session = _agent_session.get_session()
        if session is not None:
            agent_id, client = session
            try:
                client.push_cache_stats(
                    agent_id,
                    cache_read_tokens=(stats or {}).get("cached_tokens", 0),
                    cache_creation_tokens=(stats or {}).get("creation_tokens", 0),
                )
            except Exception as exc:  # noqa: BLE001 — fail-open
                log.warning(
                    "styx-core /agent/cache_stats push failed: %s", exc
                )
        return stats


# ── регистрация ───────────────────────────────────────────────────────────


def register_with_hermes() -> None:
    """Зарегистрировать все три Styx-транспорта в Hermes _REGISTRY.

    Перетирает дефолтные ``ChatCompletionsTransport``,
    ``ResponsesApiTransport`` и ``AnthropicTransport``. Активный
    транспорт выбирается Hermes по api_mode провайдера агента —
    регистрируем все три pathway чтобы независимо от выбранного
    backend'а Styx всегда участвовал в transport-уровне.
    """
    from agent.transports import register_transport

    register_transport("chat_completions", StyxOpenAITransport)
    register_transport("codex_responses", StyxCodexTransport)
    register_transport("anthropic_messages", StyxAnthropicTransport)
