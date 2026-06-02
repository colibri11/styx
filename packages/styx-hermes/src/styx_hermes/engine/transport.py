"""Styx Hermes transports — наследники ChatCompletionsTransport / Responses.

Pure logic: build_kwargs локально (не HTTP вызов в core daemon — это hot
path, дублировать round-trip не нужно). Импортируют чистые helper'ы из
``styx.engine.transport`` (core): ``compute_prefix_digest``,
``log_prefix_slice``, ``styx_cache_key_override``.

agent_id для cache_key override берётся из ``_agent_session`` (set'ится в
``MemoryProvider.initialize``); если session нет — fallback на
``params.session_id`` или None (Hermes default).
"""

from __future__ import annotations

import logging
from typing import Any

from styx_hermes import _agent_session, _hermes_path

_hermes_path.ensure_on_path()
from agent.transports.anthropic import AnthropicTransport  # noqa: E402
from agent.transports.chat_completions import ChatCompletionsTransport  # noqa: E402
from agent.transports.codex import ResponsesApiTransport  # noqa: E402

from styx.engine.transport import (  # noqa: E402
    log_prefix_slice as _log_prefix_slice_core,
    styx_cache_key_override as _override_core,
)

log = logging.getLogger(__name__)


def _resolve_cache_key(params: dict[str, Any]) -> str | None:
    session = _agent_session.get_session()
    if session is None:
        # Без active session — fallback: per-call prompt_cache_key или
        # session_id, или None (Hermes default).
        per_call = params.get("prompt_cache_key")
        if per_call:
            return str(per_call)
        return None
    agent_id, _ = session
    return _override_core(agent_id, params)


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
        api_kwargs = super().build_kwargs(model, messages, tools=tools, **params)

        cache_key = _resolve_cache_key(params) or params.get("session_id")
        if cache_key:
            api_kwargs["prompt_cache_key"] = str(cache_key)

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
        api_kwargs = super().build_kwargs(model, messages, tools=tools, **params)

        override = _resolve_cache_key(params)
        # Гейтим установку prompt_cache_key по тому же условию, что Hermes
        # (agent/transports/codex.py:158): GitHub Models и xAI Responses
        # намеренно опускают cache-key (xAI получает его отдельно через
        # extra_body, GitHub Models opt-out из cache-key routing).
        cache_key_allowed = not params.get("is_github_responses") and not params.get(
            "is_xai_responses"
        )
        if override and cache_key_allowed:
            api_kwargs["prompt_cache_key"] = override
            if params.get("is_codex_backend"):
                existing = api_kwargs.get("extra_headers")
                merged: dict[str, str] = {}
                if isinstance(existing, dict):
                    merged.update(
                        {str(k): str(v) for k, v in existing.items() if v is not None}
                    )
                merged["session_id"] = override
                merged["x-client-request-id"] = override
                api_kwargs["extra_headers"] = merged

        _wire_log(api_kwargs)
        return api_kwargs


# ── anthropic_messages (волна 29 Phase E) ────────────────────────────────


class StyxAnthropicTransport(AnthropicTransport):
    """Transport для нативного Anthropic SDK (api_mode=anthropic_messages).

    Default ``AnthropicTransport`` Hermes уже умеет cache_control
    разметку через ``apply_anthropic_cache_control`` (rolling window
    последние 3 non-system messages + system block). С волной 29 мы
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
