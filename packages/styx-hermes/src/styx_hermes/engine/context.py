"""StyxContextEngine — Hermes ContextEngine HTTP wrapper.

Тонкий adapter поверх ``POST /context/build`` styx-core daemon.
Локально держит только token-counter state (``last_*_tokens``,
``compression_count``, ``threshold_tokens``) — эти поля Hermes
``run_agent`` читает напрямую.

agent_id discovery: ``StyxMemoryProvider.initialize`` set'ит session в
``styx_hermes._agent_session``; ``compress`` читает её при первом вызове.
Если session не set — fallback на pass-through (compress возвращает
``messages`` без изменений).
"""

from __future__ import annotations

import logging
from typing import Any

from styx_hermes import _agent_session, _hermes_path

_hermes_path.ensure_on_path()
from agent.context_engine import ContextEngine  # noqa: E402

log = logging.getLogger(__name__)


class StyxContextEngine(ContextEngine):
    """Hermes ContextEngine — HTTP wrapper над styx-core daemon."""

    def __init__(
        self,
        *,
        context_length: int = 0,
        threshold_percent: float = 0.75,
    ) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.threshold_tokens = (
            int(context_length * threshold_percent) if context_length else 0
        )

    @property
    def name(self) -> str:
        return "styx"

    def update_from_response(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(
            usage.get("total_tokens")
            or self.last_prompt_tokens + self.last_completion_tokens
        )

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        # Styx хочет владеть каждым turn'ом — compress() пропускает no-op
        # внутри себя.
        return True

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        session = _agent_session.get_session()
        if session is None:
            log.debug("compress: no active session — pass-through")
            return list(messages)

        agent_id, client = session
        try:
            resp = client.build_context(
                agent_id,
                list(messages),
                current_tokens=current_tokens,
                focus_topic=focus_topic,
            )
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.warning("styx-core /context/build failed: %s — pass-through", exc)
            return list(messages)

        out = resp.get("messages")
        if not isinstance(out, list):
            log.warning("styx-core /context/build returned no 'messages' — pass-through")
            return list(messages)
        # compression_count приходит из core — синхронизируем локальный
        # счётчик чтобы Hermes-логи показывали актуальное значение.
        self.compression_count = int(resp.get("compression_count", self.compression_count))
        return out

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        del model, base_url, api_key, provider
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    def on_session_reset(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
