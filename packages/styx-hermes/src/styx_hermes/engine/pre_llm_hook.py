"""pre_llm_call hook — wraps HTTP call в core daemon.

Регистрируется в ``styx_hermes.plugin.register`` через
``ctx.register_hook("pre_llm_call", on_pre_llm_call)``.

agent_id discovery: ``_agent_session`` set'ится при
``MemoryProvider.initialize``. Если session нет — возвращаем None
(Hermes не аппендит контекст).
"""

from __future__ import annotations

import logging
from typing import Any

from styx_hermes import _agent_session

log = logging.getLogger(__name__)


def on_pre_llm_call(**hermes_kwargs: Any) -> dict[str, str] | None:
    """Hermes pre_llm_call hook — sync HTTP вызов в core daemon."""
    session = _agent_session.get_session()
    if session is None:
        return None
    agent_id, client = session

    user_message = hermes_kwargs.get("user_message")
    is_first_turn = bool(hermes_kwargs.get("is_first_turn", False))
    extra: dict[str, Any] = {
        k: v
        for k, v in hermes_kwargs.items()
        if k not in {"user_message", "is_first_turn", "session_id", "model", "platform"}
        and v is not None
    }

    try:
        resp = client.pre_llm_inject(
            agent_id,
            session_id=hermes_kwargs.get("session_id"),
            user_message=user_message,
            is_first_turn=is_first_turn,
            model=hermes_kwargs.get("model"),
            platform=hermes_kwargs.get("platform"),
            extra=extra,
        )
    except Exception as exc:  # noqa: BLE001 — fail-open
        log.warning("styx-core /pre_llm_inject failed: %s", exc)
        return None

    context = resp.get("context")
    if not context:
        return None
    return {"context": context}
