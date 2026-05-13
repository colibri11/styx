"""Реальный E2E smoke с OpenAI API — закрывает критерии 1-5 Part 1.

Запуск:

    OPENAI_API_KEY=sk-...
    STYX_DATABASE_URL=postgresql://...
    python scripts/smoke_real_openai.py

Скрипт:

1. Применяет миграции на чистую БД.
2. Регистрирует ContextEngine + Transport через styx.plugin.
3. Регистрирует MemoryProvider через styx.memory_plugin.
4. Делает два turn'а с одинаковым stable prefix'ом, разный live tail.
5. Печатает wire-log digest и cached_tokens с обоих turn'ов.

Требования: pip-package ``openai``, активный Hermes-checkout (HERMES_PATH),
доступный Postgres+pgvector. Не запускается из CI — пользовательский
manual smoke.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("styx.smoke")


def _require(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        sys.exit(f"требуется {env_name}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real OpenAI E2E smoke")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="styx-smoke")
    args = parser.parse_args(argv)

    api_key = _require("OPENAI_API_KEY")
    dsn = _require("STYX_DATABASE_URL")

    # 1. Миграция
    from styx.storage import migrate
    migrate.run(dsn)
    log.info("миграция применена")

    # 2. Регистрация компонентов
    from styx import memory_plugin, plugin
    from styx.engine.transport import (
        StyxOpenAITransport,
        compute_prefix_digest,
    )

    class _Ctx:
        context_engine: Any = None
        def register_context_engine(self, engine): self.context_engine = engine
        def register_tool(self, *a, **kw): pass
        def register_hook(self, *a, **kw): pass

    class _Collector:
        provider: Any = None
        def register_memory_provider(self, p): self.provider = p

    ctx = _Ctx()
    plugin.register(ctx)
    engine = ctx.context_engine

    collector = _Collector()
    memory_plugin.register(collector)
    provider = collector.provider

    session_id = str(uuid.uuid4())
    provider.initialize(
        session_id=session_id,
        agent_identity=args.agent_id,
        platform="cli",
    )
    log.info("provider initialized agent=%s session=%s", args.agent_id, session_id)

    transport = StyxOpenAITransport()  # тот же класс, что в _REGISTRY

    # 3. OpenAI клиент
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("openai SDK не установлен: pip install openai")

    client = OpenAI(api_key=api_key)

    # Stable prefix должен быть достаточно длинным, чтобы prompt-cache
    # сработал (>=1024 токенов на OpenAI). Достигаем filler-текстом.
    filler = ("This is a deterministic stable preamble that grows the "
              "prompt past the OpenAI prefix-cache threshold. " * 80)
    history_base = [
        {"role": "user", "content": filler},
        {"role": "assistant",
         "content": "Acknowledged. Ready for your follow-up questions."},
    ]

    def _do_turn(label: str, user_msg: str, history: list[dict]) -> dict:
        history = history + [{"role": "user", "content": user_msg}]
        compressed = engine.compress(history, current_tokens=500)
        kwargs = transport.build_kwargs(
            model=args.model,
            messages=compressed,
            session_id=session_id,
        )
        response = client.chat.completions.create(**kwargs)
        normalized = transport.normalize_response(response)
        engine.update_from_response({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        })
        cache_stats = transport.extract_cache_stats(response) or {}
        provider.sync_turn(
            user_content=user_msg,
            assistant_content=normalized.content or "",
            session_id=session_id,
        )
        digest = compute_prefix_digest(kwargs["messages"])
        log.info(
            "%s: digest=%s prompt_tokens=%d cached=%d response=%r",
            label, digest, response.usage.prompt_tokens,
            cache_stats.get("cached_tokens", 0),
            (normalized.content or "")[:80],
        )
        return {
            "digest": digest,
            "cached_tokens": cache_stats.get("cached_tokens", 0),
            "history": history + [
                {"role": "assistant", "content": normalized.content or ""}
            ],
        }

    try:
        t1 = _do_turn("turn1", "What is your name?", history_base)
        t2 = _do_turn("turn2", "And what was the last question I asked?",
                      t1["history"])
    finally:
        provider.shutdown()

    # 4. Проверки
    print()
    print("=== Part 1 critéria ===")
    same_digest = t1["digest"] == t2["digest"]
    print(f"  byte-stable prefix (digest match): {'OK' if same_digest else 'FAIL'}")
    print(f"    turn1 digest: {t1['digest']}")
    print(f"    turn2 digest: {t2['digest']}")
    cached_ok = t2["cached_tokens"] > 0
    print(f"  turn2 cached_tokens > 0:           {'OK' if cached_ok else 'FAIL'}")
    print(f"    turn2 cached_tokens: {t2['cached_tokens']}")

    return 0 if (same_digest and cached_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
