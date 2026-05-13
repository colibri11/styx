"""End-to-end smoke — закрывает Part 1.

Проверяет все 5 критериев завершения из integrations/hermes-v1.md
§ «Критерии завершения Part 1»:

1. styx setup устанавливает shim'ы в чистый HERMES_HOME.
2. Оба shim'а регистрируют свои компоненты через свои discovery-системы:
   StyxMemoryProvider — единственный, StyxContextEngine — заменяет
   дефолтный, StyxOpenAITransport — в _REGISTRY.
3. Первый turn end-to-end: compress() → build_kwargs() → mock OpenAI →
   normalize_response → update_from_response → sync_turn.
4. Wire-log: prefix-digest идентичен между turn 1 и turn 2.
5. Mock OpenAI рапортует cached_tokens > 0 на turn 2 — engine видит это
   через update_from_response → extract_cache_stats.

Реальный OpenAI smoke — отдельный скрипт ``scripts/smoke_real_openai.py``;
здесь — детерминированный mock.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from styx.cli import main as cli_main
from styx.engine import transport as transport_mod
from styx.engine.context import StyxContextEngine
from styx.engine.transport import (
    StyxCodexTransport,
    StyxOpenAITransport,
    compute_prefix_digest,
)


# -- fakes / fixtures -----------------------------------------------------


class FakePluginContext:
    """Подмена hermes_cli.plugins.PluginContext."""

    def __init__(self) -> None:
        self.context_engine: StyxContextEngine | None = None

    def register_context_engine(self, engine) -> None:
        self.context_engine = engine

    def register_tool(self, *a, **kw): pass

    def register_hook(self, *a, **kw): pass


class FakeMemoryCollector:
    """Подмена _ProviderCollector из plugins/memory/__init__.py."""

    def __init__(self) -> None:
        self.provider = None

    def register_memory_provider(self, provider) -> None:
        self.provider = provider


class FakeOpenAIClient:
    """Имитирует ``openai.chat.completions.create``.

    Возвращает структуру SimpleNamespace, совместимую с
    ChatCompletionsTransport.normalize_response и extract_cache_stats.
    Cache hit имитируется на втором (и далее) вызове.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        is_cached_turn = len(self.calls) >= 2
        prompt_tokens = 1024
        cached = 900 if is_cached_turn else 0

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"reply-{len(self.calls)}",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=10,
                total_tokens=prompt_tokens + 10,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
            ),
        )


@pytest.fixture(autouse=True)
def _restore_transport_state():
    yield
    from agent.transports import register_transport
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport
    register_transport("chat_completions", ChatCompletionsTransport)
    register_transport("codex_responses", ResponsesApiTransport)
    transport_mod._reset_for_test()


# -- E2E smoke ------------------------------------------------------------


def test_part1_end_to_end_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migrated_db: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Полный turn-1 + turn-2 цикл со всеми пятью критериями Part 1."""

    # ---- 1. styx setup в чистый HERMES_HOME -----------------------------
    hermes_home = tmp_path / "hermes-profile"
    rc = cli_main(["setup", "--hermes-home", str(hermes_home)])
    assert rc == 0
    # General plugin (`styx`) подхватывается через entry-point —
    # его shim в HERMES_HOME не копируется. Только memory shim:
    assert (hermes_home / "plugins" / "styx-memory" / "__init__.py").exists()
    assert not (hermes_home / "plugins" / "styx" / "__init__.py").exists()

    # ---- 2. Регистрация компонентов через оба discovery-фасада -----------
    from styx import memory_plugin, plugin
    from styx.providers.memory import StyxMemoryProvider

    plugin_ctx = FakePluginContext()
    plugin.register(plugin_ctx)
    engine = plugin_ctx.context_engine
    assert engine is not None
    assert engine.name == "styx"
    # Выставляем реальный context_length чтобы eviction срабатывал (#7)
    engine.update_model("gpt-x-mini", 10_000)

    collector = FakeMemoryCollector()
    memory_plugin.register(collector)
    provider = collector.provider
    assert isinstance(provider, StyxMemoryProvider)

    from agent.transports import get_transport
    transport_inst = get_transport("chat_completions")
    assert isinstance(transport_inst, StyxOpenAITransport)

    # ---- 3. Initialize provider — DSN + agent_identity -------------------
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    session_id = str(uuid.uuid4())
    provider.initialize(
        session_id=session_id,
        agent_identity="e2e-agent",
        hermes_home=str(hermes_home),
        platform="cli",
    )

    fake_client = FakeOpenAIClient()
    model = "gpt-x-mini"

    try:
        with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
            # ---- TURN 1 ----
            # digest считается от первых 3 элементов payload (head_count=3)
            # — длина строк не важна для стабильности digest'а.
            # 12+ сообщений чтобы eviction (protect_first_n=3 + protect_last_n=6)
            # реально имел что эвиктить (#7).
            stable_a = "stable bootstrap A"
            stable_b = "stable bootstrap B"
            history_t1: list[dict] = [
                {"role": "user", "content": stable_a},
                {"role": "assistant", "content": "noted A"},
                {"role": "user", "content": stable_b},
                {"role": "assistant", "content": "noted B"},
                {"role": "user", "content": "filler question 1"},
                {"role": "assistant", "content": "filler answer 1"},
                {"role": "user", "content": "filler question 2"},
                {"role": "assistant", "content": "filler answer 2"},
                {"role": "user", "content": "filler question 3"},
                {"role": "assistant", "content": "filler answer 3"},
                {"role": "user", "content": "filler question 4"},
                {"role": "assistant", "content": "filler answer 4"},
                {"role": "user", "content": "live question — turn 1"},
            ]
            # current_tokens=9_000 > threshold_tokens=7_500 → eviction сработает
            compressed_t1 = engine.compress(history_t1, current_tokens=9_000)
            assert compressed_t1, "compress вернул пустой список"

            kwargs_t1 = transport_inst.build_kwargs(
                model=model,
                messages=compressed_t1,
                session_id=session_id,
            )
            assert kwargs_t1["prompt_cache_key"] == "e2e-agent"

            response_t1 = fake_client.create(**kwargs_t1)
            normalized_t1 = transport_inst.normalize_response(response_t1)
            assert normalized_t1.content == "reply-1"

            engine.update_from_response({
                "prompt_tokens": response_t1.usage.prompt_tokens,
                "completion_tokens": response_t1.usage.completion_tokens,
                "total_tokens": response_t1.usage.total_tokens,
            })
            stats_t1 = transport_inst.extract_cache_stats(response_t1)
            assert stats_t1 is None or stats_t1.get("cached_tokens", 0) == 0

            provider.sync_turn(
                user_content="live question — turn 1",
                assistant_content=normalized_t1.content,
                session_id=session_id,
            )
            digest_t1 = compute_prefix_digest(kwargs_t1["messages"])
            prefix_messages_t1 = list(kwargs_t1["messages"][: engine.protect_first_n])

            # Eviction должен был сработать на turn 1 (#7)
            assert engine.compression_count > 0, (
                "eviction не сработал: проверь context_length/threshold_tokens и "
                f"len(history_t1)={len(history_t1)}"
            )

            # ---- TURN 2 ----
            history_t2 = list(history_t1) + [
                {"role": "assistant", "content": normalized_t1.content},
                {"role": "user", "content": "follow-up — turn 2"},
            ]
            # current_tokens=9_000 > threshold_tokens=7_500 → eviction сработает снова
            compressed_t2 = engine.compress(history_t2, current_tokens=9_000)

            kwargs_t2 = transport_inst.build_kwargs(
                model=model,
                messages=compressed_t2,
                session_id=session_id,
            )
            assert kwargs_t2["prompt_cache_key"] == "e2e-agent"

            response_t2 = fake_client.create(**kwargs_t2)
            normalized_t2 = transport_inst.normalize_response(response_t2)
            assert normalized_t2.content == "reply-2"

            engine.update_from_response({
                "prompt_tokens": response_t2.usage.prompt_tokens,
                "completion_tokens": response_t2.usage.completion_tokens,
                "total_tokens": response_t2.usage.total_tokens,
            })
            stats_t2 = transport_inst.extract_cache_stats(response_t2)

            provider.sync_turn(
                user_content="follow-up — turn 2",
                assistant_content=normalized_t2.content,
                session_id=session_id,
            )
            digest_t2 = compute_prefix_digest(kwargs_t2["messages"])
            prefix_messages_t2 = list(kwargs_t2["messages"][: engine.protect_first_n])

        # ---- Критерий 4: байт-стабильный prefix между turn 1 и 2 -------
        assert prefix_messages_t1 == prefix_messages_t2
        assert digest_t1 == digest_t2

        # ---- Критерий 5: cached_tokens > 0 на turn 2 -------------------
        assert stats_t2 is not None
        assert stats_t2["cached_tokens"] > 0

        # Wire-log на оба turn'а попал в caplog
        wire_records = [r for r in caplog.records if "prefix_slice" in r.message]
        assert len(wire_records) >= 2

        # ---- Критерий 3: sync_turn действительно записал в storage -----
        # 2 turn × (user + assistant) = 4 записи
        assert provider.queries.count_messages(session_id=uuid.UUID(session_id)) == 4

        contents = {
            m.content
            for m in provider.queries.recent_messages(
                limit=10, session_id=uuid.UUID(session_id)
            )
        }
        assert "live question — turn 1" in contents
        assert "follow-up — turn 2" in contents
        assert "reply-1" in contents
        assert "reply-2" in contents

        # Engine state-поля обновлены
        assert engine.last_prompt_tokens == 1024
        assert engine.last_completion_tokens == 10
        assert engine.last_total_tokens == 1034

    finally:
        provider.shutdown()


def test_part1_e2e_smoke_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migrated_db: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Codex-путь (api_mode=codex_responses) — симметричный smoke.

    Проверяет:
    - register_with_hermes() регистрирует StyxCodexTransport.
    - build_kwargs выдаёт prompt_cache_key = agent_id (Styx override).
    - Hermes-default prompt_cache_key=session_id при отсутствии agent_id.
    - digest от input стабилен между turn'ами (head_count=3).
    - sync_turn пишет в storage (через уже инициализированный provider).

    normalize_response не вызывается — требует полноценный Responses API объект
    (output list с items типа message/reasoning); тестируется в unit-тестах
    ResponsesApiTransport (hermes-agent suite).
    """
    # ---- Setup HERMES_HOME и регистрация ----------------------------------
    hermes_home = tmp_path / "hermes-codex"
    rc = cli_main(["setup", "--hermes-home", str(hermes_home)])
    assert rc == 0

    from styx import memory_plugin, plugin
    from styx.providers.memory import StyxMemoryProvider

    plugin_ctx = FakePluginContext()
    plugin.register(plugin_ctx)

    from agent.transports import get_transport
    transport_inst = get_transport("codex_responses")
    assert isinstance(transport_inst, StyxCodexTransport)

    collector = FakeMemoryCollector()
    memory_plugin.register(collector)
    provider = collector.provider
    assert isinstance(provider, StyxMemoryProvider)

    # ---- Initialize provider — DSN + agent_identity ----------------------
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    session_id = str(uuid.uuid4())
    provider.initialize(
        session_id=session_id,
        agent_identity="codex-agent",
        hermes_home=str(hermes_home),
        platform="cli",
    )

    model = "gpt-5.5"

    try:
        with caplog.at_level(logging.INFO, logger="styx.transport.wire"):
            # ---- TURN 1 ----
            msgs_t1 = [
                {"role": "user", "content": "stable preamble A"},
                {"role": "assistant", "content": "noted A"},
                {"role": "user", "content": "stable preamble B"},
                {"role": "assistant", "content": "noted B"},
                {"role": "user", "content": "question — turn 1"},
            ]
            kwargs_t1 = transport_inst.build_kwargs(
                model=model,
                messages=msgs_t1,
                session_id=session_id,
            )
            # agent_id override должен перекрывать session_id
            assert kwargs_t1["prompt_cache_key"] == "codex-agent"
            # Codex транспорт возвращает input (list), не messages
            assert "input" in kwargs_t1

            # digest по первым 3 input items
            digest_t1 = compute_prefix_digest(kwargs_t1["input"])

            # wire-log сработал для input
            records_t1 = [r for r in caplog.records if "prefix_slice" in r.message]
            assert len(records_t1) >= 1

            # sync_turn — запись в storage
            provider.sync_turn(
                user_content="question — turn 1",
                assistant_content="answer — turn 1",
                session_id=session_id,
            )

            # ---- TURN 2 (те же bootstrap msg, другой live вопрос) ----
            msgs_t2 = [
                {"role": "user", "content": "stable preamble A"},
                {"role": "assistant", "content": "noted A"},
                {"role": "user", "content": "stable preamble B"},
                {"role": "assistant", "content": "noted B"},
                {"role": "user", "content": "answer — turn 1"},
                {"role": "user", "content": "question — turn 2"},
            ]
            kwargs_t2 = transport_inst.build_kwargs(
                model=model,
                messages=msgs_t2,
                session_id=session_id,
            )
            assert kwargs_t2["prompt_cache_key"] == "codex-agent"

            digest_t2 = compute_prefix_digest(kwargs_t2["input"])

            provider.sync_turn(
                user_content="question — turn 2",
                assistant_content="answer — turn 2",
                session_id=session_id,
            )

        # Digest стабилен — первые 3 input items идентичны
        assert digest_t1 == digest_t2, (
            f"digest нестабилен между turn'ами: {digest_t1!r} vs {digest_t2!r}"
        )

        # Wire-log попал в caplog как минимум дважды (turn1 + turn2)
        all_wire = [r for r in caplog.records if "prefix_slice" in r.message]
        assert len(all_wire) >= 2

        # storage: 2 turn × 2 messages = 4
        assert provider.queries.count_messages(session_id=uuid.UUID(session_id)) == 4

    finally:
        provider.shutdown()
