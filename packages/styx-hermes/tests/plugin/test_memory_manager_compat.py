"""Compat-контракты Styx с реальным MemoryManager Hermes 0.21+."""

from __future__ import annotations

from agent.memory_manager import MemoryManager, memory_provider_tools_enabled

from styx_hermes.providers.memory import StyxMemoryProvider


class _FakeClient:
    base_url = "http://fake"

    def __init__(self) -> None:
        self.assemble_calls: list[tuple[str, list[dict], dict]] = []
        self.sync_calls: list[tuple[str, dict]] = []

    def assemble_context(self, agent_id, messages, **kwargs):
        self.assemble_calls.append((agent_id, messages, kwargs))
        return {
            "system_prompt_addition": "<styx-salient>kept</styx-salient>"
        }

    def sync_turn(self, agent_id, **kwargs):
        self.sync_calls.append((agent_id, kwargs))


def _manager_with_provider() -> tuple[MemoryManager, StyxMemoryProvider, _FakeClient]:
    client = _FakeClient()
    provider = StyxMemoryProvider()
    provider._client = client
    provider._agent_id = "alpha"
    manager = MemoryManager()
    manager.add_provider(provider)
    return manager, provider, client


def test_styx_stays_legacy_checkpoint_v1() -> None:
    manager, provider, _ = _manager_with_provider()
    assert provider.pre_compress_checkpoint_api_version == 1
    assert manager.supports_pre_compress_checkpoint(2) is False


def test_pre_compress_uses_raw_transcript_and_returns_memory_context() -> None:
    manager, _, client = _manager_with_provider()
    raw = [{"role": "user", "content": "raw focus"}]
    evidence = [{"role": "user", "content": "normalized evidence"}]

    result = manager.on_pre_compress(raw, evidence_messages=evidence)

    assert "<styx-salient>kept</styx-salient>" in result
    assert client.assemble_calls[0][1] == [
        {"role": "user", "content": "raw focus"}
    ]


def test_sync_all_accepts_host_messages_with_legacy_styx_signature() -> None:
    manager, _, client = _manager_with_provider()
    manager.sync_all(
        "question",
        "answer",
        session_id="physical-session",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    assert manager.flush_pending(timeout=2.0)
    assert client.sync_calls == [
        (
            "alpha",
            {
                "user_content": "question",
                "assistant_content": "answer",
                "session_id": "physical-session",
            },
        )
    ]


def test_memory_toolset_gate_keeps_tools_and_system_block_in_sync() -> None:
    assert memory_provider_tools_enabled(None) is True
    assert memory_provider_tools_enabled([], memory_tool_present=False) is False
    assert memory_provider_tools_enabled(None, ["memory"]) is False
