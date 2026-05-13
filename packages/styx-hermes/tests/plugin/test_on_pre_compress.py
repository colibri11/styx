"""Тесты ``StyxMemoryProvider.on_pre_compress()`` — защита salient
при сжатии context (волна 29 Phase D).

Hermes зовёт hook когда context_compressor собирается отбросить старые
messages. Provider возвращает текст для compression summary prompt;
без него Styx-агент теряет memories через compression boundary.
"""

from __future__ import annotations

import pytest

from styx_hermes import _agent_session
from styx_hermes.providers.memory import StyxMemoryProvider


@pytest.fixture(autouse=True)
def _reset_session():
    yield
    _agent_session.clear_session()


class _FakeClient:
    def __init__(self, addition: str | None = None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._addition = addition
        self.base_url = "http://fake"
        self.closed = False

    def initialize_agent(self, agent_id, **kwargs):
        return {"agent_id": agent_id, "tools": []}

    def shutdown_agent(self, agent_id):
        pass

    def assemble_context(self, agent_id, messages, **kwargs):
        self.calls.append(("assemble_context", (agent_id, messages), kwargs))
        return {
            "messages": [],
            "estimated_tokens": 0,
            "system_prompt_addition": self._addition,
            "prompt_authority": "assembled",
        }

    def close(self):
        self.closed = True


def _make_provider(monkeypatch, fake) -> StyxMemoryProvider:
    monkeypatch.setattr(
        "styx_hermes.providers.memory.StyxCoreClient",
        lambda *a, **kw: fake,
    )
    p = StyxMemoryProvider()
    p.initialize(session_id="sid", agent_identity="alpha")
    return p


def test_empty_messages_returns_empty() -> None:
    p = StyxMemoryProvider()
    assert p.on_pre_compress([]) == ""


def test_returns_empty_when_no_user_message_found() -> None:
    p = StyxMemoryProvider()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "только assistant без user"},
    ]
    assert p.on_pre_compress(msgs) == ""


def test_extracts_last_user_text_string_content(monkeypatch) -> None:
    """Last user-message с string content — берётся как focus query."""
    fake = _FakeClient(addition="<styx-salient>\nfoo\n</styx-salient>")
    p = _make_provider(monkeypatch, fake)
    msgs = [
        {"role": "user", "content": "первая реплика"},
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": "что я говорил про X?"},
    ]
    out = p.on_pre_compress(msgs)
    assert "Memories from Styx" in out
    assert "<styx-salient>" in out
    # Focus был последним user
    assemble_calls = [c for c in fake.calls if c[0] == "assemble_context"]
    assert len(assemble_calls) == 1
    _, (_, sent_messages), _ = assemble_calls[0]
    assert sent_messages == [{"role": "user", "content": "что я говорил про X?"}]


def test_extracts_last_user_text_multimodal_content(monkeypatch) -> None:
    """Multi-modal content — извлекаем text-part."""
    fake = _FakeClient(addition="<styx-salient>\nbar\n</styx-salient>")
    p = _make_provider(monkeypatch, fake)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "вопрос про дизайн"},
                {"type": "image", "source": "..."},
            ],
        },
    ]
    out = p.on_pre_compress(msgs)
    assert "<styx-salient>" in out
    assemble_calls = [c for c in fake.calls if c[0] == "assemble_context"]
    _, (_, sent_messages), _ = assemble_calls[0]
    assert sent_messages == [
        {"role": "user", "content": "вопрос про дизайн"}
    ]


def test_returns_empty_when_no_salient(monkeypatch) -> None:
    """Если recall не нашёл памяти — пустая строка (no provider contribution)."""
    fake = _FakeClient(addition=None)
    p = _make_provider(monkeypatch, fake)
    msgs = [{"role": "user", "content": "вопрос"}]
    assert p.on_pre_compress(msgs) == ""


def test_preamble_signals_compression_intent(monkeypatch) -> None:
    """Preamble должен сообщать compressor'у что текст должен выжить
    summary — иначе он может сжаться обратно."""
    fake = _FakeClient(addition="<styx-salient>\nbaz\n</styx-salient>")
    p = _make_provider(monkeypatch, fake)
    out = p.on_pre_compress([{"role": "user", "content": "q"}])
    assert "preserve in summary" in out.lower() or "survive compression" in out.lower()
