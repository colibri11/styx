"""Contract tests for the bounded Hermes ``post_llm_call`` transport."""

from __future__ import annotations

import logging

import pytest

from styx_hermes import _agent_session
from styx_hermes.engine import post_llm_hook


class _CaptureClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def observe_affective_turn(self, agent_id: str, **kwargs):
        self.calls.append((agent_id, kwargs))
        if self.error is not None:
            raise self.error
        return {"accepted": False, "duplicate": False}


@pytest.fixture(autouse=True)
def _clear_session():
    _agent_session.clear_session()
    yield
    _agent_session.clear_session()


def test_post_llm_hook_forwards_identity_history_and_tools() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    history = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "please inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "file contents",
        },
        {"role": "assistant", "content": "done"},
    ]

    out = post_llm_hook.on_post_llm_call(
        session_id="physical-session",
        task_id="task-1",
        turn_id="turn-9",
        user_message=[{"type": "text", "text": "current user"}],
        assistant_response="final answer",
        conversation_history=history,
        model="model-x",
        platform="cli",
    )

    assert out is None
    assert len(client.calls) == 1
    agent_id, payload = client.calls[0]
    assert agent_id == "agent-a"
    assert payload["idempotency_key"] == "hermes:physical-session:turn-9"
    assert _agent_session.get_turn_key("physical-session") == payload["idempotency_key"]
    assert payload["turn_id"] == "turn-9"
    assert payload["user_message"] == "current user"
    assert payload["assistant_response"] == "final answer"
    assert payload["task_id"] == "task-1"
    assert [item["role"] for item in payload["conversation_history"]] == [
        "system", "user", "assistant",
    ]
    assert payload["tool_events"] == [
        {
            "kind": "call",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": '{"path":"a"}',
        },
        {
            "kind": "result",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "file contents",
        },
    ]


def test_post_llm_hook_bounds_history_tools_and_text() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    history = []
    for i in range(40):
        history.extend(
            [
                {"role": "user", "content": f"u{i}-" + "x" * 8_000},
                {
                    "role": "assistant",
                    "content": f"a{i}-" + "y" * 8_000,
                    "tool_calls": [
                        {
                            "id": f"call-{i}",
                            "function": {
                                "name": "tool",
                                "arguments": "z" * 8_000,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{i}",
                    "name": "tool",
                    "content": "r" * 8_000,
                },
            ]
        )

    post_llm_hook.on_post_llm_call(
        session_id="s",
        turn_id="t",
        user_message="u" * 30_000,
        assistant_response="a" * 50_000,
        conversation_history=history,
    )

    payload = client.calls[0][1]
    assert len(payload["conversation_history"]) <= post_llm_hook.MAX_HISTORY_MESSAGES
    assert all(
        len(item["content"]) <= post_llm_hook.MAX_HISTORY_CONTENT_CHARS
        for item in payload["conversation_history"]
    )
    assert len(payload["tool_events"]) == post_llm_hook.MAX_TOOL_EVENTS
    assert all(
        len(item["content"]) <= post_llm_hook.MAX_TOOL_EVENT_CONTENT_CHARS
        for item in payload["tool_events"]
    )
    assert len(payload["user_message"]) == post_llm_hook.MAX_USER_MESSAGE_CHARS
    assert (
        len(payload["assistant_response"])
        == post_llm_hook.MAX_ASSISTANT_RESPONSE_CHARS
    )


def test_post_llm_hook_bounds_content_parts_before_iteration() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    parts = [
        {"type": "text", "text": f"part-{index}"}
        for index in range(post_llm_hook.MAX_CONTENT_PARTS + 500)
    ]

    post_llm_hook.on_post_llm_call(
        session_id="s",
        turn_id="t",
        user_message=parts,
        assistant_response=parts,
        conversation_history=[{"role": "user", "content": parts}],
    )

    payload = client.calls[0][1]
    assert "part-0" not in payload["user_message"]
    assert f"part-{len(parts) - 1}" in payload["user_message"]
    assert "part-0" not in payload["conversation_history"][0]["content"]


def test_post_llm_hook_bounds_each_multimodal_part_before_join() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    huge = "x" * 2_000_000
    parts = [{"type": "text", "text": huge} for _ in range(128)]

    post_llm_hook.on_post_llm_call(
        session_id="s", turn_id="t", user_message=parts,
        assistant_response=parts, conversation_history=[],
    )

    payload = client.calls[0][1]
    assert len(payload["user_message"]) <= post_llm_hook.MAX_USER_MESSAGE_CHARS
    assert len(payload["assistant_response"]) <= post_llm_hook.MAX_ASSISTANT_RESPONSE_CHARS


def test_post_llm_hook_tool_scan_and_serialization_are_bounded_and_safe() -> None:
    class Explosive:
        def __str__(self) -> str:
            raise AssertionError("adapter must not stringify arbitrary objects")

    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    calls = [
        {
            "id": Explosive(),
            "function": {
                "name": Explosive(),
                "arguments": {
                    f"key-{index}": Explosive()
                    for index in range(post_llm_hook.MAX_SERIALIZED_ITEMS + 500)
                },
            },
        }
        for _ in range(post_llm_hook.MAX_TOOL_EVENTS + 500)
    ]

    post_llm_hook.on_post_llm_call(
        session_id="s",
        turn_id="t",
        conversation_history=[
            {"role": "assistant", "content": "", "tool_calls": calls}
        ],
    )

    events = client.calls[0][1]["tool_events"]
    assert len(events) == post_llm_hook.MAX_TOOL_EVENTS
    assert all(
        len(event["content"]) <= post_llm_hook.MAX_TOOL_EVENT_CONTENT_CHARS
        for event in events
    )
    assert all("Explosive" in event["content"] for event in events)


def test_post_llm_hook_bounds_identifiers_and_idempotency_key() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)

    post_llm_hook.on_post_llm_call(
        session_id="s" * 600,
        turn_id="t" * 600,
        task_id="k" * 600,
        model="m" * 800,
        platform="p" * 100,
    )

    payload = client.calls[0][1]
    assert len(payload["idempotency_key"]) <= 512
    assert len(payload["turn_id"]) <= 256
    assert len(payload["session_id"]) <= 256
    assert len(payload["task_id"]) <= 256
    assert len(payload["model"]) <= 512
    assert len(payload["platform"]) <= 64
    assert ":sha256:" in payload["turn_id"]


def test_turn_key_handoff_does_not_confuse_delayed_background_sync() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    post_llm_hook.on_post_llm_call(
        session_id="same-session", turn_id="turn-n",
        user_message="user n", assistant_response="answer n",
    )
    post_llm_hook.on_post_llm_call(
        session_id="same-session", turn_id="turn-n-plus-1",
        user_message="user n+1", assistant_response="answer n+1",
    )

    assert _agent_session.get_turn_key(
        "same-session", user_content="user n", assistant_content="answer n",
    ) == "hermes:same-session:turn-n"
    assert _agent_session.get_turn_key(
        "same-session", user_content="user n+1", assistant_content="answer n+1",
    ) == "hermes:same-session:turn-n-plus-1"


def test_identical_consecutive_turn_content_keeps_physical_order() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    for turn_id in ("first", "second"):
        post_llm_hook.on_post_llm_call(
            session_id="same-session", turn_id=turn_id,
            user_message="identical", assistant_response="identical",
        )
    kwargs = {
        "user_content": "identical",
        "assistant_content": "identical",
    }
    assert _agent_session.get_turn_key("same-session", **kwargs) == (
        "hermes:same-session:first"
    )
    assert _agent_session.get_turn_key("same-session", **kwargs) == (
        "hermes:same-session:second"
    )
    # Ambiguous retry of the latest sync retains its physical key.
    assert _agent_session.get_turn_key("same-session", **kwargs) == (
        "hermes:same-session:second"
    )


def test_post_llm_hook_history_limit_applies_after_role_filtering() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    history = [
        {"role": "user", "content": f"u-{index}"}
        for index in range(post_llm_hook.MAX_HISTORY_MESSAGES + 2)
    ]
    history.extend({"role": "tool", "content": "result"} for _ in range(30))

    post_llm_hook.on_post_llm_call(
        session_id="s",
        turn_id="t",
        conversation_history=history,
    )

    forwarded = client.calls[0][1]["conversation_history"]
    assert len(forwarded) == post_llm_hook.MAX_HISTORY_MESSAGES
    assert forwarded[0]["content"] == "u-2"


def test_post_llm_hook_requires_turn_id(caplog) -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    with caplog.at_level(logging.WARNING):
        assert post_llm_hook.on_post_llm_call(session_id="s") is None
    assert client.calls == []
    assert "turn_id missing" in caplog.text


def test_post_llm_hook_is_fail_open(caplog) -> None:
    client = _CaptureClient(RuntimeError("old core returned 404"))
    _agent_session.set_session("agent-a", client)
    with caplog.at_level(logging.WARNING):
        assert post_llm_hook.on_post_llm_call(
            session_id="s", turn_id="t", user_message="u",
            assistant_response="a", conversation_history=[],
        ) is None
    assert len(client.calls) == 1
    assert "/affect/observe_turn failed" in caplog.text


def test_post_llm_hook_without_initialized_session_is_noop() -> None:
    assert post_llm_hook.on_post_llm_call(turn_id="t") is None
