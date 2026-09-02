"""Contract tests for the fenced Hermes ``pre_llm_call`` transport."""

from __future__ import annotations

import pytest

from styx_hermes import _agent_session
from styx_hermes.engine import post_llm_hook, pre_llm_hook


class _CaptureClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def cognition_preturn(self, agent_id: str, **kwargs):
        self.calls.append((agent_id, kwargs))
        if self.error is not None:
            raise self.error
        return {
            "messages": [],
            "line_version": 9,
            "snapshot_token": "snapshot-9",
            "system_prompt_addition": (
                "<styx-cognitive-continuity>{}</styx-cognitive-continuity>"
            ),
        }


@pytest.fixture(autouse=True)
def _clear_session():
    _agent_session.clear_session()
    yield
    _agent_session.clear_session()


def test_pre_llm_hook_injects_canonical_envelope_and_remembers_fence() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)

    result = pre_llm_hook.on_pre_llm_call(
        session_id="session-1",
        turn_id="turn-1",
        user_message="current event",
        model="model-x",
        platform="cli",
        is_first_turn=True,
    )

    assert result == {
        "context": (
            "<styx-cognitive-continuity>{}</styx-cognitive-continuity>"
        )
    }
    agent_id, payload = client.calls[0]
    assert agent_id == "agent-a"
    assert payload["messages"] == [{"role": "user", "content": "current event"}]
    assert payload["query"] == "current event"
    assert payload["host_key"] == "hermes:session-1:turn-1"
    assert payload["parent_host_key"] is None
    assert payload["extra"]["current_event"] == {
        "turn_id": "turn-1",
        "is_first_turn": True,
    }
    parent, snapshot = _agent_session.declare_act(
        "session-1", "hermes:session-1:turn-1"
    )
    assert parent is None
    assert snapshot == "snapshot-9"


def test_pre_llm_hook_forwards_normalized_host_conversation() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    pre_llm_hook.on_pre_llm_call(
        session_id="session-1",
        turn_id="turn-1",
        user_message="current event",
        conversation_history=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {
                "role": "tool",
                "name": "read",
                "tool_call_id": "call-1",
                "content": "result",
            },
        ],
    )

    assert client.calls[0][1]["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {
            "role": "tool", "content": "result", "name": "read",
            "tool_call_id": "call-1",
        },
    ]


def test_pre_llm_hook_bounds_current_event_metadata() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    extra = {f"field-{index}": "x" * 10_000 for index in range(100)}

    pre_llm_hook.on_pre_llm_call(
        session_id="s", user_message="q", **extra
    )

    event = client.calls[0][1]["extra"]["current_event"]
    assert len(event) <= pre_llm_hook._MAX_CURRENT_EVENT_FIELDS
    assert all(
        not isinstance(value, str)
        or len(value) <= pre_llm_hook._MAX_CURRENT_EVENT_TEXT
        for value in event.values()
    )


def test_pre_llm_hook_is_fail_open() -> None:
    client = _CaptureClient(error=RuntimeError("down"))
    _agent_session.set_session("agent-a", client)
    assert pre_llm_hook.on_pre_llm_call(user_message="q") is None


def test_pre_llm_hook_without_session_is_noop() -> None:
    assert pre_llm_hook.on_pre_llm_call(user_message="q") is None


def test_minimal_hermes_lifecycle_shares_one_fence_and_host_identity() -> None:
    class LifecycleClient(_CaptureClient):
        def __init__(self) -> None:
            super().__init__()
            self.commits: list[dict] = []

        def cognition_commit(self, agent_id: str, **kwargs):
            assert agent_id == "agent-a"
            self.commits.append(kwargs)
            return {"committed": True, "duplicate": False}

    client = LifecycleClient()
    _agent_session.set_session("agent-a", client)
    history = [{"role": "user", "content": "current"}]
    assert pre_llm_hook.on_pre_llm_call(
        session_id="session-1",
        turn_id="turn-1",
        user_message="current",
        conversation_history=history,
    ) is not None
    post_llm_hook.on_post_llm_call(
        session_id="session-1",
        turn_id="turn-1",
        user_message="current",
        assistant_response="answer",
        conversation_history=history + [
            {"role": "assistant", "content": "answer"}
        ],
    )

    assert len(client.calls) == 1
    assert len(client.commits) == 1
    assert client.commits[0]["host_key"] == "hermes:session-1:turn-1"
    assert client.calls[0][1]["host_key"] == client.commits[0]["host_key"]
    assert client.commits[0]["snapshot_token"] == "snapshot-9"
    assert client.commits[0]["parent_host_key"] is None


def test_second_preturn_names_the_declared_predecessor() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)
    first_key = "hermes:session-1:turn-1"
    assert _agent_session.declare_act("session-1", first_key) == (None, None)

    pre_llm_hook.on_pre_llm_call(
        session_id="session-1",
        turn_id="turn-2",
        user_message="next",
    )

    assert client.calls[0][1]["parent_host_key"] == first_key


def test_pre_llm_hook_without_turn_id_omits_physical_host_key() -> None:
    client = _CaptureClient()
    _agent_session.set_session("agent-a", client)

    pre_llm_hook.on_pre_llm_call(session_id="session-1", user_message="q")

    assert client.calls[0][1]["host_key"] is None
