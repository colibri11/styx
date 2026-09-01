"""HTTP contract tests for ``POST /affect/observe_turn``."""

from __future__ import annotations

from styx.http import registry


class _CaptureCore:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result or {"accepted": True, "duplicate": False}
        self.calls: list[dict] = []

    def observe_affective_turn(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _body() -> dict:
    return {
        "agent_id": "agent-a",
        "idempotency_key": "hermes:sid:turn-1",
        "turn_id": "turn-1",
        "session_id": "sid",
        "user_message": "current user",
        "assistant_response": "final answer",
        "conversation_history": [
            {"role": "user", "content": "prior user"},
            {"role": "assistant", "content": "prior answer"},
        ],
        "tool_events": [
            {
                "kind": "result",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "ok",
            }
        ],
        "task_id": "task-1",
        "model": "model-x",
        "platform": "cli",
    }


def test_observe_turn_delegates_bounded_envelope(client_no_auth) -> None:
    core = _CaptureCore()
    registry.register("agent-a", core=core)

    response = client_no_auth.post("/affect/observe_turn", json=_body())

    assert response.status_code == 200
    assert response.json() == {
        "accepted": True,
        "duplicate": False,
        "reason": None,
    }
    assert len(core.calls) == 1
    call = core.calls[0]
    assert call["idempotency_key"] == "hermes:sid:turn-1"
    assert call["turn_id"] == "turn-1"
    assert call["conversation_history"][0] == {
        "role": "user",
        "content": "prior user",
        "name": None,
    }
    assert call["tool_events"][0]["tool_call_id"] == "call-1"


def test_observe_turn_requires_initialized_agent(client_no_auth) -> None:
    response = client_no_auth.post("/affect/observe_turn", json=_body())
    assert response.status_code == 404


def test_observe_turn_requires_idempotency_key(client_no_auth) -> None:
    body = _body()
    body.pop("idempotency_key")
    response = client_no_auth.post("/affect/observe_turn", json=body)
    assert response.status_code == 422


def test_observe_turn_rejects_unbounded_history(client_no_auth) -> None:
    body = _body()
    body["conversation_history"] = [
        {"role": "user", "content": str(i)} for i in range(25)
    ]
    response = client_no_auth.post("/affect/observe_turn", json=body)
    assert response.status_code == 422


def test_observe_turn_rejects_unbounded_tool_content(client_no_auth) -> None:
    body = _body()
    body["tool_events"][0]["content"] = "x" * 4_001
    response = client_no_auth.post("/affect/observe_turn", json=body)
    assert response.status_code == 422
