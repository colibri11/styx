from __future__ import annotations

import pytest
from pydantic import ValidationError

from styx.http.models import (
    CognitionCommitRequest,
    CognitionObserveRequest,
    CognitionPreturnRequest,
)


def test_tool_shape_preserves_kind_id_and_content() -> None:
    request = CognitionCommitRequest(
        agent_id="a", host_key="turn-1",
        tool_events=[{
            "kind": "error", "tool_event_id": "call-7",
            "name": "fetch", "content": "redacted failure",
        }],
    )
    assert request.tool_events[0].model_dump() == {
        "kind": "error", "tool_event_id": "call-7", "name": "fetch",
        "content": "redacted failure", "metadata": {},
    }


def test_nested_journal_metadata_is_valid_but_bounded() -> None:
    request = CognitionCommitRequest(
        agent_id="a",
        host_key="turn-1",
        tool_events=[{
            "kind": "result",
            "metadata": {"outer": [{"authorization": "Bearer secret"}]},
        }],
        consequences=[{
            "kind": "evidence",
            "content": "observed",
            "incorporate": True,
            "line_eligible": False,
            "metadata": {"nested": {"token": "secret"}},
        }],
    )
    assert request.tool_events[0].metadata["outer"][0]["authorization"] == "Bearer secret"
    assert request.consequences[0].line_eligible is False


@pytest.mark.parametrize(
    "payload",
    [
        {"agent_id": "a", "messages": [{"role": "user", "content": "x"}] * 257},
        {"agent_id": "a", "query": "x" * 20_001},
    ],
)
def test_preturn_boundary_is_bounded(payload) -> None:
    with pytest.raises(ValidationError):
        CognitionPreturnRequest.model_validate(payload)


def test_commit_requires_nonempty_host_key() -> None:
    with pytest.raises(ValidationError):
        CognitionCommitRequest(agent_id="a", host_key="")


@pytest.mark.parametrize(
    "current_event",
    [
        {"constraints": "do not publish", "risk": "data loss"},
        '{"constraints":"do not publish","risk":"data loss"}',
    ],
)
def test_preturn_accepts_nested_and_json_current_event(current_event) -> None:
    request = CognitionPreturnRequest(
        agent_id="a", host_key="turn-1", extra={"current_event": current_event}
    )
    assert request.host_key == "turn-1"


def test_aggregate_metadata_node_budget_rejects_wide_tree() -> None:
    wide = {f"k{i}": list(range(16)) for i in range(16)}
    with pytest.raises(ValidationError, match="aggregate nodes"):
        CognitionCommitRequest(
            agent_id="a", host_key="turn-1",
            tool_events=[{"kind": "result", "metadata": wide}],
        )


def test_combined_consequence_contract_allows_32_plus_64() -> None:
    request = CognitionCommitRequest(
        agent_id="a", host_key="turn-1",
        tool_events=[{"kind": "error", "tool_event_id": str(i)} for i in range(64)],
        consequences=[{"kind": "explicit", "content": str(i)} for i in range(32)],
    )
    assert len(request.tool_events) == 64
    assert len(request.consequences) == 32


def test_observation_request_is_strict_bounded_and_agent_scoped() -> None:
    payload = {
        "agent_id": "agent-a",
        "source_id": "monitor",
        "source_stream": "workspace/main",
        "source_sequence": 7,
        "observation_key": "event-7",
        "difference_kind": "state_change",
        "content": "A bounded difference was detected.",
        "salience": 0.8,
        "confidence": 0.9,
        "reducer_name": "workspace-diff",
        "reducer_version": "1",
        "action_ref": {
            "agent_id": "agent-a",
            "host_key": "turn-1",
            "action_ordinal": 0,
        },
    }
    request = CognitionObserveRequest.model_validate(payload)
    assert request.source_sequence == 7

    with pytest.raises(ValidationError, match="Extra inputs"):
        CognitionObserveRequest.model_validate({**payload, "raw_sensor_blob": "x"})
    with pytest.raises(ValidationError, match="must match"):
        CognitionObserveRequest.model_validate({
            **payload,
            "action_ref": {**payload["action_ref"], "agent_id": "agent-b"},
        })
    with pytest.raises(ValidationError):
        CognitionObserveRequest.model_validate({**payload, "confidence": float("nan")})
