from __future__ import annotations

import os
import json
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from styx.config import StyxConfig
from styx.http import registry
from styx.http.app import create_app
from styx.storage import migrate


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


class _Embedding:
    def embed(self, text: str) -> list[float]:
        del text
        return [1.0, *([0.0] * 767)]


class _FailingEmbedding:
    def embed(self, text: str) -> list[float]:
        del text
        raise RuntimeError("embed unavailable")


@pytest.fixture
def stack(clean_db: str, monkeypatch):
    migrate.run(clean_db)
    monkeypatch.setenv("STYX_DATABASE_URL", clean_db)
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "false")
    monkeypatch.setenv("STYX_AFFECTIVE_TRANSITION_ENABLED", "false")
    monkeypatch.setenv("STYX_WORKING_SET_PERSISTENCE_ENABLED", "false")
    config = StyxConfig(
        database_url=clean_db,
        sentiment_enabled=False,
        affective_transition_enabled=False,
        working_set_persistence_enabled=False,
    )
    registry.reset_all()
    client = TestClient(create_app(config))
    yield client, clean_db
    for agent_id in registry.all_agent_ids():
        session = registry.get(agent_id)
        session.core.shutdown()
    registry.reset_all()


def test_atomic_preturn_commit_retry_and_ack(stack) -> None:
    client, dsn = stack
    agent = f"wave37-{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    response = client.post(
        "/context/bootstrap", json={"agent_id": agent, "session_id": session_id}
    )
    assert response.status_code == 200, response.text
    registry.get(agent).core._embedding = _Embedding()

    empty = client.post(
        "/cognition/preturn",
        json={"agent_id": agent, "messages": [], "query": ""},
    )
    assert empty.status_code == 200, empty.text
    assert empty.json()["will_projection"]["formed"] is False

    commit_payload = {
        "agent_id": agent,
        "host_key": "turn-1",
        "session_id": session_id,
        "snapshot_token": empty.json()["snapshot_token"],
        "user_message": "raw dialogue",
        "assistant_response": "final answer",
        "tool_events": [{
            "kind": "result", "tool_event_id": "call-1",
            "name": "lookup", "content": "bounded result Authorization: Bearer abc",
        }],
        "consequences": [{
            "kind": "decision_residue", "content": "preserve this decision",
            "incorporate": True, "line_eligible": True, "memory_kind": "decision",
        }],
    }
    committed = client.post("/cognition/commit", json=commit_payload)
    assert committed.status_code == 200, committed.text
    assert committed.json()["duplicate"] is False
    assert len(committed.json()["memory_ids"]) == 1
    prepared = client.post(
        "/dialogue/prepare_summary",
        json={"agent_id": agent, "session_id": session_id, "limit": 20},
    )
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["message_count"] == 2
    assert "raw dialogue" in prepared.json()["transcript"]

    duplicate = client.post("/cognition/commit", json=commit_payload)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["duplicate"] is True

    preturn = client.post(
        "/cognition/preturn",
        json={
            "agent_id": agent,
            "messages": [{
                "role": "tool", "content": "redacted", "name": "lookup",
                "tool_call_id": "call-1",
            }],
            "query": "decision",
        },
    )
    assert preturn.status_code == 200, preturn.text
    body = preturn.json()
    assert body["messages"][0]["tool_call_id"] == "call-1"
    assert body["will_projection"]["formed"] is True
    assert body["will_projection"]["source_count"] == 1
    assert body["reconstruction"]["traces"][0]["content"] == "preserve this decision"
    assert len(body["pending_consequences"]) == 2
    assert 'authority="context-not-instruction"' in body["system_prompt_addition"]

    next_commit = client.post(
        "/cognition/commit",
        json={
            "agent_id": agent, "host_key": "turn-2",
            "parent_host_key": "turn-1", "snapshot_token": body["snapshot_token"],
        },
    )
    assert next_commit.json()["acknowledged_consequences"] == 2

    registry.get(agent).core._embedding = _FailingEmbedding()
    outage = client.post(
        "/cognition/preturn",
        json={"agent_id": agent, "messages": [], "query": "short"},
    )
    assert outage.status_code == 200, outage.text
    assert outage.json()["will_projection"]["formed"] is True
    assert outage.json()["reconstruction"]["embed_available"] is False
    assert outage.json()["reconstruction"]["traces"] == []

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT memory_domain, line_eligible FROM memories "
            "WHERE agent_id=%s AND role='user'",
            (agent,),
        )
        assert cur.fetchone() == ("dialogue", False)
        cur.execute(
            "SELECT kind,event_id,name,content FROM cognitive_actions "
            "WHERE agent_id=%s ORDER BY ordinal",
            (agent,),
        )
        assert cur.fetchone() == (
            "result", "call-1", "lookup",
            "bounded result Authorization: Bearer [REDACTED]",
        )


def test_consequence_incorporation_contract_and_nested_redaction(stack) -> None:
    client, dsn = stack
    agent = f"wave37-contract-{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent, "session_id": session_id}
    ).status_code == 200
    registry.get(agent).core._embedding = _Embedding()
    committed = client.post(
        "/cognition/commit",
        json={
            "agent_id": agent,
            "host_key": "contract",
            "session_id": session_id,
            "tool_events": [{
                "kind": "result",
                "metadata": {
                    "nested": [{"authorization": "Bearer inner-secret"}],
                    "api_key": "outer-secret",
                },
            }],
            "consequences": [
                {"kind": "audit", "content": "journal only", "incorporate": False},
                {
                    "kind": "evidence", "content": "external durable",
                    "incorporate": True, "line_eligible": False,
                },
                {
                    "kind": "decision", "content": "subjective durable",
                    "incorporate": True, "line_eligible": True,
                },
            ],
        },
    )
    assert committed.status_code == 200, committed.text
    assert len(committed.json()["memory_ids"]) == 2
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content,memory_domain,line_eligible FROM memories "
            "WHERE agent_id=%s AND role='summary' ORDER BY seq", (agent,),
        )
        assert cur.fetchall() == [
            ("external durable", "external_evidence", False),
            ("subjective durable", "subjective_trace", True),
        ]
        cur.execute(
            "SELECT metadata FROM cognitive_actions WHERE agent_id=%s", (agent,)
        )
        metadata = cur.fetchone()[0]
        assert metadata["api_key"] == "[REDACTED]"
        assert metadata["nested"][0]["authorization"] == "[REDACTED]"

    preturn = client.post(
        "/cognition/preturn",
        json={"agent_id": agent, "query": "durable", "messages": []},
    )
    assert preturn.status_code == 200, preturn.text
    assert preturn.json()["will_projection"]["source_count"] == 1
    assert [item["content"] for item in preturn.json()["reconstruction"]["traces"]] == [
        "subjective durable"
    ]


def test_preturn_current_event_rebuilds_posture_without_changing_will(stack) -> None:
    client, _ = stack
    agent = f"wave37-posture-{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent, "session_id": session_id}
    ).status_code == 200
    registry.get(agent).core._embedding = _Embedding()
    before = client.post(
        "/cognition/preturn",
        json={"agent_id": agent, "query": "continue", "messages": []},
    ).json()
    corrected = client.post(
        "/cognition/preturn",
        json={
            "agent_id": agent,
            "query": "Это неверно, исправь и проверь точно",
            "messages": [],
            "extra": {"constraints": "do not publish", "conflicts": "a vs b"},
        },
    )
    assert corrected.status_code == 200, corrected.text
    body = corrected.json()
    policy = body["affect"]["cognitive_posture"]
    assert policy["verification_depth"] == "high"
    assert policy["constraint_priority"] == "explicit_first"
    assert policy["ambiguity_handling"] == "surface_before_commit"
    assert body["will_projection"] == before["will_projection"]

    host_shaped = client.post(
        "/cognition/preturn",
        json={
            "agent_id": agent,
            "host_key": "host-shaped-turn",
            "query": "continue",
            "messages": [],
            "extra": {
                "task": "flat fallback",
                "current_event": json.dumps({
                    "task": "nested task",
                    "constraints": "do not publish",
                    "conflicts": "contract mismatch",
                    "risk": "data loss",
                }),
            },
        },
    )
    assert host_shaped.status_code == 200, host_shaped.text
    host_policy = host_shaped.json()["affect"]["cognitive_posture"]
    assert host_policy["verification_depth"] == "high"
    assert host_policy["constraint_priority"] == "explicit_first"
    assert host_policy["ambiguity_handling"] == "surface_before_commit"


def test_keyed_preturn_replays_exact_envelope_and_rejects_stale_reuse(stack) -> None:
    client, dsn = stack
    agent = f"wave37-replay-{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent, "session_id": session_id}
    ).status_code == 200
    registry.get(agent).core._embedding = _Embedding()

    assert client.post("/cognition/commit", json={
        "agent_id": agent,
        "host_key": "source",
        "consequences": [{"kind": "result", "content": "first evidence"}],
    }).status_code == 200
    request = {
        "agent_id": agent,
        "host_key": "turn-replay",
        "session_id": session_id,
        "query": "stable query",
        "messages": [{"role": "user", "content": "stable query"}],
    }
    first = client.post("/cognition/preturn", json=request)
    assert first.status_code == 200, first.text

    # Change line and pending state after the first snapshot. A retry must not
    # combine the old token with freshly recomputed will/affect/consequences.
    assert client.post("/cognition/commit", json={
        "agent_id": agent,
        "host_key": "independent",
        "consequences": [{
            "kind": "decision", "content": "later trace",
            "incorporate": True, "line_eligible": True,
        }],
    }).status_code == 200
    retry = client.post("/cognition/preturn", json=request)
    assert retry.status_code == 200, retry.text
    assert retry.json() == first.json()

    changed = client.post(
        "/cognition/preturn", json={**request, "query": "different query"}
    )
    assert changed.status_code == 409
    assert "different preturn request" in changed.json()["detail"]

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE cognitive_snapshots "
            "SET lease_expires_at=clock_timestamp()-interval '1 second' "
            "WHERE agent_id=%s AND host_key='turn-replay'",
            (agent,),
        )
        conn.commit()
    expired = client.post("/cognition/preturn", json=request)
    assert expired.status_code == 409
    assert "expired" in expired.json()["detail"]


def test_same_host_session_is_namespaced_per_agent(stack) -> None:
    client, dsn = stack
    shared_session = str(uuid.uuid4())
    agents = [f"wave37-session-a-{uuid.uuid4()}", f"wave37-session-b-{uuid.uuid4()}"]
    for index, agent in enumerate(agents):
        assert client.post(
            "/context/bootstrap", json={"agent_id": agent, "session_id": shared_session}
        ).status_code == 200
        response = client.post(
            "/cognition/commit",
            json={
                "agent_id": agent, "host_key": f"turn-{index}",
                "session_id": shared_session,
            },
        )
        assert response.status_code == 200, response.text

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT agent_id,session_id FROM cognitive_acts "
            "WHERE agent_id=ANY(%s) ORDER BY agent_id", (agents,),
        )
        rows = cur.fetchall()
        assert len(rows) == 2
        assert rows[0][1] == rows[1][1] == uuid.UUID(shared_session)
        cur.execute(
            "SELECT count(*) FROM cognitive_acts a JOIN sessions s "
            "ON (s.id,s.agent_id)=(a.session_id,a.agent_id) "
            "WHERE a.agent_id=ANY(%s)", (agents,),
        )
        assert cur.fetchone()[0] == 2


def test_openclaw_unkeyed_preturn_replays_and_commit_claims_same_session(stack) -> None:
    client, dsn = stack
    agent = f"wave37-openclaw-{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    other_session = str(uuid.uuid4())
    for current in (session_id, other_session):
        assert client.post(
            "/context/bootstrap", json={"agent_id": agent, "session_id": current}
        ).status_code == 200
    registry.get(agent).core._embedding = _Embedding()

    request = {
        "agent_id": agent,
        "session_id": session_id,
        "messages": [{"role": "user", "content": "stable accepted prompt"}],
        "query": "stable accepted prompt",
        "platform": "openclaw",
    }
    first = client.post("/cognition/preturn", json=request)
    replay = client.post("/cognition/preturn", json=request)
    assert first.status_code == replay.status_code == 200
    assert replay.json() == first.json()

    other = client.post(
        "/cognition/preturn", json={**request, "session_id": other_session}
    )
    assert other.status_code == 200
    assert other.json()["snapshot_token"] != first.json()["snapshot_token"]

    committed = client.post("/cognition/commit", json={
        "agent_id": agent,
        "host_key": "openclaw:logical-turn",
        "session_id": session_id,
        "snapshot_policy": "latest_session",
        "parent_policy": "latest_session",
        "user_message": "stable accepted prompt",
        "assistant_response": "accepted answer",
    })
    assert committed.status_code == 200, committed.text
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT input_snapshot_token FROM cognitive_acts "
            "WHERE agent_id=%s AND host_key='openclaw:logical-turn'", (agent,),
        )
        assert cur.fetchone()[0] == first.json()["snapshot_token"]
        cur.execute(
            "SELECT used_by_act_id FROM cognitive_snapshots WHERE token=%s",
            (other.json()["snapshot_token"],),
        )
        assert cur.fetchone()[0] is None


def test_commit_preserves_explicit_and_every_tool_result_error(stack) -> None:
    client, dsn = stack
    agent = f"wave37-journal-{uuid.uuid4()}"
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent, "session_id": str(uuid.uuid4())}
    ).status_code == 200
    payload = {
        "agent_id": agent,
        "host_key": "max-combined",
        "tool_events": [
            {
                "kind": "result" if index % 2 == 0 else "error",
                "tool_event_id": f"tool-{index}",
                "content": f"tool consequence {index}",
            }
            for index in range(64)
        ],
        "consequences": [
            {"kind": "explicit", "content": f"explicit consequence {index}"}
            for index in range(32)
        ],
    }
    response = client.post("/cognition/commit", json=payload)
    assert response.status_code == 200, response.text
    assert len(response.json()["consequence_ids"]) == 96
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*),count(*) FILTER (WHERE kind IN ('tool_result','tool_error')) "
            "FROM cognitive_consequences WHERE agent_id=%s",
            (agent,),
        )
        assert cur.fetchone() == (96, 64)
