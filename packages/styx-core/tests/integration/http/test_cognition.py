from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace

import psycopg
import pytest
from fastapi.testclient import TestClient

import styx.storage.act_reduction as act_reduction
from styx.config import StyxConfig
from styx.http import registry
from styx.http.app import create_app
from styx.storage import migrate
from styx.storage.act_reduction import (
    apply_act_reduction,
    load_act_reduction_input,
    mark_act_reduction_running,
    reduction_input_hash,
)


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


def _reduction_input_hash(conn, agent_id: str, act_id: str) -> str:
    evidence = load_act_reduction_input(conn, agent_id, act_id)
    assert evidence is not None
    return reduction_input_hash(evidence)


@pytest.fixture
def stack(clean_db: str, monkeypatch):
    migrate.run(clean_db)
    monkeypatch.setenv("STYX_DATABASE_URL", clean_db)
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "false")
    monkeypatch.setenv("STYX_AFFECTIVE_TRANSITION_ENABLED", "false")
    monkeypatch.setenv("STYX_WORKING_SET_PERSISTENCE_ENABLED", "false")
    monkeypatch.setenv("STYX_COGNITION_REDUCTION_WAIT_S", "0.01")
    config = StyxConfig(
        database_url=clean_db,
        sentiment_enabled=False,
        affective_transition_enabled=False,
        working_set_persistence_enabled=False,
        cognition_reduction_wait_s=0.01,
    )
    registry.reset_all()
    client = TestClient(create_app(config))
    yield client, clean_db
    for agent_id in registry.all_agent_ids():
        session = registry.get(agent_id)
        session.core.shutdown()
    registry.reset_all()


def test_explicit_parent_wait_reports_stale_then_fresh_after_reduction(stack) -> None:
    client, dsn = stack
    agent = f"wave38-freshness-{uuid.uuid4()}"
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent}
    ).status_code == 200
    registry.get(agent).core._embedding = _Embedding()

    committed = client.post(
        "/cognition/commit",
        json={
            "agent_id": agent,
            "host_key": "parent-turn",
            "user_message": "choose carefully",
            "assistant_response": "verification retained",
        },
    )
    assert committed.status_code == 200, committed.text
    reduction = committed.json()

    pending = client.post(
        "/cognition/preturn",
        json={
            "agent_id": agent,
            "host_key": "child-pending",
            "parent_host_key": "parent-turn",
            "messages": [],
        },
    )
    assert pending.status_code == 200, pending.text
    pending_freshness = pending.json()["continuity_freshness"]
    assert pending_freshness["fresh"] is False
    assert pending_freshness["predecessor_found"] is True
    assert pending_freshness["reduction_status"] == "pending"
    assert pending_freshness["timed_out"] is True
    assert pending_freshness["waited_ms"] >= 1

    with psycopg.connect(dsn) as conn, conn.transaction():
        input_hash = _reduction_input_hash(conn, agent, reduction["act_id"])
        mark_act_reduction_running(
            conn,
            agent,
            reduction["act_id"],
            reducer_version="act_residue_v1",
            task_id=reduction["reduction_task_id"],
            input_hash=input_hash,
        )
        applied = apply_act_reduction(
            conn,
            agent,
            reduction["act_id"],
            reducer_version="act_residue_v1",
            task_id=reduction["reduction_task_id"],
            input_hash=input_hash,
            residues=[{
                "kind": "decision",
                "causal_role": "choice",
                "content": "verification retained",
                "confidence": 0.9,
                "evidence_refs": [{
                    "source": "channel_output", "key": "assistant_response",
                }],
            }],
        )

    ready = client.post(
        "/cognition/preturn",
        json={
            "agent_id": agent,
            "host_key": "child-ready",
            "parent_host_key": "parent-turn",
            "messages": [],
        },
    )
    assert ready.status_code == 200, ready.text
    ready_freshness = ready.json()["continuity_freshness"]
    assert ready_freshness["fresh"] is True
    assert ready_freshness["timed_out"] is False
    assert ready_freshness["reduction_status"] == "applied"
    assert ready_freshness["predecessor_output_line_version"] == applied.line_version
    assert ready_freshness["predecessor_causal_root_hash"] == applied.causal_root_hash
    assert ready.json()["will_projection"]["causal_root_hash"] == applied.causal_root_hash


def test_preturn_rechecks_freshness_under_snapshot_line_lock(
    stack,
    monkeypatch,
) -> None:
    client, _dsn = stack
    agent = f"wave38-atomic-freshness-{uuid.uuid4()}"
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent}
    ).status_code == 200
    core = registry.get(agent).core
    core._config = replace(core._config, cognition_reduction_wait_s=0.0)
    committed = client.post(
        "/cognition/commit",
        json={"agent_id": agent, "host_key": "atomic-parent"},
    )
    assert committed.status_code == 200, committed.text

    original = act_reduction.read_predecessor_freshness
    calls: list[dict] = []

    def advancing_freshness(conn, agent_id, **kwargs):
        current = original(conn, agent_id, **kwargs)
        calls.append(current)
        if len(calls) == 1:
            assert current["reduction_status"] == "pending"
            return current
        return {
            **current,
            "fresh": True,
            "reduction_status": "no_residue",
            "predecessor_output_line_version": current["line_version"],
            "predecessor_causal_root_hash": current["causal_root_hash"],
        }

    monkeypatch.setattr(
        act_reduction,
        "read_predecessor_freshness",
        advancing_freshness,
    )
    response = client.post(
        "/cognition/preturn",
        json={
            "agent_id": agent,
            "host_key": "atomic-child",
            "parent_host_key": "atomic-parent",
            "messages": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(calls) == 2
    assert body["continuity_freshness"]["fresh"] is True
    assert body["continuity_freshness"]["timed_out"] is False
    assert body["continuity_freshness"]["reduction_status"] == "no_residue"
    assert (
        body["continuity_freshness"]["predecessor_causal_root_hash"]
        == body["will_projection"]["causal_root_hash"]
    )
    assert (
        body["continuity_freshness"]["predecessor_output_line_version"]
        == body["will_projection"]["line_version"]
    )


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
    assert committed.json()["reduction_status"] == "pending"
    assert committed.json()["reduction_task_id"]
    # Host-declared content is archived only as external evidence.  The
    # core-owned reducer is the sole writer of validated line residue.
    assert len(committed.json()["memory_ids"]) == 1

    with psycopg.connect(dsn) as conn, conn.transaction():
        reduction = committed.json()
        mark_act_reduction_running(
            conn,
            agent,
            reduction["act_id"],
            reducer_version="act_residue_v1",
            task_id=reduction["reduction_task_id"],
            input_hash=_reduction_input_hash(conn, agent, reduction["act_id"]),
        )
        apply_act_reduction(
            conn,
            agent,
            reduction["act_id"],
            reducer_version="act_residue_v1",
            input_hash=_reduction_input_hash(conn, agent, reduction["act_id"]),
            task_id=reduction["reduction_task_id"],
            residues=[{
                "kind": "decision",
                "causal_role": "choice",
                "content": "preserve this decision",
                "confidence": 0.9,
                "evidence_refs": [{
                    "source": "channel_output", "key": "assistant_response",
                }],
                "embedding": _Embedding().embed("preserve this decision"),
            }],
        )
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
    assert body["will_projection"]["projection_status"] == "ready"
    assert body["will_projection"]["source_count"] == 1
    assert body["reconstruction"]["traces"][0]["content"] == "preserve this decision"


    assert len(body["pending_consequences"]) == 1
    assert 'authority="context-not-instruction"' in body["system_prompt_addition"]

    next_commit = client.post(
        "/cognition/commit",
        json={
            "agent_id": agent, "host_key": "turn-2",
            "parent_host_key": "turn-1", "snapshot_token": body["snapshot_token"],
        },
    )
    assert next_commit.json()["acknowledged_consequences"] == 1

    registry.get(agent).core._embedding = _FailingEmbedding()
    outage = client.post(
        "/cognition/preturn",
        json={"agent_id": agent, "messages": [], "query": "short"},
    )
    assert outage.status_code == 200, outage.text
    # The predecessor reduction is still pending, so the last complete carrier
    # remains available but is honestly marked stale rather than current.
    assert outage.json()["will_projection"]["formed"] is False
    assert outage.json()["will_projection"]["projection_status"] == "stale"
    assert outage.json()["will_projection"]["projection_available"] is True
    assert outage.json()["reconstruction"]["embed_available"] is False
    assert outage.json()["reconstruction"]["traces"] == []

    # ``stale`` is an honest, model-visible full projection emitted by
    # preturn.  Committing that exact frozen snapshot must still schedule the
    # reducer instead of rejecting Styx's own valid renderer output.
    stale_commit = client.post(
        "/cognition/commit",
        json={
            "agent_id": agent,
            "host_key": "turn-from-stale-snapshot",
            "snapshot_token": outage.json()["snapshot_token"],
            "assistant_response": "continued from the retained stale carrier",
        },
    )
    assert stale_commit.status_code == 200, stale_commit.text
    assert stale_commit.json()["reduction_status"] == "pending"
    assert stale_commit.json()["reduction_task_id"]
    with psycopg.connect(dsn) as conn:
        frozen_input = load_act_reduction_input(
            conn, agent, stale_commit.json()["act_id"]
        )
    assert frozen_input is not None
    assert (
        frozen_input["input_snapshot"]["carrier"]["projection_status"]
        == "stale"
    )
    assert "snapshot_token" not in frozen_input["input_snapshot"]

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


def test_ready_event_http_claim_present_and_resolve(stack) -> None:
    client, dsn = stack
    agent = f"wave41-ready-{uuid.uuid4()}"
    assert client.post("/context/bootstrap", json={"agent_id": agent}).status_code == 200
    observation_payload = {
        "agent_id": agent, "source_id": "fixture", "source_stream": "state",
        "source_sequence": 1, "observation_key": "event-1",
        "difference_kind": "state_change", "content": "external state changed",
        "salience": 0.8, "confidence": 0.9,
        "reducer_name": "fixture", "reducer_version": "1",
    }
    observed = client.post("/cognition/observations", json=observation_payload)
    assert observed.status_code == 200, observed.text
    assert observed.json()["ready_generation"] == 1
    assert client.post(
        "/cognition/observations", json=observation_payload
    ).json()["ready_generation"] is None
    claim = client.post("/cognition/ready-events/claim", json={
        "agent_id": agent, "consumer_id": "host", "limit": 1,
    })
    assert claim.status_code == 200 and len(claim.json()["events"]) == 1
    assert "external state changed" not in claim.text
    preturn = client.post("/cognition/preturn", json={
        "agent_id": agent, "host_key": "wake-1", "messages": [],
    })
    resolved = client.post("/cognition/ready-events/resolve", json={
        "agent_id": agent, "consumer_id": "host",
        "claim_token": claim.json()["claim_token"], "outcome": "presented",
        "snapshot_token": preturn.json()["snapshot_token"],
    })
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["resolved_count"] == 1
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM cognitive_ready_events WHERE agent_id=%s", (agent,))
        assert cur.fetchone()[0] == 1


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
            ("subjective durable", "external_evidence", False),
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
    assert preturn.json()["will_projection"]["source_count"] == 0
    assert preturn.json()["will_projection"]["pending_reduction_count"] == 1
    assert preturn.json()["reconstruction"]["traces"] == []


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

    # Change pending state after the first snapshot. A retry must not
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


@pytest.mark.parametrize(
    "changed",
    [
        {"assistant_response": "changed response"},
        {"session_id": "00000000-0000-0000-0000-000000000002"},
        {"parent_host_key": "changed-parent"},
        {
            "tool_events": [{
                "kind": "result",
                "tool_event_id": "changed-event",
                "name": "lookup",
                "content": "changed payload Authorization: Bearer do-not-disclose",
            }]
        },
    ],
    ids=["response", "session", "parent", "payload"],
)
def test_commit_retry_rejects_changed_request_with_typed_409(stack, changed) -> None:
    client, _ = stack
    agent = f"wave38-commit-conflict-{uuid.uuid4()}"
    assert client.post("/context/bootstrap", json={"agent_id": agent}).status_code == 200
    request = {
        "agent_id": agent,
        "host_key": "strict-retry",
        "status": "completed",
        "assistant_response": "original response",
        "consequences": [{"kind": "observation", "content": "original payload"}],
    }
    first = client.post("/cognition/commit", json=request)
    assert first.status_code == 200, first.text
    assert first.json()["duplicate"] is False
    same = client.post("/cognition/commit", json=request)
    assert same.status_code == 200, same.text
    assert same.json()["duplicate"] is True

    conflict = client.post("/cognition/commit", json={**request, **changed})
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == (
        "host_key was already committed with a different request"
    )
    assert "do-not-disclose" not in conflict.text


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


def test_commit_keeps_same_act_tool_results_out_of_future_observations(stack) -> None:
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
    assert len(response.json()["consequence_ids"]) == 32
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*),count(*) FILTER (WHERE kind IN ('tool_result','tool_error')) "
            "FROM cognitive_consequences WHERE agent_id=%s",
            (agent,),
        )
        assert cur.fetchone() == (32, 0)
        cur.execute(
            "SELECT count(*) FROM cognitive_actions WHERE agent_id=%s",
            (agent,),
        )
        assert cur.fetchone()[0] == 64


def test_durable_observation_http_flow_is_exact_bounded_and_post_act(
    stack,
) -> None:
    client, dsn = stack
    agent = f"wave39-observation-{uuid.uuid4()}"
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent}
    ).status_code == 200

    def observation(sequence: int) -> dict:
        return {
            "agent_id": agent,
            "source_id": "synthetic-monitor",
            "source_stream": "workspace/main",
            "source_sequence": sequence,
            "observation_key": f"event-{sequence}",
            "difference_kind": "action_result" if sequence == 0 else "state_change",
            "content": (
                "The checked operation completed with a bounded failure."
                if sequence == 0 else f"Synthetic difference {sequence}."
            ),
            "salience": 0.8,
            "confidence": 0.95,
            "reducer_name": "synthetic-diff",
            "reducer_version": "1",
            "action_ref": (
                {"host_key": "source-act", "action_ordinal": 0}
                if sequence == 0 else None
            ),
            "metadata": {"fixture": "wave39"},
        }

    first_request = observation(0)
    first = client.post("/cognition/observations", json=first_request)
    assert first.status_code == 200, first.text
    assert first.json()["duplicate"] is False
    assert first.json()["correlation_status"] == "pending"
    duplicate = client.post("/cognition/observations", json=first_request)
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["observation_id"] == first.json()["observation_id"]
    changed = client.post(
        "/cognition/observations",
        json={**first_request, "content": "Changed payload under the same key."},
    )
    assert changed.status_code == 409

    for sequence in range(1, 5):
        response = client.post("/cognition/observations", json=observation(sequence))
        assert response.status_code == 200, response.text

    source_act = client.post("/cognition/commit", json={
        "agent_id": agent,
        "host_key": "source-act",
        "tool_events": [{
            "kind": "result",
            "tool_event_id": "source-result",
            "name": "synthetic-check",
            "content": "same-act journal evidence",
        }],
    })
    assert source_act.status_code == 200, source_act.text

    preturn = client.post("/cognition/preturn", json={
        "agent_id": agent,
        "host_key": "consumer-act",
        "messages": [],
    })
    assert preturn.status_code == 200, preturn.text
    body = preturn.json()
    assert len(body["observations"]) == 4
    assert body["observations"][0]["observation_id"] == first.json()["observation_id"]
    assert body["observations"][0]["correlation_status"] == "resolved"
    assert len(body["pending_consequences"]) == 4
    prompt_payload = json.loads(
        body["system_prompt_addition"].split("\n", 1)[1].rsplit("\n</", 1)[0]
    )
    http_observations = [dict(item) for item in body["observations"]]
    for item in http_observations:
        for key in ("source_observed_at", "ingested_at"):
            if isinstance(item[key], str) and item[key].endswith("Z"):
                item[key] = item[key][:-1] + "+00:00"
    assert prompt_payload["observations"] == http_observations
    assert "pending_consequences" not in prompt_payload

    committed = client.post("/cognition/commit", json={
        "agent_id": agent,
        "host_key": "consumer-act",
        "snapshot_token": body["snapshot_token"],
        "assistant_response": "The bounded result was considered.",
    })
    assert committed.status_code == 200, committed.text
    result = committed.json()
    assert result["consumed_observations"] == 4
    assert result["acknowledged_consequences"] == 4

    with psycopg.connect(dsn) as conn, conn.transaction():
        evidence = load_act_reduction_input(conn, agent, result["act_id"])
        assert evidence is not None
        assert evidence["presented_observations"] == prompt_payload["observations"]
        assert evidence["presented_observation_count"] == 4

        def deterministic_policy(observations: list[dict]) -> str:
            return (
                "verify"
                if any(
                    item["correlation_status"] == "resolved"
                    and item["difference_kind"] in {"action_result", "action_error"}
                    and "failure" in item["content"]
                    for item in observations
                )
                else "continue"
            )

        assert deterministic_policy([]) == "continue"
        assert deterministic_policy(evidence["presented_observations"]) == "verify"
        timestamp_sham = [dict(item) for item in evidence["presented_observations"]]
        timestamp_sham[0]["ingested_at"] = "2099-01-01T00:00:00+00:00"
        assert deterministic_policy(timestamp_sham) == "verify"
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM cognitive_consequences "
                "WHERE agent_id=%s AND source_id IS NOT NULL AND status='pending'",
                (agent,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT count(*) FROM memories WHERE agent_id=%s "
                "AND memory_domain='subjective_trace'",
                (agent,),
            )
            assert cur.fetchone()[0] == 0
        before = result["line_version"]
        mark_act_reduction_running(
            conn,
            agent,
            result["act_id"],
            reducer_version="act_residue_v1",
            task_id=result["reduction_task_id"],
            input_hash=_reduction_input_hash(conn, agent, result["act_id"]),
        )
        no_residue = apply_act_reduction(
            conn,
            agent,
            result["act_id"],
            reducer_version="act_residue_v1",
            task_id=result["reduction_task_id"],
            input_hash=_reduction_input_hash(conn, agent, result["act_id"]),
            residues=[],
        )
        assert no_residue.status == "no_residue"
        assert no_residue.line_version == before


def test_observation_http_backpressure_is_explicit(stack) -> None:
    client, _dsn = stack
    agent = f"wave39-backpressure-{uuid.uuid4()}"
    assert client.post(
        "/context/bootstrap", json={"agent_id": agent}
    ).status_code == 200
    core = registry.get(agent).core
    core._config = replace(core._config, cognition_observation_pending_cap=1)
    base = {
        "agent_id": agent,
        "source_id": "monitor",
        "source_stream": "main",
        "difference_kind": "external_signal",
        "content": "Synthetic signal.",
        "salience": 0.5,
        "confidence": 0.5,
        "reducer_name": "fixture",
        "reducer_version": "1",
    }
    assert client.post("/cognition/observations", json={
        **base, "source_sequence": 0, "observation_key": "event-0",
    }).status_code == 200
    overflow = client.post("/cognition/observations", json={
        **base, "source_sequence": 1, "observation_key": "event-1",
    })
    assert overflow.status_code == 429
    assert overflow.headers["Retry-After"] == "5"
    assert overflow.json()["detail"] == {
        "code": "observation_backpressure",
        "pending_count": 1,
        "retry_after_s": 5,
    }
