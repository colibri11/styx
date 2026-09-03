"""Unit-тесты StyxCoreClient — без живого daemon, через mock requests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from styx_hermes.client import StyxCoreClient


@pytest.fixture
def client_no_token(monkeypatch: pytest.MonkeyPatch) -> StyxCoreClient:
    # В Docker окружении STYX_HTTP_TOKEN может быть в env — тест должен
    # явно сбросить чтобы проверить путь "ни token, ни env".
    monkeypatch.delenv("STYX_HTTP_TOKEN", raising=False)
    monkeypatch.delenv("STYX_SOCIAL_TOKEN", raising=False)
    return StyxCoreClient(base_url="http://daemon.local:8788", token=None)


@pytest.fixture
def client_with_token(monkeypatch: pytest.MonkeyPatch) -> StyxCoreClient:
    monkeypatch.delenv("STYX_HTTP_TOKEN", raising=False)
    monkeypatch.delenv("STYX_SOCIAL_TOKEN", raising=False)
    return StyxCoreClient(
        base_url="http://daemon.local:8788", token="test-token-12345"
    )


@pytest.fixture
def client_with_social_token(monkeypatch: pytest.MonkeyPatch) -> StyxCoreClient:
    monkeypatch.delenv("STYX_HTTP_TOKEN", raising=False)
    monkeypatch.delenv("STYX_SOCIAL_TOKEN", raising=False)
    return StyxCoreClient(
        base_url="http://daemon.local:8788",
        token="ordinary-token",
        social_token="social-principal-token",
    )


def _mock_response(status: int = 200, json_payload: dict | None = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = json_payload or {}
    resp.text = ""
    if 200 <= status < 300:
        resp.raise_for_status = MagicMock()
    else:
        resp.raise_for_status = MagicMock(side_effect=requests.HTTPError())
    return resp


def test_base_url_strips_trailing_slash() -> None:
    c = StyxCoreClient(base_url="http://a:1234/", token=None)
    assert c.base_url == "http://a:1234"


def test_no_token_no_auth_header(client_no_token: StyxCoreClient) -> None:
    assert "Authorization" not in client_no_token._session.headers


def test_token_sets_auth_header(client_with_token: StyxCoreClient) -> None:
    assert (
        client_with_token._session.headers.get("Authorization")
        == "Bearer test-token-12345"
    )


def test_initialize_agent_payload(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"agent_id": "agent-a", "tools": []}
        )
        resp = client_no_token.initialize_agent(
            "agent-a",
            session_id="sid-1",
            agent_identity="agent-a",
            platform="cli",
        )
        assert resp == {"agent_id": "agent-a", "tools": []}
        args, kwargs = mock_post.call_args
        assert args[0] == "http://daemon.local:8788/agent/initialize"
        assert kwargs["json"]["agent_id"] == "agent-a"
        assert kwargs["json"]["session_id"] == "sid-1"
        assert kwargs["json"]["agent_identity"] == "agent-a"


def test_shutdown_204_returns_empty(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(204, {})
        client_no_token.shutdown_agent("agent-a")
        mock_post.assert_called_once()


def test_sync_turn_payload(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"memory_ids": []})
        client_no_token.sync_turn(
            "agent-a",
            user_content="hi",
            assistant_content="hello",
            session_id="sid-x",
            idempotency_key="hermes:sid-x:turn-1",
        )
        args, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert body["agent_id"] == "agent-a"
        assert body["user_content"] == "hi"
        assert body["assistant_content"] == "hello"
        assert body["session_id"] == "sid-x"
        assert body["idempotency_key"] == "hermes:sid-x:turn-1"


def test_recall_long_timeout(client_no_token: StyxCoreClient) -> None:
    """recall использует long_timeout (по умолчанию 30s)."""
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"memories": []})
        client_no_token.recall("agent-a", "query", limit=5)
        kwargs = mock_post.call_args.kwargs
        assert kwargs["timeout"] == client_no_token._long_timeout


def test_pre_llm_inject_returns_context_or_none(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"context": "Peer прозвучал: ..."})
        out = client_no_token.pre_llm_inject(
            "agent-a", session_id="sid", user_message="hi"
        )
        assert out["context"] == "Peer прозвучал: ..."

        mock_post.return_value = _mock_response(200, {"context": None})
        out2 = client_no_token.pre_llm_inject("agent-a")
        assert out2["context"] is None


def test_observe_affective_turn_payload(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"accepted": False, "duplicate": False}
        )
        out = client_no_token.observe_affective_turn(
            "agent-a",
            idempotency_key="hermes:sid:turn-1",
            turn_id="turn-1",
            session_id="sid",
            user_message="current user",
            assistant_response="final answer",
            conversation_history=[{"role": "user", "content": "prior"}],
            tool_events=[
                {
                    "kind": "result",
                    "tool_call_id": "call-1",
                    "name": "read_file",
                    "content": "ok",
                }
            ],
            task_id="task-1",
            model="model-x",
            platform="cli",
        )

        assert out == {"accepted": False, "duplicate": False}
        args, kwargs = mock_post.call_args
        assert args[0] == "http://daemon.local:8788/affect/observe_turn"
        body = kwargs["json"]
        assert kwargs["timeout"] == client_no_token._affect_timeout
        assert client_no_token._timeout < client_no_token._affect_timeout < 30.0
        assert body["agent_id"] == "agent-a"
        assert body["idempotency_key"] == "hermes:sid:turn-1"
        assert body["turn_id"] == "turn-1"
        assert body["conversation_history"][0]["content"] == "prior"
        assert body["tool_events"][0]["tool_call_id"] == "call-1"


def test_cognition_preturn_payload_and_long_timeout(
    client_no_token: StyxCoreClient,
) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200,
            {
                "messages": [],
                "line_version": 4,
                "snapshot_token": "snap",
                "system_prompt_addition": "<styx-continuity />",
            },
        )
        out = client_no_token.cognition_preturn(
            "agent-a",
            host_key="hermes:sid:turn-1",
            session_id="sid",
            messages=[{"role": "user", "content": "hi"}],
            query="hi",
            model="m",
            platform="hermes",
            extra={"current_event": {"is_first_turn": True}},
        )
        assert out["snapshot_token"] == "snap"
        args, kwargs = mock_post.call_args
        assert args[0] == "http://daemon.local:8788/cognition/preturn"
        assert kwargs["timeout"] == client_no_token._long_timeout
        assert kwargs["json"]["host_key"] == "hermes:sid:turn-1"
        assert kwargs["json"]["query"] == "hi"
        assert kwargs["json"]["extra"]["current_event"] == {
            "is_first_turn": True
        }


def test_cognition_preturn_none_messages_sends_empty_array(
    client_no_token: StyxCoreClient,
) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"snapshot_token": "snap"})
        client_no_token.cognition_preturn("agent-a", messages=None)
        assert mock_post.call_args.kwargs["json"]["messages"] == []


def test_cognition_commit_payload_and_terminal_timeout(
    client_no_token: StyxCoreClient,
) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"committed": True, "duplicate": False}
        )
        out = client_no_token.cognition_commit(
            "agent-a",
            session_id="sid",
            host_key="hermes:sid:turn-1",
            parent_host_key="hermes:sid:turn-0",
            snapshot_token="snap",
            status="completed",
            user_message="hi",
            assistant_response="hello",
            tool_events=[{
                "kind": "result", "tool_event_id": "call-1",
                "name": "read", "content": "ok", "metadata": {},
            }],
            consequences=[{
                "kind": "observation", "content": "confirmed",
                "incorporate": True, "line_eligible": True,
                "metadata": {},
            }],
        )
        assert out["committed"] is True
        args, kwargs = mock_post.call_args
        assert args[0] == "http://daemon.local:8788/cognition/commit"
        assert kwargs["timeout"] == client_no_token._affect_timeout
        assert kwargs["json"]["parent_host_key"] == "hermes:sid:turn-0"
        assert kwargs["json"]["tool_events"][0]["tool_event_id"] == "call-1"
        assert kwargs["json"]["consequences"][0]["line_eligible"] is True


def test_cognition_observe_is_explicit_and_uses_long_timeout(
    client_no_token: StyxCoreClient,
) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"observation_id": "obs-1", "duplicate": False}
        )
        out = client_no_token.cognition_observe(
            "agent-a",
            source_id="monitor",
            source_stream="main",
            source_sequence=7,
            observation_key="event-7",
            difference_kind="state_change",
            content="The monitored state changed.",
            salience=0.8,
            confidence=0.9,
            reducer_name="monitor-diff",
            reducer_version="1",
            action_ref={"host_key": "turn-1", "action_ordinal": 0},
        )
        assert out["observation_id"] == "obs-1"
        args, kwargs = mock_post.call_args
        assert args[0] == "http://daemon.local:8788/cognition/observations"
        assert kwargs["timeout"] == client_no_token._long_timeout
        assert kwargs["json"]["source_sequence"] == 7
        assert kwargs["json"]["action_ref"] == {
            "host_key": "turn-1", "action_ordinal": 0,
        }


def test_ready_event_claim_and_resolve_are_explicit_host_primitives(
    client_no_token: StyxCoreClient,
) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"events": [], "claim_token": None})
        client_no_token.cognition_ready_claim(
            "agent-a", consumer_id="supervisor", after_generation=7, wait_ms=500
        )
        args, kwargs = mock_post.call_args
        assert args[0].endswith("/cognition/ready-events/claim")
        assert kwargs["json"]["consumer_id"] == "supervisor"
        assert kwargs["json"]["after_generation"] == 7

        mock_post.return_value = _mock_response(
            200, {"resolved_count": 1, "outcome": "deferred", "redelivered": False}
        )
        client_no_token.cognition_ready_resolve(
            "agent-a", consumer_id="supervisor", claim_token="claim",
            outcome="deferred",
        )
        args, kwargs = mock_post.call_args
        assert args[0].endswith("/cognition/ready-events/resolve")
        assert kwargs["json"]["outcome"] == "deferred"


def test_social_token_is_per_request_and_never_reuses_http_bearer(
    client_with_social_token: StyxCoreClient,
) -> None:
    with patch.object(client_with_social_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"actor_id": "actor-1"})
        client_with_social_token.social_create_actor(
            "agent-a",
            identity_namespace="local",
            actor_key="actor-a",
            actor_kind="local_agent",
            identity_evidence_hash="a" * 64,
        )
        social_call = mock_post.call_args
        assert social_call.args[0].endswith("/social/actors")
        assert social_call.kwargs["headers"] == {
            "X-Styx-Social-Token": "social-principal-token"
        }
        assert client_with_social_token._session.headers["Authorization"] == (
            "Bearer ordinary-token"
        )

        client_with_social_token.sync_turn("agent-a")
        ordinary_call = mock_post.call_args
        assert ordinary_call.kwargs["headers"] is None


def test_social_methods_are_explicit_and_match_route_payloads(
    client_with_social_token: StyxCoreClient,
) -> None:
    scope_id = "11111111-1111-4111-8111-111111111111"
    issuer_id = "22222222-2222-4222-8222-222222222222"
    subject_id = "33333333-3333-4333-8333-333333333333"
    source_act_id = "44444444-4444-4444-8444-444444444444"
    prior_attestation_id = "55555555-5555-4555-8555-555555555555"
    attestation_id = "66666666-6666-4666-8666-666666666666"
    with patch.object(client_with_social_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"ok": True})
        client_with_social_token.social_create_actor(
            "agent-a",
            identity_namespace="workspace",
            actor_key="peer-a",
            actor_kind="external_agent",
            identity_evidence_hash="a" * 64,
            private_label="private label",
            attestation_principal_id="principal-a",
        )
        client_with_social_token.social_create_scope(
            "agent-a",
            scope_key="scope-a",
            protocol_id="protocol-a",
            protocol_version="1",
            policy_hash="b" * 64,
        )
        client_with_social_token.social_record_encounter(
            "agent-a",
            encounter_key="encounter-a",
            scope_id=scope_id,
            observer_actor_id=issuer_id,
            encountered_actor_id=subject_id,
            direction="inbound",
            channel_kind="hermes",
            source_act_id=source_act_id,
            evidence_hash="c" * 64,
            confidence=0.8,
            summary="bounded encounter",
        )
        client_with_social_token.social_attest(
            "agent-a",
            scope_id=scope_id,
            issuer_actor_id=issuer_id,
            subject_actor_id=subject_id,
            attestation_key="attest-a",
            verdict="positive",
            protocol_id="protocol-a",
            protocol_version="1",
            source_act_id=source_act_id,
            source_action_ordinal=2,
            evidence_refs=[{"source": "action", "ordinal": 2}],
            trust_level="verified",
            signature_metadata={"algorithm": "test"},
        )
        client_with_social_token.social_revise_attestation(
            "agent-a",
            supersedes_attestation_id=prior_attestation_id,
            scope_id=scope_id,
            issuer_actor_id=issuer_id,
            subject_actor_id=subject_id,
            attestation_key="attest-a-revision",
            verdict="undetermined",
            protocol_id="protocol-a",
            protocol_version="1",
            source_act_id=source_act_id,
            trust_level="unverified",
        )
        client_with_social_token.social_dissolve_scope(
            "agent-a", scope_id=scope_id
        )
        client_with_social_token.social_create_grant(
            "agent-a",
            grant_key="grant-a",
            scope_id=scope_id,
            grantee_principal_id="principal-b",
            capability="social:read",
            evidence_class="attestation",
            evidence_id=attestation_id,
            expires_at="2026-09-03T00:00:00+00:00",
        )
        client_with_social_token.social_revoke_grant(
            "agent-a",
            revocation_key="grant-a-revoke",
            grant_id=attestation_id,
        )
        client_with_social_token.social_query(
            "agent-a",
            scope_id=scope_id,
            actor_a_id=issuer_id,
            actor_b_id=subject_id,
        )
        client_with_social_token.social_explain(
            "agent-a",
            scope_id=scope_id,
        )
        client_with_social_token.social_deliver(
            "agent-a",
            delivery_key="delivery-a",
            scope_id=scope_id,
            receiving_agent_id="agent-b",
            evidence_class="attestation",
            evidence_id=attestation_id,
        )

    calls = mock_post.call_args_list
    assert [call.args[0].removeprefix("http://daemon.local:8788") for call in calls] == [
        "/social/actors",
        "/social/scopes",
        "/social/encounters",
        "/social/attestations",
        "/social/attestations/revise",
        "/social/scopes/dissolve",
        "/social/grants",
        "/social/grants/revoke",
        "/social/query",
        "/social/explain",
        "/social/deliver",
    ]
    assert all(
        call.kwargs["headers"]["X-Styx-Social-Token"]
        == "social-principal-token"
        for call in calls
    )
    assert calls[2].kwargs["json"] == {
        "agent_id": "agent-a",
        "encounter_key": "encounter-a",
        "scope_id": scope_id,
        "observer_actor_id": issuer_id,
        "encountered_actor_id": subject_id,
        "direction": "inbound",
        "channel_kind": "hermes",
        "source_act_id": source_act_id,
        "source_observation_id": None,
        "summary": "bounded encounter",
        "evidence_hash": "c" * 64,
        "confidence": 0.8,
    }
    signed_body = json.loads(calls[3].kwargs["data"])
    assert signed_body["supersedes_attestation_id"] is None
    assert len(calls[3].kwargs["headers"]["X-Styx-Social-Signature"]) == 64
    assert calls[4].kwargs["json"]["supersedes_attestation_id"] == (
        prior_attestation_id
    )
    assert calls[8].kwargs["json"] == {
        "agent_id": "agent-a",
        "scope_id": scope_id,
        "actor_a_id": issuer_id,
        "actor_b_id": subject_id,
    }
    assert calls[9].kwargs["json"] == {
        "agent_id": "agent-a",
        "scope_id": scope_id,
    }
    assert calls[10].kwargs["json"] == {
        "agent_id": "agent-a",
        "delivery_key": "delivery-a",
        "scope_id": scope_id,
        "evidence_class": "attestation",
        "evidence_id": attestation_id,
        "receiving_agent_id": "agent-b",
    }
    forbidden = {"is_conscious", "is_person", "personality"}
    payloads = [
        call.kwargs.get("json") or json.loads(call.kwargs["data"])
        for call in calls
    ]
    assert all(not (forbidden & set(payload)) for payload in payloads)


def test_social_call_without_social_token_sends_no_principal_header(
    client_with_token: StyxCoreClient,
) -> None:
    with patch.object(client_with_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(404, {"detail": "not found"})
        with pytest.raises(requests.HTTPError):
            client_with_token.social_query(
                "agent-a",
                scope_id="11111111-1111-4111-8111-111111111111",
                actor_a_id="22222222-2222-4222-8222-222222222222",
                actor_b_id="33333333-3333-4333-8333-333333333333",
            )
        assert mock_post.call_args.kwargs["headers"] is None
        assert client_with_token._session.headers["Authorization"] == (
            "Bearer test-token-12345"
        )


def test_5xx_raises(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(503, {"detail": "down"})
        with pytest.raises(requests.HTTPError):
            client_no_token.sync_turn("agent-a")


def test_401_raises(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(401, {"detail": "missing bearer"})
        with pytest.raises(requests.HTTPError):
            client_no_token.sync_turn("agent-a")


def test_close_releases_session(client_no_token: StyxCoreClient) -> None:
    sess = client_no_token._session
    with patch.object(sess, "close") as mock_close:
        client_no_token.close()
        mock_close.assert_called_once()


def test_default_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYX_DAEMON_URL", "http://envurl:9999")
    c = StyxCoreClient()
    assert c.base_url == "http://envurl:9999"


def test_default_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYX_HTTP_TOKEN", "env-token")
    c = StyxCoreClient()
    assert c._session.headers.get("Authorization") == "Bearer env-token"


def test_default_social_token_from_env_is_not_a_session_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STYX_HTTP_TOKEN", raising=False)
    monkeypatch.setenv("STYX_SOCIAL_TOKEN", "env-social-token")
    c = StyxCoreClient(base_url="http://daemon.local:8788")
    assert "X-Styx-Social-Token" not in c._session.headers
    with patch.object(c._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"scope_id": "scope"})
        c.social_explain(
            "agent-a",
            scope_id="11111111-1111-4111-8111-111111111111",
        )
    assert mock_post.call_args.kwargs["headers"] == {
        "X-Styx-Social-Token": "env-social-token"
    }
