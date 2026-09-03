from __future__ import annotations

import hashlib

from styx.http import registry
from styx.http.social_auth import SocialPrincipal


class FakeCore:
    def social_create_scope(self, **data):
        return {"scope_id": "scope", "duplicate": False, "status": "active"}


def _register() -> None:
    registry.register("agent-a", FakeCore())


def test_social_routes_are_disabled_without_registry(client_no_auth) -> None:
    _register()
    response = client_no_auth.post("/social/scopes", json={
        "agent_id": "agent-a", "scope_key": "s", "protocol_id": "p",
        "protocol_version": "1", "policy_hash": "a" * 64,
    })
    assert response.status_code == 404


def test_common_http_token_does_not_unlock_social_route(client_with_auth) -> None:
    _register()
    client_with_auth.app.state.social_principals = (
        SocialPrincipal(
            "owner", hashlib.sha256(b"social-secret").hexdigest(),
            frozenset({"agent-a"}), frozenset({"social:scope-admin"}),
        ),
    )
    payload = {"agent_id": "agent-a", "scope_key": "s", "protocol_id": "p",
               "protocol_version": "1", "policy_hash": "a" * 64}
    assert client_with_auth.post(
        "/social/scopes", json=payload,
        headers={"Authorization": "Bearer test-token-do-not-use-in-prod"},
    ).status_code == 401
    response = client_with_auth.post(
        "/social/scopes", json=payload,
        headers={"Authorization": "Bearer test-token-do-not-use-in-prod",
                 "X-Styx-Social-Token": "social-secret"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "active"


def test_forged_owner_is_not_discoverable(client_no_auth) -> None:
    _register()
    client_no_auth.app.state.social_principals = (
        SocialPrincipal(
            "reader", hashlib.sha256(b"read-secret").hexdigest(),
            frozenset({"agent-b"}), frozenset({"social:scope-admin"}),
        ),
    )
    response = client_no_auth.post(
        "/social/scopes",
        headers={"X-Styx-Social-Token": "read-secret"},
        json={"agent_id": "agent-a", "scope_key": "s", "protocol_id": "p",
              "protocol_version": "1", "policy_hash": "a" * 64},
    )
    assert response.status_code == 404


def test_social_validation_never_echoes_private_input(client_no_auth) -> None:
    _register()
    client_no_auth.app.state.social_principals = (
        SocialPrincipal(
            "attester", hashlib.sha256(b"attest-secret").hexdigest(),
            frozenset({"agent-a"}), frozenset({"social:attest"}),
        ),
    )
    marker = "private-evidence-marker"
    response = client_no_auth.post(
        "/social/attestations",
        headers={"X-Styx-Social-Token": "attest-secret"},
        json={
            "agent_id": "agent-a",
            "scope_id": "00000000-0000-0000-0000-000000000001",
            "issuer_actor_id": "00000000-0000-0000-0000-000000000002",
            "subject_actor_id": "00000000-0000-0000-0000-000000000003",
            "attestation_key": "a",
            "verdict": "positive",
            "protocol_id": "p",
            "protocol_version": "1",
            "source_act_id": "00000000-0000-0000-0000-000000000004",
            "trust_level": "verified",
            "evidence_refs": [{"private": marker * 30}],
        },
    )
    assert response.status_code == 422
    assert marker not in response.text
    assert response.json() == {"detail": "invalid social request"}


def test_verified_attestation_requires_body_signature(client_no_auth) -> None:
    _register()
    client_no_auth.app.state.social_principals = (
        SocialPrincipal(
            "attester", hashlib.sha256(b"attest-secret").hexdigest(),
            frozenset({"agent-a"}), frozenset({"social:attest"}),
        ),
    )
    response = client_no_auth.post(
        "/social/attestations",
        headers={"X-Styx-Social-Token": "attest-secret"},
        json={
            "agent_id": "agent-a",
            "scope_id": "00000000-0000-0000-0000-000000000001",
            "issuer_actor_id": "00000000-0000-0000-0000-000000000002",
            "subject_actor_id": "00000000-0000-0000-0000-000000000003",
            "attestation_key": "a",
            "verdict": "positive",
            "protocol_id": "p",
            "protocol_version": "1",
            "source_act_id": "00000000-0000-0000-0000-000000000004",
            "trust_level": "verified",
        },
    )
    assert response.status_code == 401


def test_unknown_agent_uses_generic_social_404(client_no_auth) -> None:
    _register()
    client_no_auth.app.state.social_principals = (
        SocialPrincipal(
            "reader", hashlib.sha256(b"read-secret").hexdigest(),
            frozenset({"receiver"}), frozenset({"social:read"}),
        ),
    )
    response = client_no_auth.post(
        "/social/query",
        headers={"X-Styx-Social-Token": "read-secret"},
        json={
            "agent_id": "private-agent-marker",
            "scope_id": "00000000-0000-0000-0000-000000000001",
            "actor_a_id": "00000000-0000-0000-0000-000000000002",
            "actor_b_id": "00000000-0000-0000-0000-000000000003",
        },
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "not found"}
