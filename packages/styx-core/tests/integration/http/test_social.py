from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from styx.config import StyxConfig
from styx.http import registry
from styx.http.app import create_app
from styx.providers.memory import StyxMemoryCore
from styx.storage import migrate


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL is required",
)


def _principal(principal: str, token: str, agents: list[str], capabilities: list[str]) -> dict:
    return {
        "principal_id": principal,
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "agent_ids": agents,
        "capabilities": capabilities,
    }


@pytest.fixture
def social_stack(clean_db: str, tmp_path):
    migrate.run(clean_db)
    owner, receiver = "social-owner", "social-receiver"
    principal_file = tmp_path / "social-principals.json"
    principal_file.write_text(json.dumps({"principals": [
        _principal("p1", "token-one", [owner], [
            "social:scope-admin", "social:attest", "social:encounter", "social:read",
        ]),
        _principal("p2", "token-two", [owner], ["social:attest"]),
        _principal("receiver", "token-receiver", [receiver], ["social:read"]),
        _principal("outsider", "token-outsider", ["other"], ["social:read"]),
    ]}))
    config = StyxConfig(
        database_url=clean_db,
        social_principals_file=str(principal_file),
        sentiment_enabled=False,
        affective_transition_enabled=False,
        working_set_persistence_enabled=False,
    )
    core = StyxMemoryCore(agent_id=owner)
    core._config = config
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=owner)
    registry.reset_all()
    registry.register(owner, core)
    client = TestClient(create_app(config))
    yield client, clean_db, owner, receiver, core
    core.shutdown()
    registry.reset_all()


def _post(client: TestClient, path: str, token: str, payload: dict):
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"X-Styx-Social-Token": token, "Content-Type": "application/json"}
    if path.startswith("/social/attestations") and payload.get("trust_level") == "verified":
        headers["X-Styx-Social-Signature"] = hmac.new(
            token.encode(), body, hashlib.sha256
        ).hexdigest()
    return client.post(path, headers=headers, content=body)


def test_social_http_mutual_visibility_and_observation_delivery(social_stack) -> None:
    client, dsn, owner, receiver, core = social_stack
    h64 = "a" * 64
    actors = []
    for index, acting_principal in enumerate(("p1", "p2"), start=1):
        response = _post(client, "/social/actors", "token-one", {
            "agent_id": owner, "identity_namespace": "fixture",
            "actor_key": f"actor-{index}", "actor_kind": "external_agent",
            "private_label": f"never-return-{index}",
            "identity_evidence_hash": h64,
            "attestation_principal_id": acting_principal,
        })
        assert response.status_code == 200, response.text
        actors.append(response.json()["actor_id"])
    scope_response = _post(client, "/social/scopes", "token-one", {
        "agent_id": owner, "scope_key": "fixture-scope", "protocol_id": "vouch",
        "protocol_version": "1", "policy_hash": "b" * 64,
    })
    assert scope_response.status_code == 200, scope_response.text
    scope = scope_response.json()["scope_id"]

    act_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    with core._write_lock, core._conn.cursor() as cur:
        for index, act_id in enumerate(act_ids):
            cur.execute(
                "INSERT INTO cognitive_acts "
                "(id,agent_id,host_key,status,channel_input,channel_output,completed_at) "
                "VALUES (%s,%s,%s,'completed','{}','{}',clock_timestamp())",
                (act_id, owner, f"social-source-{index}"),
            )
        core._conn.commit()

    attestation_ids = []
    for token, issuer, subject, key, act_id in (
        ("token-one", actors[0], actors[1], "one-to-two", act_ids[0]),
        ("token-two", actors[1], actors[0], "two-to-one", act_ids[1]),
    ):
        response = _post(client, "/social/attestations", token, {
            "agent_id": owner, "scope_id": scope, "issuer_actor_id": issuer,
            "subject_actor_id": subject, "attestation_key": key,
            "attestation_kind": "direct", "verdict": "positive",
            "protocol_id": "vouch", "protocol_version": "1",
            "source_act_id": act_id, "evidence_refs": [],
            "trust_level": "verified", "signature_metadata": {},
        })
        assert response.status_code == 200, response.text
        attestation_ids.append(response.json()["attestation_id"])
    assert response.json()["projection_status"] == "mutual_positive"

    denied = _post(client, "/social/query", "token-outsider", {
        "agent_id": owner, "scope_id": scope,
        "actor_a_id": actors[0], "actor_b_id": actors[1],
    })
    assert denied.status_code == 404
    for evidence_class, coordinates in (
        ("projection", {"actor_a_id": actors[0], "actor_b_id": actors[1]}),
        ("attestation", {"evidence_id": attestation_ids[-1]}),
    ):
        grant = _post(client, "/social/grants", "token-one", {
            "agent_id": owner, "grant_key": f"receiver-{evidence_class}",
            "scope_id": scope, "grantee_principal_id": "receiver",
            "capability": "social:read", "evidence_class": evidence_class,
            **coordinates,
        })
        assert grant.status_code == 200, grant.text
    visible = _post(client, "/social/query", "token-receiver", {
        "agent_id": owner, "scope_id": scope,
        "actor_a_id": actors[0], "actor_b_id": actors[1],
    })
    assert visible.status_code == 200
    assert visible.json()["status"] == "mutual_positive"
    delivery = _post(client, "/social/deliver", "token-receiver", {
        "agent_id": owner, "delivery_key": "receiver-delivery", "scope_id": scope,
        "evidence_class": "attestation", "evidence_id": attestation_ids[-1],
        "receiving_agent_id": receiver,
    })
    assert delivery.status_code == 200, delivery.text
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM cognitive_consequences WHERE id=%s AND agent_id=%s",
            (delivery.json()["observation_id"], receiver),
        )
        assert cur.fetchone()[0] == "pending"
        cur.execute("SELECT count(*) FROM memories WHERE agent_id=%s", (receiver,))
        assert cur.fetchone()[0] == 0
