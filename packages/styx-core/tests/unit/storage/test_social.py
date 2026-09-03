from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest

from styx.storage.social import (
    SocialConflict,
    create_actor,
    create_attestation,
    create_encounter,
    create_grant,
    create_scope,
    deliver_evidence,
    dissolve_scope,
    explain_scope,
    query_projection,
    revoke_grant,
)


OWNER = "agent-social-owner"
H64 = "a" * 64


def _act(conn: psycopg.Connection, key: str) -> str:
    act_id = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cognitive_acts "
            "(id,agent_id,host_key,status,channel_input,channel_output,completed_at) "
            "VALUES (%s,%s,%s,'completed','{}','{}',clock_timestamp())",
            (act_id, OWNER, key),
        )
    return str(act_id)


def _actor(conn: psycopg.Connection, key: str, principal: str) -> str:
    return create_actor(
        conn, OWNER, principal,
        identity_namespace="fixture", actor_key=key, actor_kind="external_agent",
        private_label=f"private-{key}", attestation_principal_id=principal,
        identity_evidence_hash=H64,
    )["actor_id"]


def _scope(conn: psycopg.Connection, key: str = "scope-a") -> str:
    return create_scope(
        conn, OWNER, "scope-admin", scope_key=key, protocol_id="vouch",
        protocol_version="1", policy_hash="b" * 64,
    )["scope_id"]


def _attest(
    conn: psycopg.Connection, principal: str, scope: str, issuer: str,
    subject: str, key: str, act: str, verdict: str = "positive", **extra,
) -> dict:
    return create_attestation(
        conn, OWNER, principal, scope_id=scope, issuer_actor_id=issuer,
        subject_actor_id=subject, attestation_key=key,
        attestation_kind=extra.pop("attestation_kind", "direct"), verdict=verdict,
        protocol_id="vouch", protocol_version="1", source_act_id=act,
        source_action_ordinal=None, evidence_refs=[], trust_level="verified",
        signature_metadata={}, supersedes_attestation_id=extra.pop("supersedes_attestation_id", None),
        signature_verified=True,
        **extra,
    )


def test_encounter_is_not_attestation_and_retry_is_exact(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope, act = _scope(conn), _act(conn, "encounter-source")
        args = dict(
            encounter_key="meeting-1", scope_id=scope,
            observer_actor_id=first, encountered_actor_id=second,
            direction="inbound", channel_kind="fixture", source_act_id=act,
            source_observation_id=None, summary="private fixture", evidence_hash=H64,
            confidence=0.8,
        )
        assert create_encounter(conn, OWNER, "encounter-publisher", **args)["duplicate"] is False
        assert create_encounter(conn, OWNER, "encounter-publisher", **args)["duplicate"] is True
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM social_attestations")
            assert cur.fetchone()[0] == 0
        projection = query_projection(
            conn, OWNER, scope_id=scope, actor_a_id=first, actor_b_id=second,
        )
        assert projection["status"] == "undetermined"
        explained = explain_scope(conn, OWNER, scope_id=scope)
        assert "private fixture" not in json.dumps(explained, default=str)
        assert "private-first" not in json.dumps(explained, default=str)


def test_mutual_projection_revision_revocation_and_dissolution(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope = _scope(conn)
        act_one = _act(conn, "act-1")
        one = _attest(conn, "p1", scope, first, second, "a1", act_one)
        assert one["projection_status"] == "unilateral"
        retry = _attest(conn, "p1", scope, first, second, "a1", act_one)
        assert retry["duplicate"] is True and retry["projection_version"] == 1
        conn.commit()
        # The changed source coordinate makes this a conflicting retry.
        with pytest.raises(SocialConflict):
            _attest(conn, "p1", scope, first, second, "a1", _act(conn, "changed"))
        conn.rollback()

    # Fresh transaction after the deliberate rollback.
    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM social_actors WHERE actor_key='first'")
            first = str(cur.fetchone()[0])
            cur.execute("SELECT id FROM social_actors WHERE actor_key='second'")
            second = str(cur.fetchone()[0])
            cur.execute("SELECT id FROM social_scopes WHERE scope_key='scope-a'")
            scope = str(cur.fetchone()[0])
            cur.execute("SELECT id,source_act_id FROM social_attestations WHERE attestation_key='a1'")
            first_attestation, first_act = map(str, cur.fetchone())
        exact = _attest(conn, "p1", scope, first, second, "a1", first_act)
        assert exact["duplicate"] is True and exact["projection_version"] == 1
        two = _attest(conn, "p2", scope, second, first, "a2", _act(conn, "act-2"))
        assert two["projection_status"] == "mutual_positive"
        revised = _attest(
            conn, "p1", scope, first, second, "a1-negative", _act(conn, "act-3"),
            verdict="negative", supersedes_attestation_id=first_attestation,
        )
        assert revised["projection_status"] == "mutual_denied"
        revoked = _attest(
            conn, "p1", scope, first, second, "a1-revoke", _act(conn, "act-4"),
            verdict="undetermined", attestation_kind="revocation",
            supersedes_attestation_id=revised["attestation_id"],
        )
        assert revoked["projection_status"] == "unilateral"
        assert dissolve_scope(conn, OWNER, "scope-admin", scope)["status"] == "dissolved"
        assert query_projection(
            conn, OWNER, scope_id=scope, actor_a_id=first, actor_b_id=second,
        )["status"] == "scope_dissolved"
        assert _attest(
            conn, "p1", scope, first, second, "a1", first_act,
        )["duplicate"] is True
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM social_scope_operations WHERE scope_id=%s", (scope,))
            assert cur.fetchone()[0] == 1
        assert dissolve_scope(conn, OWNER, "scope-admin", scope)["duplicate"] is True


def test_self_and_reported_evidence_never_create_membership(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope = _scope(conn)
        self_result = _attest(
            conn, "p1", scope, first, first, "self", _act(conn, "self-act"),
            attestation_kind="self",
        )
        assert self_result["projection_version"] == 0
        reported = _attest(
            conn, "p2", scope, second, first, "reported", _act(conn, "report-act"),
            attestation_kind="reported",
        )
        assert reported["projection_status"] == "undetermined"
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM social_attestations")
            assert cur.fetchone()[0] == 2


def test_attestation_rejects_failed_source_act(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope = _scope(conn)
        failed_act = str(uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cognitive_acts "
                "(id,agent_id,host_key,status,channel_input,channel_output,completed_at) "
                "VALUES (%s,%s,%s,'failed','{}','{}',clock_timestamp())",
                (failed_act, OWNER, "failed-social-source"),
            )
        with pytest.raises(SocialConflict, match="completed source act"):
            _attest(
                conn, "p1", scope, first, second, "failed-attestation", failed_act,
            )


def test_invalid_social_references_raise_typed_errors_before_database_constraints(
    migrated_db: str,
) -> None:
    missing = str(uuid.uuid4())
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope, act = _scope(conn), _act(conn, "valid-source")
        with pytest.raises(ValueError, match="distinct"):
            create_encounter(
                conn, OWNER, "publisher", encounter_key="self-encounter",
                scope_id=scope, observer_actor_id=first, encountered_actor_id=first,
                direction="inbound", channel_kind="fixture", source_act_id=act,
                source_observation_id=None, summary=None, evidence_hash=H64,
                confidence=0.8,
            )
        with pytest.raises(LookupError, match="actor"):
            create_encounter(
                conn, OWNER, "publisher", encounter_key="missing-actor",
                scope_id=scope, observer_actor_id=first, encountered_actor_id=missing,
                direction="inbound", channel_kind="fixture", source_act_id=act,
                source_observation_id=None, summary=None, evidence_hash=H64,
                confidence=0.8,
            )
        with pytest.raises(LookupError, match="source"):
            create_encounter(
                conn, OWNER, "publisher", encounter_key="missing-source",
                scope_id=scope, observer_actor_id=first, encountered_actor_id=second,
                direction="inbound", channel_kind="fixture", source_act_id=missing,
                source_observation_id=None, summary=None, evidence_hash=H64,
                confidence=0.8,
            )
        with pytest.raises(ValueError, match="self attestation"):
            _attest(conn, "p1", scope, first, first, "not-self", act)
        with pytest.raises(SocialConflict, match="actor coordinates"):
            _attest(conn, "p1", scope, first, missing, "missing-subject", act)


def test_concurrent_reciprocal_acts_serialize_to_one_mutual_projection(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope = _scope(conn)
        act_one, act_two = _act(conn, "parallel-1"), _act(conn, "parallel-2")
        conn.commit()

    def write(args: tuple[str, str, str, str, str]) -> str:
        principal, issuer, subject, key, act = args
        with psycopg.connect(migrated_db) as conn:
            return _attest(
                conn, principal, scope, issuer, subject, key, act,
            )["projection_status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(write, [
            ("p1", first, second, "parallel-a", act_one),
            ("p2", second, first, "parallel-b", act_two),
        ]))
    assert "mutual_positive" in statuses
    with psycopg.connect(migrated_db) as conn:
        final = query_projection(
            conn, OWNER, scope_id=scope, actor_a_id=first, actor_b_id=second,
        )
        assert final["status"] == "mutual_positive"
        assert final["projection_version"] == 2


def test_cross_scope_idempotency_keys_serialize_to_typed_conflicts(
    migrated_db: str,
) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scopes = (_scope(conn, "scope-a"), _scope(conn, "scope-b"))
        acts = (_act(conn, "encounter-a"), _act(conn, "encounter-b"))
        conn.commit()

    def write_encounter(index: int) -> str:
        try:
            with psycopg.connect(migrated_db) as conn:
                create_encounter(
                    conn,
                    OWNER,
                    "publisher",
                    encounter_key="same-cross-scope-key",
                    scope_id=scopes[index],
                    observer_actor_id=first,
                    encountered_actor_id=second,
                    direction="inbound",
                    channel_kind="fixture",
                    source_act_id=acts[index],
                    source_observation_id=None,
                    summary=None,
                    evidence_hash=H64,
                    confidence=0.8,
                )
            return "created"
        except SocialConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write_encounter, (0, 1)))
    assert sorted(outcomes) == ["conflict", "created"]

    def write_grant(index: int) -> str:
        try:
            with psycopg.connect(migrated_db) as conn:
                create_grant(
                    conn,
                    OWNER,
                    "scope-admin",
                    grant_key="same-cross-scope-grant",
                    scope_id=scopes[index],
                    grantee_principal_id="publisher",
                    capability="social:encounter",
                    evidence_class="encounter",
                    evidence_id=None,
                    actor_a_id=None,
                    actor_b_id=None,
                    expires_at=None,
                )
            return "created"
        except SocialConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write_grant, (0, 1)))
    assert sorted(outcomes) == ["conflict", "created"]


def test_principal_binding_scope_isolation_and_delivery_bridge(migrated_db: str) -> None:
    with psycopg.connect(migrated_db) as conn:
        first, second = _actor(conn, "first", "p1"), _actor(conn, "second", "p2")
        scope, other_scope = _scope(conn), _scope(conn, "scope-b")
        conn.commit()
        with pytest.raises(SocialConflict):
            _attest(conn, "forged", scope, first, second, "forged", _act(conn, "forged"))
        conn.rollback()

    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM social_actors WHERE actor_key='first'")
            first = str(cur.fetchone()[0])
            cur.execute("SELECT id FROM social_actors WHERE actor_key='second'")
            second = str(cur.fetchone()[0])
            cur.execute("SELECT id FROM social_scopes WHERE scope_key='scope-a'")
            scope = str(cur.fetchone()[0])
            cur.execute("SELECT id FROM social_scopes WHERE scope_key='scope-b'")
            other_scope = str(cur.fetchone()[0])
        attestation = _attest(
            conn, "p1", scope, first, second, "deliverable", _act(conn, "deliverable")
        )
        assert query_projection(
            conn, OWNER, scope_id=other_scope, actor_a_id=first, actor_b_id=second,
        )["status"] == "undetermined"
        conn.commit()
        with pytest.raises(LookupError):
            deliver_evidence(
                conn, OWNER, "receiver", delivery_key="delivery-1", scope_id=scope,
                evidence_class="attestation", evidence_id=attestation["attestation_id"],
                receiving_agent_id="agent-receiver",
            )
        conn.rollback()

    with psycopg.connect(migrated_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM social_scopes WHERE scope_key='scope-a'")
            scope = str(cur.fetchone()[0])
            cur.execute("SELECT id FROM social_attestations WHERE attestation_key='deliverable'")
            attestation_id = str(cur.fetchone()[0])
        grant = create_grant(
            conn, OWNER, "scope-admin", grant_key="receiver-read", scope_id=scope,
            grantee_principal_id="receiver", capability="social:read",
            evidence_class="attestation", evidence_id=attestation_id,
            actor_a_id=None, actor_b_id=None, expires_at=None,
        )
        delivered = deliver_evidence(
            conn, OWNER, "receiver", delivery_key="delivery-1", scope_id=scope,
            evidence_class="attestation", evidence_id=attestation_id,
            receiving_agent_id="agent-receiver",
        )
        replay = deliver_evidence(
            conn, OWNER, "receiver", delivery_key="delivery-1", scope_id=scope,
            evidence_class="attestation", evidence_id=attestation_id,
            receiving_agent_id="agent-receiver",
        )
        assert delivered["duplicate"] is False and replay["duplicate"] is True
        revoked = revoke_grant(
            conn, OWNER, "scope-admin", revocation_key="receiver-read-revoke",
            grant_id=grant["grant_id"],
        )
        assert revoked["revoked"] is True and revoked["duplicate"] is False
        assert revoke_grant(
            conn, OWNER, "scope-admin", revocation_key="receiver-read-revoke",
            grant_id=grant["grant_id"],
        )["duplicate"] is True
        with pytest.raises(LookupError):
            deliver_evidence(
                conn, OWNER, "receiver", delivery_key="delivery-after-revoke",
                scope_id=scope, evidence_class="attestation",
                evidence_id=attestation_id, receiving_agent_id="agent-receiver",
            )
        # Authority is deliberately re-evaluated before receipt replay:
        # revocation closes even an exact delivery-key retry.
        with pytest.raises(LookupError):
            deliver_evidence(
                conn, OWNER, "receiver", delivery_key="delivery-1",
                scope_id=scope, evidence_class="attestation",
                evidence_id=attestation_id, receiving_agent_id="agent-receiver",
            )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_id,status FROM cognitive_consequences "
                "WHERE id=%s AND agent_id='agent-receiver'", (delivered["observation_id"],),
            )
            source_id, status = cur.fetchone()
            assert source_id.startswith("social-scope:") and status == "pending"
            cur.execute("SELECT count(*) FROM memories WHERE agent_id='agent-receiver'")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM cognitive_acts WHERE agent_id='agent-receiver'")
            assert cur.fetchone()[0] == 0
