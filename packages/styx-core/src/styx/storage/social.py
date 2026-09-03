"""Transactional scoped social evidence ledger (wave 42)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from typing import Any, Mapping

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from styx.engine.social_projection import evaluate_pair_projection
from styx.storage.cognition import validate_journal_json


class SocialConflict(ValueError):
    pass


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.isoformat() if isinstance(item, dt.datetime) else str(item),
    ).encode()).hexdigest()


def _uuid(value: Any, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a UUID") from exc


def _lock_key(cur: psycopg.Cursor[Any], namespace: str, owner: str, key: str) -> None:
    """Serialize idempotency keys even when competing requests name other scopes."""
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        (f"styx:social:{namespace}:{owner}:{key}",),
    )


def _has_visibility_grant_cur(
    cur: psycopg.Cursor[Any],
    owner: str,
    *,
    scope_id: uuid.UUID,
    principal_id: str,
    evidence_class: str,
    capability: str = "social:read",
    evidence_id: uuid.UUID | None = None,
    actor_a_id: uuid.UUID | None = None,
    actor_b_id: uuid.UUID | None = None,
) -> bool:
    actor_low: uuid.UUID | None = None
    actor_high: uuid.UUID | None = None
    if actor_a_id is not None and actor_b_id is not None:
        actor_low, actor_high = sorted((actor_a_id, actor_b_id), key=str)
    cur.execute(
        "SELECT 1 FROM social_visibility_grants WHERE owner_agent_id=%s "
        "AND scope_id=%s AND grantee_principal_id=%s AND capability=%s "
        "AND evidence_class=%s AND evidence_id IS NOT DISTINCT FROM %s "
        "AND actor_low_id IS NOT DISTINCT FROM %s "
        "AND actor_high_id IS NOT DISTINCT FROM %s AND revoked_at IS NULL "
        "AND (expires_at IS NULL OR expires_at>clock_timestamp()) "
        "LIMIT 1 FOR SHARE",
        (
            owner, scope_id, principal_id, capability, evidence_class,
            evidence_id, actor_low, actor_high,
        ),
    )
    return cur.fetchone() is not None


def create_actor(
    conn: psycopg.Connection, owner: str, principal_id: str, **data: Any
) -> dict[str, Any]:
    payload = {
        "identity_namespace": data["identity_namespace"], "actor_key": data["actor_key"],
        "actor_kind": data["actor_kind"], "private_label": data.get("private_label"),
        "attestation_principal_id": data.get("attestation_principal_id") or principal_id,
        "identity_evidence_hash": data["identity_evidence_hash"],
    }
    digest = _hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        _lock_key(
            cur, "actor", owner,
            f"{payload['identity_namespace']}:{payload['actor_key']}",
        )
        cur.execute(
            "SELECT id,payload_hash,status FROM social_actors WHERE owner_agent_id=%s "
            "AND identity_namespace=%s AND actor_key=%s FOR UPDATE",
            (owner, payload["identity_namespace"], payload["actor_key"]),
        )
        row = cur.fetchone()
        if row:
            if row["payload_hash"] != digest:
                raise SocialConflict("actor key already has different semantics")
            return {"actor_id": str(row["id"]), "duplicate": True, "status": row["status"]}
        actor_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO social_actors(id,owner_agent_id,identity_namespace,actor_key,"
            "actor_kind,private_label,attestation_principal_id,identity_evidence_hash,payload_hash) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (actor_id, owner, payload["identity_namespace"], payload["actor_key"],
             payload["actor_kind"], payload["private_label"],
             payload["attestation_principal_id"], payload["identity_evidence_hash"], digest),
        )
        return {"actor_id": str(actor_id), "duplicate": False, "status": "active"}


def create_scope(
    conn: psycopg.Connection, owner: str, principal_id: str, **data: Any
) -> dict[str, Any]:
    payload = {key: data[key] for key in ("scope_key","protocol_id","protocol_version","policy_hash")}
    payload["created_by_principal_id"] = principal_id
    digest = _hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        _lock_key(cur, "scope", owner, payload["scope_key"])
        cur.execute(
            "SELECT id,payload_hash,status FROM social_scopes WHERE owner_agent_id=%s "
            "AND scope_key=%s FOR UPDATE", (owner, payload["scope_key"]),
        )
        row = cur.fetchone()
        if row:
            if row["payload_hash"] != digest:
                raise SocialConflict("scope key already has different semantics")
            return {"scope_id": str(row["id"]), "duplicate": True, "status": row["status"]}
        scope_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO social_scopes(id,owner_agent_id,scope_key,protocol_id,"
            "protocol_version,policy_hash,created_by_principal_id,payload_hash) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (scope_id, owner, payload["scope_key"], payload["protocol_id"],
             payload["protocol_version"], payload["policy_hash"], principal_id, digest),
        )
        return {"scope_id": str(scope_id), "duplicate": False, "status": "active"}


def _scope(
    cur: psycopg.Cursor[Any], owner: str, scope_id: uuid.UUID, *, lock: bool = True
) -> Mapping[str, Any]:
    cur.execute(
        "SELECT id,status,protocol_id,protocol_version,policy_hash FROM social_scopes "
        "WHERE owner_agent_id=%s AND id=%s" + (" FOR UPDATE" if lock else ""),
        (owner, scope_id),
    )
    row = cur.fetchone()
    if row is None:
        raise LookupError("scope not found")
    return row


def create_encounter(
    conn: psycopg.Connection, owner: str, principal_id: str, **data: Any
) -> dict[str, Any]:
    visibility_principal_id = data.pop("visibility_principal_id", None)
    scope_id = _uuid(data["scope_id"], "scope_id")
    observer = _uuid(data["observer_actor_id"], "observer_actor_id")
    encountered = _uuid(data["encountered_actor_id"], "encountered_actor_id")
    source_act = _uuid(data["source_act_id"], "source_act_id") if data.get("source_act_id") else None
    source_observation = _uuid(data["source_observation_id"], "source_observation_id") if data.get("source_observation_id") else None
    if observer == encountered:
        raise ValueError("encounter actors must be distinct")
    payload = {
        "encounter_key": data["encounter_key"], "scope_id": str(scope_id),
        "observer_actor_id": str(observer), "encountered_actor_id": str(encountered),
        "direction": data["direction"], "channel_kind": data["channel_kind"],
        "source_act_id": str(source_act) if source_act else None,
        "source_observation_id": str(source_observation) if source_observation else None,
        "summary": data.get("summary"), "evidence_hash": data["evidence_hash"],
        "confidence": float(data["confidence"]),
        "publisher_principal_id": principal_id,
    }
    digest = _hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        _lock_key(cur, "encounter", owner, payload["encounter_key"])
        scope = _scope(cur, owner, scope_id)
        if visibility_principal_id is not None and not _has_visibility_grant_cur(
            cur,
            owner,
            scope_id=scope_id,
            principal_id=visibility_principal_id,
            evidence_class="encounter",
            capability="social:encounter",
        ):
            raise LookupError("social scope not found")
        cur.execute(
            "SELECT id,payload_hash FROM social_encounters WHERE owner_agent_id=%s "
            "AND encounter_key=%s FOR UPDATE", (owner, payload["encounter_key"]),
        )
        row = cur.fetchone()
        if row:
            if row["payload_hash"] != digest:
                raise SocialConflict("encounter key already has different semantics")
            return {"encounter_id": str(row["id"]), "duplicate": True}
        if scope["status"] != "active":
            raise SocialConflict("scope is dissolved")
        cur.execute(
            "SELECT count(*)::int AS count FROM social_actors "
            "WHERE owner_agent_id=%s AND id IN (%s,%s)",
            (owner, observer, encountered),
        )
        if int(cur.fetchone()["count"]) != 2:
            raise LookupError("encounter actor not found")
        if source_act is not None:
            cur.execute(
                "SELECT 1 FROM cognitive_acts WHERE id=%s AND agent_id=%s",
                (source_act, owner),
            )
            if cur.fetchone() is None:
                raise LookupError("encounter source not found")
        if source_observation is not None:
            cur.execute(
                "SELECT 1 FROM cognitive_consequences WHERE id=%s AND agent_id=%s",
                (source_observation, owner),
            )
            if cur.fetchone() is None:
                raise LookupError("encounter source not found")
        encounter_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO social_encounters(id,owner_agent_id,encounter_key,scope_id,"
            "observer_actor_id,encountered_actor_id,direction,channel_kind,source_act_id,"
            "source_observation_id,summary,evidence_hash,payload_hash,confidence,publisher_principal_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (encounter_id, owner, data["encounter_key"], scope_id, observer, encountered,
             data["direction"], data["channel_kind"], source_act, source_observation,
             data.get("summary"), data["evidence_hash"], digest, float(data["confidence"]),
             principal_id),
        )
        return {"encounter_id": str(encounter_id), "duplicate": False}


def _recompute_projection(
    cur: psycopg.Cursor[Any], owner: str, scope: Mapping[str, Any], first: uuid.UUID, second: uuid.UUID
) -> dict[str, Any]:
    low, high = sorted((first, second), key=str)
    cur.execute(
        "SELECT id,sequence,issuer_actor_id,subject_actor_id,attestation_kind,verdict,"
        "trust_level,issuer_principal_id FROM social_attestations "
        "WHERE owner_agent_id=%s AND scope_id=%s "
        "AND issuer_actor_id IN (%s,%s) AND subject_actor_id IN (%s,%s) ORDER BY sequence,id",
        (owner, scope["id"], low, high, low, high),
    )
    outcome = evaluate_pair_projection(
        cur.fetchall(), actor_low_id=str(low), actor_high_id=str(high),
        scope_status=scope["status"],
    )
    cur.execute(
        "INSERT INTO social_pair_projections(owner_agent_id,scope_id,actor_low_id,actor_high_id,"
        "projection_version,status,low_to_high_attestation_id,high_to_low_attestation_id,policy_hash) "
        "VALUES (%s,%s,%s,%s,1,%s,%s,%s,%s) ON CONFLICT "
        "(owner_agent_id,scope_id,actor_low_id,actor_high_id) DO UPDATE SET "
        "projection_version=social_pair_projections.projection_version+1,status=EXCLUDED.status,"
        "low_to_high_attestation_id=EXCLUDED.low_to_high_attestation_id,"
        "high_to_low_attestation_id=EXCLUDED.high_to_low_attestation_id,"
        "policy_hash=EXCLUDED.policy_hash,computed_at=clock_timestamp() "
        "RETURNING projection_version,status",
        (owner, scope["id"], low, high, outcome["status"], outcome["low_to_high"],
         outcome["high_to_low"], scope["policy_hash"]),
    )
    row = cur.fetchone()
    return {"projection_version": int(row["projection_version"]), "projection_status": row["status"]}


def _current_projection(
    cur: psycopg.Cursor[Any], owner: str, scope_id: uuid.UUID,
    first: uuid.UUID, second: uuid.UUID,
) -> dict[str, Any]:
    if first == second:
        return {"projection_version": 0, "projection_status": "undetermined"}
    low, high = sorted((first, second), key=str)
    cur.execute(
        "SELECT projection_version,status FROM social_pair_projections "
        "WHERE owner_agent_id=%s AND scope_id=%s AND actor_low_id=%s "
        "AND actor_high_id=%s", (owner, scope_id, low, high),
    )
    row = cur.fetchone()
    if row is None:
        return {"projection_version": 0, "projection_status": "undetermined"}
    return {
        "projection_version": int(row["projection_version"]),
        "projection_status": row["status"],
    }


def create_attestation(conn: psycopg.Connection, owner: str, principal_id: str, **data: Any) -> dict[str, Any]:
    signature_verified = bool(data.pop("signature_verified", False))
    scope_id = _uuid(data["scope_id"], "scope_id")
    issuer = _uuid(data["issuer_actor_id"], "issuer_actor_id")
    subject = _uuid(data["subject_actor_id"], "subject_actor_id")
    source_act = _uuid(data["source_act_id"], "source_act_id") if data.get("source_act_id") else None
    supersedes = _uuid(data["supersedes_attestation_id"], "supersedes_attestation_id") if data.get("supersedes_attestation_id") else None
    kind = data.get("attestation_kind", "direct")
    if (issuer == subject) != (kind == "self"):
        raise ValueError("self attestation coordinates are invalid")
    evidence_refs = data.get("evidence_refs") or []
    signature = data.get("signature_metadata") or {}
    validate_journal_json(evidence_refs, max_string=256)
    validate_journal_json(signature, max_string=256)
    declared_signature = (
        signature.get("declared", {})
        if signature.get("scheme") == "hmac-sha256"
        else signature
    )
    payload = {
        "scope_id": str(scope_id), "issuer_actor_id": str(issuer),
        "subject_actor_id": str(subject), "attestation_key": data["attestation_key"],
        "attestation_kind": kind,
        "verdict": data["verdict"], "protocol_id": data["protocol_id"],
        "protocol_version": data["protocol_version"],
        "source_act_id": str(source_act) if source_act else None,
        "source_action_ordinal": data.get("source_action_ordinal"),
        "evidence_refs": evidence_refs, "trust_level": data["trust_level"],
        "signature_metadata": declared_signature,
        "supersedes_attestation_id": str(supersedes) if supersedes else None,
    }
    digest = _hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        scope = _scope(cur, owner, scope_id)
        if (scope["protocol_id"], scope["protocol_version"]) != (
            data["protocol_id"], data["protocol_version"]
        ):
            raise SocialConflict("attestation protocol does not match scope")
        if source_act is None:
            raise SocialConflict("attestation requires source cognitive act")
        if data["trust_level"] == "verified" and not signature_verified:
            raise SocialConflict("verified attestation requires a valid signature")
        cur.execute(
            "SELECT 1 FROM cognitive_acts WHERE id=%s AND agent_id=%s "
            "AND status='completed'",
            (source_act, owner),
        )
        if cur.fetchone() is None:
            raise SocialConflict("completed source act not found")
        ordinal = data.get("source_action_ordinal")
        if ordinal is not None:
            cur.execute(
                "SELECT 1 FROM cognitive_actions WHERE act_id=%s AND agent_id=%s AND ordinal=%s",
                (source_act, owner, ordinal),
            )
            if cur.fetchone() is None:
                raise SocialConflict("source action not found")
        cur.execute(
            "SELECT id,attestation_principal_id FROM social_actors "
            "WHERE owner_agent_id=%s AND id IN (%s,%s)",
            (owner, issuer, subject),
        )
        actors = {row["id"]: row for row in cur.fetchall()}
        issuer_row = actors.get(issuer)
        if issuer_row is None or subject not in actors:
            raise SocialConflict("attestation actor coordinates are invalid")
        if issuer_row["attestation_principal_id"] != principal_id:
            raise SocialConflict("issuer is not authorized for this principal")
        cur.execute(
            "SELECT id,payload_hash FROM social_attestations WHERE owner_agent_id=%s "
            "AND scope_id=%s AND issuer_actor_id=%s AND attestation_key=%s FOR UPDATE",
            (owner, scope_id, issuer, data["attestation_key"]),
        )
        row = cur.fetchone()
        if row:
            if row["payload_hash"] != digest:
                raise SocialConflict("attestation key already has different semantics")
            return {
                "attestation_id": str(row["id"]), "duplicate": True,
                **_current_projection(cur, owner, scope_id, issuer, subject),
            }
        if scope["status"] != "active":
            raise SocialConflict("scope is dissolved")
        if kind == "revocation" and supersedes is None:
            raise SocialConflict("revocation requires supersedes_attestation_id")
        if supersedes:
            cur.execute(
                "SELECT issuer_actor_id,subject_actor_id,scope_id FROM social_attestations "
                "WHERE id=%s AND owner_agent_id=%s", (supersedes, owner),
            )
            prior = cur.fetchone()
            if prior is None or (prior["issuer_actor_id"], prior["subject_actor_id"], prior["scope_id"]) != (issuer, subject, scope_id):
                raise SocialConflict("revision target coordinates do not match")
            cur.execute(
                "SELECT 1 FROM social_attestations WHERE owner_agent_id=%s "
                "AND supersedes_attestation_id=%s", (owner, supersedes),
            )
            if cur.fetchone() is not None:
                raise SocialConflict("attestation has already been revised")
        attestation_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO social_attestations(id,owner_agent_id,scope_id,issuer_actor_id,"
            "subject_actor_id,attestation_key,attestation_kind,verdict,protocol_id,"
            "protocol_version,source_act_id,source_action_ordinal,evidence_refs,payload_hash,"
            "trust_level,signature_metadata,supersedes_attestation_id,issuer_principal_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (attestation_id, owner, scope_id, issuer, subject, data["attestation_key"],
             kind, data["verdict"], data["protocol_id"],
             data["protocol_version"], source_act, ordinal, Jsonb(evidence_refs), digest,
             data["trust_level"], Jsonb(signature), supersedes, principal_id),
        )
        projection = (
            {"projection_version": 0, "projection_status": "undetermined"}
            if issuer == subject
            else _recompute_projection(cur, owner, scope, issuer, subject)
        )
        return {"attestation_id": str(attestation_id), "duplicate": False, **projection}


def dissolve_scope(
    conn: psycopg.Connection, owner: str, principal_id: str, scope_id: str
) -> dict[str, Any]:
    sid = _uuid(scope_id, "scope_id")
    with conn.cursor(row_factory=dict_row) as cur:
        scope = _scope(cur, owner, sid)
        if scope["status"] == "dissolved":
            return {"scope_id": str(sid), "duplicate": True, "status": "dissolved"}
        cur.execute(
            "UPDATE social_scopes SET status='dissolved',dissolved_at=clock_timestamp() "
            "WHERE owner_agent_id=%s AND id=%s", (owner, sid),
        )
        payload_hash = _hash({
            "scope_id": str(sid), "operation_kind": "dissolve",
            "issuer_principal_id": principal_id,
        })
        cur.execute(
            "INSERT INTO social_scope_operations(owner_agent_id,scope_id,operation_kind,"
            "issuer_principal_id,payload_hash) VALUES (%s,%s,'dissolve',%s,%s)",
            (owner, sid, principal_id, payload_hash),
        )
        cur.execute(
            "UPDATE social_pair_projections SET status='scope_dissolved',"
            "projection_version=projection_version+1,computed_at=clock_timestamp() "
            "WHERE owner_agent_id=%s AND scope_id=%s", (owner, sid),
        )
        return {"scope_id": str(sid), "duplicate": False, "status": "dissolved"}


def create_grant(
    conn: psycopg.Connection, owner: str, principal_id: str, **data: Any
) -> dict[str, Any]:
    scope_id = _uuid(data["scope_id"], "scope_id")
    evidence_id = (
        _uuid(data["evidence_id"], "evidence_id") if data.get("evidence_id") else None
    )
    actor_a = _uuid(data["actor_a_id"], "actor_a_id") if data.get("actor_a_id") else None
    actor_b = _uuid(data["actor_b_id"], "actor_b_id") if data.get("actor_b_id") else None
    actor_low: uuid.UUID | None = None
    actor_high: uuid.UUID | None = None
    if actor_a is not None and actor_b is not None:
        if actor_a == actor_b:
            raise ValueError("projection grant requires distinct actors")
        actor_low, actor_high = sorted((actor_a, actor_b), key=str)
    payload = {
        "grant_key": data["grant_key"], "scope_id": str(scope_id),
        "grantee_principal_id": data["grantee_principal_id"],
        "capability": data["capability"], "evidence_class": data["evidence_class"],
        "evidence_id": str(evidence_id) if evidence_id else None,
        "actor_low_id": str(actor_low) if actor_low else None,
        "actor_high_id": str(actor_high) if actor_high else None,
        "expires_at": data.get("expires_at"),
        "issuer_principal_id": principal_id,
    }
    digest = _hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        _lock_key(cur, "grant", owner, data["grant_key"])
        scope = _scope(cur, owner, scope_id)
        cur.execute("SELECT id,payload_hash FROM social_visibility_grants WHERE owner_agent_id=%s AND grant_key=%s", (owner, data["grant_key"]))
        row = cur.fetchone()
        if row:
            if row["payload_hash"] != digest: raise SocialConflict("grant key already has different semantics")
            return {"grant_id": str(row["id"]), "duplicate": True}
        if scope["status"] != "active":
            raise SocialConflict("scope is dissolved")
        if data["capability"] == "social:encounter":
            if data["evidence_class"] != "encounter" or any(
                value is not None for value in (evidence_id, actor_low, actor_high)
            ):
                raise ValueError("invalid encounter delegation target")
        elif data["evidence_class"] == "projection":
            if evidence_id is not None or actor_low is None or actor_high is None:
                raise ValueError("projection grant requires an actor pair")
        elif evidence_id is None or actor_low is not None or actor_high is not None:
            raise ValueError("read grant requires one exact evidence id")
        if data["capability"] == "social:read" and data["evidence_class"] == "actor":
            cur.execute(
                "SELECT 1 FROM social_actors WHERE owner_agent_id=%s AND id=%s",
                (owner, evidence_id),
            )
        elif data["capability"] == "social:read" and data["evidence_class"] in {
            "attestation", "encounter",
        }:
            table = (
                "social_attestations"
                if data["evidence_class"] == "attestation"
                else "social_encounters"
            )
            cur.execute(
                f"SELECT 1 FROM {table} WHERE owner_agent_id=%s AND scope_id=%s AND id=%s",
                (owner, scope_id, evidence_id),
            )
        elif data["capability"] == "social:read":
            cur.execute(
                "SELECT 1 FROM social_actors WHERE owner_agent_id=%s "
                "AND id IN (%s,%s) HAVING count(*)=2",
                (owner, actor_low, actor_high),
            )
        else:
            cur.execute("SELECT 1")
        if cur.fetchone() is None:
            raise LookupError("social grant target not found")
        grant_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO social_visibility_grants(id,owner_agent_id,grant_key,scope_id,"
            "grantee_principal_id,capability,evidence_class,evidence_id,actor_low_id,"
            "actor_high_id,issuer_principal_id,expires_at,payload_hash) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (grant_id, owner, data["grant_key"], scope_id, data["grantee_principal_id"],
             data["capability"], data["evidence_class"], evidence_id, actor_low,
             actor_high, principal_id,
             data.get("expires_at"), digest),
        )
        return {"grant_id": str(grant_id), "duplicate": False}


def revoke_grant(
    conn: psycopg.Connection,
    owner: str,
    principal_id: str,
    *,
    revocation_key: str,
    grant_id: str,
) -> dict[str, Any]:
    gid = _uuid(grant_id, "grant_id")
    payload_hash = _hash({
        "operation_kind": "revoke",
        "revocation_key": revocation_key,
        "grant_id": str(gid),
        "issuer_principal_id": principal_id,
    })
    with conn.cursor(row_factory=dict_row) as cur:
        _lock_key(cur, "grant-revocation", owner, revocation_key)
        cur.execute(
            "SELECT grant_id,payload_hash FROM social_grant_operations "
            "WHERE owner_agent_id=%s AND operation_key=%s FOR UPDATE",
            (owner, revocation_key),
        )
        prior = cur.fetchone()
        if prior is not None:
            if prior["payload_hash"] != payload_hash:
                raise SocialConflict("revocation key already has different semantics")
            return {"grant_id": str(prior["grant_id"]), "duplicate": True, "revoked": True}
        cur.execute(
            "SELECT revoked_at FROM social_visibility_grants "
            "WHERE owner_agent_id=%s AND id=%s FOR UPDATE",
            (owner, gid),
        )
        grant = cur.fetchone()
        if grant is None:
            raise LookupError("social grant not found")
        if grant["revoked_at"] is not None:
            raise SocialConflict("social grant is already revoked")
        cur.execute(
            "INSERT INTO social_grant_operations(owner_agent_id,operation_key,"
            "operation_kind,grant_id,issuer_principal_id,payload_hash) "
            "VALUES (%s,%s,'revoke',%s,%s,%s)",
            (owner, revocation_key, gid, principal_id, payload_hash),
        )
        cur.execute(
            "UPDATE social_visibility_grants SET revoked_at=clock_timestamp() "
            "WHERE owner_agent_id=%s AND id=%s",
            (owner, gid),
        )
        return {"grant_id": str(gid), "duplicate": False, "revoked": True}


def query_projection(
    conn: psycopg.Connection,
    owner: str,
    *,
    scope_id: str,
    actor_a_id: str,
    actor_b_id: str,
    visibility_principal_id: str | None = None,
) -> dict[str, Any]:
    sid, a, b = _uuid(scope_id,"scope_id"), _uuid(actor_a_id,"actor_a_id"), _uuid(actor_b_id,"actor_b_id")
    low, high = sorted((a,b), key=str)
    with conn.cursor(row_factory=dict_row) as cur:
        _scope(cur, owner, sid, lock=False)
        if visibility_principal_id is not None and not _has_visibility_grant_cur(
            cur,
            owner,
            scope_id=sid,
            principal_id=visibility_principal_id,
            evidence_class="projection",
            actor_a_id=a,
            actor_b_id=b,
        ):
            raise LookupError("social projection not found")
        cur.execute(
            "SELECT projection_version,status,policy_hash,computed_at FROM social_pair_projections "
            "WHERE owner_agent_id=%s AND scope_id=%s AND actor_low_id=%s AND actor_high_id=%s",
            (owner,sid,low,high),
        )
        row=cur.fetchone()
        if row is None: return {"scope_id":str(sid),"actor_low_id":str(low),"actor_high_id":str(high),"projection_version":0,"status":"undetermined","policy_hash":None,"computed_at":None}
        return {"scope_id":str(sid),"actor_low_id":str(low),"actor_high_id":str(high),**dict(row)}


def has_visibility_grant(
    conn: psycopg.Connection,
    owner: str,
    *,
    scope_id: str,
    principal_id: str,
    evidence_class: str,
    capability: str = "social:read",
    evidence_id: str | None = None,
    actor_a_id: str | None = None,
    actor_b_id: str | None = None,
) -> bool:
    """Return only a boolean; callers use 404 to avoid scope enumeration."""
    sid = _uuid(scope_id, "scope_id")
    eid = _uuid(evidence_id, "evidence_id") if evidence_id else None
    actor_a = _uuid(actor_a_id, "actor_a_id") if actor_a_id else None
    actor_b = _uuid(actor_b_id, "actor_b_id") if actor_b_id else None
    with conn.cursor() as cur:
        return _has_visibility_grant_cur(
            cur,
            owner,
            scope_id=sid,
            principal_id=principal_id,
            evidence_class=evidence_class,
            capability=capability,
            evidence_id=eid,
            actor_a_id=actor_a,
            actor_b_id=actor_b,
        )


def explain_scope(
    conn: psycopg.Connection, owner: str, *, scope_id: str
) -> dict[str, Any]:
    """Content-free audit projection; labels, summaries and evidence stay private."""
    sid = _uuid(scope_id, "scope_id")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,protocol_id,protocol_version,policy_hash,status,created_at,dissolved_at "
            "FROM social_scopes WHERE owner_agent_id=%s AND id=%s", (owner, sid),
        )
        scope = cur.fetchone()
        if scope is None:
            raise LookupError("scope not found")
        counts: dict[str, int] = {}
        for key, table in (
            ("actors", "social_actors"),
            ("encounters", "social_encounters"),
            ("attestations", "social_attestations"),
            ("projections", "social_pair_projections"),
            ("deliveries", "social_delivery_receipts"),
        ):
            if key == "actors":
                cur.execute(
                    "SELECT count(DISTINCT actor_id)::int FROM ("
                    "SELECT observer_actor_id AS actor_id FROM social_encounters "
                    "WHERE owner_agent_id=%s AND scope_id=%s UNION SELECT encountered_actor_id "
                    "FROM social_encounters WHERE owner_agent_id=%s AND scope_id=%s UNION "
                    "SELECT issuer_actor_id FROM social_attestations WHERE owner_agent_id=%s "
                    "AND scope_id=%s UNION SELECT subject_actor_id FROM social_attestations "
                    "WHERE owner_agent_id=%s AND scope_id=%s) scoped",
                    (owner, sid, owner, sid, owner, sid, owner, sid),
                )
            else:
                cur.execute(
                    f"SELECT count(*)::int FROM {table} WHERE owner_agent_id=%s AND scope_id=%s",
                    (owner, sid),
                )
            counts[key] = int(cur.fetchone()["count"])
        cur.execute(
            "SELECT actor_low_id,actor_high_id,projection_version,status,policy_hash,computed_at "
            "FROM social_pair_projections WHERE owner_agent_id=%s AND scope_id=%s "
            "ORDER BY actor_low_id,actor_high_id LIMIT 256", (owner, sid),
        )
        projections = [dict(row) for row in cur.fetchall()]
    return {"scope": dict(scope), "counts": counts, "projections": projections}


def deliver_evidence(
    conn: psycopg.Connection,
    owner: str,
    principal_id: str,
    *,
    delivery_key: str,
    scope_id: str,
    evidence_class: str,
    evidence_id: str,
    receiving_agent_id: str,
    pending_cap: int = 1024,
    event_cap: int = 1024,
    global_event_cap: int = 100_000,
) -> dict[str, Any]:
    """Publish one explicitly granted, bounded social coordinate as observation."""
    from styx.storage.observations import ingest_observation
    from styx.storage.ready_events import create_observation_ready_event

    sid = _uuid(scope_id, "scope_id")
    eid = _uuid(evidence_id, "evidence_id")
    if evidence_class not in {"attestation", "encounter"}:
        raise ValueError("delivery evidence_class must be attestation or encounter")
    payload = {
        "delivery_key": delivery_key,
        "scope_id": str(sid),
        "evidence_class": evidence_class,
        "evidence_id": str(eid),
        "receiving_agent_id": receiving_agent_id,
        "grantee_principal_id": principal_id,
    }
    digest = _hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        _lock_key(cur, "delivery", owner, delivery_key)
        scope = _scope(cur, owner, sid)
        if scope["status"] != "active":
            raise SocialConflict("scope is dissolved")
        if not _has_visibility_grant_cur(
            cur,
            owner,
            scope_id=sid,
            principal_id=principal_id,
            evidence_class=evidence_class,
            capability="social:read",
            evidence_id=eid,
        ):
            raise LookupError("social evidence not found")
        cur.execute(
            "SELECT id,payload_hash,observation_id FROM social_delivery_receipts "
            "WHERE owner_agent_id=%s AND delivery_key=%s FOR UPDATE",
            (owner, delivery_key),
        )
        prior = cur.fetchone()
        if prior is not None:
            if prior["payload_hash"] != digest:
                raise SocialConflict("delivery key already has different semantics")
            return {
                "delivery_id": str(prior["id"]),
                "observation_id": str(prior["observation_id"]),
                "duplicate": True,
            }
        table = "social_attestations" if evidence_class == "attestation" else "social_encounters"
        cur.execute(
            f"SELECT sequence,"
            + ("payload_hash AS evidence_hash" if evidence_class == "attestation" else "evidence_hash")
            + ",created_at"
            + (",verdict,trust_level" if evidence_class == "attestation" else ",confidence")
            + f" FROM {table} WHERE owner_agent_id=%s AND scope_id=%s AND id=%s",
            (owner, sid, eid),
        )
        evidence = cur.fetchone()
        if evidence is None:
            raise LookupError("social evidence not found")
        if evidence_class == "attestation":
            content = (
                "Scoped social attestation received: "
                f"verdict={evidence['verdict']}; trust={evidence['trust_level']}."
            )
            confidence = 1.0 if evidence["trust_level"] == "verified" else 0.5
        else:
            content = "Scoped social encounter evidence received."
            confidence = float(evidence["confidence"])
        observation = ingest_observation(
            conn, receiving_agent_id,
            source_id=f"social-scope:{sid}",
            source_stream=evidence_class,
            source_sequence=int(evidence["sequence"]),
            observation_key=f"{evidence_class}:{eid}",
            difference_kind="external_signal",
            content=content,
            salience=0.6,
            confidence=confidence,
            reducer_name="social-ledger",
            reducer_version="1",
            source_observed_at=evidence["created_at"],
            metadata={
                "scope_id": str(sid),
                "protocol_id": scope["protocol_id"],
                "protocol_version": scope["protocol_version"],
                "evidence_class": evidence_class,
                "evidence_hash": evidence["evidence_hash"],
            },
            pending_cap=pending_cap,
        )
        if not observation.duplicate:
            create_observation_ready_event(
                conn, receiving_agent_id,
                source_generation=observation.ingest_seq,
                observation_high_water=observation.ingest_seq,
                pending_count=observation.pending_count,
                event_cap=event_cap,
                global_event_cap=global_event_cap,
            )
        delivery_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO social_delivery_receipts(id,owner_agent_id,delivery_key,scope_id,"
            "attestation_id,encounter_id,receiving_agent_id,observation_id,payload_hash) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (delivery_id, owner, delivery_key, sid,
             eid if evidence_class == "attestation" else None,
             eid if evidence_class == "encounter" else None,
             receiving_agent_id, observation.observation_id, digest),
        )
        return {
            "delivery_id": str(delivery_id),
            "observation_id": str(observation.observation_id),
            "duplicate": False,
        }
