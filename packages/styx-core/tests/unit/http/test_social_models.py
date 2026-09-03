from __future__ import annotations

import pytest
from pydantic import ValidationError

from styx.http.social_models import (
    SocialAttestationRequest,
    SocialEncounterRequest,
    SocialGrantRequest,
)


BASE = {
    "agent_id": "agent-a",
    "scope_id": "00000000-0000-0000-0000-000000000001",
    "issuer_actor_id": "00000000-0000-0000-0000-000000000002",
    "subject_actor_id": "00000000-0000-0000-0000-000000000003",
    "attestation_key": "act-1",
    "verdict": "positive",
    "protocol_id": "local-vouch",
    "protocol_version": "1",
    "source_act_id": "00000000-0000-0000-0000-000000000004",
    "trust_level": "verified",
}


def test_attestation_contract_is_strict_and_bounded() -> None:
    assert SocialAttestationRequest.model_validate(BASE).verdict == "positive"
    with pytest.raises(ValidationError):
        SocialAttestationRequest.model_validate({**BASE, "is_person": True})
    with pytest.raises(ValidationError):
        SocialAttestationRequest.model_validate(
            {**BASE, "evidence_refs": [{"x": "z" * 257}]}
        )


def test_encounter_requires_explicit_source() -> None:
    with pytest.raises(ValidationError):
        SocialEncounterRequest.model_validate({
            "agent_id": "agent-a", "encounter_key": "e",
            "scope_id": BASE["scope_id"],
            "observer_actor_id": BASE["issuer_actor_id"],
            "encountered_actor_id": BASE["subject_actor_id"],
            "direction": "inbound", "channel_kind": "test",
            "evidence_hash": "a" * 64, "confidence": 0.5,
        })


def test_visibility_grants_require_exact_read_coordinates() -> None:
    base = {
        "agent_id": "agent-a",
        "grant_key": "grant-a",
        "scope_id": BASE["scope_id"],
        "grantee_principal_id": "reader",
        "capability": "social:read",
        "evidence_class": "attestation",
    }
    with pytest.raises(ValidationError):
        SocialGrantRequest.model_validate(base)
    assert SocialGrantRequest.model_validate({
        **base,
        "evidence_id": BASE["source_act_id"],
    }).evidence_id == BASE["source_act_id"]
    projection = {
        **base,
        "evidence_class": "projection",
        "actor_a_id": BASE["issuer_actor_id"],
        "actor_b_id": BASE["subject_actor_id"],
    }
    assert SocialGrantRequest.model_validate(projection).evidence_id is None
