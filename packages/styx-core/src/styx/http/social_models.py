"""Strict HTTP contracts for scoped social evidence (wave 42)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from styx.storage.cognition import validate_journal_json


Hash = str


class SocialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1, max_length=256)


class SocialActorRequest(SocialRequest):
    identity_namespace: str = Field(min_length=1, max_length=128)
    actor_key: str = Field(min_length=1, max_length=256)
    actor_kind: Literal["local_agent", "external_agent", "human", "collective", "unknown"]
    private_label: str | None = Field(default=None, max_length=256)
    identity_evidence_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    attestation_principal_id: str | None = Field(default=None, min_length=1, max_length=128)


class SocialScopeRequest(SocialRequest):
    scope_key: str = Field(min_length=1, max_length=256)
    protocol_id: str = Field(min_length=1, max_length=128)
    protocol_version: str = Field(min_length=1, max_length=64)
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")


class SocialEncounterRequest(SocialRequest):
    encounter_key: str = Field(min_length=1, max_length=256)
    scope_id: str
    observer_actor_id: str
    encountered_actor_id: str
    direction: Literal["inbound", "outbound", "bidirectional"]
    channel_kind: str = Field(min_length=1, max_length=64)
    source_act_id: str | None = None
    source_observation_id: str | None = None
    summary: str | None = Field(default=None, max_length=1000)
    evidence_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def source_required(self) -> "SocialEncounterRequest":
        if self.source_act_id is None and self.source_observation_id is None:
            raise ValueError("encounter requires a source act or observation")
        return self


class SocialAttestationRequest(SocialRequest):
    scope_id: str
    issuer_actor_id: str
    subject_actor_id: str
    attestation_key: str = Field(min_length=1, max_length=256)
    attestation_kind: Literal["direct", "self", "revocation", "reported"] = "direct"
    verdict: Literal["positive", "negative", "undetermined"]
    protocol_id: str = Field(min_length=1, max_length=128)
    protocol_version: str = Field(min_length=1, max_length=64)
    source_act_id: str
    source_action_ordinal: int | None = Field(default=None, ge=0)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    trust_level: Literal["verified", "unverified"]
    signature_metadata: dict[str, Any] = Field(default_factory=dict)
    supersedes_attestation_id: str | None = None

    @model_validator(mode="after")
    def bounded_evidence(self) -> "SocialAttestationRequest":
        validate_journal_json(self.evidence_refs, max_string=256)
        validate_journal_json(self.signature_metadata, max_string=256)
        return self


class SocialAttestationReviseRequest(SocialAttestationRequest):
    supersedes_attestation_id: str


class SocialScopeDissolveRequest(SocialRequest):
    scope_id: str


class SocialGrantRequest(SocialRequest):
    grant_key: str = Field(min_length=1, max_length=256)
    scope_id: str
    grantee_principal_id: str = Field(min_length=1, max_length=128)
    capability: Literal["social:read", "social:encounter"]
    evidence_class: Literal["actor", "encounter", "attestation", "projection"]
    evidence_id: str | None = None
    actor_a_id: str | None = None
    actor_b_id: str | None = None
    expires_at: dt.datetime | None = None

    @model_validator(mode="after")
    def aware_expiry(self) -> "SocialGrantRequest":
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        if self.capability == "social:encounter":
            if self.evidence_class != "encounter" or any(
                value is not None
                for value in (self.evidence_id, self.actor_a_id, self.actor_b_id)
            ):
                raise ValueError("encounter delegation must be scope-scoped")
        elif self.evidence_class == "projection":
            if (
                self.evidence_id is not None
                or self.actor_a_id is None
                or self.actor_b_id is None
                or self.actor_a_id == self.actor_b_id
            ):
                raise ValueError("projection grant requires an actor pair")
        elif (
            self.evidence_id is None
            or self.actor_a_id is not None
            or self.actor_b_id is not None
        ):
            raise ValueError("read grant requires one exact evidence id")
        return self


class SocialGrantRevokeRequest(SocialRequest):
    revocation_key: str = Field(min_length=1, max_length=256)
    grant_id: str


class SocialQueryRequest(SocialRequest):
    scope_id: str
    actor_a_id: str
    actor_b_id: str


class SocialExplainRequest(SocialRequest):
    scope_id: str


class SocialDeliveryRequest(SocialRequest):
    delivery_key: str = Field(min_length=1, max_length=256)
    scope_id: str
    evidence_class: Literal["attestation", "encounter"]
    evidence_id: str
    receiving_agent_id: str = Field(min_length=1, max_length=256)
