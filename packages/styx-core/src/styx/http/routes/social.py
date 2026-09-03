"""Deny-by-default scoped social evidence routes (wave 42)."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.social_auth import (
    SocialPrincipal,
    require_social_grant,
    require_social_principal,
)
from styx.http.social_models import (
    SocialActorRequest,
    SocialAttestationRequest,
    SocialAttestationReviseRequest,
    SocialDeliveryRequest,
    SocialEncounterRequest,
    SocialExplainRequest,
    SocialGrantRequest,
    SocialGrantRevokeRequest,
    SocialQueryRequest,
    SocialScopeDissolveRequest,
    SocialScopeRequest,
)
from styx.storage.social import SocialConflict

router = APIRouter(dependencies=[Depends(require_auth)])


def _invoke(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except SocialConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _owner(req: Any, principal: SocialPrincipal, capability: str):
    require_social_grant(principal, req.agent_id, capability)
    return _core(req.agent_id)


def _core(agent_id: str):
    session = registry.get_optional(agent_id)
    if session is None:
        raise HTTPException(status_code=404, detail="not found")
    return session.core


def _delegated_principal(
    req: Any, principal: SocialPrincipal, capability: str
) -> str | None:
    if principal.allows(req.agent_id, capability):
        return None
    if not principal.has_capability(capability):
        raise HTTPException(status_code=404, detail="not found")
    return principal.principal_id


@router.post("/social/actors")
def create_actor(req: SocialActorRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    core = _owner(req, principal, "social:scope-admin")
    return _invoke(lambda: core.social_create_actor(
        principal_id=principal.principal_id, **req.model_dump(exclude={"agent_id"})
    ))


@router.post("/social/scopes")
def create_scope(req: SocialScopeRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    core = _owner(req, principal, "social:scope-admin")
    return _invoke(lambda: core.social_create_scope(
        principal_id=principal.principal_id,
        **req.model_dump(exclude={"agent_id"}),
    ))


@router.post("/social/encounters")
def create_encounter(req: SocialEncounterRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    visibility_principal_id = _delegated_principal(
        req, principal, "social:encounter"
    )
    core = _core(req.agent_id)
    return _invoke(lambda: core.social_create_encounter(
        principal_id=principal.principal_id,
        visibility_principal_id=visibility_principal_id,
        **req.model_dump(exclude={"agent_id"}),
    ))


async def _attest(
    req: SocialAttestationRequest,
    request: Request,
    principal: SocialPrincipal,
    signature: str | None,
) -> dict[str, Any]:
    core = _owner(req, principal, "social:attest")
    signature_verified = False
    signature_metadata = req.signature_metadata
    if req.trust_level == "verified":
        body = await request.body()
        if signature is None or not principal.verifies_body(body, signature):
            raise HTTPException(status_code=401, detail="invalid social act signature")
        signature_verified = True
        signature_metadata = {
            "scheme": "hmac-sha256",
            "signature_sha256": hashlib.sha256(signature.encode("ascii")).hexdigest(),
            "declared": signature_metadata,
        }
    data = req.model_dump(exclude={"agent_id", "signature_metadata"})
    return _invoke(lambda: core.social_create_attestation(
        principal_id=principal.principal_id,
        signature_verified=signature_verified,
        signature_metadata=signature_metadata,
        **data,
    ))


@router.post("/social/attestations")
async def create_attestation(
    req: SocialAttestationRequest,
    request: Request,
    principal: SocialPrincipal = Depends(require_social_principal),
    x_styx_social_signature: str | None = Header(default=None),
) -> dict[str, Any]:
    if req.supersedes_attestation_id is not None:
        raise HTTPException(status_code=422, detail="use /social/attestations/revise")
    return await _attest(req, request, principal, x_styx_social_signature)


@router.post("/social/attestations/revise")
async def revise_attestation(
    req: SocialAttestationReviseRequest,
    request: Request,
    principal: SocialPrincipal = Depends(require_social_principal),
    x_styx_social_signature: str | None = Header(default=None),
) -> dict[str, Any]:
    return await _attest(req, request, principal, x_styx_social_signature)


@router.post("/social/scopes/dissolve")
def dissolve_scope(req: SocialScopeDissolveRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    core = _owner(req, principal, "social:scope-admin")
    return _invoke(lambda: core.social_dissolve_scope(
        principal_id=principal.principal_id, scope_id=req.scope_id,
    ))


@router.post("/social/grants")
def create_grant(req: SocialGrantRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    core = _owner(req, principal, "social:scope-admin")
    return _invoke(lambda: core.social_create_grant(
        principal_id=principal.principal_id,
        **req.model_dump(exclude={"agent_id"}),
    ))


@router.post("/social/grants/revoke")
def revoke_grant(req: SocialGrantRevokeRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    core = _owner(req, principal, "social:scope-admin")
    return _invoke(lambda: core.social_revoke_grant(
        principal_id=principal.principal_id,
        revocation_key=req.revocation_key,
        grant_id=req.grant_id,
    ))


@router.post("/social/query")
def query(req: SocialQueryRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    visibility_principal_id = _delegated_principal(req, principal, "social:read")
    core = _core(req.agent_id)
    return _invoke(lambda: core.social_query(
        visibility_principal_id=visibility_principal_id,
        **req.model_dump(exclude={"agent_id"}),
    ))


@router.post("/social/explain")
def explain(req: SocialExplainRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    core = _owner(req, principal, "social:read")
    return _invoke(lambda: core.social_explain(scope_id=req.scope_id))


@router.post("/social/deliver")
def deliver(req: SocialDeliveryRequest, principal: SocialPrincipal = Depends(require_social_principal)) -> dict[str, Any]:
    require_social_grant(principal, req.receiving_agent_id, "social:read")
    core = _core(req.agent_id)
    return _invoke(lambda: core.social_deliver(
        principal_id=principal.principal_id, **req.model_dump(exclude={"agent_id"})
    ))
