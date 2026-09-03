"""Separate hash-only principal registry for deny-by-default social routes."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from fastapi import Header, HTTPException, Request


CAPABILITIES = frozenset({
    "social:scope-admin", "social:attest", "social:encounter", "social:read",
})


@dataclass(frozen=True)
class SocialPrincipal:
    principal_id: str
    token_sha256: str
    agent_ids: frozenset[str]
    capabilities: frozenset[str]
    _presented_token: str | None = field(default=None, repr=False, compare=False)

    def allows(self, agent_id: str, capability: str) -> bool:
        return agent_id in self.agent_ids and capability in self.capabilities

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def verifies_body(self, body: bytes, signature: str) -> bool:
        if (
            self._presented_token is None
            or len(signature) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in signature)
        ):
            return False
        expected = hmac.new(
            self._presented_token.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return secrets.compare_digest(expected, signature.lower())


def load_social_principals(path: str | None) -> tuple[SocialPrincipal, ...]:
    if not path:
        return ()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid social principal registry") from None
    entries = raw.get("principals") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or len(entries) > 256:
        raise ValueError("invalid social principal registry")
    result: list[SocialPrincipal] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "principal_id", "token_sha256", "agent_ids", "capabilities"
        }:
            raise ValueError("invalid social principal entry")
        principal_id = item["principal_id"]
        token_hash = item["token_sha256"]
        agents = item["agent_ids"]
        capabilities = item["capabilities"]
        if not isinstance(principal_id, str) or not 1 <= len(principal_id) <= 128:
            raise ValueError("invalid social principal id")
        if not isinstance(token_hash, str) or len(token_hash) != 64 or any(
            char not in "0123456789abcdef" for char in token_hash
        ):
            raise ValueError("invalid social principal token hash")
        if not isinstance(agents, list) or not agents or len(agents) > 256:
            raise ValueError("invalid social principal agent grants")
        if any(not isinstance(value, str) or not 1 <= len(value) <= 256 for value in agents):
            raise ValueError("invalid social principal agent id")
        if (
            not isinstance(capabilities, list)
            or len(capabilities) > len(CAPABILITIES)
            or any(not isinstance(value, str) for value in capabilities)
            or not set(capabilities) <= CAPABILITIES
        ):
            raise ValueError("invalid social principal capabilities")
        if principal_id in seen_ids or token_hash in seen_hashes:
            raise ValueError("duplicate social principal coordinate")
        seen_ids.add(principal_id)
        seen_hashes.add(token_hash)
        result.append(SocialPrincipal(
            principal_id=principal_id,
            token_sha256=token_hash,
            agent_ids=frozenset(agents),
            capabilities=frozenset(capabilities),
        ))
    return tuple(result)


def require_social_principal(
    request: Request,
    x_styx_social_token: str | None = Header(default=None),
) -> SocialPrincipal:
    registry: tuple[SocialPrincipal, ...] = getattr(
        request.app.state, "social_principals", ()
    )
    if not registry:
        raise HTTPException(status_code=404, detail="not found")
    if not x_styx_social_token:
        raise HTTPException(status_code=401, detail="missing social principal token")
    presented_hash = hashlib.sha256(x_styx_social_token.encode("utf-8")).hexdigest()
    for principal in registry:
        if secrets.compare_digest(presented_hash, principal.token_sha256):
            return replace(principal, _presented_token=x_styx_social_token)
    raise HTTPException(status_code=401, detail="invalid social principal token")


def require_social_grant(
    principal: SocialPrincipal, agent_id: str, capability: str
) -> None:
    if not principal.allows(agent_id, capability):
        raise HTTPException(status_code=404, detail="not found")
