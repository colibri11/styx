from __future__ import annotations

import hashlib
import json

import pytest

from styx.http.social_auth import SocialPrincipal, load_social_principals


def _entry(principal_id: str, token: str) -> dict:
    return {
        "principal_id": principal_id,
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "agent_ids": ["agent-a"],
        "capabilities": ["social:read"],
    }


def test_registry_loads_hashes_without_plaintext(tmp_path) -> None:
    path = tmp_path / "principals.json"
    path.write_text(json.dumps({"principals": [_entry("p1", "secret")]}))
    result = load_social_principals(str(path))
    assert result[0].principal_id == "p1"
    assert result[0].token_sha256 != "secret"


def test_registry_rejects_duplicate_identity_or_token(tmp_path) -> None:
    for entries in (
        [_entry("p1", "one"), _entry("p1", "two")],
        [_entry("p1", "one"), _entry("p2", "one")],
    ):
        path = tmp_path / "principals.json"
        path.write_text(json.dumps({"principals": entries}))
        with pytest.raises(ValueError, match="duplicate"):
            load_social_principals(str(path))


def test_registry_read_error_does_not_expose_path(tmp_path) -> None:
    private_path = tmp_path / "private-registry-marker.json"
    with pytest.raises(ValueError) as raised:
        load_social_principals(str(private_path))
    assert str(private_path) not in str(raised.value)


def test_body_signature_rejects_non_ascii_hex_without_raising() -> None:
    principal = SocialPrincipal(
        "p1", hashlib.sha256(b"secret").hexdigest(),
        frozenset({"agent-a"}), frozenset({"social:attest"}),
        _presented_token="secret",
    )
    assert principal.verifies_body(b"{}", "я" * 64) is False
