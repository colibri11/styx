"""Strict, content-free execution provenance for cognitive acts (wave 41)."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from typing import Any, Mapping
from urllib.parse import urlparse


SCHEMA_VERSION = 1
PROVIDER_FAMILIES = frozenset(
    {"ollama", "openai", "anthropic", "openai_compatible", "other"}
)
RUNTIME_FAMILIES = frozenset(
    {"ollama", "sglang", "vllm", "cloud", "unknown"}
)
PROTOCOLS = frozenset(
    {"chat_completions", "responses", "messages", "generate", "native", "unknown"}
)
FIELDS = frozenset(
    {
        "schema_version", "provider_family", "runtime_family", "model_id",
        "model_revision", "endpoint_id", "adapter", "adapter_version",
        "protocol", "sampling_hash", "toolset_hash",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]*$")
_SECRETISH = re.compile(
    r"(?i)(bearer\s|api[_-]?key|password|passwd|secret|credential|token[=:])"
)


def _bounded_id(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError(f"execution_provenance.{field} must contain 1..256 characters")
    if _SECRETISH.search(value) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"execution_provenance.{field} is not a safe identifier")
    return value


def _endpoint_alias(value: Any) -> str:
    alias = _bounded_id(value, "endpoint_id")
    assert alias is not None
    parsed = urlparse(alias)
    if parsed.scheme or "://" in alias:
        raise ValueError("execution_provenance.endpoint_id must be an alias, not a URL")
    candidate = alias.strip("[]")
    try:
        ipaddress.ip_address(candidate.split(":", 1)[0])
    except ValueError:
        pass
    else:
        raise ValueError("execution_provenance.endpoint_id must not contain an IP address")
    return alias


def _optional_hash(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"execution_provenance.{field} must be a lowercase sha256")
    return value


def normalize_execution_provenance(
    value: Mapping[str, Any] | None,
    *,
    legacy_model: str | None = None,
    legacy_platform: str | None = None,
    default_adapter: str = "unknown",
    default_adapter_version: str | None = None,
) -> dict[str, Any]:
    """Return canonical schema v1, including deterministic legacy normalization."""
    if value is None:
        platform = (legacy_platform or "").strip().lower()
        runtime = platform if platform in RUNTIME_FAMILIES else "unknown"
        provider = "ollama" if runtime == "ollama" else (
            "openai_compatible" if runtime in {"sglang", "vllm"} else "other"
        )
        raw: dict[str, Any] = {
            "schema_version": 1,
            "provider_family": provider,
            "runtime_family": runtime,
            "model_id": legacy_model or "unknown",
            "model_revision": None,
            "endpoint_id": "legacy",
            "adapter": default_adapter,
            "adapter_version": default_adapter_version,
            "protocol": "unknown",
            "sampling_hash": None,
            "toolset_hash": None,
        }
    else:
        if not isinstance(value, Mapping):
            raise ValueError("execution_provenance must be an object")
        unknown = set(value) - FIELDS
        missing = FIELDS - set(value)
        if unknown or missing:
            raise ValueError("execution_provenance must contain exactly the schema v1 fields")
        raw = dict(value)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("execution_provenance.schema_version must be 1")
    provider = raw.get("provider_family")
    runtime = raw.get("runtime_family")
    protocol = raw.get("protocol")
    if provider not in PROVIDER_FAMILIES:
        raise ValueError("execution_provenance.provider_family is unsupported")
    if runtime not in RUNTIME_FAMILIES:
        raise ValueError("execution_provenance.runtime_family is unsupported")
    if protocol not in PROTOCOLS:
        raise ValueError("execution_provenance.protocol is unsupported")
    result = {
        "schema_version": 1,
        "provider_family": provider,
        "runtime_family": runtime,
        "model_id": _bounded_id(raw.get("model_id"), "model_id"),
        "model_revision": _bounded_id(raw.get("model_revision"), "model_revision", nullable=True),
        "endpoint_id": _endpoint_alias(raw.get("endpoint_id")),
        "adapter": _bounded_id(raw.get("adapter"), "adapter"),
        "adapter_version": _bounded_id(raw.get("adapter_version"), "adapter_version", nullable=True),
        "protocol": protocol,
        "sampling_hash": _optional_hash(raw.get("sampling_hash"), "sampling_hash"),
        "toolset_hash": _optional_hash(raw.get("toolset_hash"), "toolset_hash"),
    }
    # Defensive: json encoder must never silently serialize NaN-like custom data.
    if any(isinstance(item, float) and not math.isfinite(item) for item in result.values()):
        raise ValueError("execution_provenance must contain finite values")
    return result


def execution_provenance_hash(value: Mapping[str, Any]) -> str:
    normalized = normalize_execution_provenance(value)
    material = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
