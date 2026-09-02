"""Build conservative Hermes execution provenance without network coordinates."""

from __future__ import annotations

from typing import Any

from styx_hermes import __version__


def execution_provenance(kwargs: dict[str, Any]) -> dict[str, Any]:
    runtime_raw = str(
        kwargs.get("runtime_family") or kwargs.get("runtime") or kwargs.get("platform") or ""
    ).lower()
    runtime = runtime_raw if runtime_raw in {"ollama", "sglang", "vllm", "cloud"} else "unknown"
    provider_raw = str(kwargs.get("provider_family") or kwargs.get("provider") or "").lower()
    if provider_raw in {"ollama", "openai", "anthropic", "openai_compatible", "other"}:
        provider = provider_raw
    elif runtime == "ollama":
        provider = "ollama"
    elif runtime in {"sglang", "vllm"}:
        provider = "openai_compatible"
    else:
        provider = "other"
    protocol_raw = str(kwargs.get("protocol") or "").lower()
    if protocol_raw in {"chat_completions", "responses", "messages", "generate", "native"}:
        protocol = protocol_raw
    elif runtime == "ollama":
        protocol = "generate"
    elif runtime in {"sglang", "vllm"}:
        protocol = "chat_completions"
    else:
        protocol = "unknown"
    model = str(kwargs.get("model") or "unknown")[:256]
    return {
        "schema_version": 1,
        "provider_family": provider,
        "runtime_family": runtime,
        "model_id": model,
        "model_revision": None,
        "endpoint_id": "hermes-default",
        "adapter": "styx-hermes",
        "adapter_version": __version__,
        "protocol": protocol,
        "sampling_hash": None,
        "toolset_hash": None,
    }
