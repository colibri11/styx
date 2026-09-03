from __future__ import annotations

import pytest

from styx.engine.execution_provenance import (
    execution_provenance_hash,
    normalize_execution_provenance,
)


def _provenance(*, runtime: str = "sglang", model: str = "model-a") -> dict:
    return {
        "schema_version": 1,
        "provider_family": "openai_compatible",
        "runtime_family": runtime,
        "model_id": model,
        "model_revision": None,
        "endpoint_id": "local-coding",
        "adapter": "styx-hermes",
        "adapter_version": "1.6.0",
        "protocol": "chat_completions",
        "sampling_hash": None,
        "toolset_hash": None,
    }


def test_ollama_and_sglang_are_distinct_typed_runtimes() -> None:
    sglang = normalize_execution_provenance(_provenance())
    ollama = normalize_execution_provenance({
        **_provenance(runtime="ollama"),
        "provider_family": "ollama",
        "protocol": "generate",
    })
    assert sglang["runtime_family"] == "sglang"
    assert ollama["runtime_family"] == "ollama"
    assert execution_provenance_hash(sglang) != execution_provenance_hash(ollama)


def test_legacy_coordinates_normalize_deterministically() -> None:
    first = normalize_execution_provenance(None, legacy_model="qwen", legacy_platform="ollama")
    second = normalize_execution_provenance(None, legacy_model="qwen", legacy_platform="ollama")
    assert first == second
    assert first["runtime_family"] == "ollama"
    assert execution_provenance_hash(first) == execution_provenance_hash(second)


@pytest.mark.parametrize("field,value", [
    ("endpoint_id", "http://sglang.example.invalid:30000"),
    ("endpoint_id", ".".join(("192", "0", "2", "31"))),
    ("model_id", "api_key=not-allowed"),
    ("sampling_hash", "short"),
])
def test_provenance_rejects_network_credentials_and_bad_hashes(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        normalize_execution_provenance({**_provenance(), field: value})


def test_provenance_rejects_unknown_or_missing_fields() -> None:
    with pytest.raises(ValueError):
        normalize_execution_provenance({**_provenance(), "url": "local"})
    raw = _provenance()
    del raw["toolset_hash"]
    with pytest.raises(ValueError):
        normalize_execution_provenance(raw)
