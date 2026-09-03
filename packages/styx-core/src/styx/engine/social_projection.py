"""Pure scoped mutual-attestation projection (wave 42)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def evaluate_pair_projection(
    attestations: Iterable[Mapping[str, Any]],
    *,
    actor_low_id: str,
    actor_high_id: str,
    scope_status: str = "active",
) -> dict[str, Any]:
    """Evaluate only latest verified direct acts in one pre-filtered scope."""
    if actor_low_id >= actor_high_id:
        raise ValueError("actors must use canonical low/high ordering")
    if scope_status == "dissolved":
        return {"status": "scope_dissolved", "low_to_high": None, "high_to_low": None}
    if scope_status != "active":
        raise ValueError("unsupported scope status")
    latest: dict[tuple[str, str], Mapping[str, Any] | None] = {}
    ordered = sorted(
        attestations,
        key=lambda row: (int(row.get("sequence", 0)), str(row.get("id", ""))),
    )
    for row in ordered:
        issuer = str(row.get("issuer_actor_id"))
        subject = str(row.get("subject_actor_id"))
        if {issuer, subject} != {actor_low_id, actor_high_id} or issuer == subject:
            continue
        direction = (issuer, subject)
        kind = row.get("attestation_kind", "direct")
        if kind == "reported" or row.get("trust_level") != "verified":
            continue
        if kind == "revocation":
            latest[direction] = None
        elif kind == "direct":
            latest[direction] = row
    low = latest.get((actor_low_id, actor_high_id))
    high = latest.get((actor_high_id, actor_low_id))
    rows = [row for row in (low, high) if row is not None]
    verdicts = [row.get("verdict") for row in rows]
    decisive = [verdict for verdict in verdicts if verdict in {"positive", "negative"}]
    independent = (
        len(rows) == 2
        and rows[0].get("issuer_principal_id") is not None
        and rows[1].get("issuer_principal_id") is not None
        and rows[0].get("issuer_principal_id") != rows[1].get("issuer_principal_id")
    )
    if independent and "negative" in verdicts:
        status = "mutual_denied"
    elif independent and verdicts == ["positive", "positive"]:
        status = "mutual_positive"
    elif decisive:
        status = "unilateral"
    else:
        status = "undetermined"
    return {
        "status": status,
        "low_to_high": str(low.get("id")) if low is not None else None,
        "high_to_low": str(high.get("id")) if high is not None else None,
    }
