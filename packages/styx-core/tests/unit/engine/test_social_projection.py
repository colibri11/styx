from styx.engine.social_projection import evaluate_pair_projection


LOW = "00000000-0000-0000-0000-000000000001"
HIGH = "00000000-0000-0000-0000-000000000002"


def _act(i, issuer, subject, verdict="positive", **extra):
    return {
        "id": f"a{i}", "sequence": i, "issuer_actor_id": issuer,
        "subject_actor_id": subject, "verdict": verdict,
        "attestation_kind": "direct", "trust_level": "verified",
        "issuer_principal_id": f"principal:{issuer}", **extra,
    }


def test_reciprocal_positive_only_changes_scoped_projection() -> None:
    one = [_act(1, LOW, HIGH)]
    assert evaluate_pair_projection(one, actor_low_id=LOW, actor_high_id=HIGH)["status"] == "unilateral"
    mutual = evaluate_pair_projection(
        one + [_act(2, HIGH, LOW)], actor_low_id=LOW, actor_high_id=HIGH
    )
    assert mutual["status"] == "mutual_positive"
    # Neutral coordinates are excluded from the evaluator.
    mutated = [{**row, "created_at": "other", "model": "other", "label": "other"} for row in one + [_act(2, HIGH, LOW)]]
    assert evaluate_pair_projection(mutated, actor_low_id=LOW, actor_high_id=HIGH) == mutual


def test_negative_revision_revocation_and_dissolution() -> None:
    acts = [_act(1, LOW, HIGH), _act(2, HIGH, LOW), _act(3, HIGH, LOW, "negative")]
    assert evaluate_pair_projection(acts, actor_low_id=LOW, actor_high_id=HIGH)["status"] == "mutual_denied"
    acts.append(_act(4, HIGH, LOW, attestation_kind="revocation"))
    assert evaluate_pair_projection(acts, actor_low_id=LOW, actor_high_id=HIGH)["status"] == "unilateral"
    assert evaluate_pair_projection(acts, actor_low_id=LOW, actor_high_id=HIGH, scope_status="dissolved")["status"] == "scope_dissolved"


def test_self_report_and_unverified_do_not_create_mutuality() -> None:
    acts = [
        _act(1, LOW, LOW),
        _act(2, LOW, HIGH),
        _act(3, HIGH, LOW, attestation_kind="reported"),
        _act(4, HIGH, LOW, trust_level="unverified"),
    ]
    assert evaluate_pair_projection(acts, actor_low_id=LOW, actor_high_id=HIGH)["status"] == "unilateral"


def test_one_principal_cannot_manufacture_reciprocity() -> None:
    acts = [
        _act(1, LOW, HIGH, issuer_principal_id="same-principal"),
        _act(2, HIGH, LOW, issuer_principal_id="same-principal"),
    ]
    assert evaluate_pair_projection(
        acts, actor_low_id=LOW, actor_high_id=HIGH,
    )["status"] == "unilateral"


def test_unilateral_negative_is_not_mutual_denial() -> None:
    assert evaluate_pair_projection(
        [_act(1, LOW, HIGH, "negative")],
        actor_low_id=LOW,
        actor_high_id=HIGH,
    )["status"] == "unilateral"
