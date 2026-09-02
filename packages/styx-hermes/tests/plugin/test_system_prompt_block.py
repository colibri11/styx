"""Wave-37 conceptual-boundary tests for the static Hermes block."""

from __future__ import annotations

from styx_hermes.providers.memory import StyxMemoryProvider


def test_block_is_nonempty() -> None:
    p = StyxMemoryProvider()
    assert p.system_prompt_block().strip() != ""


def test_block_describes_locus_as_working_architecture_not_identity() -> None:
    block = StyxMemoryProvider().system_prompt_block()
    assert "Locus-style working architecture" in block
    assert "does not establish personality or consciousness" in block
    assert "agent-as-personality" not in block
    assert "past self" not in block
    assert "lives in Styx" not in block


def test_block_clarifies_reconstruction_boundary() -> None:
    block = StyxMemoryProvider().system_prompt_block()
    assert "Reconstruct relevant personal memory" in block
    assert "stored text remains evidence" in block


def test_block_makes_canonical_preturn_primary_and_legacy_unambiguous() -> None:
    block = StyxMemoryProvider().system_prompt_block()
    assert "primary and only canonical automatic preturn channel" in block
    assert "**primary automatic preturn**" in block
    assert "legacy compatibility recall" in block
    assert "not the canonical automatic preturn" in block
    assert "legacy compatibility projection from `/pre_llm_inject`" in block
    for tag in (
        "<styx-cognitive-continuity>",
        "<styx-salient>",
        "<styx-self-state>",
        "<styx-recall>",
        "<styx-archive>",
        "<styx-dialogue>",
        "<styx-relations>",
        "<styx-explain>",
        "<styx-working-set>",
    ):
        assert tag in block, f"family tag {tag!r} missing in system_prompt_block"
    for section in (
        "technical_projection",
        "cognitive_posture",
        "pending_consequences",
        "reconstructed_subjective_traces",
    ):
        assert section in block


def test_block_explains_no_marker_means_live_conversation() -> None:
    """Decision logic: «No <styx-*> wrapper → it is in the live
    conversation, not memory.» — критично для различения source'а."""
    block = StyxMemoryProvider().system_prompt_block()
    assert "live conversation" in block
    assert "without" in block.lower()


def test_block_does_not_emit_styx_tags_for_user_output() -> None:
    """Hermes-агент не должен включать `<styx-*>` теги в ответ user'у —
    это маркеры input'а, не output'а."""
    block = StyxMemoryProvider().system_prompt_block()
    assert "Do not include" in block or "do not include" in block.lower()


def test_block_keeps_engineering_policy_out_of_treatise_claims() -> None:
    block = StyxMemoryProvider().system_prompt_block()
    assert "Weighted blending and ranking are Styx engineering policies" in block
    assert "same trajectory" in block


def test_block_is_stable_across_calls() -> None:
    """Static block: same content между вызовами (не зависит от state)."""
    p = StyxMemoryProvider()
    a = p.system_prompt_block()
    b = p.system_prompt_block()
    assert a == b
