"""Тесты ``StyxMemoryProvider.system_prompt_block()`` — статический block
в system prompt каждый turn (волна 29 Phase C).

Содержит Locus framing, marker taxonomy table, continuity reference.
Помечает Hermes-агента как часть Locus (IAmBook §IV); таблица tag'ов
— symmetric с разделом «How to read markers» из styx-recall SKILL.md
(волна 30 D6).
"""

from __future__ import annotations

from styx_hermes.providers.memory import StyxMemoryProvider


def test_block_is_nonempty() -> None:
    p = StyxMemoryProvider()
    assert p.system_prompt_block().strip() != ""


def test_block_mentions_locus_concept() -> None:
    """IAmBook §IV — Locus framing должен быть видно."""
    block = StyxMemoryProvider().system_prompt_block()
    assert "Locus" in block or "locus" in block
    assert "agent-as-personality" in block or "personality" in block


def test_block_clarifies_styx_is_not_rag() -> None:
    """Концептуальный invariant: Styx не RAG."""
    block = StyxMemoryProvider().system_prompt_block()
    assert "not RAG" in block


def test_block_contains_all_seven_family_tags() -> None:
    """Marker taxonomy table должна включать все 7 family-тегов
    из волны 30 (salient, recall, archive, dialogue, relations,
    explain, working-set)."""
    block = StyxMemoryProvider().system_prompt_block()
    for tag in (
        "<styx-salient>",
        "<styx-recall>",
        "<styx-archive>",
        "<styx-dialogue>",
        "<styx-relations>",
        "<styx-explain>",
        "<styx-working-set>",
    ):
        assert tag in block, f"family tag {tag!r} missing in system_prompt_block"


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


def test_block_references_continuity_and_reinterpret() -> None:
    """Continuity (IAmBook §V — переосмысление через blend, не replace)
    — должна быть упомянута, чтобы LLM знал когда reinterpret vs store."""
    block = StyxMemoryProvider().system_prompt_block()
    assert "styx_reinterpret" in block
    assert "trajectory" in block.lower() or "continuity" in block.lower()


def test_block_mentions_styx_tools_namespace() -> None:
    """Префикс `styx_*` упомянут — LLM должен понимать что это его
    namespace для work с памятью."""
    block = StyxMemoryProvider().system_prompt_block()
    assert "styx_" in block


def test_block_is_stable_across_calls() -> None:
    """Static block: same content между вызовами (не зависит от state)."""
    p = StyxMemoryProvider()
    a = p.system_prompt_block()
    b = p.system_prompt_block()
    assert a == b
