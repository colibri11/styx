"""Юнит-тесты eviction_relevance — pure helpers + apply_relevance_eviction.

Волна 12. Все тесты host-only — БД не трогается напрямую (handle.queries
mockается через FakeQueries). Логика lookup_embeddings_by_content имеет
свой test в tests/storage/test_lookup_embeddings.py (требует migrated_db).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from styx.engine import eviction_relevance_bridge
from styx.engine.eviction_relevance import (
    _cosine,
    _message_text,
    _pair_group_relevance,
    _segment_pair_groups,
    _select_top_k,
    apply_relevance_eviction,
)
from styx.engine.eviction_relevance_bridge import EvictionRelevanceHandle


@pytest.fixture(autouse=True)
def _reset_bridge() -> None:
    eviction_relevance_bridge.reset_all()
    yield
    eviction_relevance_bridge.reset_all()


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class _FakeQueries:
    """Stub AgentScopedQueries — только lookup_embeddings_by_content."""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}
        self.last_call: list[str] | None = None
        self.fail: Exception | None = None

    def lookup_embeddings_by_content(
        self, contents: list[str]
    ) -> dict[str, list[float]]:
        self.last_call = list(contents)
        if self.fail is not None:
            raise self.fail
        return {c: v for c, v in self.mapping.items() if c in contents}


def _handle(
    queries: _FakeQueries,
    *,
    keep_k: int = 2,
    threshold: float = 0.4,
) -> EvictionRelevanceHandle:
    return EvictionRelevanceHandle(
        queries=queries,  # type: ignore[arg-type]
        keep_k=keep_k,
        threshold=threshold,
        agent_id="test",
    )


# ── _segment_pair_groups ─────────────────────────────────────────────


def test_segment_user_only() -> None:
    middle = [
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
    ]
    groups = _segment_pair_groups(middle)
    assert len(groups) == 2
    assert groups[0][0]["content"] == "u1"
    assert groups[1][0]["content"] == "u2"


def test_segment_assistant_with_tool_pair() -> None:
    middle = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {}}],
        },
        {"role": "tool", "content": "result1", "tool_call_id": "tc1"},
        {"role": "user", "content": "u2"},
    ]
    groups = _segment_pair_groups(middle)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert groups[0][0]["content"] == "u1"
    assert len(groups[1]) == 2
    assert groups[1][0]["role"] == "assistant"
    assert groups[1][1]["role"] == "tool"
    assert len(groups[2]) == 1


def test_segment_orphan_tool_at_start_is_single_group() -> None:
    """Defense-in-depth: orphan tool в начале middle формирует свою группу."""
    middle = [
        {"role": "tool", "content": "orphan", "tool_call_id": "x"},
        {"role": "user", "content": "u1"},
    ]
    groups = _segment_pair_groups(middle)
    assert len(groups) == 2
    assert groups[0][0]["role"] == "tool"
    assert groups[1][0]["role"] == "user"


def test_segment_assistant_with_multiple_tool_results() -> None:
    middle = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {}},
                {"id": "tc2", "type": "function", "function": {}},
            ],
        },
        {"role": "tool", "content": "r1", "tool_call_id": "tc1"},
        {"role": "tool", "content": "r2", "tool_call_id": "tc2"},
    ]
    groups = _segment_pair_groups(middle)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_segment_empty_middle() -> None:
    assert _segment_pair_groups([]) == []


# ── _pair_group_relevance ─────────────────────────────────────────────


def test_relevance_max_across_embed_able() -> None:
    centroid = _unit([1.0, 0.0, 0.0])
    embeds = {
        "high": _unit([0.9, 0.1, 0.0]),
        "low": _unit([0.1, 0.9, 0.0]),
    }
    group = [
        {"role": "user", "content": "high"},
        {"role": "assistant", "content": "low"},
    ]
    rel = _pair_group_relevance(group, embeds, centroid)
    assert rel is not None
    expected = _cosine(embeds["high"], centroid)
    assert abs(rel - expected) < 1e-9


def test_relevance_none_when_no_embed_able() -> None:
    centroid = _unit([1.0, 0.0, 0.0])
    embeds: dict[str, list[float]] = {}
    group = [
        {"role": "user", "content": "missing"},
        {"role": "tool", "content": "no embed", "tool_call_id": "x"},
    ]
    assert _pair_group_relevance(group, embeds, centroid) is None


def test_relevance_skips_empty_text_messages() -> None:
    centroid = _unit([1.0, 0.0, 0.0])
    embeds = {"text": _unit([1.0, 0.0, 0.0])}
    group = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
        {"role": "user", "content": "text"},
    ]
    rel = _pair_group_relevance(group, embeds, centroid)
    assert rel is not None
    assert abs(rel - 1.0) < 1e-9


def test_relevance_skips_non_string_content() -> None:
    """Anthropic-style блоки (list[dict]) не embed-able — пропускаются."""
    centroid = _unit([1.0, 0.0, 0.0])
    embeds: dict[str, list[float]] = {}
    group = [
        {"role": "user", "content": [{"type": "text", "text": "blocky"}]},
    ]
    assert _pair_group_relevance(group, embeds, centroid) is None


# ── _select_top_k ─────────────────────────────────────────────────────


def test_select_top_k_returns_chronological() -> None:
    g0 = [{"role": "user", "content": "g0"}]
    g1 = [{"role": "user", "content": "g1"}]
    g2 = [{"role": "user", "content": "g2"}]
    scored = [
        (0, g0, 0.5),
        (1, g1, 0.9),
        (2, g2, 0.7),
    ]
    out = _select_top_k(scored, k=2, floor=0.0)
    # top-2 by relevance: g1 (0.9), g2 (0.7); chronological: g1, g2
    assert [m["content"] for m in out] == ["g1", "g2"]


def test_select_top_k_skips_below_floor() -> None:
    g0 = [{"role": "user", "content": "g0"}]
    g1 = [{"role": "user", "content": "g1"}]
    scored = [
        (0, g0, 0.3),
        (1, g1, 0.5),
    ]
    out = _select_top_k(scored, k=2, floor=0.4)
    assert [m["content"] for m in out] == ["g1"]


def test_select_top_k_skips_none_relevance() -> None:
    g0 = [{"role": "user", "content": "g0"}]
    g1 = [{"role": "user", "content": "g1"}]
    scored = [
        (0, g0, None),
        (1, g1, 0.5),
    ]
    out = _select_top_k(scored, k=2, floor=0.0)
    assert [m["content"] for m in out] == ["g1"]


def test_select_top_k_zero_returns_empty() -> None:
    g0 = [{"role": "user", "content": "g0"}]
    out = _select_top_k([(0, g0, 0.9)], k=0, floor=0.0)
    assert out == []


def test_select_top_k_preserves_pair_messages() -> None:
    """Если выбрана группа с tool_result'ами — оба message в результате."""
    g_pair = [
        {"role": "assistant", "content": "calling", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "result", "tool_call_id": "x"},
    ]
    out = _select_top_k([(0, g_pair, 0.9)], k=1, floor=0.0)
    assert len(out) == 2
    assert out[0]["role"] == "assistant"
    assert out[1]["role"] == "tool"


# ── apply_relevance_eviction (end-to-end) ─────────────────────────────


def _build_messages() -> list[dict[str, Any]]:
    """4 head + 6 middle + 4 tail (всего 14)."""
    return (
        [{"role": "user", "content": f"head{i}"} for i in range(4)]
        + [
            {"role": "user", "content": "noise1"},
            {"role": "assistant", "content": "noise2"},
            {"role": "user", "content": "ANCHOR"},
            {"role": "assistant", "content": "noise3"},
            {"role": "user", "content": "noise4"},
            {"role": "assistant", "content": "noise5"},
        ]
        + [{"role": "user", "content": f"tail{i}"} for i in range(4)]
    )


def test_apply_returns_empty_when_handle_none() -> None:
    messages = _build_messages()
    out = apply_relevance_eviction(messages, 4, 10, None, [1.0, 0.0, 0.0])
    assert out == []


def test_apply_returns_empty_when_centroid_none() -> None:
    messages = _build_messages()
    queries = _FakeQueries()
    handle = _handle(queries)
    out = apply_relevance_eviction(messages, 4, 10, handle, None)
    assert out == []


def test_apply_returns_empty_when_keep_k_zero() -> None:
    messages = _build_messages()
    queries = _FakeQueries()
    handle = _handle(queries, keep_k=0)
    out = apply_relevance_eviction(messages, 4, 10, handle, [1.0, 0.0, 0.0])
    assert out == []


def test_apply_returns_empty_when_middle_empty() -> None:
    messages = _build_messages()
    queries = _FakeQueries()
    handle = _handle(queries)
    out = apply_relevance_eviction(messages, 4, 4, handle, [1.0, 0.0, 0.0])
    assert out == []


def test_apply_keeps_top_k_relevant_groups() -> None:
    centroid = _unit([1.0, 0.0, 0.0])
    messages = _build_messages()
    queries = _FakeQueries(
        {
            "noise1": _unit([0.0, 1.0, 0.0]),
            "noise2": _unit([0.0, 1.0, 0.0]),
            "ANCHOR": _unit([0.95, 0.05, 0.0]),
            "noise3": _unit([0.0, 1.0, 0.0]),
            "noise4": _unit([0.85, 0.15, 0.0]),
            "noise5": _unit([0.0, 1.0, 0.0]),
        }
    )
    handle = _handle(queries, keep_k=2, threshold=0.4)
    out = apply_relevance_eviction(messages, 4, 10, handle, centroid)
    contents = [m["content"] for m in out]
    # ANCHOR (0.95) и noise4 (0.85) — единственные >= 0.4. Chronological:
    # ANCHOR раньше noise4.
    assert "ANCHOR" in contents
    assert "noise4" in contents
    assert contents.index("ANCHOR") < contents.index("noise4")


def test_apply_drops_groups_below_floor() -> None:
    centroid = _unit([1.0, 0.0, 0.0])
    messages = _build_messages()
    queries = _FakeQueries(
        {
            "noise1": _unit([0.0, 1.0, 0.0]),
            "noise2": _unit([0.0, 1.0, 0.0]),
            "ANCHOR": _unit([0.0, 1.0, 0.0]),
            "noise3": _unit([0.0, 1.0, 0.0]),
            "noise4": _unit([0.0, 1.0, 0.0]),
            "noise5": _unit([0.0, 1.0, 0.0]),
        }
    )
    handle = _handle(queries, keep_k=2, threshold=0.4)
    out = apply_relevance_eviction(messages, 4, 10, handle, centroid)
    # Все ортогональны → cosine=0 < floor 0.4 → ничего не keep'ится.
    assert out == []


def test_apply_handles_lookup_failure_fail_open() -> None:
    centroid = _unit([1.0, 0.0, 0.0])
    messages = _build_messages()
    queries = _FakeQueries()
    queries.fail = RuntimeError("DB down")
    handle = _handle(queries)
    out = apply_relevance_eviction(messages, 4, 10, handle, centroid)
    assert out == []


def test_apply_keeps_assistant_with_tool_result_pair() -> None:
    """Если выбран assistant с tool_calls — tool_result тащится с ним."""
    centroid = _unit([1.0, 0.0, 0.0])
    messages = (
        [{"role": "user", "content": "head0"}]
        + [
            {"role": "user", "content": "noise"},
            {
                "role": "assistant",
                "content": "ANCHOR_call",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            {"role": "user", "content": "noise2"},
        ]
        + [{"role": "user", "content": "tail0"}]
    )
    queries = _FakeQueries(
        {
            "noise": _unit([0.0, 1.0, 0.0]),
            "ANCHOR_call": _unit([1.0, 0.0, 0.0]),
            "noise2": _unit([0.0, 1.0, 0.0]),
        }
    )
    handle = _handle(queries, keep_k=1, threshold=0.4)
    out = apply_relevance_eviction(messages, 1, 5, handle, centroid)
    contents = [m.get("content") for m in out]
    assert "ANCHOR_call" in contents
    # tool_result должен идти сразу за assistant
    assert any(m.get("role") == "tool" for m in out)


def test_apply_calls_lookup_with_unique_contents() -> None:
    """Дубликаты content не дублируются в SQL-payload."""
    centroid = _unit([1.0, 0.0, 0.0])
    messages = (
        [{"role": "user", "content": "h0"}]
        + [
            {"role": "user", "content": "dup"},
            {"role": "assistant", "content": "dup"},
            {"role": "user", "content": "uniq"},
        ]
        + [{"role": "user", "content": "t0"}]
    )
    queries = _FakeQueries({"dup": _unit([1.0, 0.0, 0.0])})
    handle = _handle(queries, keep_k=2, threshold=0.0)
    apply_relevance_eviction(messages, 1, 4, handle, centroid)
    assert queries.last_call is not None
    # "dup" — один раз, "uniq" — один раз
    assert sorted(queries.last_call) == ["dup", "uniq"]


# ── _cosine, _message_text — sanity ──────────────────────────────────


def test_cosine_orthogonal_zero() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_identical_one() -> None:
    e = _unit([0.3, 0.4, 0.5])
    assert abs(_cosine(e, e) - 1.0) < 1e-9


def test_cosine_zero_norm_returns_zero() -> None:
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_message_text_string_content() -> None:
    assert _message_text({"role": "user", "content": "hi"}) == "hi"


def test_message_text_none_or_list_returns_empty() -> None:
    assert _message_text({"role": "user", "content": None}) == ""
    assert _message_text({"role": "user", "content": [{"type": "text"}]}) == ""
    assert _message_text({"role": "user"}) == ""
