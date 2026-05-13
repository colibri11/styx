"""Unit-тесты для importance.py — буквальный port из memorybox."""

from __future__ import annotations

import pytest

from styx.storage.importance import (
    DEFAULT_EXPLICIT_HINT_WEIGHT,
    DEFAULT_IMPORTANCE_BASE_BY_KIND,
    DEFAULT_IMPORTANCE_BONUSES,
    DEFAULT_LAMBDA_BY_KIND,
    ImportanceConfig,
    ImportanceInput,
    ImportanceProvisionalConfig,
    ImportanceRuntime,
    build_lambda_case_expr,
    compute_provisional_importance,
    normalize_query_for_hash,
    query_hash,
)


def test_default_constants_match_memorybox() -> None:
    """Числа должны буквально совпадать с TS-оригиналом (decisions §17.5)."""
    assert DEFAULT_IMPORTANCE_BASE_BY_KIND == {
        "decision": 0.85,
        "fact": 0.70,
        "concept": 0.60,
        "note": 0.45,
        "episode": 0.40,
    }
    assert DEFAULT_IMPORTANCE_BONUSES == {
        "role_human": 0.1,
        "supersede_context": 0.2,
    }
    assert DEFAULT_EXPLICIT_HINT_WEIGHT == 0.8
    assert DEFAULT_LAMBDA_BY_KIND == {
        "decision": 0.003,
        "fact": 0.005,
        "concept": 0.006,
        "note": 0.012,
        "episode": 0.020,
    }


@pytest.mark.parametrize(
    "kind,expected_base",
    [
        ("decision", 0.85),
        ("fact", 0.70),
        ("concept", 0.60),
        ("note", 0.45),
        ("episode", 0.40),
    ],
)
def test_compute_provisional_per_kind_baseline(kind: str, expected_base: float) -> None:
    """Без role/runtime/hint — провижонал = base_by_kind."""
    out = compute_provisional_importance(ImportanceInput(kind=kind))  # type: ignore[arg-type]
    assert out == pytest.approx(expected_base)


def test_compute_provisional_role_human_bonus() -> None:
    out = compute_provisional_importance(ImportanceInput(kind="episode", role="human"))
    assert out == pytest.approx(0.40 + 0.1)


def test_compute_provisional_supersede_bonus() -> None:
    out = compute_provisional_importance(
        ImportanceInput(kind="note"),
        ImportanceRuntime(supersede_context=True),
    )
    assert out == pytest.approx(0.45 + 0.2)


def test_compute_provisional_explicit_hint() -> None:
    out = compute_provisional_importance(
        ImportanceInput(kind="episode", explicit_hint=0.5)
    )
    assert out == pytest.approx(0.40 + 0.5 * 0.8)


def test_compute_provisional_clamps_to_1() -> None:
    """decision (0.85) + role human (0.1) + supersede (0.2) + hint=1*0.8 = 1.95 → clamp to 1.0."""
    out = compute_provisional_importance(
        ImportanceInput(kind="decision", role="human", explicit_hint=1.0),
        ImportanceRuntime(supersede_context=True),
    )
    assert out == 1.0


def test_compute_provisional_clamps_to_0() -> None:
    """Custom config с отрицательной базой — clamp снизу."""
    out = compute_provisional_importance(
        ImportanceInput(kind="episode"),
        config=ImportanceConfig(
            provisional=ImportanceProvisionalConfig(base_by_kind={"episode": -0.5})
        ),
    )
    assert out == 0.0


def test_compute_provisional_ignores_invalid_hint() -> None:
    """hint вне [0, 1] не применяется (но и не падает)."""
    out_high = compute_provisional_importance(
        ImportanceInput(kind="episode", explicit_hint=2.0)
    )
    out_low = compute_provisional_importance(
        ImportanceInput(kind="episode", explicit_hint=-0.5)
    )
    assert out_high == pytest.approx(0.40)
    assert out_low == pytest.approx(0.40)


def test_compute_provisional_unknown_kind_falls_back_05() -> None:
    """kind вне DEFAULT_IMPORTANCE_BASE_BY_KIND → fallback 0.5."""
    out = compute_provisional_importance(ImportanceInput(kind="other_kind"))  # type: ignore[arg-type]
    assert out == pytest.approx(0.5)


def test_normalize_query_for_hash_lowercase_collapse_trim() -> None:
    assert normalize_query_for_hash("  Hello   WORLD  ") == "hello world"
    assert normalize_query_for_hash("WHERE\tX?\n") == "where x?"
    # Punctuation сохраняется.
    assert normalize_query_for_hash("a.b!c?") == "a.b!c?"


def test_query_hash_deterministic_and_normalised() -> None:
    """Регистро/whitespace-нечувствительный, sha256 32 байта."""
    h1 = query_hash("Hello World")
    h2 = query_hash("hello   world")
    h3 = query_hash("hello world")
    assert h1 == h2 == h3
    assert len(h1) == 32

    # Punctuation учитывается.
    assert query_hash("a?") != query_hash("a")


def test_build_lambda_case_default_kind_column() -> None:
    expr = build_lambda_case_expr()
    assert expr.startswith("CASE kind ")
    assert "WHEN 'decision' THEN 0.003" in expr
    assert "WHEN 'fact' THEN 0.005" in expr
    assert "WHEN 'concept' THEN 0.006" in expr
    assert "WHEN 'note' THEN 0.012" in expr
    assert "WHEN 'episode' THEN 0.02" in expr
    assert expr.endswith("ELSE 0.01 END")


def test_build_lambda_case_with_table_alias() -> None:
    expr = build_lambda_case_expr(table_alias="m")
    assert expr.startswith("CASE m.kind ")


def test_build_lambda_case_override() -> None:
    expr = build_lambda_case_expr({"fact": 0.999})
    assert "WHEN 'fact' THEN 0.999" in expr
    # Остальные дефолты на месте.
    assert "WHEN 'decision' THEN 0.003" in expr
