"""Pipeline-тест harness'а — без живой Ollama.

FakeEmbeddingClient даёт детерминированные ортогональные векторы на
разных текстах (cosine ≈ 0). На FakeEmbeddingClient числа из bench'а
малоосмысленны (paraphrase ≈ unrelated), но сам pipeline (load suite
→ run_bench → percentiles → markdown) проверяется.
"""

from __future__ import annotations

from pathlib import Path

from styx.embedding import FakeEmbeddingClient

from tests.benchmarks.recall_threshold import (
    BenchOutput,
    CATEGORIES,
    Pair,
    load_suite,
    render_markdown,
    run_bench,
)


def test_load_suite_parses_all_categories() -> None:
    pairs = load_suite()
    by_cat: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for p in pairs:
        by_cat[p.category] += 1
    # Минимум 5 на категорию — wave 8 contract.
    for cat, n in by_cat.items():
        assert n >= 5, f"категория {cat!r}: только {n} пар (нужно ≥ 5)"


def test_load_suite_rejects_unknown_category(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        '[{"id":"x","category":"nonsense","query":"q","content":"c"}]',
        encoding="utf-8",
    )
    try:
        load_suite(bad)
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("ожидался ValueError на неизвестной категории")


def test_run_bench_produces_per_category_stats() -> None:
    """На фейковом клиенте все cosine ≈ 0 (случайные нормированные
    векторы); paraphrase/unrelated не отличаются по числам — это OK,
    мы тестируем pipeline, не качество.
    """
    client = FakeEmbeddingClient(dim=768)
    pairs = load_suite()
    out = run_bench(pairs, client, model="fake")
    assert out.model == "fake"
    for cat in CATEGORIES:
        assert out.by_category[cat].count >= 5


def test_run_bench_exact_pairs_have_score_near_one() -> None:
    """exact-категория: query == content → cosine = 1.0 точно."""
    client = FakeEmbeddingClient(dim=768)
    pairs = [
        Pair(id="e1", category="exact", query="hello", content="hello"),
        Pair(id="e2", category="exact", query="мир", content="мир"),
    ]
    out = run_bench(pairs, client, model="fake")
    s = out.by_category["exact"]
    assert s.count == 2
    # cosine(v, v) = 1 точно (с точностью до float).
    assert s.p50 > 0.999


def test_run_bench_unrelated_low_on_fake() -> None:
    """FakeEmbeddingClient на разных текстах даёт почти ортогональные
    векторы (sha256-seed → independent gaussians → нормировка). На
    768-dim cosine между двумя независимыми единичными векторами имеет
    стандартное отклонение ≈ 1/sqrt(768) ≈ 0.036 — медиана близка к 0.
    """
    client = FakeEmbeddingClient(dim=768)
    pairs = [
        Pair(id="u1", category="unrelated", query="apples", content="quantum mechanics"),
        Pair(id="u2", category="unrelated", query="борщ", content="machine learning"),
        Pair(id="u3", category="unrelated", query="вторник", content="прокси сервер"),
    ]
    out = run_bench(pairs, client, model="fake")
    s = out.by_category["unrelated"]
    assert s.count == 3
    assert abs(s.p50) < 0.2  # ортогональные ± noise


def test_render_markdown_includes_decision_block() -> None:
    out = BenchOutput(
        model="test",
        by_category={
            "exact": _stats(5, 1.0),
            "paraphrase": _stats(5, 0.85),
            "partial": _stats(5, 0.55),
            "unrelated": _stats(5, 0.10),
        },
        suggested_threshold=0.48,
        overlap=False,
    )
    md = render_markdown(out)
    assert "Recall threshold bench" in md
    assert "| exact |" in md
    assert "| paraphrase |" in md
    assert "Decision" in md
    assert "Suggested threshold = 0.48" in md


def test_render_markdown_marks_overlap() -> None:
    out = BenchOutput(
        model="test",
        by_category={
            "exact": _stats(5, 1.0),
            "paraphrase": _stats(5, 0.40),
            "partial": _stats(5, 0.45),
            "unrelated": _stats(5, 0.50),
        },
        suggested_threshold=None,
        overlap=True,
    )
    md = render_markdown(out)
    assert "OVERLAP" in md


def _stats(count: int, val: float):
    """Constant percentiles для удобства."""
    from tests.benchmarks.recall_threshold import CategoryStats

    return CategoryStats(
        category="x",
        count=count,
        p10=val,
        p25=val,
        p50=val,
        p75=val,
        p90=val,
    )
