"""Recall threshold benchmark harness — измеряет распределение
base_match (cosine similarity) embedding-модели на фиксированной
suite пар query→content по 4 категориям семантической дистанции.

Self-contained: не требует Postgres. Считает cosine sim в Python между
двумя embed-вызовами Ollama. Численно идентично pgvector ``<=>``
оператору до float noise.

Usage:
    python -m tests.benchmarks.recall_threshold
    python -m tests.benchmarks.recall_threshold \\
        --ollama-url $STYX_OLLAMA_URL \\
        --model embeddinggemma:300m-qat-q8_0
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Suite — фиксированный артефакт рядом с harness'ом.
SUITE_PATH = Path(__file__).parent / "recall_threshold_suite.json"

# Категории и порядок их вывода — фиксирован.
CATEGORIES = ("exact", "paraphrase", "partial", "unrelated")


@dataclass(frozen=True)
class Pair:
    id: str
    category: str
    query: str
    content: str


@dataclass(frozen=True)
class CategoryStats:
    category: str
    count: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float

    @classmethod
    def empty(cls, category: str) -> "CategoryStats":
        return cls(category, 0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BenchOutput:
    model: str
    by_category: dict[str, CategoryStats]
    suggested_threshold: float | None
    overlap: bool


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q ∈ [0, 1].

    statistics.quantiles требует n ≥ 2 и не даёт прямой контроль над
    значением q — поэтому считаем сами через индексирование.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _summarize(scores: list[float], category: str) -> CategoryStats:
    if not scores:
        return CategoryStats.empty(category)
    s = sorted(scores)
    return CategoryStats(
        category=category,
        count=len(s),
        p10=_percentile(s, 0.10),
        p25=_percentile(s, 0.25),
        p50=_percentile(s, 0.50),
        p75=_percentile(s, 0.75),
        p90=_percentile(s, 0.90),
    )


def load_suite(path: Path = SUITE_PATH) -> list[Pair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[Pair] = []
    for entry in raw:
        cat = entry["category"]
        if cat not in CATEGORIES:
            raise ValueError(
                f"suite {entry['id']}: неизвестная категория {cat!r}; "
                f"допустимы {CATEGORIES}"
            )
        pairs.append(
            Pair(
                id=entry["id"],
                category=cat,
                query=entry["query"],
                content=entry["content"],
            )
        )
    return pairs


def run_bench(
    pairs: Iterable[Pair],
    embed_client,
    *,
    model: str,
) -> BenchOutput:
    """Прогон bench'а. ``embed_client`` обязан иметь ``embed(text) -> list[float]``.

    Возвращает структурированный BenchOutput с percentiles по категориям
    и suggested threshold'ом (если классы paraphrase/unrelated разделимы).
    """
    scores_by_cat: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        qvec = embed_client.embed(p.query)
        cvec = embed_client.embed(p.content)
        scores_by_cat[p.category].append(_cosine_similarity(qvec, cvec))

    stats = {cat: _summarize(scores_by_cat.get(cat, []), cat) for cat in CATEGORIES}

    paraphrase = stats["paraphrase"]
    unrelated = stats["unrelated"]
    overlap = paraphrase.count == 0 or unrelated.count == 0 or (
        paraphrase.p10 <= unrelated.p90
    )
    suggested = None if overlap else round((paraphrase.p10 + unrelated.p90) / 2, 2)

    return BenchOutput(
        model=model,
        by_category=stats,
        suggested_threshold=suggested,
        overlap=overlap,
    )


def render_markdown(out: BenchOutput) -> str:
    lines = [
        f"# Recall threshold bench — `{out.model}`",
        "",
        "| Category | N | p10 | p25 | p50 | p75 | p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat in CATEGORIES:
        s = out.by_category[cat]
        if s.count == 0:
            lines.append(f"| {cat} | 0 | — | — | — | — | — |")
        else:
            lines.append(
                f"| {cat} | {s.count} | "
                f"{s.p10:.3f} | {s.p25:.3f} | {s.p50:.3f} | "
                f"{s.p75:.3f} | {s.p90:.3f} |"
            )

    lines.extend(["", "## Decision"])
    para = out.by_category["paraphrase"]
    unr = out.by_category["unrelated"]
    lines.append(f"- p10(paraphrase) = {para.p10:.3f}")
    lines.append(f"- p90(unrelated)  = {unr.p90:.3f}")
    if out.overlap:
        lines.append("- **OVERLAP** — классы не разделяются на этой модели.")
        lines.append("  Возвращаемся к обсуждению модели.")
    else:
        lines.append(
            f"- Зазор: {para.p10 - unr.p90:.3f}. "
            f"**Suggested threshold = {out.suggested_threshold}**"
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama-url",
        default="http://ollama:11434",
        help="Ollama HTTP endpoint (default: http://ollama:11434)",
    )
    parser.add_argument(
        "--model",
        default="embeddinggemma:300m-qat-q8_0",
        help="embedding model name (default: embeddinggemma:300m-qat-q8_0)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="expected embedding dim (default: 768)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="embed timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=SUITE_PATH,
        help=f"path to suite JSON (default: {SUITE_PATH.name})",
    )
    args = parser.parse_args(argv)

    # Ленивый импорт — позволяет запускать --help без styx на пути.
    from styx.embedding import OllamaEmbeddingClient

    pairs = load_suite(args.suite)
    client = OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.model,
        dim=args.dim,
        timeout=args.timeout,
    )
    out = run_bench(pairs, client, model=args.model)
    print(render_markdown(out))
    return 0 if not out.overlap else 2


if __name__ == "__main__":
    sys.exit(main())
