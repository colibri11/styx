"""Drift threshold validation harness — проверяет что дефолт 0.4 различает
смену темы (unrelated) и продолжение темы (paraphrase) на bench-suite
из волны 8.

Логика идентична volna-10 drift detection'у: для каждой пары
``query → content`` в bench-suite считается cosine между их embedding'ами.
- exact / paraphrase: ожидаем cosine > 0.4 (drift НЕ должен сработать).
- unrelated: ожидаем cosine < 0.4 (drift должен сработать).
- partial — пограничная зона; выводится для информации, не ломает test.

Self-contained: не требует Postgres. Использует тот же Ollama endpoint
что и волна 10 в проде.

Usage:
    python -m tests.benchmarks.drift_threshold
    python -m tests.benchmarks.drift_threshold \\
        --ollama-url $STYX_OLLAMA_URL \\
        --model embeddinggemma:300m-qat-q8_0 \\
        --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Используем тот же suite что волна 8 (recall_threshold).
SUITE_PATH = Path(__file__).parent / "recall_threshold_suite.json"

CATEGORIES = ("exact", "paraphrase", "partial", "unrelated")


@dataclass(frozen=True)
class Pair:
    id: str
    category: str
    query: str
    content: str


def load_suite() -> list[Pair]:
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    return [Pair(**p) for p in raw]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    s = sorted(values)

    def pct(p: float) -> float:
        idx = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
        return s[idx]

    return {
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("STYX_OLLAMA_URL", "http://localhost:11434"),
        help="Ollama endpoint (default: $STYX_OLLAMA_URL or localhost)",
    )
    parser.add_argument(
        "--model", default="embeddinggemma:300m-qat-q8_0",
        help="Embedding model name",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Drift threshold для validation (default 0.4)",
    )
    args = parser.parse_args()

    # Lazy import — embedding клиент тянет urllib и ждёт ollama; не делаем
    # это импортом модуля.
    from styx.embedding import OllamaEmbeddingClient

    suite = load_suite()
    embed = OllamaEmbeddingClient(
        base_url=args.ollama_url, model=args.model, dim=768, timeout=30.0,
    )

    by_cat: dict[str, list[float]] = defaultdict(list)
    print(f"# Drift threshold validation (threshold={args.threshold})")
    print(f"# Model: {args.model}")
    print(f"# Endpoint: {args.ollama_url}")
    print(f"# Suite: {len(suite)} pairs across {len(CATEGORIES)} categories")
    print()

    for pair in suite:
        q_vec = embed.embed(pair.query)
        c_vec = embed.embed(pair.content)
        sim = cosine(q_vec, c_vec)
        by_cat[pair.category].append(sim)

    print(f"{'Category':<14} {'N':>3} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7}")
    print("-" * 60)
    for cat in CATEGORIES:
        vals = by_cat[cat]
        p = percentiles(vals)
        print(
            f"{cat:<14} {len(vals):>3} "
            f"{p['p10']:>7.3f} {p['p25']:>7.3f} {p['p50']:>7.3f} "
            f"{p['p75']:>7.3f} {p['p90']:>7.3f}"
        )

    print()
    print(f"# Validation against threshold {args.threshold}")

    failures = 0
    warnings = 0

    # exact — все pairs ≈ 1.0 на embeddinggemma; жёсткий FAIL на любом below.
    vals = by_cat["exact"]
    below = [v for v in vals if v < args.threshold]
    if below:
        print(f"FAIL exact: {len(below)}/{len(vals)} pairs below threshold")
        failures += 1
    else:
        print(f"OK   exact: all {len(vals)} pairs above threshold")

    # paraphrase — soft. Bench-suite измеряет cross-role similarity
    # (query → content), drift в волне 10 — within-role (user → user).
    # Cross-role нижняя граница на embeddinggemma-Q8 плавает между
    # прогонами (0.35 ↔ 0.44 в разных runs); within-role на стабильной
    # теме обычно выше. Допускаем до 25% paraphrase below threshold как
    # WARN — это сигнал «бенч на границе с threshold», не катастрофа.
    vals = by_cat["paraphrase"]
    below = [v for v in vals if v < args.threshold]
    ratio = len(below) / len(vals) if vals else 0
    if ratio > 0.25:
        print(
            f"FAIL paraphrase: {len(below)}/{len(vals)} ({ratio:.0%}) below "
            f"threshold > 25% bound: {[f'{v:.3f}' for v in below]}"
        )
        failures += 1
    elif below:
        print(
            f"WARN paraphrase: {len(below)}/{len(vals)} below threshold "
            f"(допустимо ≤25%): {[f'{v:.3f}' for v in below]}"
        )
        warnings += 1
    else:
        print(f"OK   paraphrase: all {len(vals)} pairs above threshold")

    # unrelated — жёсткий FAIL на любом above. False negative drift =
    # cache не инвалидируется при реальной смене темы → стейл salient.
    vals = by_cat["unrelated"]
    above = [v for v in vals if v >= args.threshold]
    if above:
        print(
            f"FAIL unrelated: {len(above)}/{len(vals)} pairs above threshold "
            f"(false negative drift): {[f'{v:.3f}' for v in above]}"
        )
        failures += 1
    else:
        print(f"OK   unrelated: all {len(vals)} pairs below threshold")

    # partial — info only
    vals = by_cat["partial"]
    if vals:
        p = percentiles(vals)
        print(
            f"INFO partial: median={p['p50']:.3f}, p25={p['p25']:.3f} "
            f"(пограничная зона, не валидируется строго)"
        )

    print()
    if failures:
        print(f"VALIDATION FAILED: {failures} categor(ies) miscalibrated")
        return 1
    if warnings:
        print(
            f"VALIDATION PASSED with {warnings} warning(s): threshold "
            f"{args.threshold} в пределах допуска (см. WARN выше)"
        )
    else:
        print(f"VALIDATION PASSED: threshold {args.threshold} works clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
