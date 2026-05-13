"""Importance + lambda + query-hash утилиты.

Прямой port из ``openclaw-memorybox/src/importance.ts``. Числа и формулы
буквальны (decisions.md § 17.5). Используется в:

- composite scoring (``scoring.py``) — provisional/final mix + per-kind
  decay lambda;
- recall pipeline (``recall.py``) — query_hash для UNIQUE constraint
  на recall_events;
- LLM worker'е волны 7a (computeProvisionalImportance) — для INSERT
  своих memories с осмысленным provisional до прихода importance_final
  от LLM-классификатора.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping

MemoryKind = Literal["fact", "episode", "decision", "concept", "note"]
DialogueRole = Literal["human", "agent"]

DEFAULT_IMPORTANCE_BASE_BY_KIND: dict[MemoryKind, float] = {
    "decision": 0.85,
    "fact": 0.70,
    "concept": 0.60,
    "note": 0.45,
    "episode": 0.40,
}

DEFAULT_IMPORTANCE_BONUSES = {
    "role_human": 0.1,
    "supersede_context": 0.2,
}

DEFAULT_EXPLICIT_HINT_WEIGHT = 0.8

DEFAULT_LAMBDA_BY_KIND: dict[MemoryKind, float] = {
    "decision": 0.003,
    "fact": 0.005,
    "concept": 0.006,
    "note": 0.012,
    "episode": 0.020,
}


@dataclass(frozen=True)
class ImportanceInput:
    kind: MemoryKind
    role: DialogueRole | None = None
    explicit_hint: float | None = None


@dataclass(frozen=True)
class ImportanceRuntime:
    supersede_context: bool = False


@dataclass(frozen=True)
class ImportanceProvisionalConfig:
    base_by_kind: Mapping[MemoryKind, float] | None = None
    bonuses: Mapping[str, float] | None = None
    explicit_hint_weight: float | None = None


@dataclass(frozen=True)
class ImportanceConfig:
    provisional: ImportanceProvisionalConfig | None = None


_WHITESPACE_RE = re.compile(r"\s+")


def compute_provisional_importance(
    inp: ImportanceInput,
    runtime: ImportanceRuntime | None = None,
    config: ImportanceConfig | None = None,
) -> float:
    """Additive model: clamp(0, 1, base_by_kind + role_bonus + supersede + hint*weight).

    Прямой port ``computeProvisionalImportance`` (importance.ts:37).
    """
    runtime = runtime or ImportanceRuntime()
    config = config or ImportanceConfig()
    prov = config.provisional or ImportanceProvisionalConfig()

    base_by_kind: dict[MemoryKind, float] = dict(DEFAULT_IMPORTANCE_BASE_BY_KIND)
    if prov.base_by_kind:
        base_by_kind.update(prov.base_by_kind)

    role_human = (
        prov.bonuses["role_human"]
        if prov.bonuses and "role_human" in prov.bonuses
        else DEFAULT_IMPORTANCE_BONUSES["role_human"]
    )
    supersede_bonus = (
        prov.bonuses["supersede_context"]
        if prov.bonuses and "supersede_context" in prov.bonuses
        else DEFAULT_IMPORTANCE_BONUSES["supersede_context"]
    )
    hint_weight = (
        prov.explicit_hint_weight
        if prov.explicit_hint_weight is not None
        else DEFAULT_EXPLICIT_HINT_WEIGHT
    )

    score = base_by_kind.get(inp.kind, 0.5)

    if inp.role == "human":
        score += role_human
    if runtime.supersede_context:
        score += supersede_bonus

    if (
        inp.explicit_hint is not None
        and _is_finite(inp.explicit_hint)
        and 0.0 <= inp.explicit_hint <= 1.0
    ):
        score += inp.explicit_hint * hint_weight

    return max(0.0, min(1.0, score))


def normalize_query_for_hash(query: str) -> str:
    """lowercase + collapse whitespace + trim. Punctuation сохраняется.

    Port ``normalizeQueryForHash`` (importance.ts:74).
    """
    return _WHITESPACE_RE.sub(" ", query.lower()).strip()


def query_hash(query: str) -> bytes:
    """SHA-256 от нормализованного query. Port ``queryHash``."""
    return hashlib.sha256(normalize_query_for_hash(query).encode("utf-8")).digest()


def build_lambda_case_expr(
    lambdas: Mapping[MemoryKind, float] | None = None,
    *,
    table_alias: str | None = None,
) -> str:
    """SQL ``CASE kind WHEN ... END`` mapping kind → lambda_base.

    Используется внутри ``scoring.build_factor_exprs`` для decay фактора.
    Port ``buildLambdaCaseExpr`` (importance.ts:86).

    ``table_alias`` опционален: если задан, ``kind`` квалифицируется как
    ``{alias}.kind``; иначе — голое ``kind``. Соответствует TS-логике
    "buildLambdaCaseExpr emits 'CASE kind WHEN ... END' — inject prefix
    on 'kind'" (memory.ts:501-505).
    """
    merged: dict[MemoryKind, float] = dict(DEFAULT_LAMBDA_BY_KIND)
    if lambdas:
        merged.update(lambdas)
    column = f"{table_alias}.kind" if table_alias else "kind"
    cases = " ".join(
        f"WHEN '{kind}' THEN {lam}" for kind, lam in merged.items()
    )
    return f"CASE {column} {cases} ELSE 0.01 END"


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
