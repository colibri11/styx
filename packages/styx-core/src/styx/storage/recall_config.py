"""RecallConfig — структура и дефолты.

Прямой port из ``openclaw-memorybox/src/types.ts`` +
``context-engine.ts:403 DEFAULT_RECALL_CONFIG``. Числа буквальны
(decisions.md § 17.5).

В волне 7 Styx использует только ``full`` — ``companion`` оставлен
в типе для совместимости с port'ируемой формулой и для будущих волн
(8 hot-tier, 7d emotional). ``hotTier`` и dialogue-paths пока не
вызываются.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

SessionScope = Literal["all", "current"]


@dataclass(frozen=True)
class CompanionDialogueRecallConfig:
    enabled: bool = True
    limit: int = 5
    min_score: float = 0.6
    session_scope: SessionScope = "all"
    recency_boost: bool | None = None


@dataclass(frozen=True)
class CompanionStructuredRecallConfig:
    enabled: bool = False
    limit: int = 4
    min_score: float = 0.7
    internal_dedup_similarity: float = 0.92


@dataclass(frozen=True)
class CompanionHotTierRecallConfig:
    enabled: bool = True
    limit: int = 5


@dataclass(frozen=True)
class CompanionRecallConfig:
    dialogue: CompanionDialogueRecallConfig = field(
        default_factory=CompanionDialogueRecallConfig
    )
    structured: CompanionStructuredRecallConfig = field(
        default_factory=CompanionStructuredRecallConfig
    )
    hot_tier: CompanionHotTierRecallConfig = field(
        default_factory=CompanionHotTierRecallConfig
    )


@dataclass(frozen=True)
class FullRecallConfig:
    memory_limit: int = 6
    dialogue_limit: int = 3
    chunk_limit: int = 3
    # Калибровано волной 8 под embeddinggemma:300m-qat-q8_0 (768-dim).
    # Memorybox-оригинал = 0.6 на bge-m3 (1024-dim) — другое
    # распределение cosine sim, не переносится. См. decisions.md § 22.
    min_score: float = 0.32
    internal_dedup_similarity: float = 0.92


@dataclass(frozen=True)
class RecallConfig:
    companion: CompanionRecallConfig = field(default_factory=CompanionRecallConfig)
    full: FullRecallConfig = field(default_factory=FullRecallConfig)
    token_budget_fraction: float = 0.1


DEFAULT_RECALL_CONFIG = RecallConfig()


def resolve_recall_config(partial: Mapping[str, object] | None = None) -> RecallConfig:
    """Merge partial overrides поверх дефолтов.

    Port ``resolveRecallConfig`` (context-engine.ts:432). Принимает
    nested mapping вида ``{"full": {"min_score": 0.5}, ...}``. Не
    знающие ключи игнорируются молча — это TS-семантика.
    """
    if not partial:
        return DEFAULT_RECALL_CONFIG

    full_raw = _as_mapping(partial.get("full"))
    full = (
        replace(DEFAULT_RECALL_CONFIG.full, **_keep_known(full_raw, FullRecallConfig))
        if full_raw is not None
        else DEFAULT_RECALL_CONFIG.full
    )

    companion_raw = _as_mapping(partial.get("companion"))
    if companion_raw is None:
        companion = DEFAULT_RECALL_CONFIG.companion
    else:
        dialogue_raw = _as_mapping(companion_raw.get("dialogue"))
        dialogue = (
            replace(
                DEFAULT_RECALL_CONFIG.companion.dialogue,
                **_keep_known(dialogue_raw, CompanionDialogueRecallConfig),
            )
            if dialogue_raw is not None
            else DEFAULT_RECALL_CONFIG.companion.dialogue
        )
        structured_raw = _as_mapping(companion_raw.get("structured"))
        structured = (
            replace(
                DEFAULT_RECALL_CONFIG.companion.structured,
                **_keep_known(structured_raw, CompanionStructuredRecallConfig),
            )
            if structured_raw is not None
            else DEFAULT_RECALL_CONFIG.companion.structured
        )
        hot_tier_raw = _as_mapping(companion_raw.get("hot_tier"))
        hot_tier = (
            replace(
                DEFAULT_RECALL_CONFIG.companion.hot_tier,
                **_keep_known(hot_tier_raw, CompanionHotTierRecallConfig),
            )
            if hot_tier_raw is not None
            else DEFAULT_RECALL_CONFIG.companion.hot_tier
        )
        companion = CompanionRecallConfig(
            dialogue=dialogue, structured=structured, hot_tier=hot_tier
        )

    token_budget_fraction = partial.get("token_budget_fraction")
    if not isinstance(token_budget_fraction, (int, float)):
        token_budget_fraction = DEFAULT_RECALL_CONFIG.token_budget_fraction

    return RecallConfig(
        companion=companion,
        full=full,
        token_budget_fraction=float(token_budget_fraction),
    )


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _keep_known(raw: Mapping[str, object], cls: type) -> dict[str, object]:
    """Отбирает только те ключи raw, которые реально объявлены на cls."""
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return {k: v for k, v in raw.items() if k in fields}
