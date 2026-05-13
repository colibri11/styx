"""Batch sentiment piggyback (волна 14, memorybox 26b).

Tightly coupled с dialogue_batch_consolidation handler'ом: тот же
LLM-вызов, который генерирует summary окна, возвращает интегральный
VAD peer-части окна. Handler усредняет VAD по chunk'ам и применяет к
emotional_state с ``source='sentiment:batch'`` и K_BATCH=0.4.

K_BATCH > K_HOT (0.15) потому что batch — реальное усреднение тона
(не одна реплика); заслуживает больше веса в эмоциональной проекции
агента. ~2.7× от K_HOT, port memorybox 26b §9a.

Apply ПЕРВЫМ в транзакции handler'а (до INSERT memory) — чтобы
emotional snapshot новой memory читался уже с свежей дельтой
(memorybox `dialogue-sentiment-v1.md` §6 «вариант 1»).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from styx.emotional.state import EmotionalVector


K_BATCH = 0.4
"""Scale factor для batch-VAD delta. ``delta = K_BATCH × avg_vad``."""


def scale_batch_vad_delta(vad: EmotionalVector) -> EmotionalVector:
    """Multiply by K_BATCH — final delta ∈ [-K_BATCH, K_BATCH] на каждой оси."""
    return EmotionalVector(
        valence=K_BATCH * vad.valence,
        arousal=K_BATCH * vad.arousal,
        dominance=K_BATCH * vad.dominance,
    )


def average_vads(vads: list[EmotionalVector]) -> EmotionalVector | None:
    """Среднее по списку VAD'ов. Пустой список → None.

    Включаем как scored, так и skip chunks (skip про память, не про
    эмоции — скип-чанк может нести peer-emotion).
    """
    if not vads:
        return None
    n = len(vads)
    return EmotionalVector(
        valence=sum(v.valence for v in vads) / n,
        arousal=sum(v.arousal for v in vads) / n,
        dominance=sum(v.dominance for v in vads) / n,
    )


@dataclass
class SentimentBatchMetrics:
    """Счётчики для /healthz. Один экземпляр на batch-handler."""

    calls: int = 0
    skips_no_vad: int = 0
    schema_errors: int = 0
    applied: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def record_call(self) -> None:
        with self._lock:
            self.calls += 1

    def record_skip_no_vad(self) -> None:
        with self._lock:
            self.skips_no_vad += 1

    def record_schema_error(self) -> None:
        with self._lock:
            self.schema_errors += 1

    def record_applied(self) -> None:
        with self._lock:
            self.applied += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "skips_no_vad": self.skips_no_vad,
                "schema_errors": self.schema_errors,
                "applied": self.applied,
            }
