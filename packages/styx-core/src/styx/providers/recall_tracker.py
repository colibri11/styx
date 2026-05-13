"""RecallTracker — in-memory ring buffer recall_event_ids per session.

Используется provider'ом (волна 7c) чтобы связать ``styx_recall`` с
последующим ``sync_turn``: каждый recall в session кладёт ids в буфер,
sync_turn в той же session забирает их и enqueue'ит классификатор.

Persistence: нет. Рестарт Hermes-процесса = пропуск одного turn'а
классификации. Это не критично: classifier — фоновый сигнал, не
влияющий на active suffix.

Thread-safety: дёргается из одного потока (Hermes plugin loop), но
держим Lock на всякий случай — на будущее multi-thread интеграции.
"""

from __future__ import annotations

import threading
import uuid


DEFAULT_MAX_PER_SESSION = 50


class RecallTracker:
    """Ring-buffer recall_event_ids на session_id."""

    def __init__(self, max_per_session: int = DEFAULT_MAX_PER_SESSION) -> None:
        if max_per_session < 1:
            raise ValueError("max_per_session >= 1")
        self._max = max_per_session
        self._buffer: dict[uuid.UUID, list[int]] = {}
        self._lock = threading.Lock()

    def append(self, session_id: uuid.UUID, recall_event_ids: list[int]) -> None:
        if not recall_event_ids:
            return
        with self._lock:
            buf = self._buffer.setdefault(session_id, [])
            buf.extend(recall_event_ids)
            if len(buf) > self._max:
                self._buffer[session_id] = buf[-self._max :]

    def take(self, session_id: uuid.UUID) -> list[int]:
        """Извлечь и очистить buffer для session_id."""
        with self._lock:
            return self._buffer.pop(session_id, [])

    def peek(self, session_id: uuid.UUID) -> list[int]:
        """Снимок без очистки. Только для тестов / диагностики."""
        with self._lock:
            return list(self._buffer.get(session_id, []))

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buffer.values())
