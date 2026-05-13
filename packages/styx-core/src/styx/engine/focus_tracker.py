"""Focus tracker — sliding centroid из K последних user-embed'ов + drift signal.

Волна 10. Closes условие переоткрытия из ADR § 23.2: salient block
кэшируется на эпоху (последовательность turn'ов между двумя drift-
событиями) и переиспользуется между compress'ами. Cache hit на стороне
LLM-провайдера становится реальным.

Drift detection: для каждого нового user-embed'а считаем cosine с
центроидом окна (mean из K последних embed'ов). Если cosine ниже
порога — drift, кэш salient'а инвалидируется, идёт fresh recall.

Per-agent state: словарь ``agent_id → FocusState``. Один core daemon
обслуживает несколько агентов параллельно. Параметры окна/порога
живут внутри ``FocusState`` (per-agent override).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

from styx.observability.logging import log_event

log = logging.getLogger(__name__)


@dataclass
class FocusState:
    """Скользящее окно последних K user-embed'ов и кэшированный salient."""

    window_size: int = 3
    drift_threshold: float = 0.4
    window: list[list[float]] = field(default_factory=list)
    cached_salient: dict | None = None
    epoch_id: int = 0


_STATES: dict[str, FocusState] = {}
_LOCK = threading.Lock()


def configure(
    agent_id: str,
    *,
    window_size: int = 3,
    drift_threshold: float = 0.4,
) -> None:
    """Инициализировать tracker для ``agent_id``. Двойной configure пересоздаёт состояние."""
    if window_size < 1:
        raise ValueError(f"window_size должен быть >= 1, получено {window_size}")
    if not 0.0 <= drift_threshold <= 1.0:
        raise ValueError(
            f"drift_threshold должен быть в [0, 1], получено {drift_threshold}"
        )
    with _LOCK:
        _STATES[agent_id] = FocusState(
            window_size=window_size,
            drift_threshold=drift_threshold,
        )


def get_state(agent_id: str) -> FocusState | None:
    return _STATES.get(agent_id)


def get_centroid(agent_id: str) -> list[float] | None:
    """Mean из текущего окна или None если tracker не configured / окно пусто.

    Используется волной 12 (eviction relevance-aware) — ranking middle-
    сообщений против текущего фокуса. Mean считается по тем же векторам,
    что и centroid в drift-сравнении внутри ``observe`` — это та же точка
    в embedding-пространстве, на которой основана вся drift-семантика.
    """
    state = _STATES.get(agent_id)
    if state is None or not state.window:
        return None
    return _mean(state.window)


def observe(agent_id: str, new_embed: list[float]) -> bool:
    """Зарегистрировать новый user-embed, вернуть True если drift detected.

    На пустом окне (первый turn в session) считается drift=True — это
    форсирует первый recall и заполнение кэша. Затем new_embed
    добавляется в окно (FIFO с вытеснением старейшего при превышении
    K).

    Если tracker не configured для ``agent_id`` — undefined; caller
    обязан сам проверить ``get_state(agent_id)``.
    """
    state = _STATES.get(agent_id)
    if state is None:
        # Безопасный fallback: считаем drift, не трогаем (несуществующее) state.
        return True

    if not state.window:
        state.window.append(list(new_embed))
        state.epoch_id = 1
        return True

    centroid = _mean(state.window)
    sim = _cosine(new_embed, centroid)
    drift = sim < state.drift_threshold

    state.window.append(list(new_embed))
    if len(state.window) > state.window_size:
        state.window.pop(0)

    if drift:
        state.epoch_id += 1
        log_event(
            log,
            "drift_detected",
            agent_id=agent_id,
            cosine=round(sim, 4),
            threshold=state.drift_threshold,
            epoch_id=state.epoch_id,
        )
    return drift


def set_cached(agent_id: str, salient: dict | None) -> None:
    """Записать новый кэш salient'а. None = инвалидировать."""
    state = _STATES.get(agent_id)
    if state is not None:
        state.cached_salient = salient


def reset(agent_id: str) -> None:
    """Полный сброс tracker'а одного агента. Вызывается из shutdown() и тестов."""
    with _LOCK:
        _STATES.pop(agent_id, None)


def reset_all() -> None:
    """Сбросить tracker все агенты. Используется в daemon shutdown / тестах."""
    with _LOCK:
        _STATES.clear()


def restore(
    agent_id: str,
    window: list[list[float]],
    cached_salient: dict | None,
    epoch_id: int,
) -> None:
    """Записать persisted state поверх текущего (волна 13).

    Требует ``configure(agent_id, ...)`` уже вызванным — иначе no-op +
    warning. Окно обрезается до текущего ``window_size``: если между
    save и restart'ом ENV ``STYX_FOCUS_WINDOW_SIZE`` уменьшился,
    оставляем хвост (свежие embed'ы — самые информативные про текущий
    фокус).
    """
    state = _STATES.get(agent_id)
    if state is None:
        log.warning(
            "focus_tracker.restore: tracker не configured для agent_id=%s — no-op",
            agent_id,
        )
        return
    state.window = [list(v) for v in window[-state.window_size:]]
    state.cached_salient = (
        dict(cached_salient) if cached_salient is not None else None
    )
    state.epoch_id = max(0, int(epoch_id))


def snapshot(agent_id: str) -> tuple[list[list[float]], dict | None, int] | None:
    """Shallow-copy текущего state'а для save (волна 13).

    Возвращает ``(window, cached_salient, epoch_id)`` или None если
    tracker не configured для ``agent_id``. Каждый embed-вектор
    копируется (mutable list), salient — новый dict (тот же контракт по
    immutability как в ``set_cached``). Возвращённый tuple отвязан от
    state'а в ``_STATES``.
    """
    state = _STATES.get(agent_id)
    if state is None:
        return None
    return (
        [list(v) for v in state.window],
        dict(state.cached_salient) if state.cached_salient is not None else None,
        state.epoch_id,
    )


def _mean(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
