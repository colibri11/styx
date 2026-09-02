"""Working set state persistence — волна 13 (ADR § 29).

Persistence module-global state'а ``focus_tracker`` (window +
cached_salient + epoch_id) и ``hot_tier`` (entries) между restart'ами
Hermes-процесса. Закрывает поверхность waves-v1 § «Working set state»
v2: «Persistence working set'а между restart'ами процесса —
продолжение сессии после выпадения».

Контракт:

- ``load(conn, agent_id, ttl_s, hot_ttl_s, embedding_dim)`` —
  читает строку ``working_set`` из БД, возвращает ``WorkingSetSnapshot``
  либо None (cold start). Past ``ttl_s`` → None; past ``hot_ttl_s`` →
  focus only (hot обнулён). Embedding-dim mismatch → None.
- ``save(dsn, agent_id, payload)`` — INSERT ON CONFLICT DO UPDATE через
  свою connection (open-write-close). save раз в 30s → connect overhead
  ≤5ms допустим.
- ``start(*, dsn, agent_id, embedding_dim, interval_s, write_lock,
  snapshot_fn)`` — запускает daemon-thread, который раз в interval_s
  снимает snapshot под ``write_lock`` и save'ит вне lock'а.
- ``stop()`` — set stop-event, join thread, финальный flush
  (caller обычно уже держит write_lock).
- ``serialize`` / ``deserialize`` — pure helpers, JSON-able dict.

Per-agent state: словарь ``agent_id → _ControlState``. Один core daemon
обслуживает несколько агентов параллельно — каждому своя save-thread,
свой stop_event, своя connection-policy.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import psycopg
from psycopg.types.json import Jsonb

from styx.engine.hot_tier import HotEntry
from styx.emotional.state import EmotionalVector
from styx.observability.logging import log_event
from styx.storage.queries import (
    MemoryAffectiveCauseRef,
    MemoryAffectiveProvenance,
)

log = logging.getLogger(__name__)

_CAUSAL_SOURCE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")


PAYLOAD_VERSION = 2
"""Bump при breaking-change'е shape'а serialize/deserialize."""

_FINAL_FLUSH_TIMEOUT_S = 10.0
"""Сколько ждём остановки daemon-thread'а в ``stop()``."""


@dataclass(frozen=True)
class FocusSnapshot:
    window: list[list[float]]
    cached_salient: dict[str, Any] | None
    epoch_id: int


@dataclass(frozen=True)
class WorkingSetSnapshot:
    """Десериализованный state. focus / hot могут быть None независимо."""

    focus: FocusSnapshot | None
    hot: list[HotEntry] | None


@dataclass
class _ControlState:
    dsn: str
    agent_id: str
    embedding_dim: int
    interval_s: float
    write_lock: threading.Lock
    snapshot_fn: Callable[
        [], tuple[
            tuple[list[list[float]], dict | None, int] | None,
            list[HotEntry],
        ]
    ]
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


_CONTROLS: dict[str, _ControlState] = {}
_CONTROLS_LOCK = threading.Lock()


# ── public control API ────────────────────────────────────────────────────


def start(
    agent_id: str,
    *,
    dsn: str,
    embedding_dim: int,
    interval_s: float,
    write_lock: threading.Lock,
    snapshot_fn: Callable[
        [], tuple[
            tuple[list[list[float]], dict | None, int] | None,
            list[HotEntry],
        ]
    ],
) -> None:
    """Запустить daemon-thread persistence для ``agent_id``. Если уже running — restart."""
    if interval_s <= 0:
        raise ValueError(f"interval_s должен быть > 0, получено {interval_s}")
    if embedding_dim <= 0:
        raise ValueError(
            f"embedding_dim должен быть > 0, получено {embedding_dim}"
        )

    with _CONTROLS_LOCK:
        existing = _CONTROLS.pop(agent_id, None)
    if existing is not None:
        # Двойной start — корректно, например при повторном initialize().
        # Старый thread останавливаем без финального flush'а (вызовет stop()
        # из caller'а если нужно). Здесь не save'им — каждое save должно
        # инициироваться явно, иначе можно потерять данные если уже
        # снизились в новом state'е.
        log.info(
            "working_set_persistence: restart — stop existing thread for agent_id=%s",
            agent_id,
        )
        _stop_thread_only(existing)

    ctrl = _ControlState(
        dsn=dsn,
        agent_id=agent_id,
        embedding_dim=embedding_dim,
        interval_s=interval_s,
        write_lock=write_lock,
        snapshot_fn=snapshot_fn,
    )
    ctrl.thread = threading.Thread(
        target=_loop,
        args=(ctrl,),
        name=f"styx-working-set-{agent_id}",
        daemon=True,
    )
    with _CONTROLS_LOCK:
        _CONTROLS[agent_id] = ctrl
    ctrl.thread.start()


def stop(agent_id: str) -> None:
    """Остановить daemon-thread + финальный sync flush для ``agent_id``.

    Должен вызываться **до** того как caller акквайрит ``write_lock``:
    save-thread может в этот момент держать lock внутри ``_tick``'а;
    если shutdown держит lock и ждёт join — deadlock. После остановки
    thread'а ``snapshot_fn`` зовётся напрямую (без lock'а) — caller в
    этой точке single-threaded, mutate state некому.
    """
    with _CONTROLS_LOCK:
        ctrl = _CONTROLS.pop(agent_id, None)
    if ctrl is None:
        return
    _stop_thread_only(ctrl)
    try:
        focus_snap, hot_snap = ctrl.snapshot_fn()
        payload = serialize(focus_snap, hot_snap, ctrl.embedding_dim)
        save(ctrl.dsn, ctrl.agent_id, payload)
    except Exception as exc:  # noqa: BLE001 — fail-open
        log.warning(
            "working_set_persistence: final flush failed for agent_id=%s: %s",
            agent_id, exc,
        )


def stop_all() -> None:
    """Остановить save-thread'ы всех агентов (используется в daemon shutdown)."""
    with _CONTROLS_LOCK:
        agent_ids = list(_CONTROLS.keys())
    for agent_id in agent_ids:
        stop(agent_id)


def is_running(agent_id: str) -> bool:
    ctrl = _CONTROLS.get(agent_id)
    return ctrl is not None and ctrl.thread is not None and ctrl.thread.is_alive()


def _stop_thread_only(ctrl: _ControlState) -> None:
    ctrl.stop_event.set()
    if ctrl.thread is not None:
        ctrl.thread.join(timeout=_FINAL_FLUSH_TIMEOUT_S)
        if ctrl.thread.is_alive():
            log.warning(
                "working_set_persistence: thread did not stop within %.1fs",
                _FINAL_FLUSH_TIMEOUT_S,
            )


# ── thread loop ───────────────────────────────────────────────────────────


def _loop(ctrl: _ControlState) -> None:
    """Body daemon-thread'а. Tick раз в interval_s до stop_event."""
    while not ctrl.stop_event.is_set():
        # wait возвращает True если event set'нулся — выходим без tick'а
        # (финальный flush — забота caller'а через stop()).
        if ctrl.stop_event.wait(ctrl.interval_s):
            return
        try:
            _tick(ctrl)
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.warning("working_set_persistence: tick failed: %s", exc)


def _tick(ctrl: _ControlState) -> None:
    """Один цикл — snapshot под write_lock, save вне lock'а."""
    with ctrl.write_lock:
        focus_snap, hot_snap = ctrl.snapshot_fn()
    if focus_snap is None and not hot_snap:
        # Нечего сохранять (state не configured).
        return
    payload = serialize(focus_snap, hot_snap, ctrl.embedding_dim)
    started = time.monotonic()
    save(ctrl.dsn, ctrl.agent_id, payload)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    log_event(
        log,
        "working_set_save",
        agent_id=ctrl.agent_id,
        focus_window=len(focus_snap[0]) if focus_snap is not None else 0,
        hot_entries=len(hot_snap) if hot_snap else 0,
        elapsed_ms=elapsed_ms,
    )


# ── (de)serialization ─────────────────────────────────────────────────────


def serialize(
    focus_snap: tuple[list[list[float]], dict | None, int] | None,
    hot_snap: list[HotEntry],
    embedding_dim: int,
) -> dict[str, Any]:
    """Превратить snapshots в JSON-able dict.

    ``focus_snap`` — результат ``focus_tracker.snapshot()``;
    ``hot_snap`` — результат ``hot_tier.snapshot()``;
    ``embedding_dim`` — для guard'а на load (mismatch → drop).
    """
    now_mono = time.monotonic()
    payload: dict[str, Any] = {
        "version": PAYLOAD_VERSION,
        "embedding_dim": embedding_dim,
        "saved_at_monotonic": now_mono,
        "focus": None,
        "hot": None,
    }
    if focus_snap is not None:
        window, cached_salient, epoch_id = focus_snap
        payload["focus"] = {
            "window": [list(v) for v in window],
            "cached_salient": (
                dict(cached_salient) if cached_salient is not None else None
            ),
            "epoch_id": int(epoch_id),
        }
    if hot_snap:
        payload["hot"] = [
            {
                "id": str(e.id),
                "agent_id": e.agent_id,
                "kind": e.kind,
                "kind_src": e.kind_src,
                "role": e.role,
                "content": e.content,
                "metadata": dict(e.metadata) if e.metadata else {},
                "created_at": _serialize_dt(e.created_at),
                "embedding": list(e.embedding),
                "memory_domain": e.memory_domain,
                "line_eligible": e.line_eligible,
                "affective_provenance": _serialize_affective_provenance(
                    e.affective_provenance
                ),
                "evicted_age_s": max(0.0, now_mono - e.evicted_at),
            }
            for e in hot_snap
        ]
    return payload


def deserialize(
    payload: dict[str, Any],
    *,
    embedding_dim: int,
    drop_hot: bool = False,
) -> WorkingSetSnapshot | None:
    """Распарсить payload в WorkingSetSnapshot.

    ``drop_hot=True`` (caller — load когда past hot_ttl_s) — hot
    игнорируется, focus может остаться. None если version/dim mismatch
    или payload явно повреждён.
    """
    if not isinstance(payload, dict):
        log.warning("working_set deserialize: payload is not a dict")
        return None
    version = payload.get("version")
    if version != PAYLOAD_VERSION:
        log.info(
            "working_set deserialize: version mismatch (got %r, want %d) — drop",
            version, PAYLOAD_VERSION,
        )
        return None
    saved_dim = payload.get("embedding_dim")
    if saved_dim != embedding_dim:
        log.info(
            "working_set deserialize: embedding_dim mismatch (saved=%r, current=%d) — drop",
            saved_dim, embedding_dim,
        )
        return None

    focus = _deserialize_focus(payload.get("focus"), embedding_dim)
    hot: list[HotEntry] | None = None
    if not drop_hot:
        hot = _deserialize_hot(payload.get("hot"), embedding_dim)
    return WorkingSetSnapshot(focus=focus, hot=hot)


def _deserialize_focus(
    raw: Any, embedding_dim: int
) -> FocusSnapshot | None:
    if not isinstance(raw, dict):
        return None
    window_raw = raw.get("window") or []
    if not isinstance(window_raw, list):
        return None
    window: list[list[float]] = []
    for v in window_raw:
        if not isinstance(v, list) or len(v) != embedding_dim:
            log.info(
                "working_set deserialize: focus window entry dim mismatch — drop entry"
            )
            continue
        try:
            window.append([float(x) for x in v])
        except (TypeError, ValueError):
            continue
    cached_salient_raw = raw.get("cached_salient")
    cached_salient: dict[str, Any] | None
    if isinstance(cached_salient_raw, dict):
        cached_salient = dict(cached_salient_raw)
    else:
        cached_salient = None
    epoch_id_raw = raw.get("epoch_id", 0)
    try:
        epoch_id = max(0, int(epoch_id_raw))
    except (TypeError, ValueError):
        epoch_id = 0
    if not window and cached_salient is None and epoch_id == 0:
        return None
    return FocusSnapshot(
        window=window, cached_salient=cached_salient, epoch_id=epoch_id
    )


def _deserialize_hot(raw: Any, embedding_dim: int) -> list[HotEntry] | None:
    if not isinstance(raw, list) or not raw:
        return None
    now_mono = time.monotonic()
    out: list[HotEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            entry_id = uuid.UUID(item["id"])
        except (KeyError, ValueError, TypeError):
            continue
        embedding = item.get("embedding")
        if (
            not isinstance(embedding, list)
            or len(embedding) != embedding_dim
        ):
            continue
        try:
            embedding_floats = [float(x) for x in embedding]
        except (TypeError, ValueError):
            continue
        evicted_age_raw = item.get("evicted_age_s", 0.0)
        try:
            evicted_age = max(0.0, float(evicted_age_raw))
        except (TypeError, ValueError):
            evicted_age = 0.0
        evicted_at = now_mono - evicted_age
        try:
            out.append(
                HotEntry(
                    id=entry_id,
                    agent_id=str(item.get("agent_id", "")),
                    kind=str(item.get("kind", "")),
                    kind_src=str(item.get("kind_src", "")),
                    role=str(item.get("role", "")),
                    content=str(item.get("content", "")),
                    metadata=dict(item.get("metadata") or {}),
                    created_at=_deserialize_dt(item.get("created_at")),
                    embedding=embedding_floats,
                    memory_domain=str(
                        item.get("memory_domain", "subjective_trace")
                    ),
                    line_eligible=bool(item.get("line_eligible", True)),
                    evicted_at=evicted_at,
                    affective_provenance=_deserialize_affective_provenance(
                        item.get("affective_provenance")
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            log.info("working_set deserialize: skip hot entry: %s", exc)
            continue
    return out or None


def _serialize_dt(value: Any) -> Any:
    """timestamptz → ISO-строка; иначе как есть (str/None/etc)."""
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    return value


def _serialize_affective_provenance(
    value: MemoryAffectiveProvenance | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "state_id": value.state_id,
        "context_at": _serialize_dt(value.context_at),
        "vad": [value.vad.valence, value.vad.arousal, value.vad.dominance],
        "confidence": value.confidence,
        "causal_refs": [
            {
                "evidence_id": ref.evidence_id,
                "source_ref": ref.source_ref,
                "cause_class": ref.cause_class,
                "cause_subject": ref.cause_subject,
                "status_at_capture": ref.status_at_capture,
                "current_status": ref.current_status,
                "current_active": ref.current_active,
                "intensity": ref.intensity,
                "confidence": ref.confidence,
                "observed_at": _serialize_dt(ref.observed_at),
                "lease_expires_at": _serialize_dt(ref.lease_expires_at),
            }
            for ref in value.causal_refs[:8]
        ],
    }


def _deserialize_affective_provenance(
    raw: Any,
) -> MemoryAffectiveProvenance | None:
    """Read the optional v1 extension; old payloads remain valid."""
    if not isinstance(raw, dict):
        return None
    vad_raw = raw.get("vad")
    if not isinstance(vad_raw, list) or len(vad_raw) != 3:
        return None
    try:
        vad = [float(axis) for axis in vad_raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(axis) and -1.0 <= axis <= 1.0 for axis in vad):
        return None

    state_id_raw = raw.get("state_id")
    state_id = (
        int(state_id_raw)
        if isinstance(state_id_raw, int)
        and not isinstance(state_id_raw, bool)
        and state_id_raw > 0
        else None
    )
    confidence_raw = raw.get("confidence")
    confidence: float | None = None
    if confidence_raw is not None:
        try:
            candidate = float(confidence_raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
            return None
        confidence = candidate

    allowed_classes = {
        "semantic_alignment", "task_uncertainty", "constraint_pressure",
        "execution_risk", "conflicting_signals", "goal_progress",
        "discovery", "interpersonal_tension", "resolution", "unknown",
    }
    allowed_statuses = {
        "active", "resolved", "superseded", "expired", "unknown",
    }
    allowed_subjects = {
        "response_correctness", "task_completion", "constraint_compliance",
        "tool_outcome", "relationship_alignment", "uncertainty_resolution",
        "external_event", "unknown",
    }
    refs: list[MemoryAffectiveCauseRef] = []
    refs_raw = raw.get("causal_refs")
    if isinstance(refs_raw, list):
        for item in refs_raw[:8]:
            if not isinstance(item, dict):
                continue
            evidence_raw = item.get("evidence_id")
            evidence_id = (
                int(evidence_raw)
                if isinstance(evidence_raw, int)
                and not isinstance(evidence_raw, bool)
                and evidence_raw > 0
                else None
            )
            source_raw = item.get("source_ref")
            source_ref = (
                source_raw
                if isinstance(source_raw, str)
                and _CAUSAL_SOURCE_REF_RE.fullmatch(source_raw)
                else None
            )
            if evidence_id is None and source_ref is None:
                continue
            cause_class_raw = item.get("cause_class")
            status_raw = item.get("status_at_capture", item.get("status"))
            current_status_raw = item.get("current_status")
            subject_raw = item.get("cause_subject")
            def _unit(value: Any) -> float | None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                parsed = float(value)
                return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else None
            refs.append(MemoryAffectiveCauseRef(
                evidence_id=evidence_id,
                source_ref=source_ref,
                cause_class=(
                    cause_class_raw
                    if cause_class_raw in allowed_classes else "unknown"
                ),
                cause_subject=(subject_raw if subject_raw in allowed_subjects else "unknown"),
                status_at_capture=(status_raw if status_raw in allowed_statuses else "unknown"),
                current_status=(current_status_raw if current_status_raw in allowed_statuses else None),
                current_active=(item.get("current_active") if isinstance(item.get("current_active"), bool) else None),
                intensity=_unit(item.get("intensity")),
                confidence=_unit(item.get("confidence")),
                observed_at=_deserialize_dt(item.get("observed_at")),
                lease_expires_at=_deserialize_dt(item.get("lease_expires_at")),
            ))

    return MemoryAffectiveProvenance(
        state_id=state_id,
        context_at=_deserialize_dt(raw.get("context_at")),
        vad=EmotionalVector(*vad),
        confidence=confidence,
        causal_refs=tuple(refs),
    )


def _deserialize_dt(value: Any) -> Any:
    """ISO-строка → datetime; всё прочее возвращаем без изменений."""
    if isinstance(value, str):
        try:
            return _dt.datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


# ── DB read/write ─────────────────────────────────────────────────────────


def load(
    conn: psycopg.Connection,
    *,
    agent_id: str,
    ttl_s: float,
    hot_ttl_s: float,
    embedding_dim: int,
) -> WorkingSetSnapshot | None:
    """Читать persisted state из БД. None — cold start."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload, updated_at FROM working_set WHERE agent_id = %s",
                (agent_id,),
            )
            row = cur.fetchone()
        # SELECT не должен оставлять висящую tx — psycopg autocommit обычно
        # выключен, делаем явный commit чтобы не мешать последующим
        # initialize-пишущим запросам.
        try:
            conn.commit()
        except psycopg.Error:
            pass
    except psycopg.Error as exc:
        log.warning("working_set load: SELECT failed: %s", exc)
        try:
            conn.rollback()
        except psycopg.Error:
            pass
        return None

    if row is None:
        return None

    payload, updated_at = row
    if not isinstance(payload, dict):
        log.warning(
            "working_set load: payload is not dict (got %s) — drop",
            type(payload).__name__,
        )
        return None

    age_s: float
    if isinstance(updated_at, _dt.datetime):
        now = _dt.datetime.now(tz=updated_at.tzinfo or _dt.timezone.utc)
        age_s = max(0.0, (now - updated_at).total_seconds())
    else:
        age_s = 0.0

    if age_s > ttl_s:
        log.info(
            "working_set load: state past TTL (age=%.0fs > %.0fs) — cold start",
            age_s, ttl_s,
        )
        return None

    drop_hot = age_s > hot_ttl_s
    if drop_hot:
        log.info(
            "working_set load: hot past TTL (age=%.0fs > %.0fs) — focus only",
            age_s, hot_ttl_s,
        )
    return deserialize(payload, embedding_dim=embedding_dim, drop_hot=drop_hot)


def save(dsn: str, agent_id: str, payload: dict[str, Any]) -> None:
    """INSERT ON CONFLICT DO UPDATE. Open-write-close own connection."""
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO working_set (agent_id, payload, updated_at)
                VALUES (%s, %s, clock_timestamp())
                ON CONFLICT (agent_id) DO UPDATE
                SET payload = EXCLUDED.payload,
                    updated_at = clock_timestamp()
                """,
                (agent_id, Jsonb(payload)),
            )
        conn.commit()
    finally:
        try:
            conn.close()
        except psycopg.Error:
            pass
