"""Pre-LLM cognitive posture из причинных координат состояния.

Историческое имя ``self_state`` сохранено для config/plugin ABI. Канал больше
не переводит знаки VAD в название эмоции. Он проецирует накопленное состояние
и явные координаты текущего Hermes event в компактную политику решений,
которая действует до появления языка.

Payload намеренно не содержит сырого текста пользователя и инструкций о тоне,
формулировках или обязательном проговаривании. Сигналы текущего события
ограничены точными маркерами и наличием известных host-полей: это координаты
основания распознавания, а не притворный semantic emotion classifier.

Fail-open rules:

* disabled channel -> ``None``;
* no active inherited state and no explicit current-event signal -> ``None``;
* an old active state is excluded and logged, while a current event may still
  produce a posture;
* database failures do not suppress an independently recognised current
  event.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import re
from contextlib import nullcontext
from typing import Any

from styx.emotional.state import EmotionalVector
from styx.engine.pre_llm_inject import ChannelHandle

log = logging.getLogger(__name__)


_CURRENT_EVENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "correction",
        re.compile(
            r"(?:\bне\s+так\b|\bошиб\w*|\bисправ\w*|\bпоправ\w*|"
            r"\bwrong\b|\bincorrect\b|\bfix\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "precision_required",
        re.compile(
            r"(?:\bточн\w*|\bпровер\w*|\bвериф\w*|\bexact\w*|"
            r"\bverif\w*|\bcareful\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_constraint",
        re.compile(
            r"(?:\bне\s+(?:надо|делай|нужно)\b|\bпока\s+не\b|"
            r"\bсначала\b|\bтолько\b|\bбез\s+\w+|\bdo\s+not\b|"
            r"\bmust\b|\bonly\b|\bwithout\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_conflict",
        re.compile(
            r"(?:\bпротивореч\w*|\bнесовмест\w*|\bконфликт\w*|"
            r"\bcontradict\w*|\bconflict\w*|\bincompatib\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "urgency",
        re.compile(
            r"(?:\bсроч\w*|\bнемедлен\w*|\bпрямо\s+сейчас\b|"
            r"\burgent\w*|\bimmediately\b|\bright\s+now\b)",
            re.IGNORECASE,
        ),
    ),
)

_HOST_CONTEXT_KEYS: tuple[str, ...] = (
    "goal",
    "current_goal",
    "task",
    "constraints",
    "conflicts",
    "risk",
    "urgency",
)

_CAUSE_CLASSES = {
    "semantic_alignment",
    "task_uncertainty",
    "constraint_pressure",
    "execution_risk",
    "conflicting_signals",
    "goal_progress",
    "discovery",
    "interpersonal_tension",
    "resolution",
    "unknown",
}
_CAUSE_SUBJECTS = {
    "response_correctness", "task_completion", "constraint_compliance",
    "tool_outcome", "relationship_alignment", "uncertainty_resolution",
    "external_event", "unknown",
}
_POSTURE_VALUES: dict[str, set[str]] = {
    "attention": {
        "preserve_direction",
        "verify_correspondence",
        "surface_ambiguity",
        "explore_connections",
    },
    "verification_depth": {"normal", "high"},
    "branch_budget": {"narrow", "balanced", "broad"},
    "closure_policy": {"normal", "resist_premature_closure"},
}
_CAUSE_OUTPUT_SOFT_LIMIT = 8


def _current_event_coordinates(hermes_kwargs: dict[str, Any]) -> dict[str, Any]:
    user_message = hermes_kwargs.get("user_message")
    text = user_message if isinstance(user_message, str) else ""
    signals = [
        name for name, pattern in _CURRENT_EVENT_PATTERNS if pattern.search(text)
    ]
    host_fields = [
        key
        for key in _HOST_CONTEXT_KEYS
        if key in hermes_kwargs and hermes_kwargs[key] not in (None, "", [], {})
    ]

    explicit_count = len(signals) + len(host_fields)
    return {
        "source": "current_hermes_event",
        "cause_active": bool(text or host_fields),
        "user_message_present": bool(text),
        "explicit_signals": signals,
        "host_context_fields": host_fields,
        "intensity": round(min(1.0, explicit_count / 4.0), 3),
        "confidence": 1.0 if explicit_count else None,
        "confidence_basis": (
            "exact_marker_or_field_presence" if explicit_count else None
        ),
    }


def _append_once(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _safe_source_ref(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= 128 and re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def _controlled_posture(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for field, allowed in _POSTURE_VALUES.items():
        candidate = value.get(field)
        if candidate not in allowed:
            return None
        result[field] = candidate
    return result


def _parse_timestamp(value: object) -> _dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed


def _unit_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else None


def _structured_causes(
    raw_causes: list[Any], *, now: _dt.datetime
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return hard-bounded coordinates plus conservative omitted summary."""
    valid = [item for item in raw_causes if isinstance(item, dict)]
    normalized: list[dict[str, Any]] = []
    for item in valid:
        cause_class = item.get("cause_class")
        cause_subject = item.get("cause_subject")
        status = item.get("status")
        observed_at = _parse_timestamp(item.get("observed_at"))
        lease_expires_at = _parse_timestamp(item.get("lease_expires_at"))
        lease_valid = lease_expires_at is not None and lease_expires_at > now
        active = (
            item.get("cause_active") is True
            and status == "active"
            and lease_valid
        )
        intensity = _unit_coordinate(item.get("intensity"))
        confidence = _unit_coordinate(item.get("confidence"))
        normalized.append({
            "event": item.get("evidence_id")
            if isinstance(item.get("evidence_id"), int)
            and not isinstance(item.get("evidence_id"), bool) else None,
            "source_ref": _safe_source_ref(item.get("source_ref")),
            "cause_class": cause_class if cause_class in _CAUSE_CLASSES else "unknown",
            "cause_subject": (
                cause_subject if cause_subject in _CAUSE_SUBJECTS else "unknown"
            ),
            "status": "expired" if status == "active" and not lease_valid else status
            if status in {
                "unknown", "active", "resolved", "superseded", "expired"
            }
            else "unknown",
            "cause_active": active,
            "intensity": intensity,
            "confidence": confidence,
            "observed_at": observed_at.isoformat()
            if observed_at is not None else None,
            "lease_expires_at": lease_expires_at.isoformat()
            if lease_expires_at is not None else None,
            "posture": _controlled_posture(item.get("cognitive_posture")),
            "posture_weight": (
                round(intensity * confidence, 6)
                if intensity is not None and confidence is not None else 0.0
            ),
        })
    active = sorted(
        (item for item in normalized if item["cause_active"]),
        key=lambda item: float(item["posture_weight"]),
        reverse=True,
    )
    inactive = [item for item in normalized if not item["cause_active"]]
    selected = (active + list(reversed(inactive)))[:_CAUSE_OUTPUT_SOFT_LIMIT]
    selected_ids = {id(item) for item in selected}
    selected = [item for item in normalized if id(item) in selected_ids]

    min_effective_posture_weight = 0.05
    active_for_conflict = [
        item for item in active
        if item["posture_weight"] >= min_effective_posture_weight
        and isinstance(item["posture"], dict)
    ]
    conflicts: list[str] = []
    for field in _POSTURE_VALUES:
        ranked = sorted(
            (
                (float(item["posture_weight"]), item["posture"][field])
                for item in active_for_conflict
            ),
            reverse=True,
        )
        if ranked:
            top_weight, top_value = ranked[0]
            if any(
                value != top_value and weight >= max(0.15, top_weight * 0.6)
                for weight, value in ranked[1:]
            ):
                conflicts.append(field)
    summary = {
        "omitted_count": max(0, len(normalized) - len(selected)),
        "omitted_active_count": max(
            0,
            sum(1 for item in normalized if item["cause_active"])
            - sum(1 for item in selected if item["cause_active"]),
        ),
        "aggregate_posture_conflicts": conflicts,
    }
    return selected, summary


def _decision_policy(
    vector: EmotionalVector | None,
    current_event: dict[str, Any],
    active_causes: list[dict[str, Any]],
    aggregate_posture_conflicts: list[str],
) -> dict[str, Any]:
    """Спроецировать причины в reasoning controls, не в стиль языка."""
    attention = ["task_goal", "explicit_constraints"]
    verification_depth = "standard"
    branch_budget = "bounded_parallel"
    ambiguity_handling = "track"
    closure_threshold = "standard"
    constraint_priority = "normal"
    posture_conflicts: list[str] = []

    if vector is not None:
        if vector.valence < -0.2:
            _append_once(attention, "semantic_alignment")
            verification_depth = "high"
            closure_threshold = "high"
        if vector.arousal > 0.2:
            _append_once(attention, "highest_risk_assumption")
            branch_budget = "one_primary"
        elif vector.arousal < -0.2:
            branch_budget = "bounded_parallel"
        if vector.dominance < -0.2:
            ambiguity_handling = "surface_before_commit"
        elif vector.dominance > 0.2:
            _append_once(attention, "completion_path")

    effective: dict[str, str] = {}
    posture_conflicts = list(aggregate_posture_conflicts)
    for field in _POSTURE_VALUES:
        ranked = sorted(
            (
                (float(item.get("posture_weight", 0.0)), item["posture"][field])
                for item in active_causes
                if item.get("cause_active") is True
                and float(item.get("posture_weight", 0.0)) >= 0.05
                and isinstance(item.get("posture"), dict)
            ),
            reverse=True,
        )
        if ranked:
            effective[field] = ranked[0][1]

    if effective:
        if effective.get("attention") == "verify_correspondence":
            _append_once(attention, "semantic_alignment")
        elif effective.get("attention") == "surface_ambiguity":
            _append_once(attention, "signal_conflicts")
            ambiguity_handling = "surface_before_commit"
        elif effective.get("attention") == "explore_connections":
            _append_once(attention, "cross_connections")
        if effective.get("verification_depth") == "high":
            verification_depth = "high"
        if effective.get("branch_budget") == "narrow":
            branch_budget = "one_primary"
        elif effective.get("branch_budget") == "broad" and branch_budget != "one_primary":
            branch_budget = "bounded_parallel"
        if effective.get("closure_policy") == "resist_premature_closure":
            closure_threshold = "high"
    if posture_conflicts:
        _append_once(attention, "posture_conflicts")
        ambiguity_handling = "surface_before_commit"
        closure_threshold = "high"

    signals = set(current_event["explicit_signals"])
    host_fields = set(current_event["host_context_fields"])
    if "correction" in signals:
        _append_once(attention, "semantic_alignment")
        verification_depth = "high"
        closure_threshold = "high"
    if "precision_required" in signals:
        _append_once(attention, "evidence_quality")
        verification_depth = "high"
    if "explicit_constraint" in signals or "constraints" in host_fields:
        constraint_priority = "explicit_first"
    if "explicit_conflict" in signals or "conflicts" in host_fields:
        _append_once(attention, "signal_conflicts")
        ambiguity_handling = "surface_before_commit"
        closure_threshold = "high"
    if "urgency" in signals or "urgency" in host_fields:
        branch_budget = "one_primary"
    if {"goal", "current_goal", "task"} & host_fields:
        _append_once(attention, "host_supplied_goal")
    if "risk" in host_fields:
        _append_once(attention, "highest_risk_assumption")
        verification_depth = "high"

    return {
        "attention_order": attention,
        "verification_depth": verification_depth,
        "branch_budget": branch_budget,
        "ambiguity_handling": ambiguity_handling,
        "closure_threshold": closure_threshold,
        "constraint_priority": constraint_priority,
        "posture_conflicts": posture_conflicts,
    }


def channel_self_state(
    handle: ChannelHandle, hermes_kwargs: dict[str, Any]
) -> str | None:
    """Вернуть структурированную cognitive posture либо ``None`` при покое."""
    if not handle.self_state_enabled:
        return None

    current_event = _current_event_coordinates(hermes_kwargs)
    vector: EmotionalVector | None = None
    at: _dt.datetime | None = None
    rich_record = None
    connection = getattr(handle.queries, "_conn", None)
    lock_context = handle.write_lock if handle.write_lock is not None else nullcontext()
    with lock_context:
        try:
            rich_reader = getattr(
                handle.queries, "get_last_emotional_state_record", None
            )
            if callable(rich_reader):
                rich_record = rich_reader()
                entry = (
                    (rich_record.vector, rich_record.at)
                    if rich_record is not None else None
                )
            else:
                entry = handle.queries.get_last_emotional_state()
            # The provider owns one shared psycopg connection. A read starts a
            # transaction too; finish it before any worker/write path waits on
            # the same handle.
            if connection is not None:
                connection.commit()
        except Exception as exc:  # noqa: BLE001 - fail-open on DB errors
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:  # noqa: BLE001 - defensive cleanup
                    pass
            log.warning("self_state: get_last_emotional_state failed: %s", exc)
            entry = None

    now = _dt.datetime.now(tz=_dt.timezone.utc)
    raw_causes = list(rich_record.causal_context) if rich_record else []
    structured_causes, cause_summary = _structured_causes(raw_causes, now=now)
    has_unexpired_active_cause = any(
        item.get("cause_active") is True for item in structured_causes
    )
    if entry is not None:
        candidate, candidate_at = entry
        norm = math.sqrt(
            candidate.valence ** 2
            + candidate.arousal ** 2
            + candidate.dominance ** 2
        )
        if norm >= handle.self_state_min_norm:
            if candidate_at.tzinfo is None:
                candidate_at = candidate_at.replace(tzinfo=_dt.timezone.utc)
            age_s = max(0.0, (now - candidate_at).total_seconds())
            if (
                age_s > handle.self_state_max_age_s
                and not has_unexpired_active_cause
            ):
                log.warning(
                    "self_state: last state age=%.0fs > "
                    "self_state_max_age_s=%.0fs - inherited cause excluded; "
                    "styx-worker may not be applying emotional_tick decay",
                    age_s,
                    handle.self_state_max_age_s,
                )
            else:
                vector = candidate
                at = candidate_at

    has_current_signal = bool(
        current_event["explicit_signals"] or current_event["host_context_fields"]
    )
    if vector is None and not has_current_signal:
        return None

    inherited: dict[str, Any] | None = None
    if vector is not None and at is not None:
        causes = structured_causes
        inherited = {
            "source": "emotional_state:last",
            "observed_at": at.isoformat(),
            "age_s": round(max(0.0, (now - at).total_seconds()), 3),
            "coordinates": {
                "valence": round(float(vector.valence), 6),
                "arousal": round(float(vector.arousal), 6),
                "dominance": round(float(vector.dominance), 6),
            },
            "intensity": round(
                min(
                    1.0,
                    math.sqrt(
                        vector.valence ** 2
                        + vector.arousal ** 2
                        + vector.dominance ** 2
                    )
                    / math.sqrt(3.0),
                ),
                6,
            ),
            "confidence": rich_record.confidence if rich_record else None,
            "confidence_basis": (
                "causal_turn_observation"
                if rich_record and rich_record.confidence is not None
                else "legacy_state_has_no_confidence_coordinate"
            ),
            "cause_active": (
                any(item.get("cause_active") is True for item in causes)
                if causes else None
            ),
            "causal_contributions": causes,
            "causal_contributions_summary": cause_summary,
        }

    payload = {
        "kind": "cognitive_posture",
        "version": 1,
        "causal_coordinates": {
            "inherited": inherited,
            "current_event": current_event,
        },
        "decision_policy": _decision_policy(
            vector,
            current_event,
            list((inherited or {}).get("causal_contributions", [])),
            list(cause_summary["aggregate_posture_conflicts"]),
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'<styx-self-state version="1">{encoded}</styx-self-state>'
