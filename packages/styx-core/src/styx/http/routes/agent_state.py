"""GET /agent_state — snapshot эмоционального состояния агента."""

from __future__ import annotations

import re
import datetime as dt

from fastapi import APIRouter, Depends

from styx.emotional.baseline import read_baseline_for_scoring
from styx.emotional.state import read_last_state_record
from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import AffectiveStateEvidence, VAD, AgentStateResponse
from styx.http.models import RecallAffectiveCauseRef

router = APIRouter()

_CAUSE_CLASSES = {
    "semantic_alignment", "task_uncertainty", "constraint_pressure",
    "execution_risk", "conflicting_signals", "goal_progress", "discovery",
    "interpersonal_tension", "resolution", "unknown",
}
_CAUSE_SUBJECTS = {
    "response_correctness", "task_completion", "constraint_compliance",
    "tool_outcome", "relationship_alignment", "uncertainty_resolution",
    "external_event", "unknown",
}
_CAUSE_STATUSES = {"active", "resolved", "superseded", "expired", "unknown"}
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None else parsed


def _safe_causal_components(raw_items) -> list[RecallAffectiveCauseRef]:
    """Typed, bounded projection; never expose audit prose or posture."""
    projected: list[tuple[float, RecallAffectiveCauseRef]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        evidence_id = raw.get("evidence_id")
        if isinstance(evidence_id, bool) or not isinstance(evidence_id, int) or evidence_id <= 0:
            evidence_id = None
        status = raw.get("status") if raw.get("status") in _CAUSE_STATUSES else "unknown"
        lease_raw = raw.get("lease_expires_at")
        lease = _timestamp(lease_raw)
        active = (
            status == "active" and raw.get("cause_active") is True
            and lease is not None and lease > dt.datetime.now(tz=dt.timezone.utc)
        )
        current_status = "expired" if status == "active" and not active else status
        def _unit(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return value if 0.0 <= value <= 1.0 else None
        intensity = _unit(raw.get("intensity"))
        confidence = _unit(raw.get("confidence"))
        weight = (intensity or 0.0) * (confidence or 0.0)
        source_ref = raw.get("source_ref")
        if not isinstance(source_ref, str) or _SAFE_REF.fullmatch(source_ref) is None:
            source_ref = None
        projected.append((weight + (1.0 if active else 0.0), RecallAffectiveCauseRef(
            evidence_id=evidence_id,
            source_ref=source_ref,
            cause_class=raw.get("cause_class") if raw.get("cause_class") in _CAUSE_CLASSES else "unknown",
            cause_subject=raw.get("cause_subject") if raw.get("cause_subject") in _CAUSE_SUBJECTS else "unknown",
            status_at_capture=status,
            current_status=current_status,
            current_active=active,
            intensity=intensity,
            confidence=confidence,
            observed_at=_timestamp(raw.get("observed_at")),
            lease_expires_at=lease,
        )))
    projected.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in projected[:8]]


@router.get(
    "/agent_state",
    response_model=AgentStateResponse,
    dependencies=[Depends(require_auth)],
)
def agent_state(agent_id: str) -> AgentStateResponse:
    session = registry.get(agent_id)
    core = session.core
    if core._conn is None:
        return AgentStateResponse(agent_id=agent_id)

    # The core owns one persistent psycopg connection. Read a committed
    # snapshot under the same mutex as writers and end the read transaction
    # before releasing it; otherwise concurrent HTTP requests can share one
    # transaction boundary.
    with session.write_lock:
        try:
            last = read_last_state_record(core._conn, agent_id)
            baseline = read_baseline_for_scoring(core._conn, agent_id)
            commit = getattr(core._conn, "commit", None)
            if callable(commit):
                commit()
        except Exception:
            rollback = getattr(core._conn, "rollback", None)
            if callable(rollback):
                rollback()
            raise

    instant_vad = None
    if last is not None:
        instant_vad = VAD(
            valence=last.vector.valence,
            arousal=last.vector.arousal,
            dominance=last.vector.dominance,
        )
    baseline_vad = (
        VAD(
            valence=baseline.valence,
            arousal=baseline.arousal,
            dominance=baseline.dominance,
        )
        if baseline is not None
        else None
    )
    return AgentStateResponse(
        agent_id=agent_id,
        instant=instant_vad,
        baseline=baseline_vad,
        mood=None,
        instant_evidence=(
            AffectiveStateEvidence(
                state_id=last.id,
                at=last.at,
                source=last.source,
                parent_state_id=last.parent_state_id,
                event_id=last.event_id,
                delta=(
                    VAD(
                        valence=last.delta.valence,
                        arousal=last.delta.arousal,
                        dominance=last.delta.dominance,
                    )
                    if last.delta is not None else None
                ),
                intensity=last.intensity,
                transition_confidence=last.transition_confidence,
                confidence=last.confidence,
                causal_components=_safe_causal_components(last.causal_context),
                computation_version=last.computation_version,
            )
            if last is not None else None
        ),
    )
