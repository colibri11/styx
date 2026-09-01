"""Интеграция completed-turn evidence -> state -> memory snapshot."""

from __future__ import annotations

import uuid
import datetime as dt
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row

from styx.embedding import FakeEmbeddingClient
from styx.emotional.state import EmotionalVector
from styx.emotional.transition import (
    AffectiveAssessment,
    CognitivePosture,
    TransitionMetrics,
)
from styx.providers.memory import (
    StyxMemoryCore,
    _bounded_causal_context,
    _observer_prior_context,
)


class _Observer:
    def __init__(self, assessments: list[AffectiveAssessment | None]) -> None:
        self._assessments = list(assessments)
        self.metrics = TransitionMetrics()
        self.calls = 0
        self.received: list[dict] = []
        self.on_observe = None

    def observe(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        self.received.append(kwargs)
        if self.on_observe is not None:
            self.on_observe()
        if not self._assessments:
            raise AssertionError("unexpected observer call")
        return self._assessments.pop(0)


def _assessment(
    *,
    cause: str = "risk of semantic substitution",
    status: str = "active",
    confidence: float = 0.8,
    reaction: EmotionalVector = EmotionalVector(-0.2, 0.5, 0.6),
    updates_event_ids: tuple[int, ...] = (),
    reaffirms_event_ids: tuple[int, ...] = (),
    revises_event_ids: tuple[int, ...] = (),
    intensity: float = 0.75,
    posture: CognitivePosture | None = None,
) -> AffectiveAssessment:
    return AffectiveAssessment(
        stimulus=EmotionalVector(-0.7, 0.8, 0.1),
        reaction=reaction,
        cause_class="semantic_alignment",
        cause_summary=cause,
        intensity=intensity,
        confidence=confidence,
        cause_status=status,  # type: ignore[arg-type]
        posture=posture or CognitivePosture(
            attention="verify_correspondence",
            verification_depth="high",
            branch_budget="narrow",
            closure_policy="resist_premature_closure",
        ),
        updates_event_ids=updates_event_ids,
        reaffirms_event_ids=reaffirms_event_ids,
        revises_event_ids=revises_event_ids,
    )


@pytest.fixture
def core_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str) -> str:
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)
    monkeypatch.setenv("STYX_SENTIMENT_ENABLED", "0")
    monkeypatch.setattr(
        "styx.providers.memory.make_embedding_client",
        lambda **_: FakeEmbeddingClient(dim=768),
    )
    return migrated_db


def test_completed_turn_is_idempotent_and_separates_stimulus_from_reaction(
    core_env: str,
) -> None:
    agent = f"affect-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    observer = _Observer([_assessment()])
    core._transition_observer = observer  # controlled integration seam
    try:
        payload = dict(
            idempotency_key=f"hermes:{sid}:turn-1",
            turn_id="turn-1",
            session_id=sid,
            user_message="Нет, смысл был другим; проверь соответствие.",
            assistant_response="Проверил причинную линию и исправил вывод.",
        )
        first = core.observe_affective_turn(**payload)
        second = core.observe_affective_turn(**payload)
        assert first == {"accepted": True, "duplicate": False, "reason": None}
        assert second == {"accepted": True, "duplicate": True, "reason": None}
        assert observer.calls == 1

        with psycopg.connect(core_env, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT valence, arousal, dominance, confidence, cause_status "
                    "FROM emotional_events WHERE agent_id = %s",
                    (agent,),
                )
                events = cur.fetchall()
                cur.execute(
                    "SELECT delta_valence, delta_arousal, delta_dominance, "
                    "confidence, computation_version, causal_context "
                    "FROM emotional_state WHERE agent_id = %s",
                    (agent,),
                )
                states = cur.fetchall()
        assert len(events) == 1
        assert tuple(float(events[0][axis]) for axis in ("valence", "arousal", "dominance")) == pytest.approx((-0.7, 0.8, 0.1))
        assert events[0]["cause_status"] == "active"
        assert len(states) == 1
        # State follows reaction, not the peer stimulus.
        assert float(states[0]["delta_arousal"]) < 0.5
        assert float(states[0]["delta_arousal"]) > 0.0
        assert states[0]["computation_version"] == "causal-turn-v1"
        assert states[0]["causal_context"][0]["cause_active"] is True
    finally:
        core.shutdown()


def test_mixed_causes_survive_and_snapshot_enters_memory(core_env: str) -> None:
    agent = f"affect-mixed-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    core._transition_observer = _Observer([
        _assessment(cause="risk of semantic substitution", status="active"),
        _assessment(
            cause="task direction is now stable",
            status="resolved",
            reaction=EmotionalVector(0.4, -0.1, 0.5),
        ),
    ])
    try:
        for number in (1, 2):
            result = core.observe_affective_turn(
                idempotency_key=f"hermes:{sid}:turn-{number}",
                turn_id=f"turn-{number}",
                session_id=sid,
                user_message=f"user-{number}",
                assistant_response=f"assistant-{number}",
            )
            assert result["accepted"] is True

        core.sync_turn("final user", "final assistant", session_id=sid)
        with psycopg.connect(core_env, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT causal_context FROM emotional_state "
                    "WHERE agent_id = %s ORDER BY at DESC, id DESC LIMIT 1",
                    (agent,),
                )
                causes = cur.fetchone()["causal_context"]
                cur.execute(
                    "SELECT emotional_context_state_id, "
                    "emotional_context_confidence, emotional_context_causes "
                    "FROM memories WHERE agent_id = %s ORDER BY seq DESC LIMIT 1",
                    (agent,),
                )
                memory = cur.fetchone()
        assert [item["status"] for item in causes] == ["active", "resolved"]
        assert memory["emotional_context_state_id"] is not None
        assert float(memory["emotional_context_confidence"]) == pytest.approx(0.8)
        assert len(memory["emotional_context_causes"]) == 2
    finally:
        core.shutdown()


def test_observer_failure_is_fail_open(core_env: str) -> None:
    agent = f"affect-fail-{uuid.uuid4().hex[:8]}"
    core = StyxMemoryCore(agent)
    core.initialize(session_id=str(uuid.uuid4()))
    core._transition_observer = _Observer([None])
    try:
        result = core.observe_affective_turn(
            idempotency_key="failure-turn",
            turn_id="turn-failure",
            user_message="user",
            assistant_response="assistant",
        )
        assert result == {
            "accepted": False,
            "duplicate": False,
            "reason": "observation_failed",
        }
    finally:
        core.shutdown()


def test_resolution_stops_support_for_a_specific_active_cause(
    core_env: str,
) -> None:
    agent = f"affect-resolve-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    core._transition_observer = _Observer([
        _assessment(cause="semantic mismatch", status="active"),
    ])
    try:
        first_result = core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:turn-1",
            turn_id="turn-1",
            session_id=sid,
            user_message="user-1",
            assistant_response="assistant-1",
        )
        assert first_result["accepted"] is True
        with psycopg.connect(core_env) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM emotional_events WHERE agent_id=%s "
                "AND source_ref='turn-1'",
                (agent,),
            )
            first_event_id = int(cur.fetchone()[0])
        core._transition_observer = _Observer([_assessment(
            cause="semantic mismatch was corrected",
            status="resolved",
            reaction=EmotionalVector(0.2, -0.3, 0.4),
            updates_event_ids=(first_event_id,),
        )])
        second_result = core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:turn-2",
            turn_id="turn-2",
            session_id=sid,
            user_message="user-2",
            assistant_response="assistant-2",
        )
        assert second_result["accepted"] is True

        with psycopg.connect(core_env, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT causal_context FROM emotional_state "
                    "WHERE agent_id = %s ORDER BY at DESC, id DESC LIMIT 1",
                    (agent,),
                )
                causes = cur.fetchone()["causal_context"]

        first = next(item for item in causes if item["source_ref"] == "turn-1")
        second = next(item for item in causes if item["source_ref"] == "turn-2")
        assert first["cause_active"] is False
        assert first["status"] == "resolved"
        assert first["status_source_event_id"] == second["evidence_id"]
        assert second["updates_event_ids"] == [first_event_id]
    finally:
        core.shutdown()


def test_unknown_or_inactive_lineage_ref_is_fail_open_without_write(
    core_env: str,
) -> None:
    agent = f"affect-bad-ref-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    core._transition_observer = _Observer([
        _assessment(
            status="resolved",
            updates_event_ids=(999_999_999,),
        )
    ])
    try:
        result = core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:bad-ref",
            turn_id="bad-ref",
            session_id=sid,
            user_message="user",
            assistant_response="assistant",
        )
        assert result == {
            "accepted": False,
            "duplicate": False,
            "reason": "invalid_cause_lineage",
        }
        with psycopg.connect(core_env) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM emotional_events WHERE agent_id = %s",
                (agent,),
            )
            assert cur.fetchone()[0] == 0
            cur.execute(
                "SELECT count(*) FROM emotional_state WHERE agent_id = %s",
                (agent,),
            )
            assert cur.fetchone()[0] == 0
    finally:
        core.shutdown()


def test_reaffirmation_renews_original_lease_without_stacking_delta(
    core_env: str,
) -> None:
    agent = f"affect-reaffirm-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    core._transition_observer = _Observer([_assessment(status="active")])
    try:
        assert core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:first",
            turn_id="first",
            session_id=sid,
            user_message="user",
            assistant_response="assistant",
        )["accepted"] is True
        with psycopg.connect(core_env) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM emotional_events WHERE agent_id=%s "
                "AND source_ref='first'",
                (agent,),
            )
            cause_id = int(cur.fetchone()[0])
            cur.execute(
                "SELECT lease_expires_at FROM emotional_cause_status "
                "WHERE agent_id=%s AND cause_event_id=%s "
                "ORDER BY at DESC,id DESC LIMIT 1",
                (agent, cause_id),
            )
            first_lease = cur.fetchone()[0]

        core._transition_observer = _Observer([_assessment(
            status="active", reaffirms_event_ids=(cause_id,)
        )])
        assert core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:again",
            turn_id="again",
            session_id=sid,
            user_message="same cause remains",
            assistant_response="same posture remains",
        )["accepted"] is True

        with psycopg.connect(core_env, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT delta_valence,delta_arousal,delta_dominance,"
                    "causal_context FROM emotional_state WHERE agent_id=%s "
                    "ORDER BY at DESC,id DESC LIMIT 1",
                    (agent,),
                )
                state = cur.fetchone()
                cur.execute(
                    "SELECT lease_expires_at FROM emotional_cause_status "
                    "WHERE agent_id=%s AND cause_event_id=%s "
                    "ORDER BY at DESC,id DESC LIMIT 1",
                    (agent, cause_id),
                )
                renewed_lease = cur.fetchone()["lease_expires_at"]
                cur.execute(
                    "SELECT cause_status FROM emotional_events "
                    "WHERE agent_id=%s AND source_ref='again'",
                    (agent,),
                )
                reaffirm_event_status = cur.fetchone()["cause_status"]
        assert (
            float(state["delta_valence"]),
            float(state["delta_arousal"]),
            float(state["delta_dominance"]),
        ) == (0.0, 0.0, 0.0)
        active = [
            item for item in state["causal_context"]
            if item.get("cause_active") is True
        ]
        assert [item["evidence_id"] for item in active] == [cause_id]
        assert renewed_lease > first_lease
        assert reaffirm_event_status == "unknown"
    finally:
        core.shutdown()


def test_snapshot_commits_before_llm_and_excludes_cause_prose(
    core_env: str,
) -> None:
    del core_env
    agent = f"affect-snapshot-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    observer = _Observer([
        _assessment(cause="private first cause", status="active"),
        _assessment(cause="second cause", status="active"),
    ])
    core._transition_observer = observer
    try:
        first = core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:one",
            turn_id="turn-one",
            session_id=sid,
            user_message="user one",
            assistant_response="assistant one",
        )
        assert first["accepted"] is True

        def _assert_idle() -> None:
            assert core._conn is not None
            assert core._conn.info.transaction_status == TransactionStatus.IDLE

        observer.on_observe = _assert_idle
        second = core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:two",
            turn_id="turn-two",
            session_id=sid,
            user_message="user two",
            assistant_response="assistant two",
        )
        assert second["accepted"] is True
        prior = observer.received[1]["prior_context"]
        assert prior["causes"][0]["cause_class"] == "semantic_alignment"
        assert "cause" not in prior["causes"][0]
        assert "private first cause" not in str(prior)
    finally:
        core.shutdown()


def test_revision_applies_only_support_difference_and_replaces_posture(
    core_env: str,
) -> None:
    agent = f"affect-revise-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    first_assessment = _assessment(status="active", intensity=0.8)
    core._transition_observer = _Observer([first_assessment])
    try:
        assert core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:first", turn_id="first",
            session_id=sid, user_message="u", assistant_response="a",
        )["accepted"] is True
        with psycopg.connect(core_env) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM emotional_events WHERE agent_id=%s AND source_ref='first'",
                (agent,),
            )
            cause_id = int(cur.fetchone()[0])

        changed_posture = CognitivePosture(
            attention="explore_connections", verification_depth="normal",
            branch_budget="broad", closure_policy="normal",
        )
        revised_assessment = _assessment(
            status="active", intensity=0.2, revises_event_ids=(cause_id,),
            posture=changed_posture,
        )
        core._transition_observer = _Observer([revised_assessment])
        assert core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:revise", turn_id="revise",
            session_id=sid, user_message="u2", assistant_response="a2",
        )["accepted"] is True

        with psycopg.connect(core_env, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT delta_valence,delta_arousal,delta_dominance,causal_context "
                "FROM emotional_state WHERE agent_id=%s ORDER BY at DESC,id DESC LIMIT 1",
                (agent,),
            )
            state = cur.fetchone()
        expected = EmotionalVector(
            revised_assessment.weighted_delta.valence - first_assessment.weighted_delta.valence,
            revised_assessment.weighted_delta.arousal - first_assessment.weighted_delta.arousal,
            revised_assessment.weighted_delta.dominance - first_assessment.weighted_delta.dominance,
        )
        assert float(state["delta_valence"]) == pytest.approx(expected.valence)
        assert float(state["delta_arousal"]) == pytest.approx(expected.arousal)
        assert float(state["delta_dominance"]) == pytest.approx(expected.dominance)
        active = [item for item in state["causal_context"] if item.get("cause_active")]
        assert [item["evidence_id"] for item in active] == [cause_id]
        assert active[0]["cognitive_posture"] == changed_posture.as_dict()
        assert active[0]["intensity"] == 0.2
    finally:
        core.shutdown()


def test_persistence_redacts_cause_even_at_direct_observer_seam(
    core_env: str,
) -> None:
    agent = f"affect-redact-{uuid.uuid4().hex[:8]}"
    sid = str(uuid.uuid4())
    core = StyxMemoryCore(agent)
    core.initialize(session_id=sid)
    core._transition_observer = _Observer([
        _assessment(cause="email a@example.org phone +7 999 123-45-67")
    ])
    try:
        result = core.observe_affective_turn(
            idempotency_key=f"hermes:{sid}:redact",
            turn_id="turn-redact",
            session_id=sid,
            user_message="user",
            assistant_response="assistant",
        )
        assert result["accepted"] is True
        with psycopg.connect(core_env) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT cause_summary, metadata->>'cause_class' "
                "FROM emotional_events WHERE agent_id = %s",
                (agent,),
            )
            cause, cause_class = cur.fetchone()
        assert "a@example.org" not in cause
        assert "999" not in cause
        assert cause_class == "semantic_alignment"
    finally:
        core.shutdown()


def test_reducer_context_preserves_active_and_bounds_inactive_history() -> None:
    items = [
        {"source_ref": f"old-{index}", "cause_active": False}
        for index in range(100)
    ] + [
        {"source_ref": f"live-{index}", "cause_active": True}
        for index in range(9)
    ]
    bounded = _bounded_causal_context(items)
    assert len(bounded) == 17
    assert [item["source_ref"] for item in bounded[:9]] == [
        f"live-{index}" for index in range(9)
    ]
    assert [item["source_ref"] for item in bounded[9:]] == [
        f"old-{index}" for index in range(92, 100)
    ]


def test_observer_prior_selects_strong_recent_causes_and_keeps_subject_identity() -> None:
    now = dt.datetime.now(tz=dt.timezone.utc)
    causes = []
    lifecycle = {}
    for index in range(10):
        item = {
            "evidence_id": index + 1,
            "cause_class": "semantic_alignment",
            "cause_subject": (
                "response_correctness" if index % 2 == 0 else "constraint_compliance"
            ),
            "cause_active": True,
            "status": "active",
            "intensity": (index + 1) / 10,
            "confidence": 1.0,
            "observed_at": (now + dt.timedelta(seconds=index)).isoformat(),
            "lease_expires_at": (now + dt.timedelta(minutes=10)).isoformat(),
            "cognitive_posture": CognitivePosture().as_dict(),
        }
        causes.append(item)
        lifecycle[index + 1] = {"context": item}
    prior = _observer_prior_context(
        SimpleNamespace(
            id=1, at=now, vector=EmotionalVector(0.0, 0.0, 0.0),
            confidence=0.8, causal_context=tuple(causes),
        ),
        lifecycle,
    )
    assert [item["evidence_id"] for item in prior["causes"]] == list(range(10, 2, -1))
    assert {item["cause_subject"] for item in prior["causes"]} == {
        "response_correctness", "constraint_compliance",
    }
    assert prior["omitted_cause_count"] == 2
