"""Unit-тесты для emotional/state.py — pure + DB."""

from __future__ import annotations

import datetime as _dt
import math
import threading
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from styx.emotional.state import (
    EMOTIONAL_AXIS_MAX,
    EMOTIONAL_AXIS_MIN,
    INSTANT_DECAY_EPSILON,
    INSTANT_DECAY_PER_MINUTE,
    EmotionalVector,
    NEUTRAL_VECTOR,
    append_emotional_event,
    append_emotional_state,
    append_emotional_transition,
    aggregate_state_confidence,
    active_cause_support,
    apply_decay,
    apply_instant_decay,
    clamp_axis,
    clamp_vector,
    decay_factor,
    list_active_agent_ids,
    max_abs,
    read_emotional_event,
    read_cause_lifecycle_statuses,
    read_last_state,
    read_last_state_record,
)


# ── Pure functions ────────────────────────────────────────────────────


def test_clamp_axis_basic() -> None:
    assert clamp_axis(0.5) == 0.5
    assert clamp_axis(-2.0) == EMOTIONAL_AXIS_MIN
    assert clamp_axis(5.0) == EMOTIONAL_AXIS_MAX


def test_clamp_vector() -> None:
    v = EmotionalVector(2.0, -3.0, 0.5)
    out = clamp_vector(v)
    assert out.valence == 1.0
    assert out.arousal == -1.0
    assert out.dominance == 0.5


def test_max_abs() -> None:
    v = EmotionalVector(0.3, -0.7, 0.2)
    assert max_abs(v) == 0.7


def test_decay_factor() -> None:
    """factor = 0.95^minutes."""
    assert math.isclose(decay_factor(1), INSTANT_DECAY_PER_MINUTE)
    assert math.isclose(decay_factor(2), INSTANT_DECAY_PER_MINUTE ** 2)


def test_apply_decay() -> None:
    v = EmotionalVector(0.5, -0.4, 0.3)
    out = apply_decay(v, 10.0)
    f = INSTANT_DECAY_PER_MINUTE ** 10
    assert math.isclose(out.valence, 0.5 * f, rel_tol=1e-9)
    assert math.isclose(out.arousal, -0.4 * f, rel_tol=1e-9)


def test_active_cause_support_uses_only_weighted_active_components() -> None:
    support = active_cause_support([
        {"cause_active": True, "weighted_delta": [-0.1, 0.2, 0.3]},
        {"cause_active": False, "weighted_delta": [0.8, 0.8, 0.8]},
        {"cause_active": True, "reaction_vad": [1.0, 1.0, 1.0]},
    ])
    assert support == EmotionalVector(-0.1, 0.2, 0.3)


def test_aggregate_confidence_is_not_latest_transition_confidence() -> None:
    out = aggregate_state_confidence(
        EmotionalVector(0.5, 0.0, 0.0),
        0.9,
        EmotionalVector(0.5, 0.0, 0.0),
        0.1,
    )
    assert out == pytest.approx(0.5)


def test_unknown_confidence_keeps_mass_in_denominator() -> None:
    assert aggregate_state_confidence(
        EmotionalVector(0.5, 0.0, 0.0),
        0.8,
        EmotionalVector(0.5, 0.0, 0.0),
        None,
    ) == pytest.approx(0.4)
    assert aggregate_state_confidence(
        EmotionalVector(0.5, 0.0, 0.0),
        None,
        EmotionalVector(0.5, 0.0, 0.0),
        None,
    ) is None


# ── DB-side ────────────────────────────────────────────────────────────


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM emotional_state WHERE agent_id LIKE 'state-test-%'")
        cur.execute("DELETE FROM emotional_events WHERE agent_id LIKE 'state-test-%'")
        cur.execute("DELETE FROM memories WHERE agent_id LIKE 'state-test-%'")
    conn.commit()
    conn.close()


def test_read_last_state_empty_returns_none(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    assert read_last_state(db, agent) is None


def test_append_emotional_state_neutral_base(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    delta = EmotionalVector(0.1, 0.2, -0.3)
    out = append_emotional_state(db, agent, delta, source="hot_sentiment")
    db.commit()
    assert math.isclose(out.valence, 0.1)
    assert math.isclose(out.arousal, 0.2)
    assert math.isclose(out.dominance, -0.3)

    # И из БД достанем то же значение.
    last = read_last_state(db, agent)
    assert last is not None
    vec, _ = last
    assert math.isclose(vec.valence, 0.1, abs_tol=1e-6)


def test_append_emotional_state_accumulates(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    append_emotional_state(db, agent, EmotionalVector(0.3, 0, 0))
    db.commit()
    out = append_emotional_state(db, agent, EmotionalVector(0.4, 0, 0))
    db.commit()
    assert math.isclose(out.valence, 0.7, abs_tol=1e-6)


def test_append_emotional_state_clamps(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    append_emotional_state(db, agent, EmotionalVector(0.8, 0, 0))
    db.commit()
    out = append_emotional_state(db, agent, EmotionalVector(0.5, 0, 0))
    db.commit()
    # 0.8 + 0.5 = 1.3 → clamp до 1.0.
    assert out.valence == 1.0


def test_event_is_idempotent_and_first_evidence_wins(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    first = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        source_ref="turn-1",
        idempotency_key="observe-turn-1",
        signal=EmotionalVector(-0.4, 0.6, -0.1),
        confidence=0.75,
        cause_summary="несколько неточных интерпретаций",
        cause_status="active",
    )
    second = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        source_ref="turn-1-overwrite",
        idempotency_key="observe-turn-1",
        signal=EmotionalVector(1.0, 1.0, 1.0),
        confidence=1.0,
    )
    db.commit()

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.event.id == first.event.id
    assert second.event.source_ref == "turn-1"
    assert second.event.signal == EmotionalVector(-0.4, 0.6, -0.1)
    assert math.isclose(second.event.confidence or 0.0, 0.75)
    assert second.event.cause_status_at is not None
    assert read_emotional_event(db, agent, first.event.id) == first.event
    assert read_emotional_event(db, "another-agent", first.event.id) is None


def test_rich_transition_keeps_event_delta_parent_and_mixed_context(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    event = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        source_ref="turn-2",
        signal=EmotionalVector(-0.5, 0.7, 0.4),
        intensity=0.7,
        confidence=0.8,
        cause_summary="риск снова потерять различение",
        cause_status="active",
    ).event
    causes = [
        {
            "event_id": event.id,
            "state": "tension",
            "cause_status": "active",
            "confidence": 0.8,
        },
        {
            "state": "focus",
            "cause_status": "active",
            "confidence": 0.7,
        },
    ]
    first = append_emotional_transition(
        db,
        agent,
        EmotionalVector(-0.1, 0.2, 0.15),
        event_id=event.id,
        source="affect_observer",
        confidence=0.8,
        intensity=0.7,
        causal_context=causes,
        computation_version="affect-v1",
    )
    second = append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.02, -0.01, 0.03),
        source="turn_recompute",
    )
    db.commit()

    assert first.parent_state_id is None
    assert first.event_id == event.id
    assert first.delta == EmotionalVector(-0.1, 0.2, 0.15)
    assert first.computation_version == "affect-v1"
    assert len(first.causal_context) == 2
    assert second.parent_state_id == first.id
    assert second.causal_context == first.causal_context

    last = read_last_state_record(db, agent)
    assert last == second
    legacy = read_last_state(db, agent)
    assert legacy == (second.vector, second.at)


def test_transition_and_aggregate_confidence_are_separate(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    first = append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.5, 0.0, 0.0),
        confidence=0.9,
    )
    second = append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.5, 0.0, 0.0),
        confidence=0.1,
    )
    db.commit()

    assert first.transition_confidence == pytest.approx(0.9)
    assert first.confidence == pytest.approx(0.9)
    assert second.transition_confidence == pytest.approx(0.1)
    assert second.confidence == pytest.approx(0.5)


def test_normalized_lifecycle_preserves_more_than_eight_active_causes(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    latest = None
    for index in range(10):
        event = append_emotional_event(
            db,
            agent,
            source_kind="turn_observation",
            source_ref=f"turn-{index}",
            cause_status="active",
            confidence=0.8,
            intensity=0.5,
        ).event
        latest = append_emotional_transition(
            db,
            agent,
            EmotionalVector(0.01, 0.0, 0.0),
            event_id=event.id,
            confidence=0.8,
            causal_context=[{
                "evidence_id": event.id,
                "source_ref": f"turn-{index}",
                "status": "active",
                "cause_active": True,
                "confidence": 0.8,
                "intensity": 0.5,
                "weighted_delta": [0.01, 0.0, 0.0],
            }],
        )
    db.commit()

    assert latest is not None
    active = [item for item in latest.causal_context if item["cause_active"]]
    assert len(active) == 10
    assert {item["source_ref"] for item in active} == {
        f"turn-{index}" for index in range(10)
    }
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT cause_event_id) "
            "FROM emotional_cause_status WHERE agent_id=%s",
            (agent,),
        )
        assert cur.fetchone()[0] == 10


def test_resolution_is_append_only_and_removes_db_support(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    cause = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        source_ref="cause",
        cause_status="active",
    ).event
    append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.2, 0.0, 0.0),
        event_id=cause.id,
        causal_context=[{
            "evidence_id": cause.id,
            "source_ref": "cause",
            "status": "active",
            "cause_active": True,
            "weighted_delta": [0.2, 0.0, 0.0],
        }],
    )
    resolution = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        source_ref="resolution",
        cause_status="resolved",
    ).event
    state = append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.0, 0.0, 0.0),
        event_id=resolution.id,
        causal_context=[{
            "evidence_id": cause.id,
            "source_ref": "cause",
            "status": "resolved",
            "cause_active": False,
        }],
    )
    db.commit()

    assert not any(
        item.get("evidence_id") == cause.id and item.get("cause_active")
        for item in state.causal_context
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM emotional_cause_status "
            "WHERE agent_id=%s AND cause_event_id=%s ORDER BY at, id",
            (agent, cause.id),
        )
        assert [row[0] for row in cur.fetchall()] == ["active", "active", "resolved"]
    current = read_cause_lifecycle_statuses(db, agent, {cause.id})
    assert current[cause.id]["status"] == "resolved"
    assert current[cause.id]["active"] is False


def test_inherited_projection_does_not_renew_active_lease(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    event = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        source_ref="cause",
        cause_status="active",
    ).event
    first = append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.1, 0.0, 0.0),
        event_id=event.id,
        causal_context=[{
            "evidence_id": event.id,
            "source_ref": "cause",
            "status": "active",
            "cause_active": True,
            "weighted_delta": [0.1, 0.0, 0.0],
        }],
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*), max(lease_expires_at) "
            "FROM emotional_cause_status WHERE agent_id=%s AND cause_event_id=%s",
            (agent, event.id),
        )
        before = cur.fetchone()

    append_emotional_transition(
        db,
        agent,
        EmotionalVector(0.0, 0.0, 0.0),
        causal_context=list(first.causal_context),
    )
    db.commit()
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*), max(lease_expires_at) "
            "FROM emotional_cause_status WHERE agent_id=%s AND cause_event_id=%s",
            (agent, event.id),
        )
        assert cur.fetchone() == before


def test_advisory_lock_serializes_cross_connection_transitions(
    migrated_db: str,
) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    started = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []

    with psycopg.connect(migrated_db) as first_conn:
        # Hold the exact namespace used by state.py. The worker transition
        # must wait, then observe the transition committed by first_conn.
        with first_conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"styx:emotional_state:{agent}",),
            )

        def _worker() -> None:
            try:
                with psycopg.connect(migrated_db) as second_conn:
                    started.set()
                    append_emotional_transition(
                        second_conn,
                        agent,
                        EmotionalVector(0.3, 0.0, 0.0),
                        source="worker",
                    )
                    second_conn.commit()
            except BaseException as exc:  # pragma: no cover - thread relay
                failures.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        assert started.wait(timeout=2.0)
        assert not finished.wait(timeout=0.1), "second transition bypassed advisory lock"

        append_emotional_transition(
            first_conn,
            agent,
            EmotionalVector(0.2, 0.0, 0.0),
            source="daemon",
        )
        first_conn.commit()
        assert finished.wait(timeout=5.0)
        thread.join(timeout=1.0)

    assert not failures
    with psycopg.connect(migrated_db) as check_conn:
        last = read_last_state_record(check_conn, agent)
        assert last is not None
        assert math.isclose(last.vector.valence, 0.5, abs_tol=1e-6)
        with check_conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), count(parent_state_id) "
                "FROM emotional_state WHERE agent_id=%s",
                (agent,),
            )
            assert cur.fetchone() == (2, 1)
            cur.execute("DELETE FROM emotional_state WHERE agent_id=%s", (agent,))
        check_conn.commit()


# ── Decay ──────────────────────────────────────────────────────────────


def test_apply_instant_decay_no_history_noop(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    out = apply_instant_decay(db, agent)
    assert out.decayed is False
    assert out.point is None


def test_apply_instant_decay_recent_point_noop(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    append_emotional_state(db, agent, EmotionalVector(0.5, 0, 0))
    db.commit()
    out = apply_instant_decay(db, agent)
    # at = now() в БД, прошло < 1 минуты → no decay.
    assert out.decayed is False


def test_apply_instant_decay_below_epsilon_noop(db) -> None:
    """В пределах epsilon — decay не пишется (журнал не раздувается)."""
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    # Прямой INSERT с малым значением и старым at.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state (agent_id, at, valence, arousal, dominance) "
            "VALUES (%s, now() - interval '10 minutes', %s, %s, %s)",
            (agent, INSTANT_DECAY_EPSILON / 2, 0.0, 0.0),
        )
    db.commit()
    out = apply_instant_decay(db, agent)
    assert out.decayed is False


def test_apply_instant_decay_writes_decayed_point(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    # Точка 10 минут назад, valence=0.5.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state (agent_id, at, valence, arousal, dominance, source) "
            "VALUES (%s, now() - interval '10 minutes', 0.5, 0.0, 0.0, 'hot_sentiment')",
            (agent,),
        )
    db.commit()

    out = apply_instant_decay(db, agent)
    db.commit()
    assert out.decayed is True
    assert out.point is not None
    # 0.5 * 0.95^10 ≈ 0.299
    expected = 0.5 * (INSTANT_DECAY_PER_MINUTE ** 10)
    assert math.isclose(out.point.valence, expected, rel_tol=0.01)

    # В БД должно быть две точки.
    with db.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT count(*)::int AS n, "
            "       count(*) FILTER (WHERE source = 'decay')::int AS decay_n, "
            "       max(parent_state_id) FILTER (WHERE source = 'decay') AS parent_id, "
            "       max(delta_valence) FILTER (WHERE source = 'decay') AS decay_delta "
            "  FROM emotional_state WHERE agent_id = %s",
            (agent,),
        )
        row = cur.fetchone()
    assert row["n"] == 2
    assert row["decay_n"] == 1
    assert row["parent_id"] is not None
    assert math.isclose(float(row["decay_delta"]), expected - 0.5, rel_tol=0.01)


def test_active_cause_is_supported_while_residual_decays(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, at, valence, arousal, dominance, intensity, "
            " causal_context, computation_version) VALUES "
            "(%s, now() - interval '10 minutes', 0.6, 0.0, 0.0, 0.8, %s, "
            " 'causal-turn-v1')",
            (
                agent,
                psycopg.types.json.Jsonb([
                    {
                        "cause": "ongoing",
                        "cause_active": True,
                        "intensity": 0.8,
                        "weighted_delta": [0.2, 0.0, 0.0],
                    }
                ]),
            ),
        )
    db.commit()

    out = apply_instant_decay(db, agent)
    db.commit()
    assert out.decayed is True
    assert out.point is not None
    expected = 0.2 + 0.4 * (INSTANT_DECAY_PER_MINUTE ** 10)
    assert out.point.valence == pytest.approx(expected, rel=0.02)
    assert out.point.valence > 0.2


def test_fully_supported_state_does_not_emit_technical_decay_rows(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, at, valence, arousal, dominance, causal_context) "
            "VALUES (%s, now() - interval '10 minutes', 0.2, 0.0, 0.0, %s)",
            (
                agent,
                psycopg.types.json.Jsonb([
                    {"cause_active": True, "weighted_delta": [0.2, 0.0, 0.0]}
                ]),
            ),
        )
    db.commit()
    out = apply_instant_decay(db, agent)
    assert out.decayed is False
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM emotional_state WHERE agent_id = %s", (agent,)
        )
        assert cur.fetchone()[0] == 1


def test_expired_lease_is_materialized_and_no_longer_supports_state(db) -> None:
    agent = f"state-test-{uuid.uuid4().hex[:6]}"
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    event = append_emotional_event(
        db,
        agent,
        source_kind="turn_observation",
        cause_status="unknown",
    ).event
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO emotional_cause_status "
            "(agent_id, cause_event_id, at, status, lease_expires_at, "
            " support_valence, support_arousal, support_dominance, context) "
            "VALUES (%s, %s, %s, 'active', %s, 0.2, 0, 0, %s)",
            (
                agent,
                event.id,
                now - _dt.timedelta(minutes=20),
                now - _dt.timedelta(minutes=5),
                psycopg.types.json.Jsonb({
                    "evidence_id": event.id,
                    "cause_active": True,
                    "weighted_delta": [0.2, 0.0, 0.0],
                }),
            ),
        )
        cur.execute(
            "INSERT INTO emotional_state "
            "(agent_id, at, valence, arousal, dominance, causal_context) "
            "VALUES (%s, %s, 0.2, 0, 0, %s)",
            (
                agent,
                now - _dt.timedelta(minutes=20),
                psycopg.types.json.Jsonb([{
                    "evidence_id": event.id,
                    "cause_active": True,
                    "weighted_delta": [0.9, 0.0, 0.0],
                }]),
            ),
        )
    db.commit()

    out = apply_instant_decay(db, agent, now=now)
    db.commit()
    assert out.decayed is True
    assert out.point is not None
    assert out.point.valence == pytest.approx(
        # Cause support is protected for the first 15 minutes, then decays
        # only across the five-minute interval after lease expiry.
        0.2 * (INSTANT_DECAY_PER_MINUTE ** 5), rel=0.02
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM emotional_cause_status "
            "WHERE agent_id=%s AND cause_event_id=%s ORDER BY at DESC, id DESC LIMIT 1",
            (agent, event.id),
        )
        assert cur.fetchone()[0] == "expired"


# ── list_active_agent_ids ─────────────────────────────────────────────


def test_list_active_agent_ids_returns_distinct(db) -> None:
    agent_a = f"state-test-A-{uuid.uuid4().hex[:6]}"
    agent_b = f"state-test-B-{uuid.uuid4().hex[:6]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO memories (agent_id, role, content) VALUES "
            "(%s, 'user', 'x'), (%s, 'user', 'y'), (%s, 'assistant', 'z')",
            (agent_a, agent_a, agent_b),
        )
    db.commit()
    ids = list_active_agent_ids(db)
    assert agent_a in ids
    assert agent_b in ids
