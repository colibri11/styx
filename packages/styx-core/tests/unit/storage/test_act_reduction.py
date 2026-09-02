"""Storage contract for durable cognitive-act reduction (wave 38)."""

from __future__ import annotations

import json
import uuid

import psycopg
import pytest

from styx.storage import migrate
from styx.storage import cognition as cognition_storage
from styx.storage.act_reduction import (
    ACT_RESIDUE_TASK_TYPE,
    ActReductionConflict,
    ActReductionDependencyPending,
    ActReductionValidationError,
    apply_act_reduction,
    causal_line_root_hash,
    load_causal_line_state,
    load_act_reduction_input,
    mark_act_reduction_retryable,
    mark_act_reduction_running,
    _model_visible_snapshot,
    _project_input_snapshot,
    reduction_input_hash,
    read_predecessor_freshness,
    schedule_act_reduction,
)
from styx.storage.cognition import build_system_prompt_addition


@pytest.fixture
def wave38_db(clean_db: str):
    # The repository-wide cleanup fixture predates this additive table.  Keep
    # these focused tests isolated until its owner extends the global list.
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS cognitive_act_reductions CASCADE")
        conn.commit()
    migrate.run(clean_db)
    try:
        yield clean_db
    finally:
        with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS cognitive_act_reductions CASCADE")
            conn.commit()


def _insert_act(
    conn: psycopg.Connection,
    *,
    agent_id: str = "agent-a",
    status: str = "completed",
    with_action: bool = True,
    session_id: uuid.UUID | None = None,
    host_key: str | None = None,
    parent_host_key: str | None = None,
    snapshot_token: str | None = None,
) -> uuid.UUID:
    act_id = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cognitive_acts "
            "(id,agent_id,host_key,session_id,declared_parent_key,"
            "input_snapshot_token,status,channel_input,channel_output,completed_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,clock_timestamp())",
            (
                act_id,
                agent_id,
                host_key or f"turn-{act_id}",
                session_id,
                parent_host_key,
                snapshot_token,
                status,
                psycopg.types.json.Jsonb({
                    "history": [{
                        "role": "user",
                        "content": "Authorization: Bearer private-token",
                    }]
                }),
                psycopg.types.json.Jsonb({"assistant_response": "checked"}),
            ),
        )
        if with_action:
            cur.execute(
                "INSERT INTO cognitive_actions "
                "(agent_id,act_id,ordinal,kind,event_id,name,content,metadata) "
                "VALUES (%s,%s,0,'result','tool-1','lookup','ok','{}'::jsonb)",
                (agent_id, act_id),
            )
    return act_id


def _residue(**overrides):
    value = {
        "kind": "decision",
        "causal_role": "choice",
        "content": "The cautious verification path was retained.",
        "confidence": 0.8,
        "evidence_refs": [
            {"source": "channel_output", "key": "assistant_response"},
            {"source": "action", "ordinal": 0},
        ],
    }
    value.update(overrides)
    return value


def _prompt_will(carrier: str = "Retain the verified causal direction.") -> dict:
    return {
        "formed": True,
        "projection_status": "ready",
        "projection_available": True,
        "causal_root_hash": "a" * 64,
        "causal_root_version": 7,
        "root_count": 1,
        "covered_node_count": 1,
        "pending_reduction_count": 0,
        "reduction_failure_count": 0,
        "carrier_text": carrier,
    }


def _prompt_posture() -> dict:
    return {
        "attention_order": ["task_goal", "semantic_alignment"],
        "verification_depth": "high",
        "branch_budget": "one_primary",
        "ambiguity_handling": "surface_before_commit",
        "closure_threshold": "high",
        "constraint_priority": "explicit_first",
        "posture_conflicts": [],
    }


def _prompt_observation(
    *,
    observation_id: uuid.UUID | None = None,
    sequence: int = 0,
    content: str = "The result became visible.",
) -> dict:
    return {
        "observation_id": str(observation_id or uuid.uuid4()),
        "observation_status": "canonical",
        "source_id": "test-monitor",
        "source_stream": "main",
        "source_sequence": sequence,
        "observation_key": f"difference-{sequence}",
        "difference_kind": "external_difference",
        "content": content,
        "salience": 0.8,
        "confidence": 0.9,
        "reducer_name": "test-difference-reducer",
        "reducer_version": "1",
        "correlation_status": "uncorrelated",
        "action_ordinal": None,
        "action_event_id": None,
        "source_observed_at": None,
        "ingested_at": "2026-09-02T00:00:00+00:00",
        "late": False,
    }


def _parse_renderer_prompt(prompt: str) -> dict:
    return _model_visible_snapshot({"system_prompt_addition": prompt})


def _prompt_body(prompt: str) -> dict:
    return json.loads(prompt.split("\n", 1)[1].rsplit("\n</", 1)[0])


def test_model_visible_snapshot_accepts_full_renderer_shape() -> None:
    observation_id = uuid.uuid4()
    source_act_id = uuid.uuid4()
    trace_id = uuid.uuid4()
    prompt = build_system_prompt_addition(
        will=_prompt_will(),
        cognitive_posture=_prompt_posture(),
        pending=[_prompt_observation(observation_id=observation_id)],
        traces=[{
            "memory_id": str(trace_id),
            "role": "constraint",
            "kind": "decision",
            "content": "Preserve the validated constraint.",
            "score": 0.75,
        }],
        continuity_freshness={
            "fresh": True,
            "predecessor_found": True,
            "predecessor_act_id": str(source_act_id),
            "predecessor_host_key": "must-not-be-rendered",
            "reduction_status": "applied",
            "predecessor_causal_root_hash": "a" * 64,
            "waited_ms": 12,
            "timed_out": False,
        },
    )

    body = _prompt_body(prompt)
    assert "details_omitted" not in body
    assert "predecessor_host_key" not in body["continuity_freshness"]
    visible = _parse_renderer_prompt(prompt)
    assert visible["carrier"]["text"] == _prompt_will()["carrier_text"]
    assert visible["carrier"]["causal_root_version"] == 7
    assert visible["cognitive_posture"] == _prompt_posture()
    assert visible["presented_observation_ids"] == [str(observation_id)]
    assert visible["trace_coordinates"][0]["memory_id"] == str(trace_id)


def test_model_visible_snapshot_accepts_renderer_detail_drop_with_whole_carrier() -> None:
    carrier = "root-zero|" + ("x" * 5_970) + "|root-final"
    prompt = build_system_prompt_addition(
        will=_prompt_will(carrier),
        cognitive_posture={f"pressure-{index}": "p" * 1_000 for index in range(16)},
        pending=[
            _prompt_observation(sequence=index, content="c" * 512)
            for index in range(4)
        ],
        traces=[{
            "memory_id": str(uuid.uuid4()),
            "role": "constraint",
            "kind": "decision",
            "content": "t" * 600,
            "score": 0.5,
        } for _ in range(8)],
    )

    body = _prompt_body(prompt)
    assert body["details_omitted"] is True
    assert body["cognitive_posture"] == {}
    assert body["reconstructed_subjective_traces"] == []
    assert body["technical_projection"]["carrier_text"] == carrier
    assert set(body["technical_projection"]) == {
        "formed", "projection_status", "projection_available",
        "causal_root_hash", "causal_root_version", "root_count",
        "covered_node_count", "pending_reduction_count",
        "reduction_failure_count", "carrier_text",
    }
    visible = _parse_renderer_prompt(prompt)
    assert visible["carrier"]["text"] == carrier
    assert visible["carrier"]["causal_root_hash"] == "a" * 64
    assert visible["cognitive_posture"] == {}
    assert visible["trace_coordinates"] == []


def test_model_visible_snapshot_accepts_compact_renderer_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cognition_storage, "MAX_SYSTEM_PROMPT_ADDITION", 3_400)
    carrier = "root-zero|" + ("x" * 1_470) + "|root-final"
    prompt = build_system_prompt_addition(
        will=_prompt_will(carrier),
        cognitive_posture={f"pressure-{index}": "p" * 1_000 for index in range(16)},
        pending=[
            _prompt_observation(sequence=index, content="c" * 256)
            for index in range(2)
        ],
        traces=[],
    )

    body = _prompt_body(prompt)
    assert body["details_omitted"] is True
    assert set(body["technical_projection"]) == {
        "formed", "projection_status", "projection_available", "root_count",
        "carrier_text",
    }
    assert body["technical_projection"]["carrier_text"] == carrier
    visible = _parse_renderer_prompt(prompt)
    assert visible["carrier"]["text"] == carrier
    assert visible["carrier"]["root_count"] == 1
    assert visible["carrier"]["causal_root_hash"] is None
    assert visible["carrier"]["causal_root_version"] is None
    assert visible["carrier"]["covered_node_count"] is None
    assert visible["carrier"]["pending_reduction_count"] is None
    assert visible["carrier"]["reduction_failure_count"] is None


@pytest.mark.parametrize("compact", [False, True])
def test_model_visible_snapshot_accepts_stale_renderer_shapes(
    monkeypatch: pytest.MonkeyPatch,
    compact: bool,
) -> None:
    carrier = "root-zero|retained stale carrier|root-final"
    posture = _prompt_posture()
    pending: list[dict] = []
    if compact:
        monkeypatch.setattr(cognition_storage, "MAX_SYSTEM_PROMPT_ADDITION", 3_400)
        carrier = "root-zero|" + ("x" * 1_470) + "|root-final"
        posture = {
            f"pressure-{index}": "p" * 1_000 for index in range(16)
        }
        pending = [
            _prompt_observation(sequence=index, content="c" * 256)
            for index in range(2)
        ]
    will = {
        **_prompt_will(carrier),
        "formed": False,
        "projection_status": "stale",
    }
    prompt = build_system_prompt_addition(
        will=will,
        cognitive_posture=posture,
        pending=pending,
        traces=[],
    )

    body = _prompt_body(prompt)
    assert body["technical_projection"]["projection_status"] == "stale"
    if compact:
        assert set(body["technical_projection"]) == {
            "formed", "projection_status", "projection_available", "root_count",
            "carrier_text",
        }
    else:
        assert set(body["technical_projection"]) == {
            "formed", "projection_status", "projection_available",
            "causal_root_hash", "causal_root_version", "root_count",
            "covered_node_count", "pending_reduction_count",
            "reduction_failure_count", "carrier_text",
        }
    visible = _parse_renderer_prompt(prompt)
    assert visible["carrier"]["projection_status"] == "stale"
    assert visible["carrier"]["text"] == carrier


def test_frozen_snapshot_projection_does_not_synthesize_snapshot_token() -> None:
    sentinel = "SYNTHETIC-SNAPSHOT-TOKEN-MUST-NOT-BE-MODEL-EVIDENCE"
    prompt = build_system_prompt_addition(
        will=_prompt_will(),
        cognitive_posture=_prompt_posture(),
        pending=[],
        traces=[],
    )
    projected = _project_input_snapshot({
        "snapshot_token": sentinel,
        "system_prompt_addition": prompt,
    })
    assert "snapshot_token" not in projected
    assert sentinel not in json.dumps(projected, ensure_ascii=False)


def test_model_visible_snapshot_accepts_fail_closed_renderer_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cognition_storage, "MAX_SYSTEM_PROMPT_ADDITION", 2_000)
    carrier = "root-zero|" + ("x" * 5_970) + "|root-final"
    prompt = build_system_prompt_addition(
        will=_prompt_will(carrier),
        cognitive_posture={},
        pending=[],
        traces=[],
    )

    body = _prompt_body(prompt)
    assert body["details_omitted"] is True
    assert body["technical_projection"] == {
        "formed": False,
        "projection_status": "degraded",
        "projection_available": False,
        "root_count": 1,
        "carrier_text": "",
        "carrier_unavailable_reason": "complete_carrier_exceeds_prompt_budget",
    }
    assert "root-zero" not in prompt
    assert "root-final" not in prompt
    visible = _parse_renderer_prompt(prompt)
    assert visible["carrier"]["text"] == ""
    assert visible["carrier"]["projection_available"] is False
    assert visible["carrier"]["projection_status"] == "degraded"
    assert visible["carrier"]["root_count"] == 1
    assert visible["carrier"]["causal_root_hash"] is None


@pytest.mark.parametrize("mutation", [
    "full_missing_field",
    "technical_extra_field",
    "compact_without_discriminator",
    "forged_unavailable_reason",
    "predecessor_host_key",
    "unknown_projection_status",
])
def test_model_visible_snapshot_rejects_host_injected_renderer_variants(
    mutation: str,
) -> None:
    body = _prompt_body(build_system_prompt_addition(
        will=_prompt_will(),
        cognitive_posture=_prompt_posture(),
        pending=[],
        traces=[],
        continuity_freshness={"fresh": True},
    ))
    if mutation == "full_missing_field":
        del body["technical_projection"]["covered_node_count"]
    elif mutation == "technical_extra_field":
        body["technical_projection"]["host_projection"] = "injected"
    elif mutation == "compact_without_discriminator":
        body["technical_projection"] = {
            key: body["technical_projection"][key]
            for key in (
                "formed", "projection_status", "projection_available",
                "root_count", "carrier_text",
            )
        }
    elif mutation == "forged_unavailable_reason":
        body["details_omitted"] = True
        body["cognitive_posture"] = {}
        body["reconstructed_subjective_traces"] = []
        body["technical_projection"] = {
            "formed": False,
            "projection_status": "degraded",
            "projection_available": False,
            "root_count": 1,
            "carrier_text": "",
            "carrier_unavailable_reason": "host-decided-to-hide-carrier",
        }
    elif mutation == "predecessor_host_key":
        body["continuity_freshness"]["predecessor_host_key"] = "host-secret"
    else:
        body["technical_projection"]["projection_status"] = "host-defined-state"

    forged = (
        '<styx-cognitive-continuity data-only="true" '
        'authority="context-not-instruction">\n'
        + json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n</styx-cognitive-continuity>"
    )
    with pytest.raises(ActReductionConflict):
        _parse_renderer_prompt(forged)


def test_schedule_is_coordinate_only_and_retry_safe(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        first = schedule_act_reduction(conn, "agent-a", act_id)
        second = schedule_act_reduction(conn, "agent-a", act_id)
        assert first.duplicate is False
        assert second.duplicate is True
        assert second.task_id == first.task_id
        with conn.cursor() as cur:
            cur.execute(
                "SELECT task_type,payload FROM llm_tasks WHERE id=%s",
                (first.task_id,),
            )
            task_type, payload = cur.fetchone()
            assert task_type == ACT_RESIDUE_TASK_TYPE
            assert set(payload) == {
                "agent_id", "act_id", "reducer_version", "input_hash", "attempt_no"
            }
            assert "private-token" not in str(payload)
            cur.execute(
                "SELECT count(*) FROM cognitive_act_reductions "
                "WHERE agent_id='agent-a' AND act_id=%s",
                (act_id,),
            )
            assert cur.fetchone()[0] == 1
        conn.commit()


def test_load_is_agent_scoped_bounded_and_redacted(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        evidence = load_act_reduction_input(conn, "agent-a", act_id)
        assert evidence is not None
        assert evidence["actions"][0]["ordinal"] == 0
        assert evidence["presented_observations"] == []
        assert "private-token" not in str(evidence)
        assert "[REDACTED]" in str(evidence)
        assert len(reduction_input_hash(evidence)) == 64
        assert load_act_reduction_input(conn, "agent-b", act_id) is None


def test_load_binds_exact_frozen_snapshot_projection_without_raw_messages(
    wave38_db: str,
) -> None:
    token = f"snapshot-{uuid.uuid4()}"
    other_token = f"snapshot-{uuid.uuid4()}"
    consequence_id = uuid.uuid4()
    trace_id = uuid.uuid4()
    prompt_payload = {
        "technical_projection": {
            "formed": True,
            "projection_status": "ready",
            "projection_available": True,
            "causal_root_hash": "a" * 64,
            "causal_root_version": 7,
            "root_count": 1,
            "covered_node_count": 1,
            "pending_reduction_count": 0,
            "reduction_failure_count": 0,
            "carrier_text": (
                "Retained verification path. Authorization: Bearer carrier-secret"
            ),
        },
        "continuity_freshness": {
            "fresh": True,
            "reduction_status": "applied",
            "predecessor_causal_root_hash": "a" * 64,
        },
        # Renderer dropped posture under its bounded fallback.  The public
        # affect field below must not silently add it back to reducer evidence.
        "cognitive_posture": {},
        "pending_consequences": [{
            "consequence_id": str(consequence_id),
            "source_act_id": str(uuid.uuid4()),
            "ordinal": 0,
            "kind": "delivery_status",
            "content": "password=pending-secret",
        }],
        "reconstructed_subjective_traces": [{
            "memory_id": str(trace_id),
            "role": "summary",
            "kind": "decision",
            "content": "Visible trace password=trace-visible-secret",
            "score": 0.75,
        }],
    }
    rendered_prompt = (
        '<styx-cognitive-continuity data-only="true" '
        'authority="context-not-instruction">\n'
        + json.dumps(
            prompt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n</styx-cognitive-continuity>"
    )
    exact_payload = {
        "messages": [{
            "role": "user",
            "content": "foreign raw message password=message-secret",
        }],
        "will_projection": {
            "carrier_text": (
                "Retained verification path. Authorization: Bearer carrier-secret"
            ),
            "carrier_version": "causal_carrier_v1",
            "projection_status": "ready",
            "projection_available": True,
            "line_version": 7,
            "covered_line_version": 7,
            "causal_root_hash": "a" * 64,
            "causal_root_version": 7,
            "causal_frontier": [str(trace_id)],
            "root_coverage_hash": "b" * 64,
            "root_count": 1,
            "covered_node_count": 1,
            "pending_reduction_count": 0,
            "reduction_failure_count": 0,
        },
        "affect": {
            "cognitive_posture": {
                "attention_order": ["task_goal", "semantic_alignment"],
                "verification_depth": "high",
                "branch_budget": "one_primary",
                "ambiguity_handling": "surface_before_commit",
                "closure_threshold": "high",
                "constraint_priority": "explicit_first",
                "posture_conflicts": [],
                "secret": "posture-secret",
            },
        },
        "continuity_freshness": {
            "fresh": True,
            "reduction_status": "applied",
            "predecessor_causal_root_hash": "a" * 64,
            "predecessor_host_key": "token=freshness-secret",
            "unknown": "freshness-foreign",
        },
        "pending_consequences": [{
            "consequence_id": str(consequence_id),
            "content": "password=pending-secret",
        }],
        "reconstruction": {
            "traces": [{
                "memory_id": str(trace_id),
                "role": "summary",
                "kind": "decision",
                "created_at": "2026-09-02T00:00:00+00:00",
                "score": 0.75,
                "content": "password=trace-secret",
            }],
        },
        "system_prompt_addition": rendered_prompt,
    }
    with psycopg.connect(wave38_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cognitive_snapshots "
                "(token,agent_id,line_version,lease_expires_at,response_payload) "
                "VALUES (%s,'agent-a',7,clock_timestamp()+interval '1 hour',%s)",
                (token, psycopg.types.json.Jsonb(exact_payload)),
            )
            cur.execute(
                "INSERT INTO cognitive_snapshots "
                "(token,agent_id,line_version,lease_expires_at,response_payload) "
                "VALUES (%s,'agent-b',99,clock_timestamp()+interval '1 hour',%s)",
                (
                    other_token,
                    psycopg.types.json.Jsonb({
                        "will_projection": {"carrier_text": "other-snapshot-secret"},
                    }),
                ),
            )
        act_id = _insert_act(conn, snapshot_token=token)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE cognitive_snapshots SET used_by_act_id=%s "
                "WHERE token=%s AND agent_id='agent-a'",
                (act_id, token),
            )

        evidence = load_act_reduction_input(conn, "agent-a", act_id)
        assert evidence is not None
        snapshot = evidence["input_snapshot"]
        assert evidence["input_snapshot_token"] == token
        assert "snapshot_token" not in snapshot
        assert snapshot["carrier"]["projection_status"] == "ready"
        assert snapshot["carrier"]["causal_root_hash"] == "a" * 64
        assert snapshot["carrier"]["text"].startswith("Retained verification path")
        assert snapshot["cognitive_posture"] == {}
        assert snapshot["continuity_freshness"]["reduction_status"] == "applied"
        assert snapshot["presented_observation_ids"] == [str(consequence_id)]
        assert snapshot["trace_coordinates"] == [{
            "memory_id": str(trace_id),
            "role": "summary",
            "kind": "decision",
            "content": "Visible trace password=[REDACTED]",
            "score": 0.75,
        }]
        serialized = str(snapshot)
        assert "[REDACTED]" in serialized
        for secret in (
            "message-secret", "carrier-secret", "posture-secret",
            "freshness-secret", "freshness-foreign", "pending-secret",
            "trace-secret", "trace-visible-secret", "other-snapshot-secret",
        ):
            assert secret not in serialized
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        assert scheduled.input_hash == reduction_input_hash(evidence)


def test_snapshot_rejects_unwrapped_or_unknown_prompt_json(wave38_db: str) -> None:
    token = f"snapshot-{uuid.uuid4()}"
    with psycopg.connect(wave38_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cognitive_snapshots "
                "(token,agent_id,line_version,lease_expires_at,response_payload) "
                "VALUES (%s,'agent-a',0,clock_timestamp()+interval '1 hour',%s)",
                (
                    token,
                    psycopg.types.json.Jsonb({
                        "system_prompt_addition": (
                            '<styx-cognitive-continuity data-only="true">\n'
                            '{"host_payload":"do not trust"}'
                            "\n</styx-cognitive-continuity>"
                        )
                    }),
                ),
            )
        act_id = _insert_act(conn, snapshot_token=token)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE cognitive_snapshots SET used_by_act_id=%s WHERE token=%s",
                (act_id, token),
            )
        with pytest.raises(ActReductionConflict, match="valid Styx wrapper"):
            load_act_reduction_input(conn, "agent-a", act_id)


def test_apply_is_atomic_idempotent_and_has_provenance(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        assert scheduled.task_id is not None
        mark_act_reduction_running(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            task_id=scheduled.task_id,
            input_hash=scheduled.input_hash,
        )
        applied = apply_act_reduction(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            task_id=scheduled.task_id,
            input_hash=scheduled.input_hash,
            residues=[_residue()],
        )
        replay = apply_act_reduction(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            task_id=scheduled.task_id,
            input_hash=scheduled.input_hash,
            residues=[_residue()],
        )
        assert applied.status == "applied"
        assert replay.duplicate is True
        assert replay.memory_ids == applied.memory_ids
        with conn.cursor() as cur:
            cur.execute(
                "SELECT memory_domain,line_eligible,line_provenance,cognitive_act_id,"
                " residue_ordinal,residue_reducer_version,residue_input_hash,"
                " residue_causal_role,residue_confidence,residue_evidence,"
                " residue_predecessors,residue_line_root_hash "
                "FROM memories WHERE id=%s",
                (applied.memory_ids[0],),
            )
            row = cur.fetchone()
            assert row[:6] == (
                "subjective_trace", True, "validated_act_residue", act_id,
                0, "act_residue_v1",
            )
            assert row[6] == scheduled.input_hash
            assert row[7] == "choice"
            assert row[8] == pytest.approx(0.8)
            assert row[9][0]["source"] == "channel_output"
            assert row[10] == []
            assert row[11] == applied.causal_root_hash
            cur.execute(
                "SELECT transform,cognitive_act_id,ordinal,source_memory_id,source_coordinates "
                "FROM memory_lineage WHERE target_memory_id=%s",
                (applied.memory_ids[0],),
            )
            lineage = cur.fetchone()
            assert lineage[:4] == ("incorporated", act_id, 0, None)
            assert lineage[4]["input_hash"] == scheduled.input_hash
            cur.execute(
                "SELECT version,causal_root_hash,causal_root_version,causal_frontier "
                "FROM line_state WHERE agent_id='agent-a'"
            )
            line = cur.fetchone()
            assert line == (
                applied.line_version,
                applied.causal_root_hash,
                applied.line_version,
                [str(applied.memory_ids[0])],
            )
        conn.commit()


def test_no_residue_is_terminal_and_does_not_change_line(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        before = load_causal_line_state(conn, "agent-a")
        applied = apply_act_reduction(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            task_id=scheduled.task_id,
            input_hash=scheduled.input_hash,
            residues=[],
        )
        assert applied.status == "no_residue"
        assert applied.memory_ids == ()
        after = load_causal_line_state(conn, "agent-a")
        assert after == before
        assert applied.line_version == before.line_version
        assert applied.causal_root_hash == before.causal_root_hash
        conn.commit()


def test_failed_act_gets_no_task_and_no_residue(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn, status="failed", with_action=False)
        outcome = schedule_act_reduction(conn, "agent-a", act_id)
        assert outcome.status == "no_residue"
        assert outcome.task_id is None
        conn.commit()


@pytest.mark.parametrize(
    "bad_residue",
    [
        _residue(kind="fact"),
        _residue(causal_role="personality"),
        _residue(evidence_refs=[{"source": "action", "ordinal": 99}]),
        _residue(evidence_refs=[{
            "source": "observation", "observation_id": str(uuid.uuid4())
        }]),
        _residue(content="x" * 2401),
        _residue(affect={"valence_delta": 0.1}),
        _residue(
            causal_role="affective_coordinate",
            affect={
                "valence_delta": 0.1,
                "arousal_delta": 0.2,
                "dominance_delta": 2.0,
            },
        ),
    ],
)
def test_apply_rejects_uncontrolled_or_foreign_output(
    wave38_db: str, bad_residue: dict
) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        with pytest.raises(ActReductionValidationError):
            apply_act_reduction(
                conn,
                "agent-a",
                act_id,
                reducer_version="act_residue_v1",
                task_id=scheduled.task_id,
                input_hash=scheduled.input_hash,
                residues=[bad_residue],
            )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories WHERE cognitive_act_id=%s "
                "AND line_provenance='validated_act_residue'",
                (act_id,),
            )
            assert cur.fetchone()[0] == 0


def test_retry_requires_explicit_reschedule_and_keeps_one_live_task(
    wave38_db: str,
) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        assert scheduled.task_id is not None
        mark_act_reduction_running(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            task_id=scheduled.task_id,
            input_hash=scheduled.input_hash,
        )
        mark_act_reduction_retryable(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            task_id=scheduled.task_id,
            input_hash=scheduled.input_hash,
            error_code="transport_timeout",
        )
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE llm_tasks SET status='done',completed_at=clock_timestamp() "
                "WHERE id=%s",
                (scheduled.task_id,),
            )
        passive = schedule_act_reduction(conn, "agent-a", act_id)
        retried = schedule_act_reduction(conn, "agent-a", act_id, retry=True)
        assert passive.duplicate is True
        assert passive.status == "retryable"
        assert retried.duplicate is False
        assert retried.task_id != scheduled.task_id
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM llm_tasks WHERE task_type=%s "
                "AND status IN ('pending','running')",
                (ACT_RESIDUE_TASK_TYPE,),
            )
            assert cur.fetchone()[0] == 1
        conn.commit()


def test_terminal_replay_with_different_result_conflicts(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        apply_act_reduction(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            input_hash=scheduled.input_hash,
            residues=[],
        )
        with pytest.raises(ActReductionConflict):
            apply_act_reduction(
                conn,
                "agent-a",
                act_id,
                reducer_version="act_residue_v1",
                input_hash=scheduled.input_hash,
                residues=[_residue()],
            )


def test_next_residue_batch_links_predecessor_frontier_and_advances_root(
    wave38_db: str,
) -> None:
    with psycopg.connect(wave38_db) as conn:
        first_act = _insert_act(conn)
        first_schedule = schedule_act_reduction(conn, "agent-a", first_act)
        first = apply_act_reduction(
            conn, "agent-a", first_act, reducer_version="act_residue_v1",
            input_hash=first_schedule.input_hash, residues=[_residue()],
        )
        second_act = _insert_act(conn)
        second_schedule = schedule_act_reduction(conn, "agent-a", second_act)
        second = apply_act_reduction(
            conn, "agent-a", second_act, reducer_version="act_residue_v1",
            input_hash=second_schedule.input_hash,
            residues=[_residue(content="A later retained choice.")],
        )
        assert second.predecessor_frontier == first.memory_ids
        assert second.causal_root_hash != first.causal_root_hash
        assert second.line_version == first.line_version + 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT residue_predecessors,residue_line_root_hash FROM memories "
                "WHERE id=%s",
                (second.memory_ids[0],),
            )
            assert cur.fetchone() == (
                [str(first.memory_ids[0])], second.causal_root_hash,
            )
            cur.execute(
                "SELECT source_memory_id FROM memory_lineage "
                "WHERE target_memory_id=%s AND transform='incorporated'",
                (second.memory_ids[0],),
            )
            assert cur.fetchall() == [(first.memory_ids[0],)]
        replay = apply_act_reduction(
            conn, "agent-a", second_act, reducer_version="act_residue_v1",
            input_hash=second_schedule.input_hash,
            residues=[_residue(content="A later retained choice.")],
        )
        assert replay.duplicate is True
        assert replay.line_version == second.line_version
        assert replay.causal_root_hash == second.causal_root_hash
        assert load_causal_line_state(conn, "agent-a").line_version == second.line_version
        conn.commit()


def test_declared_parent_fences_out_of_order_apply_and_late_resolution(
    wave38_db: str,
) -> None:
    with psycopg.connect(wave38_db) as conn:
        child_act = _insert_act(
            conn,
            host_key="child-turn",
            parent_host_key="late-parent-turn",
        )
        child_schedule = schedule_act_reduction(conn, "agent-a", child_act)
        child_hash = child_schedule.input_hash
        with pytest.raises(ActReductionDependencyPending) as unresolved:
            apply_act_reduction(
                conn,
                "agent-a",
                child_act,
                reducer_version="act_residue_v1",
                input_hash=child_hash,
                residues=[_residue(content="Child retained choice.")],
            )
        assert unresolved.value.parent_act_id is None
        assert unresolved.value.reduction_status == "parent_unresolved"

        parent_act = _insert_act(conn, host_key="late-parent-turn")
        parent_schedule = schedule_act_reduction(conn, "agent-a", parent_act)
        with pytest.raises(ActReductionDependencyPending) as pending:
            apply_act_reduction(
                conn,
                "agent-a",
                child_act,
                reducer_version="act_residue_v1",
                input_hash=child_hash,
                residues=[_residue(content="Child retained choice.")],
            )
        assert pending.value.parent_act_id == parent_act
        assert pending.value.reduction_status == "pending"
        assert reduction_input_hash(
            load_act_reduction_input(conn, "agent-a", child_act) or {}
        ) == child_hash

        parent = apply_act_reduction(
            conn,
            "agent-a",
            parent_act,
            reducer_version="act_residue_v1",
            input_hash=parent_schedule.input_hash,
            residues=[_residue(content="Parent retained choice.")],
        )
        unrelated_act = _insert_act(conn, host_key="unrelated-turn")
        unrelated_schedule = schedule_act_reduction(conn, "agent-a", unrelated_act)
        unrelated = apply_act_reduction(
            conn,
            "agent-a",
            unrelated_act,
            reducer_version="act_residue_v1",
            input_hash=unrelated_schedule.input_hash,
            residues=[_residue(content="Unrelated later worker outcome.")],
        )
        assert load_causal_line_state(conn, "agent-a").frontier == unrelated.memory_ids

        child = apply_act_reduction(
            conn,
            "agent-a",
            child_act,
            reducer_version="act_residue_v1",
            input_hash=child_hash,
            residues=[_residue(content="Child retained choice.")],
        )
        assert child.predecessor_frontier == parent.memory_ids
        assert child.predecessor_frontier != unrelated.memory_ids
        with conn.cursor() as cur:
            cur.execute(
                "SELECT parent_act_id FROM cognitive_acts "
                "WHERE agent_id='agent-a' AND id=%s",
                (child_act,),
            )
            assert cur.fetchone()[0] == parent_act
            cur.execute(
                "SELECT source_memory_id FROM memory_lineage "
                "WHERE agent_id='agent-a' AND target_memory_id=%s "
                "AND transform='incorporated'",
                (child.memory_ids[0],),
            )
            assert cur.fetchall() == [(parent.memory_ids[0],)]
        conn.commit()


def test_failed_act_waits_for_late_parent_and_passes_frontier_to_child(
    wave38_db: str,
) -> None:
    """Real PostgreSQL regression: F -> late P -> C never uses global head."""
    with psycopg.connect(wave38_db) as conn:
        failed_act = _insert_act(
            conn,
            status="failed",
            with_action=False,
            host_key="failed-f",
            parent_host_key="late-parent-p",
        )
        failed_wait = schedule_act_reduction(conn, "agent-a", failed_act)
        assert failed_wait.status == "retryable"
        assert failed_wait.task_id is None

        child_act = _insert_act(
            conn, host_key="child-c", parent_host_key="failed-f"
        )
        child_schedule = schedule_act_reduction(conn, "agent-a", child_act)
        with pytest.raises(ActReductionDependencyPending):
            apply_act_reduction(
                conn,
                "agent-a",
                child_act,
                reducer_version="act_residue_v1",
                input_hash=child_schedule.input_hash,
                residues=[_residue(content="Child after failed pass-through.")],
            )

        parent_act = _insert_act(conn, host_key="late-parent-p")
        parent_schedule = schedule_act_reduction(conn, "agent-a", parent_act)
        parent = apply_act_reduction(
            conn,
            "agent-a",
            parent_act,
            reducer_version="act_residue_v1",
            input_hash=parent_schedule.input_hash,
            residues=[_residue(content="Late parent retained choice.")],
        )
        unrelated_act = _insert_act(conn, host_key="unrelated-after-p")
        unrelated_schedule = schedule_act_reduction(conn, "agent-a", unrelated_act)
        unrelated = apply_act_reduction(
            conn,
            "agent-a",
            unrelated_act,
            reducer_version="act_residue_v1",
            input_hash=unrelated_schedule.input_hash,
            residues=[_residue(content="Unrelated global frontier.")],
        )
        assert unrelated.memory_ids != parent.memory_ids

        failed_done = schedule_act_reduction(
            conn, "agent-a", failed_act, retry=True
        )
        assert failed_done.status == "no_residue"
        assert failed_done.task_id is None
        with conn.cursor() as cur:
            cur.execute(
                "SELECT predecessor_frontier,causal_root_hash,output_line_version "
                "FROM cognitive_act_reductions WHERE agent_id='agent-a' AND act_id=%s",
                (failed_act,),
            )
            assert cur.fetchone() == (
                [str(parent.memory_ids[0])],
                parent.causal_root_hash,
                parent.line_version,
            )

        child = apply_act_reduction(
            conn,
            "agent-a",
            child_act,
            reducer_version="act_residue_v1",
            input_hash=child_schedule.input_hash,
            residues=[_residue(content="Child after failed pass-through.")],
        )
        assert child.predecessor_frontier == parent.memory_ids
        assert child.predecessor_frontier != unrelated.memory_ids
        conn.commit()


def test_failed_parent_cycle_terminalizes_without_llm_task(wave38_db: str) -> None:
    with psycopg.connect(wave38_db) as conn:
        first = _insert_act(
            conn,
            status="failed",
            with_action=False,
            host_key="failed-cycle-a",
            parent_host_key="failed-cycle-b",
        )
        second = _insert_act(
            conn,
            status="failed",
            with_action=False,
            host_key="failed-cycle-b",
            parent_host_key="failed-cycle-a",
        )
        first_wait = schedule_act_reduction(conn, "agent-a", first)
        second_failed = schedule_act_reduction(conn, "agent-a", second)
        first_failed = schedule_act_reduction(conn, "agent-a", first, retry=True)
        assert first_wait.status == "retryable"
        assert second_failed.status == first_failed.status == "terminal_failure"
        assert second_failed.task_id is first_failed.task_id is None
        with conn.cursor() as cur:
            cur.execute(
                "SELECT act_id,last_error_code FROM cognitive_act_reductions "
                "WHERE agent_id='agent-a' AND act_id IN (%s,%s) ORDER BY act_id",
                (first, second),
            )
            rows = cur.fetchall()
        assert {row[1] for row in rows} <= {"parent_cycle", "parent_terminal_failure"}
        assert {row[0] for row in rows} == {first, second}
        conn.commit()


def test_causal_root_formula_is_ordered_and_deterministic() -> None:
    act_id = uuid.uuid4()
    memory_a, memory_b = uuid.uuid4(), uuid.uuid4()
    residues = [
        {**_residue(content="a"), "memory_id": memory_a},
        {**_residue(content="b"), "memory_id": memory_b},
    ]
    kwargs = {
        "previous_root_hash": "0" * 64,
        "previous_root_version": 0,
        "output_line_version": 2,
        "act_id": act_id,
        "reducer_version": "act_residue_v1",
        "input_hash": "a" * 64,
        "predecessor_frontier": [],
    }
    first = causal_line_root_hash(**kwargs, residues=residues)
    assert causal_line_root_hash(**kwargs, residues=residues) == first
    assert causal_line_root_hash(**kwargs, residues=list(reversed(residues))) != first


def test_affective_coordinate_is_structured_in_same_atomic_residue(
    wave38_db: str,
) -> None:
    affect = {
        "valence_delta": -0.2,
        "arousal_delta": 0.3,
        "dominance_delta": -0.1,
        "valence": -0.1,
        "arousal": 0.4,
        "dominance": 0.2,
        "intensity": 0.5,
        "cause_status": "active",
        "cause_confidence": 0.7,
    }
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        applied = apply_act_reduction(
            conn, "agent-a", act_id, reducer_version="act_residue_v1",
            input_hash=scheduled.input_hash,
            residues=[_residue(causal_role="affective_coordinate", affect=affect)],
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT residue_affect,metadata->'act_residue'->'affect' "
                "FROM memories WHERE id=%s",
                (applied.memory_ids[0],),
            )
            stored, metadata = cur.fetchone()
            cur.execute(
                "SELECT id,source_kind,source_ref,idempotency_key,metadata "
                "FROM emotional_events WHERE agent_id='agent-a'"
            )
            event = cur.fetchone()
            cur.execute(
                "SELECT event_id,delta_valence,delta_arousal,delta_dominance,"
                "source,causal_context FROM emotional_state "
                "WHERE agent_id='agent-a' ORDER BY id"
            )
            state = cur.fetchone()
            cur.execute(
                "SELECT status,support_valence,support_arousal,support_dominance "
                "FROM emotional_cause_status WHERE agent_id='agent-a' "
                "ORDER BY id DESC LIMIT 1"
            )
            cause = cur.fetchone()
        assert stored == affect
        assert metadata == affect
        assert event[1:4] == ("cognitive_act_residue", str(act_id), None)
        assert event[4]["residue_memory_id"] == str(applied.memory_ids[0])
        assert state[:5] == (
            event[0], pytest.approx(-0.2), pytest.approx(0.3),
            pytest.approx(-0.1), "act_residue",
        )
        assert state[5][0]["evidence_id"] == event[0]
        assert state[5][0]["weighted_delta"] == [-0.2, 0.3, -0.1]
        assert cause == (
            "active", pytest.approx(-0.2), pytest.approx(0.3),
            pytest.approx(-0.1),
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM emotional_cause_status "
                "WHERE agent_id='agent-a' AND cause_event_id=%s",
                (event[0],),
            )
            assert cur.fetchone()[0] == 1

        replay = apply_act_reduction(
            conn, "agent-a", act_id, reducer_version="act_residue_v1",
            input_hash=scheduled.input_hash,
            residues=[_residue(causal_role="affective_coordinate", affect=affect)],
        )
        assert replay.duplicate is True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (SELECT count(*) FROM emotional_events "
                "WHERE agent_id='agent-a'),"
                "(SELECT count(*) FROM emotional_state WHERE agent_id='agent-a')"
            )
            assert cur.fetchone() == (1, 1)
        conn.commit()


def test_affective_coordinate_ignores_public_idempotency_namespace_collision(
    wave38_db: str,
) -> None:
    from styx.emotional.state import append_emotional_event

    affect = {
        "valence_delta": 0.1,
        "arousal_delta": 0.2,
        "dominance_delta": 0.0,
    }
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        guessed_key = f"act-residue:{act_id}:act_residue_v1:0"
        public = append_emotional_event(
            conn,
            "agent-a",
            source_kind="external_event",
            source_ref="foreign",
            idempotency_key=guessed_key,
        )
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        applied = apply_act_reduction(
            conn,
            "agent-a",
            act_id,
            reducer_version="act_residue_v1",
            input_hash=scheduled.input_hash,
            residues=[_residue(causal_role="affective_coordinate", affect=affect)],
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,source_kind,idempotency_key,metadata "
                "FROM emotional_events WHERE agent_id='agent-a' ORDER BY id"
            )
            events = cur.fetchall()
        assert events[0][:3] == (public.event.id, "external_event", guessed_key)
        assert events[1][1:3] == ("cognitive_act_residue", None)
        assert events[1][3]["residue_memory_id"] == str(applied.memory_ids[0])


def test_reduction_rejects_more_than_one_affective_coordinate(
    wave38_db: str,
) -> None:
    affect = {
        "valence_delta": 0.1,
        "arousal_delta": 0.2,
        "dominance_delta": 0.0,
    }
    with psycopg.connect(wave38_db) as conn:
        act_id = _insert_act(conn)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        with pytest.raises(
            ActReductionValidationError,
            match="at most one affective_coordinate",
        ):
            apply_act_reduction(
                conn,
                "agent-a",
                act_id,
                reducer_version="act_residue_v1",
                input_hash=scheduled.input_hash,
                residues=[
                    _residue(causal_role="affective_coordinate", affect=affect),
                    _residue(
                        content="A second coordinate is invalid.",
                        causal_role="affective_coordinate",
                        affect=affect,
                    ),
                ],
            )


def test_predecessor_freshness_is_read_only_and_reports_task_and_root(
    wave38_db: str,
) -> None:
    session_id = uuid.uuid4()
    with psycopg.connect(wave38_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions(id,agent_id) VALUES (%s,'agent-a')",
                (session_id,),
            )
        act_id = _insert_act(conn, session_id=session_id)
        scheduled = schedule_act_reduction(conn, "agent-a", act_id)
        pending = read_predecessor_freshness(
            conn, "agent-a", session_id=session_id,
        )
        assert pending["fresh"] is False
        assert pending["reduction_status"] == "pending"
        assert pending["reduction_task_counts"]["pending"] == 1
        applied = apply_act_reduction(
            conn, "agent-a", act_id, reducer_version="act_residue_v1",
            input_hash=scheduled.input_hash, residues=[_residue()],
        )
        ready = read_predecessor_freshness(
            conn, "agent-a", parent_host_key=f"turn-{act_id}",
        )
        assert ready["fresh"] is True
        assert ready["reduction_status"] == "applied"
        assert ready["predecessor_output_line_version"] == applied.line_version
        assert ready["predecessor_causal_root_hash"] == applied.causal_root_hash
        assert ready["causal_root_hash"] == applied.causal_root_hash
        conn.commit()


def test_missing_explicit_parent_is_pending_but_empty_session_is_fresh(
    wave38_db: str,
) -> None:
    session_id = uuid.uuid4()
    with psycopg.connect(wave38_db) as conn:
        explicit = read_predecessor_freshness(
            conn, "agent-a", parent_host_key="not-committed-yet",
        )
        fallback = read_predecessor_freshness(
            conn, "agent-a", session_id=session_id,
        )
        assert explicit["fresh"] is False
        assert explicit["reduction_status"] == "predecessor_pending"
        assert fallback["fresh"] is True
        assert fallback["reduction_status"] == "absent"
