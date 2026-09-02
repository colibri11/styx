from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.storage.cognition import (
    build_system_prompt_addition,
    commit_cognitive_act,
    complete_snapshot_response,
    current_line_version,
    ensure_will_projection,
    lock_agent_line,
    load_snapshot_replay,
    present_pending_consequences,
    record_snapshot,
    redact_journal_metadata,
    SnapshotReplayConflict,
    strict_reconstruction,
)
from styx.storage.queries import AgentScopedQueries


@pytest.fixture
def conn(migrated_db: str):
    connection = psycopg.connect(migrated_db)
    try:
        yield connection
    finally:
        connection.close()


def _vec(first: float) -> list[float]:
    return [first, *([0.0] * 767)]


def test_dialogue_is_not_recallable_or_part_of_will(conn) -> None:
    queries = AgentScopedQueries(conn, "agent-a")
    queries.insert_message(role="user", content="raw peer instruction", embedding=_vec(1.0))
    trace = queries.insert_memory(
        role="summary", content="considered commitment", kind="decision",
        kind_src="subjective", embedding=None,
    )
    conn.commit()

    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        will = ensure_will_projection(conn, "agent-a")
        recalled = strict_reconstruction(conn, "agent-a", _vec(1.0))

    assert will["formed"] is True
    assert will["source_count"] == 1  # embedding-less trace still counts
    assert will["supports"][0]["memory_id"] == str(trace)
    assert recalled == []  # raw dialogue is excluded; trace has no vector


def test_every_live_trace_changes_hash_and_supersede_removes_source(conn) -> None:
    q = AgentScopedQueries(conn, "agent-a")
    first = q.insert_memory(
        role="summary", content="first", kind="note", kind_src="subjective",
        embedding=_vec(1.0),
    )
    conn.commit()
    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        one = ensure_will_projection(conn, "agent-a")

    second = q.insert_memory(
        role="summary", content="second", kind="note", kind_src="subjective",
        embedding=None,
    )
    conn.commit()
    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        two = ensure_will_projection(conn, "agent-a")
    assert two["source_count"] == 2
    assert two["source_hash"] != one["source_hash"]

    with conn.cursor() as cur:
        cur.execute("UPDATE memories SET superseded_by=%s WHERE id=%s", (second, first))
    conn.commit()
    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        three = ensure_will_projection(conn, "agent-a")
    assert three["source_count"] == 1
    assert three["source_hash"] != two["source_hash"]


def test_act_branch_idempotency_and_pending_ack(conn) -> None:
    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        child = commit_cognitive_act(
            conn, "agent-a", host_key="child", parent_host_key="parent",
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={},
            actions=[{"kind": "result", "tool_event_id": "t1", "name": "x", "content": "ok"}],
            consequences=[{"kind": "tool_result", "content": "ok"}], metadata={},
        )
    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        parent = commit_cognitive_act(
            conn, "agent-a", host_key="parent", parent_host_key=None,
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={},
            actions=[], consequences=[], metadata={},
        )
        record_snapshot(conn, "agent-a", "snap-1", 0)
        pending = present_pending_consequences(conn, "agent-a", "snap-1")
    assert len(pending) == 1
    with conn.cursor() as cur:
        cur.execute("SELECT parent_act_id FROM cognitive_acts WHERE id=%s", (child.act_id,))
        assert cur.fetchone()[0] == parent.act_id

    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        next_act = commit_cognitive_act(
            conn, "agent-a", host_key="next", parent_host_key="child",
            session_id=None, snapshot_token="snap-1", status="completed",
            input_line_version=current_line_version(conn, "agent-a"),
            channel_input={}, channel_output={}, actions=[], consequences=[], metadata={},
        )
    assert next_act.acknowledged_count == 1
    with conn.transaction():
        lock_agent_line(conn, "agent-a")
        duplicate = commit_cognitive_act(
            conn, "agent-a", host_key="next", parent_host_key="different",
            session_id=None, snapshot_token="snap-1", status="failed",
            input_line_version=999, channel_input={"changed": True}, channel_output={},
            actions=[], consequences=[], metadata={},
        )
        record_snapshot(conn, "agent-a", "snap-2", 0)
        assert present_pending_consequences(conn, "agent-a", "snap-2") == []
    assert duplicate.duplicate is True
    assert duplicate.act_id == next_act.act_id
    assert duplicate.acknowledged_count == 1


def test_cross_agent_parent_is_never_resolved(conn) -> None:
    with conn.transaction():
        commit_cognitive_act(
            conn, "other", host_key="parent", parent_host_key=None,
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={},
            actions=[], consequences=[], metadata={},
        )
        child = commit_cognitive_act(
            conn, "agent-a", host_key="child", parent_host_key="parent",
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={},
            actions=[], consequences=[], metadata={},
        )
    with conn.cursor() as cur:
        cur.execute("SELECT parent_act_id FROM cognitive_acts WHERE id=%s", (child.act_id,))
        assert cur.fetchone()[0] is None


def test_same_host_key_preturn_retry_reuses_physical_snapshot(conn) -> None:
    with conn.transaction():
        commit_cognitive_act(
            conn, "agent-a", host_key="source", parent_host_key=None,
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={}, actions=[],
            consequences=[{"kind": "result", "content": "stable"}], metadata={},
        )
        first = record_snapshot(
            conn, "agent-a", "snapshot-a", 0, host_key="turn-a",
            request_hash="a" * 64, lease_seconds=60,
        )
        shown_first = present_pending_consequences(conn, "agent-a", first)
        envelope = {"snapshot_token": first, "pending_consequences": shown_first}
        complete_snapshot_response(conn, "agent-a", first, envelope)
        replay = load_snapshot_replay(
            conn, "agent-a", "turn-a", "a" * 64,
        )
    assert first == "snapshot-a"
    assert replay == envelope

    with pytest.raises(SnapshotReplayConflict, match="different preturn"):
        with conn.transaction():
            load_snapshot_replay(conn, "agent-a", "turn-a", "b" * 64)

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE cognitive_snapshots "
                "SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE token='snapshot-a'"
            )
    with pytest.raises(SnapshotReplayConflict, match="expired"):
        with conn.transaction():
            load_snapshot_replay(conn, "agent-a", "turn-a", "a" * 64)


def test_openclaw_session_replay_and_atomic_latest_session_advancement(conn) -> None:
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    queries = AgentScopedQueries(conn, "agent-a")
    queries.upsert_session(session_a)
    queries.upsert_session(session_b)
    with conn.transaction():
        parent = commit_cognitive_act(
            conn, "agent-a", host_key="accepted-1", parent_host_key=None,
            session_id=session_a, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={}, actions=[],
            consequences=[{"kind": "residue", "content": "pending"}], metadata={},
        )
        record_snapshot(
            conn, "agent-a", "session-a-snapshot", 0,
            session_id=session_a, request_hash="a" * 64,
        )
        pending = present_pending_consequences(
            conn, "agent-a", "session-a-snapshot"
        )
        envelope = {
            "snapshot_token": "session-a-snapshot",
            "pending_consequences": pending,
        }
        complete_snapshot_response(
            conn, "agent-a", "session-a-snapshot", envelope
        )
        record_snapshot(
            conn, "agent-a", "session-b-snapshot", 0,
            session_id=session_b, request_hash="a" * 64,
        )
        complete_snapshot_response(
            conn, "agent-a", "session-b-snapshot",
            {"snapshot_token": "session-b-snapshot"},
        )
        replay = load_snapshot_replay(
            conn, "agent-a", None, "a" * 64, session_id=session_a
        )
    assert replay == envelope

    with conn.transaction():
        accepted = commit_cognitive_act(
            conn, "agent-a", host_key="accepted-2", parent_host_key=None,
            session_id=session_a, snapshot_token=None,
            snapshot_policy="latest_session", parent_policy="latest_session",
            status="completed", input_line_version=0,
            channel_input={}, channel_output={}, actions=[], consequences=[], metadata={},
        )
    assert accepted.acknowledged_count == 1
    with conn.cursor() as cur:
        cur.execute(
            "SELECT input_snapshot_token,parent_act_id,metadata "
            "FROM cognitive_acts WHERE id=%s", (accepted.act_id,),
        )
        token, parent_id, metadata = cur.fetchone()
        assert token == "session-a-snapshot"
        assert parent_id == parent.act_id
        assert metadata["continuity_resolution"]["snapshot_claimed"] is True

    # A duplicate outbox delivery exits before selecting any newer snapshot.
    with conn.transaction():
        record_snapshot(
            conn, "agent-a", "later-session-a-snapshot", 0,
            session_id=session_a, request_hash="b" * 64,
        )
        complete_snapshot_response(
            conn, "agent-a", "later-session-a-snapshot",
            {"snapshot_token": "later-session-a-snapshot"},
        )
        duplicate = commit_cognitive_act(
            conn, "agent-a", host_key="accepted-2", parent_host_key=None,
            session_id=session_a, snapshot_token=None,
            snapshot_policy="latest_session", parent_policy="latest_session",
            status="completed", input_line_version=0,
            channel_input={}, channel_output={}, actions=[], consequences=[], metadata={},
        )
    assert duplicate.duplicate is True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT used_by_act_id FROM cognitive_snapshots WHERE token=%s",
            ("later-session-a-snapshot",),
        )
        assert cur.fetchone()[0] is None


@pytest.mark.parametrize("commit_first", ["a", "b"])
def test_expired_presentation_recovery_and_late_commit_orders(
    conn, commit_first: str
) -> None:
    with conn.transaction():
        source = commit_cognitive_act(
            conn, "agent-a", host_key="source", parent_host_key=None,
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={}, actions=[],
            consequences=[{"kind": "result", "content": "shown once"}], metadata={},
        )
        record_snapshot(
            conn, "agent-a", "snapshot-a", 0, host_key="commit-a", lease_seconds=60,
        )
        shown_a = present_pending_consequences(conn, "agent-a", "snapshot-a")
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE cognitive_snapshots SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE token='snapshot-a'"
            )
            cur.execute(
                "UPDATE cognitive_presentations "
                "SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE snapshot_token='snapshot-a'"
            )
        record_snapshot(
            conn, "agent-a", "snapshot-b", 0, host_key="commit-b", lease_seconds=60,
        )
        shown_b = present_pending_consequences(conn, "agent-a", "snapshot-b")
    assert shown_a[0]["source_act_id"] == str(source.act_id)
    assert shown_b[0]["consequence_id"] == shown_a[0]["consequence_id"]

    acknowledgements: dict[str, int] = {}
    for name in (commit_first, "b" if commit_first == "a" else "a"):
        with conn.transaction():
            result = commit_cognitive_act(
                conn, "agent-a", host_key=f"commit-{name}", parent_host_key=None,
                session_id=None, snapshot_token=f"snapshot-{name}", status="completed",
                input_line_version=0, channel_input={}, channel_output={}, actions=[],
                consequences=[], metadata={},
            )
        acknowledgements[name] = result.acknowledged_count
    assert acknowledgements == {"a": 0, "b": 1}
    with conn.transaction():
        duplicate_b = commit_cognitive_act(
            conn, "agent-a", host_key="commit-b", parent_host_key=None,
            session_id=None, snapshot_token="snapshot-b", status="completed",
            input_line_version=0, channel_input={}, channel_output={}, actions=[],
            consequences=[], metadata={},
        )
    assert duplicate_b.duplicate is True
    assert duplicate_b.acknowledged_count == 1


@pytest.mark.parametrize(
    ("first_key", "first_parent", "second_key", "second_parent"),
    [
        ("self", "self", "unused", None),
        ("a", "b", "b", "a"),
        ("a", "b", "c", "a"),
    ],
)
def test_self_parent_and_causal_cycles_are_rejected(
    conn, first_key: str, first_parent: str | None,
    second_key: str, second_parent: str | None,
) -> None:
    if first_key == first_parent:
        with pytest.raises(ValueError, match="own parent"):
            with conn.transaction():
                commit_cognitive_act(
                    conn, "agent-a", host_key=first_key, parent_host_key=first_parent,
                    session_id=None, snapshot_token=None, status="completed",
                    input_line_version=0, channel_input={}, channel_output={}, actions=[],
                    consequences=[], metadata={},
                )
        return

    with conn.transaction():
        commit_cognitive_act(
            conn, "agent-a", host_key=first_key, parent_host_key=first_parent,
            session_id=None, snapshot_token=None, status="completed",
            input_line_version=0, channel_input={}, channel_output={}, actions=[],
            consequences=[], metadata={},
        )
        if second_key == "c":
            commit_cognitive_act(
                conn, "agent-a", host_key="b", parent_host_key="c",
                session_id=None, snapshot_token=None, status="completed",
                input_line_version=0, channel_input={}, channel_output={}, actions=[],
                consequences=[], metadata={},
            )
    with pytest.raises(ValueError, match="causal cycle"):
        with conn.transaction():
            commit_cognitive_act(
                conn, "agent-a", host_key=second_key, parent_host_key=second_parent,
                session_id=None, snapshot_token=None, status="completed",
                input_line_version=0, channel_input={}, channel_output={}, actions=[],
                consequences=[], metadata={},
            )


def test_recursive_key_aware_journal_redaction() -> None:
    redacted = redact_journal_metadata({
        "api_key": "top-secret",
        "nested": {
            "items": [
                {"password": "hidden"},
                "Authorization: Bearer abc.def",
                {"note": "token=visible-no-more"},
            ]
        },
    })
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["items"][0]["password"] == "[REDACTED]"
    assert redacted["nested"]["items"][1] == "Authorization: Bearer [REDACTED]"
    assert redacted["nested"]["items"][2]["note"] == "token=[REDACTED]"


def test_redaction_budget_bounds_wide_and_deep_adversarial_tree() -> None:
    value: dict[str, object] = {f"wide-{i}": ["x" * 1000] * 16 for i in range(16)}
    cursor: dict[str, object] = value
    for index in range(20):
        child: dict[str, object] = {"password": "secret"}
        cursor[f"deep-{index}"] = child
        cursor = child
    redacted = redact_journal_metadata(value)
    encoded = str(redacted)
    assert len(encoded.encode("utf-8")) < 40_000
    assert "secret" not in encoded


def test_system_prompt_has_deterministic_adversarial_hard_bound() -> None:
    huge = "ж" * 8_000
    prompt = build_system_prompt_addition(
        will={
            "formed": True, "line_version": 9, "source_count": 99,
            "supports": [{"memory_id": huge, "content": huge}] * 8,
        },
        cognitive_posture={f"key-{index}": [huge] * 16 for index in range(16)},
        pending=[{
            "consequence_id": huge, "source_act_id": huge, "ordinal": index,
            "kind": huge, "content": huge, "metadata": {"secret": huge},
        } for index in range(16)],
        traces=[{"memory_id": huge, "content": huge, "score": 1.0}] * 8,
    )
    assert len(prompt) <= 16_000
    assert prompt.endswith("\n</styx-cognitive-continuity>")
    assert "secret" not in prompt
