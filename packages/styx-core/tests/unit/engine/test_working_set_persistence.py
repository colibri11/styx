"""Юнит-тесты working_set_persistence — serialize/deserialize/load (волна 13).

`save` тестируется в `tests/storage/test_working_set_table.py` (требует
migrated_db). `start`/`stop`/`is_running` — в
`tests/providers/test_working_set_wiring.py` (provider lifecycle).
"""

from __future__ import annotations

import datetime as _dt
import time
import uuid
from typing import Any

import pytest

from styx.engine import working_set_persistence as wsp
from styx.engine.hot_tier import HotEntry


# ── helpers ────────────────────────────────────────────────────────────────


def _focus_snap(
    *,
    window: list[list[float]] | None = None,
    cached_salient: dict | None = None,
    epoch_id: int = 1,
) -> tuple[list[list[float]], dict | None, int]:
    return (
        window if window is not None else [[0.1, 0.2, 0.3]],
        cached_salient,
        epoch_id,
    )


def _hot_entry(
    *,
    embedding: list[float] | None = None,
    evicted_at: float | None = None,
    agent_id: str = "agent-a",
    content: str = "content",
) -> HotEntry:
    return HotEntry(
        id=uuid.uuid4(),
        agent_id=agent_id,
        kind="subjective_dialogue",
        kind_src="subjective",
        role="user",
        content=content,
        metadata={"k": "v"},
        created_at=_dt.datetime(2026, 5, 3, 12, 0, tzinfo=_dt.timezone.utc),
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3],
        evicted_at=evicted_at if evicted_at is not None else time.monotonic(),
    )


# ── serialize ─────────────────────────────────────────────────────────────


def test_serialize_focus_only() -> None:
    payload = wsp.serialize(_focus_snap(), [], embedding_dim=3)
    assert payload["version"] == wsp.PAYLOAD_VERSION
    assert payload["embedding_dim"] == 3
    assert payload["focus"] == {
        "window": [[0.1, 0.2, 0.3]],
        "cached_salient": None,
        "epoch_id": 1,
    }
    assert payload["hot"] is None


def test_serialize_hot_only() -> None:
    e = _hot_entry()
    payload = wsp.serialize(None, [e], embedding_dim=3)
    assert payload["focus"] is None
    assert payload["hot"] is not None
    assert len(payload["hot"]) == 1
    item = payload["hot"][0]
    assert item["id"] == str(e.id)
    assert item["agent_id"] == e.agent_id
    assert item["embedding"] == e.embedding
    assert item["evicted_age_s"] >= 0.0
    assert item["created_at"] == e.created_at.isoformat()


def test_serialize_both_none() -> None:
    payload = wsp.serialize(None, [], embedding_dim=3)
    assert payload["focus"] is None
    assert payload["hot"] is None


def test_serialize_evicted_age_reflects_elapsed_time() -> None:
    e = _hot_entry(evicted_at=time.monotonic() - 100.0)
    payload = wsp.serialize(None, [e], embedding_dim=3)
    assert 99.0 < payload["hot"][0]["evicted_age_s"] < 110.0


def test_serialize_clamps_negative_evicted_age_to_zero() -> None:
    """Если evicted_at в будущем (clock skew/тест) — не пишем negative."""
    e = _hot_entry(evicted_at=time.monotonic() + 50.0)
    payload = wsp.serialize(None, [e], embedding_dim=3)
    assert payload["hot"][0]["evicted_age_s"] == 0.0


# ── deserialize / roundtrip ───────────────────────────────────────────────


def test_deserialize_roundtrip_focus_and_hot() -> None:
    focus = _focus_snap(
        window=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        cached_salient={"role": "user", "content": "X"},
        epoch_id=4,
    )
    hot = [_hot_entry(), _hot_entry()]
    payload = wsp.serialize(focus, hot, embedding_dim=3)

    snap = wsp.deserialize(payload, embedding_dim=3)
    assert snap is not None
    assert snap.focus is not None
    assert snap.focus.window == focus[0]
    assert snap.focus.cached_salient == focus[1]
    assert snap.focus.epoch_id == focus[2]
    assert snap.hot is not None
    assert {e.id for e in snap.hot} == {e.id for e in hot}


def test_deserialize_drop_hot_skips_hot_section() -> None:
    payload = wsp.serialize(_focus_snap(), [_hot_entry()], embedding_dim=3)
    snap = wsp.deserialize(payload, embedding_dim=3, drop_hot=True)
    assert snap is not None
    assert snap.focus is not None
    assert snap.hot is None


def test_deserialize_version_mismatch_returns_none() -> None:
    payload = wsp.serialize(_focus_snap(), [], embedding_dim=3)
    payload["version"] = wsp.PAYLOAD_VERSION + 1
    assert wsp.deserialize(payload, embedding_dim=3) is None


def test_deserialize_dim_mismatch_returns_none() -> None:
    payload = wsp.serialize(_focus_snap(), [], embedding_dim=3)
    assert wsp.deserialize(payload, embedding_dim=768) is None


def test_deserialize_non_dict_payload_returns_none() -> None:
    assert wsp.deserialize("not a dict", embedding_dim=3) is None  # type: ignore[arg-type]


def test_deserialize_skips_focus_window_entry_with_wrong_dim() -> None:
    payload = wsp.serialize(_focus_snap(window=[[0.1, 0.2, 0.3]]), [], embedding_dim=3)
    # Подпортим один из векторов: 2 floats вместо 3.
    payload["focus"]["window"].append([0.4, 0.5])
    snap = wsp.deserialize(payload, embedding_dim=3)
    assert snap is not None
    assert snap.focus is not None
    # Кривой entry отброшен, валидный остался.
    assert snap.focus.window == [[0.1, 0.2, 0.3]]


def test_deserialize_skips_hot_entry_with_wrong_embedding_dim() -> None:
    e_ok = _hot_entry(embedding=[0.1, 0.2, 0.3])
    e_bad = _hot_entry(embedding=[0.4, 0.5])
    payload = wsp.serialize(None, [e_ok, e_bad], embedding_dim=3)
    snap = wsp.deserialize(payload, embedding_dim=3)
    assert snap is not None
    assert snap.hot is not None
    assert {e.id for e in snap.hot} == {e_ok.id}


def test_deserialize_empty_focus_returns_focus_none() -> None:
    payload: dict[str, Any] = {
        "version": wsp.PAYLOAD_VERSION,
        "embedding_dim": 3,
        "focus": {"window": [], "cached_salient": None, "epoch_id": 0},
        "hot": None,
        "saved_at_monotonic": 0.0,
    }
    snap = wsp.deserialize(payload, embedding_dim=3)
    assert snap is not None
    assert snap.focus is None
    assert snap.hot is None


def test_deserialize_restores_evicted_at_relative_to_now() -> None:
    e = _hot_entry(evicted_at=time.monotonic() - 50.0)
    payload = wsp.serialize(None, [e], embedding_dim=3)
    snap = wsp.deserialize(payload, embedding_dim=3)
    assert snap is not None
    assert snap.hot is not None
    restored = snap.hot[0]
    delta = time.monotonic() - restored.evicted_at
    # Относительная свежесть сохранилась — entry "evicted ~50s назад".
    assert 49.0 < delta < 55.0


def test_deserialize_invalid_uuid_skips_hot_entry() -> None:
    payload: dict[str, Any] = {
        "version": wsp.PAYLOAD_VERSION,
        "embedding_dim": 3,
        "focus": None,
        "hot": [{"id": "not-a-uuid", "embedding": [0.1, 0.2, 0.3]}],
        "saved_at_monotonic": 0.0,
    }
    snap = wsp.deserialize(payload, embedding_dim=3)
    # Пустой list после фильтрации → hot = None.
    assert snap is not None
    assert snap.hot is None


# ── load ──────────────────────────────────────────────────────────────────


class _StubCursor:
    def __init__(self, row: tuple | None) -> None:
        self._row = row
        self.last_sql: str | None = None
        self.last_params: tuple | None = None

    def __enter__(self) -> "_StubCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple) -> None:
        self.last_sql = sql
        self.last_params = params

    def fetchone(self) -> tuple | None:
        return self._row


class _StubConn:
    def __init__(self, row: tuple | None = None, raise_on_select: bool = False) -> None:
        self._row = row
        self._raise = raise_on_select
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _StubCursor:
        if self._raise:
            import psycopg
            raise psycopg.Error("simulated db failure")
        return _StubCursor(self._row)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_load_returns_none_when_row_missing() -> None:
    conn = _StubConn(row=None)
    snap = wsp.load(
        conn,  # type: ignore[arg-type]
        agent_id="agent-a",
        ttl_s=86400.0,
        hot_ttl_s=300.0,
        embedding_dim=3,
    )
    assert snap is None


def test_load_returns_none_past_ttl() -> None:
    payload = wsp.serialize(_focus_snap(), [_hot_entry()], embedding_dim=3)
    old = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(seconds=2_000_000)
    conn = _StubConn(row=(payload, old))
    snap = wsp.load(
        conn,  # type: ignore[arg-type]
        agent_id="agent-a",
        ttl_s=86400.0,
        hot_ttl_s=300.0,
        embedding_dim=3,
    )
    assert snap is None


def test_load_returns_focus_only_past_hot_ttl() -> None:
    payload = wsp.serialize(_focus_snap(), [_hot_entry()], embedding_dim=3)
    medium = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(seconds=1000)
    conn = _StubConn(row=(payload, medium))
    snap = wsp.load(
        conn,  # type: ignore[arg-type]
        agent_id="agent-a",
        ttl_s=86400.0,
        hot_ttl_s=300.0,
        embedding_dim=3,
    )
    assert snap is not None
    assert snap.focus is not None
    assert snap.hot is None


def test_load_returns_full_state_within_both_ttls() -> None:
    payload = wsp.serialize(_focus_snap(), [_hot_entry()], embedding_dim=3)
    fresh = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(seconds=10)
    conn = _StubConn(row=(payload, fresh))
    snap = wsp.load(
        conn,  # type: ignore[arg-type]
        agent_id="agent-a",
        ttl_s=86400.0,
        hot_ttl_s=300.0,
        embedding_dim=3,
    )
    assert snap is not None
    assert snap.focus is not None
    assert snap.hot is not None and len(snap.hot) == 1


def test_load_handles_db_error_returning_none() -> None:
    conn = _StubConn(raise_on_select=True)
    snap = wsp.load(
        conn,  # type: ignore[arg-type]
        agent_id="agent-a",
        ttl_s=86400.0,
        hot_ttl_s=300.0,
        embedding_dim=3,
    )
    assert snap is None


def test_load_drops_when_payload_not_dict() -> None:
    fresh = _dt.datetime.now(tz=_dt.timezone.utc)
    conn = _StubConn(row=("not a dict", fresh))
    snap = wsp.load(
        conn,  # type: ignore[arg-type]
        agent_id="agent-a",
        ttl_s=86400.0,
        hot_ttl_s=300.0,
        embedding_dim=3,
    )
    assert snap is None
