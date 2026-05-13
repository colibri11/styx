"""Тесты `working_set` table — INSERT ON CONFLICT, per-agent изоляция (волна 13).

Требует ``STYX_TEST_DATABASE_URL`` — на host без БД скипается.
"""

from __future__ import annotations

import time

import psycopg
import pytest

from styx.engine import working_set_persistence as wsp
from styx.storage import migrate


@pytest.fixture
def db(clean_db: str) -> str:
    migrate.run(clean_db)
    return clean_db


def _payload(*, marker: str = "v1") -> dict:
    return {
        "version": wsp.PAYLOAD_VERSION,
        "embedding_dim": 3,
        "saved_at_monotonic": time.monotonic(),
        "focus": {
            "window": [[0.1, 0.2, 0.3]],
            "cached_salient": {"role": "user", "content": marker},
            "epoch_id": 1,
        },
        "hot": None,
    }


def test_save_inserts_row(db: str) -> None:
    wsp.save(db, "agent-a", _payload(marker="initial"))
    with psycopg.connect(db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM working_set WHERE agent_id = %s",
                ("agent-a",),
            )
            row = cur.fetchone()
    assert row is not None
    assert row[0]["focus"]["cached_salient"]["content"] == "initial"


def test_save_updates_on_conflict(db: str) -> None:
    wsp.save(db, "agent-a", _payload(marker="first"))
    wsp.save(db, "agent-a", _payload(marker="second"))
    with psycopg.connect(db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload, updated_at FROM working_set WHERE agent_id = %s",
                ("agent-a",),
            )
            row = cur.fetchone()
            cur.execute("SELECT count(*) FROM working_set")
            count = cur.fetchone()
    assert row is not None
    assert row[0]["focus"]["cached_salient"]["content"] == "second"
    assert count is not None
    assert count[0] == 1


def test_per_agent_isolation(db: str) -> None:
    wsp.save(db, "agent-a", _payload(marker="A"))
    wsp.save(db, "agent-b", _payload(marker="B"))
    with psycopg.connect(db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT agent_id, payload FROM working_set ORDER BY agent_id")
            rows = cur.fetchall()
    assert len(rows) == 2
    by_agent = {r[0]: r[1] for r in rows}
    assert by_agent["agent-a"]["focus"]["cached_salient"]["content"] == "A"
    assert by_agent["agent-b"]["focus"]["cached_salient"]["content"] == "B"


def test_load_roundtrip(db: str) -> None:
    wsp.save(db, "agent-a", _payload(marker="loaded"))
    with psycopg.connect(db) as conn:
        snap = wsp.load(
            conn,
            agent_id="agent-a",
            ttl_s=86400.0,
            hot_ttl_s=300.0,
            embedding_dim=3,
        )
    assert snap is not None
    assert snap.focus is not None
    assert snap.focus.cached_salient == {"role": "user", "content": "loaded"}


def test_load_missing_agent_returns_none(db: str) -> None:
    with psycopg.connect(db) as conn:
        snap = wsp.load(
            conn,
            agent_id="agent-missing",
            ttl_s=86400.0,
            hot_ttl_s=300.0,
            embedding_dim=3,
        )
    assert snap is None
