"""Unit-тесты для consolidation_state KV."""

from __future__ import annotations

import psycopg
import pytest

from styx.workers.sweep.state import get_state, set_state


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


def test_get_state_missing_returns_none(db) -> None:
    assert get_state(db, "missing_key") is None


def test_set_state_inserts_new_key(db) -> None:
    set_state(db, "test_key", {"x": 1, "y": [2, 3]})
    db.commit()
    out = get_state(db, "test_key")
    assert out == {"x": 1, "y": [2, 3]}


def test_set_state_upserts(db) -> None:
    set_state(db, "k", {"v": 1})
    db.commit()
    set_state(db, "k", {"v": 2})
    db.commit()
    assert get_state(db, "k") == {"v": 2}


def test_set_state_updated_at_advances(db) -> None:
    set_state(db, "k", {"v": 1})
    db.commit()
    with db.cursor() as cur:
        cur.execute("SELECT updated_at FROM consolidation_state WHERE key = 'k'")
        first = cur.fetchone()[0]

    # Маленькая пауза чтобы now() вернул другое значение
    import time

    time.sleep(0.01)

    set_state(db, "k", {"v": 2})
    db.commit()
    with db.cursor() as cur:
        cur.execute("SELECT updated_at FROM consolidation_state WHERE key = 'k'")
        second = cur.fetchone()[0]

    assert second > first
