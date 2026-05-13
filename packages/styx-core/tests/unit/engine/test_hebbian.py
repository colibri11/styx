"""Unit-тесты для Hebbian orchestrator'а (волна 21)."""

from __future__ import annotations

import uuid

from styx.engine.hebbian import HebbianConfig, reinforce_co_retrieval


class _FakeQueries:
    """Stub queries для unit-теста."""

    def __init__(self) -> None:
        self.upserts: list[tuple[uuid.UUID, uuid.UUID]] = []

    def upsert_co_retrieved_pair(
        self, *, source_id, target_id,
        initial_weight, weight_bump, weight_max,
    ) -> None:
        self.upserts.append((source_id, target_id))


def test_disabled_returns_zero_no_upserts() -> None:
    cfg = HebbianConfig(enabled=False)
    q = _FakeQueries()
    n = reinforce_co_retrieval(
        q, memory_ids=[uuid.uuid4(), uuid.uuid4()],
        config=cfg, agent_id="alpha",
    )
    assert n == 0
    assert q.upserts == []


def test_less_than_two_ids_returns_zero() -> None:
    cfg = HebbianConfig(enabled=True)
    q = _FakeQueries()
    n = reinforce_co_retrieval(
        q, memory_ids=[uuid.uuid4()],
        config=cfg, agent_id="alpha",
    )
    assert n == 0
    assert q.upserts == []


def test_three_ids_three_pairs() -> None:
    """N=3 → C(3,2) = 3 пары: (0,1), (0,2), (1,2)."""
    cfg = HebbianConfig(enabled=True)
    q = _FakeQueries()
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    n = reinforce_co_retrieval(
        q, memory_ids=[a, b, c], config=cfg, agent_id="alpha",
    )
    assert n == 3
    assert q.upserts == [(a, b), (a, c), (b, c)]


def test_ten_ids_45_pairs() -> None:
    """N=10 → C(10,2) = 45 пар."""
    cfg = HebbianConfig(enabled=True)
    q = _FakeQueries()
    ids = [uuid.uuid4() for _ in range(10)]
    n = reinforce_co_retrieval(
        q, memory_ids=ids, config=cfg, agent_id="alpha",
    )
    assert n == 45
    assert len(q.upserts) == 45


def test_pair_order_preserved_i_lt_j() -> None:
    """Pairs (i, j) с i < j по порядку memory_ids."""
    cfg = HebbianConfig(enabled=True)
    q = _FakeQueries()
    a, b, c, d = (uuid.uuid4() for _ in range(4))
    reinforce_co_retrieval(
        q, memory_ids=[a, b, c, d], config=cfg, agent_id="alpha",
    )
    assert q.upserts == [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
