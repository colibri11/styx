"""Unit-тесты для auto-link orchestrator'а (волна 18).

Фокус на ветвлениях `auto_link_after_store` (enabled flag, no neighbors).
SQL-уровень покрывается в `tests/unit/storage/test_auto_link_queries.py`.
"""

from __future__ import annotations

import uuid

from styx.engine.auto_link import (
    AutoLinkConfig,
    AutoLinkNeighbor,
    auto_link_after_store,
)


class _FakeQueries:
    """Минимальный stub queries для unit-теста orchestrator'а."""

    def __init__(self, candidates: list[AutoLinkNeighbor]) -> None:
        self._candidates = candidates
        self.find_calls: list[tuple] = []
        self.insert_calls: list[tuple[uuid.UUID, list[AutoLinkNeighbor]]] = []

    def find_auto_link_candidates(
        self, embedding, *, max_distance, max_links, exclude_id,
    ):
        self.find_calls.append(
            (tuple(embedding[:3]), max_distance, max_links, exclude_id)
        )
        return list(self._candidates[:max_links])

    def insert_auto_link_relations(self, memory_id, neighbors):
        self.insert_calls.append((memory_id, list(neighbors)))


def test_disabled_returns_zero_no_sql() -> None:
    cfg = AutoLinkConfig(enabled=False)
    q = _FakeQueries(candidates=[])
    n = auto_link_after_store(
        q, memory_id=uuid.uuid4(),
        embedding=[0.0] * 768,
        config=cfg,
        agent_id="alpha",
        source="memory_store",
    )
    assert n == 0
    assert q.find_calls == []
    assert q.insert_calls == []


def test_no_neighbors_returns_zero_no_insert() -> None:
    cfg = AutoLinkConfig(enabled=True)
    q = _FakeQueries(candidates=[])
    mid = uuid.uuid4()
    n = auto_link_after_store(
        q, memory_id=mid, embedding=[0.0] * 768, config=cfg,
        agent_id="alpha", source="memory_store",
    )
    assert n == 0
    assert len(q.find_calls) == 1
    assert q.insert_calls == []


def test_neighbors_inserted_returns_count() -> None:
    cfg = AutoLinkConfig(enabled=True, max_distance=0.25, max_links=3)
    candidates = [
        AutoLinkNeighbor(id=uuid.uuid4(), cosine_distance=0.05),
        AutoLinkNeighbor(id=uuid.uuid4(), cosine_distance=0.12),
        AutoLinkNeighbor(id=uuid.uuid4(), cosine_distance=0.20),
    ]
    q = _FakeQueries(candidates=candidates)
    mid = uuid.uuid4()
    n = auto_link_after_store(
        q, memory_id=mid, embedding=[0.0] * 768, config=cfg,
        agent_id="alpha", source="dialogue_batch_consolidation",
    )
    assert n == 3
    assert len(q.insert_calls) == 1
    assert q.insert_calls[0][0] == mid
    assert [c.id for c in q.insert_calls[0][1]] == [c.id for c in candidates]


def test_passes_exclude_id_to_find() -> None:
    cfg = AutoLinkConfig(enabled=True)
    q = _FakeQueries(candidates=[])
    mid = uuid.uuid4()
    auto_link_after_store(
        q, memory_id=mid, embedding=[0.0] * 768, config=cfg,
        agent_id="alpha", source="sync_turn",
    )
    # exclude_id == memory_id (4-й элемент tuple).
    assert q.find_calls[0][3] == mid


def test_passes_max_distance_and_max_links() -> None:
    cfg = AutoLinkConfig(enabled=True, max_distance=0.10, max_links=5)
    q = _FakeQueries(candidates=[])
    auto_link_after_store(
        q, memory_id=uuid.uuid4(), embedding=[0.0] * 768, config=cfg,
        agent_id="alpha", source="memory_store",
    )
    _, max_dist, max_links, _ = q.find_calls[0]
    assert max_dist == 0.10
    assert max_links == 5
