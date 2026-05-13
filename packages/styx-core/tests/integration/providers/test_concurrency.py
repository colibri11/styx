"""Concurrency smoke-тесты для StyxMemoryCore (Issue #17)."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


def test_sync_turn_concurrent_threads(styx_env, migrated_db: str) -> None:
    """10 параллельных sync_turn сериализуются через _write_lock.

    Ожидаем ровно 20 записей (10 user + 10 assistant).
    """
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="concurrent-agent")

    n_threads = 10

    def do_sync(i: int) -> None:
        p.sync_turn(f"user-{i}", f"assistant-{i}", session_id=sid)

    try:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(do_sync, i) for i in range(n_threads)]
            for f in as_completed(futures):
                f.result()  # поднимет исключение если sync_turn упал

        total = p.queries.count_messages(session_id=uuid.UUID(sid))
        assert total == 20, f"Ожидали 20 сообщений, получили {total}"
    finally:
        p.shutdown()


def test_two_agents_writing_same_session_id(styx_env, migrated_db: str) -> None:
    """Два агента пишут в один и тот же session_id.

    Сессия создаётся первым агентом. Второй агент, который пишет
    в тот же session_id, может столкнуться с ON CONFLICT DO NOTHING —
    это нормально. Каждый агент видит свои 2 записи (изоляция по agent_id).
    """
    shared_sid = str(uuid.uuid4())

    alpha = StyxMemoryCore()
    beta = StyxMemoryCore()

    alpha.initialize(session_id=shared_sid, agent_identity="alpha")
    # beta инициализируем с тем же session_id
    beta.initialize(session_id=shared_sid, agent_identity="beta")

    try:
        alpha.sync_turn("alpha-user", "alpha-assistant", session_id=shared_sid)
        beta.sync_turn("beta-user", "beta-assistant", session_id=shared_sid)

        alpha_count = alpha.queries.count_messages(session_id=uuid.UUID(shared_sid))
        beta_count = beta.queries.count_messages(session_id=uuid.UUID(shared_sid))

        assert alpha_count == 2, f"alpha должен видеть 2 сообщения, видит {alpha_count}"
        assert beta_count == 2, f"beta должен видеть 2 сообщения, видит {beta_count}"

        alpha_contents = {m.content for m in alpha.queries.recent_messages(limit=10, session_id=uuid.UUID(shared_sid))}
        beta_contents = {m.content for m in beta.queries.recent_messages(limit=10, session_id=uuid.UUID(shared_sid))}

        assert alpha_contents == {"alpha-user", "alpha-assistant"}
        assert beta_contents == {"beta-user", "beta-assistant"}
        assert alpha_contents.isdisjoint(beta_contents), "агенты не должны видеть сообщения друг друга"
    finally:
        alpha.shutdown()
        beta.shutdown()
