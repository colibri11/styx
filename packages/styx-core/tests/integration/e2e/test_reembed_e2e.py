"""End-to-end reembed: реальный embeddinggemma на настроенном Ollama."""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest


pytestmark = pytest.mark.skipif(
    not os.path.isdir("/opt/hermes"),
    reason="integration tests run only inside hermes-agent-styx-test container",
)


@pytest.fixture
def dsn() -> str:
    return os.environ["STYX_DATABASE_URL"]


@pytest.fixture
def migrated(dsn: str) -> str:
    from styx.storage import migrate
    migrate.run(dsn)
    return dsn


def test_reembed_backfills_real_ollama(migrated: str) -> None:
    from styx.commands.reembed import REEMBED_MODE_NULL_ONLY, run_reembed
    from styx.embedding import make_embedding_client

    agent = "alpha"

    # INSERT 3 memories без embedding'а.
    with psycopg.connect(migrated) as conn:
        for i in range(3):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memories (agent_id, role, content) "
                    "VALUES (%s, 'user', %s)",
                    (agent, f"тестовая память номер {i}"),
                )
        conn.commit()

    embed = make_embedding_client(
        base_url=os.environ.get("STYX_OLLAMA_URL", "http://ollama:11434"),
        model=os.environ.get("STYX_EMBEDDING_MODEL", "embeddinggemma:300m-qat-q8_0"),
        dim=int(os.environ.get("STYX_EMBEDDING_DIM", "768")),
        timeout=30.0,
    )

    with psycopg.connect(migrated) as conn:
        result = run_reembed(
            conn=conn,
            embed_client=embed,
            mode=REEMBED_MODE_NULL_ONLY,
            agent_id=agent,
            rate_per_second=10.0,
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories "
                " WHERE agent_id = %s AND embedding IS NOT NULL",
                (agent,),
            )
            with_embedding = cur.fetchone()[0]
        # Cleanup.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE agent_id = %s", (agent,))
        conn.commit()

    assert result.processed == 3
    assert result.failed == 0
    assert with_embedding == 3


def test_reembed_dry_run_real_db(migrated: str) -> None:
    """Dry-run на пустой scope → would_process=0, exit 0."""
    from styx.commands.reembed import REEMBED_MODE_NULL_ONLY, run_reembed
    from styx.embedding import FakeEmbeddingClient

    agent = "alpha"
    with psycopg.connect(migrated) as conn:
        result = run_reembed(
            conn=conn,
            embed_client=FakeEmbeddingClient(),
            mode=REEMBED_MODE_NULL_ONLY,
            agent_id=agent,
            dry_run=True,
        )
    assert result.dry_run is True
    assert result.would_process == 0
