from __future__ import annotations

import psycopg

from styx.storage import migrate
from styx.storage.cognition import ensure_will_projection, strict_reconstruction
from styx.storage.queries import AgentScopedQueries


def test_upgrade_backfills_strict_domains(clean_db: str) -> None:
    migrations = migrate.discover_migrations()
    with psycopg.connect(clean_db) as conn:
        migrate.apply(conn, [item for item in migrations if item.name < "0009"])
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories(agent_id,role,content,kind_src) "
                "VALUES ('a','user','raw','subjective'), "
                "       ('a','summary','formed','subjective'), "
                "       ('a','summary','sensor','experience_intake')"
            )
        conn.commit()
        migrate.apply(conn, [item for item in migrations if item.name == "0009_cognitive_continuity.sql"])
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role,memory_domain,line_eligible FROM memories ORDER BY seq"
            )
            assert cur.fetchall() == [
                ("user", "dialogue", False),
                ("summary", "subjective_trace", True),
                ("summary", "external_evidence", False),
            ]


def test_post_migration_experience_writer_is_external_and_excluded(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn:
        queries = AgentScopedQueries(conn, "agent-a")
        memory_id, duplicate = queries.ingest_upsert_memory(
            content="pipeline observation",
            kind="note",
            kind_src="experience_intake",
            content_hash=None,
            embedding=[1.0, *([0.0] * 767)],
        )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT memory_domain,line_eligible FROM memories WHERE id=%s",
                (memory_id,),
            )
            assert cur.fetchone() == ("external_evidence", False)
        with conn.transaction():
            will = ensure_will_projection(conn, "agent-a")
            recalled = strict_reconstruction(
                conn, "agent-a", [1.0, *([0.0] * 767)]
            )
    assert duplicate is False
    assert will["formed"] is False
    assert will["source_count"] == 0
    assert recalled == []


def test_database_guard_coerces_raw_and_experience_direct_inserts(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories"
            "(agent_id,role,kind_src,content,memory_domain,line_eligible) VALUES "
            "('a','user','subjective','raw','subjective_trace',true),"
            "('a','summary','experience_intake','document','subjective_trace',true) "
            "RETURNING role,kind_src,memory_domain,line_eligible"
        )
        assert cur.fetchall() == [
            ("user", "subjective", "dialogue", False),
            ("summary", "experience_intake", "external_evidence", False),
        ]


def test_legacy_recall_event_writer_derives_agent_from_memory(clean_db: str) -> None:
    migrate.run(clean_db)
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories(agent_id,role,content) "
            "VALUES ('owner','summary','trace') RETURNING id"
        )
        memory_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO recall_events(memory_id,agent_id,match_score) "
            "VALUES (%s,'wrong-owner',0.5) RETURNING agent_id",
            (memory_id,),
        )
        assert cur.fetchone()[0] == "owner"
