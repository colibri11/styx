"""Schema migration contract for wave 38."""

from __future__ import annotations

import uuid

import psycopg
import pytest

from styx.storage import migrate


@pytest.fixture
def pre_wave38_db(clean_db: str):
    with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS cognitive_act_reductions CASCADE")
        conn.commit()
    migrations = migrate.discover_migrations()
    with psycopg.connect(clean_db) as conn:
        migrate.apply(conn, [item for item in migrations if item.name < "0010"])
    try:
        yield clean_db
    finally:
        with psycopg.connect(clean_db) as conn, conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS cognitive_act_reductions CASCADE")
            conn.commit()


def _apply_0010(conn: psycopg.Connection) -> None:
    migrations = migrate.discover_migrations()
    migrate.apply(
        conn,
        [item for item in migrations if item.name == "0010_act_residue_carrier.sql"],
    )


def test_upgrade_quarantines_legacy_line_and_projection(pre_wave38_db: str) -> None:
    with psycopg.connect(pre_wave38_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memories(agent_id,role,kind,kind_src,content) "
            "VALUES ('a','summary','note','subjective','legacy') RETURNING id"
        )
        memory_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO will_projections "
            "(agent_id,line_version,formed,source_count,source_hash,supports) "
            "VALUES ('a',1,true,1,%s,'[]'::jsonb)",
            ("a" * 64,),
        )
        conn.commit()
        _apply_0010(conn)
        cur.execute(
            "SELECT line_provenance,residue_ordinal,residue_predecessors,"
            " residue_line_root_hash,residue_affect FROM memories WHERE id=%s",
            (memory_id,),
        )
        assert cur.fetchone() == ("legacy_unknown", None, [], None, {})
        cur.execute(
            "SELECT formed,projection_status,projection_available,coverage_count,carrier_text "
            "FROM will_projections WHERE agent_id='a' AND line_version=1"
        )
        assert cur.fetchone() == (False, "provisional", False, 0, None)


def test_schema_rejects_cross_agent_task_binding(pre_wave38_db: str) -> None:
    with psycopg.connect(pre_wave38_db) as conn:
        _apply_0010(conn)
        act_id = uuid.uuid4()
        input_hash = "a" * 64
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cognitive_acts(id,agent_id,host_key,status,completed_at) "
                "VALUES (%s,'owner','turn','completed',clock_timestamp())",
                (act_id,),
            )
            cur.execute(
                "INSERT INTO llm_tasks(task_type,payload) VALUES "
                "('act_residue_reduction',%s) RETURNING id",
                (psycopg.types.json.Jsonb({
                    "agent_id": "other",
                    "act_id": str(act_id),
                    "reducer_version": "act_residue_v1",
                    "input_hash": input_hash,
                    "attempt_no": 1,
                }),),
            )
            task_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO cognitive_act_reductions "
                "(agent_id,act_id,reducer_version,input_hash,task_id) "
                "VALUES ('owner',%s,'act_residue_v1',%s,%s)",
                (act_id, input_hash, task_id),
            )
        with pytest.raises(psycopg.errors.RaiseException, match="coordinates"):
            conn.commit()


def test_legacy_formed_write_is_downgraded_without_exact_carrier(
    pre_wave38_db: str,
) -> None:
    with psycopg.connect(pre_wave38_db) as conn:
        _apply_0010(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO will_projections "
                "(agent_id,line_version,formed,source_count,source_hash,"
                " projection_status,projection_available,covered_line_version,"
                " coverage_count,coverage_hash,carrier_text,carrier_version) "
                "VALUES ('a',2,true,1,%s,'ready',false,2,1,%s,NULL,NULL) "
                "RETURNING formed,projection_status",
                ("a" * 64, "a" * 64),
            )
            assert cur.fetchone() == (False, "provisional")


def test_validated_residue_requires_complete_coordinates(pre_wave38_db: str) -> None:
    with psycopg.connect(pre_wave38_db) as conn:
        _apply_0010(conn)
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memories "
                    "(agent_id,role,kind,kind_src,content,memory_domain,line_eligible,"
                    " line_provenance) VALUES "
                    "('a','summary','note','subjective','bad','subjective_trace',true,"
                    " 'validated_act_residue')"
                )


def test_diagnostic_only_updates_do_not_advance_semantic_line_version(
    pre_wave38_db: str,
) -> None:
    with psycopg.connect(pre_wave38_db) as conn:
        _apply_0010(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories(agent_id,role,kind,kind_src,content) "
                "VALUES ('a','summary','note','subjective','stable') RETURNING id",
            )
            memory_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE line_state SET dirty=false WHERE agent_id='a' "
                "RETURNING version"
            )
            semantic_version = cur.fetchone()[0]

            cur.execute(
                "UPDATE memories SET embedding=%s WHERE id=%s",
                ("[" + ",".join(["1", *(["0"] * 767)]) + "]", memory_id),
            )
            cur.execute("SELECT version,dirty FROM line_state WHERE agent_id='a'")
            assert cur.fetchone() == (semantic_version, True)

            cur.execute("UPDATE line_state SET dirty=false WHERE agent_id='a'")
            cur.execute(
                "UPDATE memories SET created_at=created_at+interval '1 hour' WHERE id=%s",
                (memory_id,),
            )
            cur.execute("SELECT version,dirty FROM line_state WHERE agent_id='a'")
            assert cur.fetchone() == (semantic_version, True)

            cur.execute("UPDATE line_state SET dirty=false WHERE agent_id='a'")
            cur.execute(
                "UPDATE memories SET content='semantic change' WHERE id=%s",
                (memory_id,),
            )
            cur.execute("SELECT version,dirty FROM line_state WHERE agent_id='a'")
            assert cur.fetchone() == (semantic_version + 1, True)
