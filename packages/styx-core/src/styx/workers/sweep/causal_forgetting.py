"""Conservative, model-free scheduler for Wave 40 causal forgetting."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from styx import turn_state
from styx.storage.causal_graph import apply_causal_forgetting
from styx.storage.queries import list_subject_agents

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CausalForgettingConfig:
    enabled: bool = False
    min_age_days: int = 90
    min_idle_days: int = 30
    relevance_ceiling: float = 0.15
    max_batch: int = 2

    def __post_init__(self) -> None:
        if self.min_age_days < 1 or self.min_idle_days < 1:
            raise ValueError("causal forgetting age bounds must be positive")
        if not 0.0 <= self.relevance_ceiling <= 1.0:
            raise ValueError("causal forgetting relevance must be in 0..1")
        if not 1 <= self.max_batch <= 16:
            raise ValueError("causal forgetting max_batch must be in 1..16")


@dataclass(slots=True)
class CausalForgettingSummary:
    applied_operations: int = 0
    forgotten_nodes: int = 0
    rewired_edges: int = 0
    skipped_agents: int = 0
    errors: int = 0


def _candidates(
    conn: psycopg.Connection,
    agent_id: str,
    *,
    config: CausalForgettingConfig,
    now: dt.datetime,
) -> tuple[list, int, int, str] | None:
    """Return candidates plus one frozen line coordinate, or no safe work."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT version,causal_root_hash FROM line_state "
            "WHERE agent_id=%s AND dirty=false FOR UPDATE", (agent_id,),
        )
        line = cur.fetchone()
        if line is None:
            return None
        cur.execute(
            "SELECT 1 FROM will_projections WHERE agent_id=%s "
            "AND line_version=%s AND projection_status='ready' "
            "AND projection_available=true AND covered_line_version=%s",
            (agent_id, line["version"], line["version"]),
        )
        if cur.fetchone() is None:
            return None
        cur.execute(
            "SELECT 1 FROM causal_operations WHERE agent_id=%s "
            "AND status IN ('pending','running','retryable') LIMIT 1",
            (agent_id,),
        )
        if cur.fetchone() is not None:
            return None
        cur.execute(
            "SELECT count(*)::int AS count FROM memories WHERE agent_id=%s "
            "AND line_provenance IN "
            "('validated_act_residue','validated_transform') "
            "AND line_status='active'", (agent_id,),
        )
        active_count = int(cur.fetchone()["count"])
        if active_count <= 1:
            return None
        limit = min(config.max_batch, active_count - 1)
        cur.execute(
            "SELECT memory.id,"
            "coalesce(memory.importance_final,memory.importance_provisional)::float8 "
            "AS relevance FROM memories memory "
            "WHERE memory.agent_id=%s AND memory.line_provenance IN "
            "('validated_act_residue','validated_transform') "
            "AND memory.line_status='active' AND memory.embedding IS NOT NULL "
            "AND memory.created_at <= %s - (%s * interval '1 day') "
            "AND coalesce(memory.last_accessed_at,memory.created_at) "
            "    <= %s - (%s * interval '1 day') "
            "AND coalesce(memory.importance_final,memory.importance_provisional) <= %s "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM emotional_events event "
            "  JOIN LATERAL ("
            "    SELECT status,lease_expires_at FROM emotional_cause_status "
            "    WHERE agent_id=event.agent_id AND cause_event_id=event.id "
            "    ORDER BY at DESC,id DESC LIMIT 1"
            "  ) cause ON true "
            "  WHERE event.agent_id=memory.agent_id "
            "    AND event.metadata->>'residue_memory_id'=memory.id::text "
            "    AND cause.status='active' AND cause.lease_expires_at>%s"
            ") ORDER BY relevance ASC,memory.causal_node_hash ASC LIMIT %s "
            "FOR UPDATE OF memory SKIP LOCKED",
            (
                agent_id, now, config.min_age_days, now, config.min_idle_days,
                config.relevance_ceiling, now, limit,
            ),
        )
        rows = list(cur.fetchall())
    if not rows:
        return None
    return (
        [row["id"] for row in rows], int(line["version"]), len(rows),
        str(line["causal_root_hash"]),
    )


def run_causal_forgetting_sweep(
    conn: psycopg.Connection,
    *,
    config: CausalForgettingConfig,
    now: dt.datetime | None = None,
) -> CausalForgettingSummary:
    """Apply at most one small, fully evidenced operation per idle agent."""
    summary = CausalForgettingSummary()
    if not config.enabled:
        return summary
    moment = now or dt.datetime.now(tz=dt.timezone.utc)
    for agent_id in list_subject_agents(conn):
        if turn_state.is_active(agent_id, now=moment):
            summary.skipped_agents += 1
            continue
        try:
            with conn.transaction():
                selected = _candidates(
                    conn, agent_id, config=config, now=moment,
                )
                if selected is None:
                    summary.skipped_agents += 1
                    continue
                memory_ids, version, count, root_hash = selected
                operation = apply_causal_forgetting(
                    conn,
                    agent_id,
                    operation_key=(
                        f"automatic_forgetting:v1:{version}:"
                        + ":".join(sorted(str(item) for item in memory_ids))
                    ),
                    memory_ids=memory_ids,
                    reason_code="bounded_relevance_v1",
                    feature_coordinates={
                        "policy_version": "causal_forgetting_v1",
                        "min_age_days": config.min_age_days,
                        "min_idle_days": config.min_idle_days,
                        "relevance_ceiling": config.relevance_ceiling,
                        "candidate_count": count,
                        "embedding_required": True,
                    },
                    expected_line_version=version,
                    expected_root_hash=root_hash,
                )
                summary.applied_operations += 1
                summary.forgotten_nodes += operation.tombstone_count
                summary.rewired_edges += operation.rewired_edge_count
        except Exception as exc:  # noqa: BLE001
            summary.errors += 1
            log.warning("causal forgetting failed agent=%s: %s", agent_id, exc)
    return summary


__all__ = [
    "CausalForgettingConfig", "CausalForgettingSummary",
    "run_causal_forgetting_sweep",
]
