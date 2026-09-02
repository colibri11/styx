-- Wave 41: host-owned wakeups and model-independent execution provenance.

CREATE TABLE IF NOT EXISTS ready_event_state (
    agent_id text PRIMARY KEY,
    next_generation bigint NOT NULL DEFAULT 1 CHECK (next_generation >= 1),
    last_source_generation bigint NOT NULL DEFAULT 0 CHECK (last_source_generation >= 0),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS cognitive_ready_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id text NOT NULL,
    ready_generation bigint NOT NULL CHECK (ready_generation >= 1),
    reason text NOT NULL CHECK (reason IN (
        'observation_available', 'observation_redeliverable', 'operator_signal'
    )),
    source_generation bigint NOT NULL CHECK (source_generation >= 0),
    observation_high_water bigint,
    pending_count integer NOT NULL CHECK (pending_count >= 0),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','claimed','resolved')),
    claim_token uuid,
    claimed_by text,
    lease_expires_at timestamptz,
    resolve_outcome text CHECK (resolve_outcome IN ('presented','deferred','discarded')),
    resolved_snapshot_token text,
    policy_reason text,
    available_after timestamptz NOT NULL DEFAULT clock_timestamp(),
    delivery_count integer NOT NULL DEFAULT 0 CHECK (delivery_count >= 0),
    redelivery_count integer NOT NULL DEFAULT 0 CHECK (redelivery_count >= 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    resolved_at timestamptz,
    UNIQUE (agent_id, ready_generation),
    UNIQUE (agent_id, reason, source_generation),
    UNIQUE (id, agent_id),
    CONSTRAINT cognitive_ready_events_claim_shape CHECK (
        (status = 'claimed') =
        (claim_token IS NOT NULL AND claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL)
    ),
    CONSTRAINT cognitive_ready_events_resolve_shape CHECK (
        (status = 'resolved') = (resolve_outcome IS NOT NULL AND resolved_at IS NOT NULL)
    ),
    CONSTRAINT cognitive_ready_events_consumer_length CHECK (
        claimed_by IS NULL OR length(claimed_by) BETWEEN 1 AND 128
    ),
    CONSTRAINT cognitive_ready_events_policy_length CHECK (
        policy_reason IS NULL OR length(policy_reason) BETWEEN 1 AND 128
    ),
    CONSTRAINT cognitive_ready_events_snapshot_fk FOREIGN KEY
        (resolved_snapshot_token, agent_id)
        REFERENCES cognitive_snapshots(token, agent_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS cognitive_ready_events_claim_idx
    ON cognitive_ready_events(agent_id, status, available_after, ready_generation);
CREATE INDEX IF NOT EXISTS cognitive_ready_events_lease_idx
    ON cognitive_ready_events(lease_expires_at) WHERE status='claimed';

ALTER TABLE cognitive_acts ADD COLUMN IF NOT EXISTS execution_provenance jsonb;
ALTER TABLE cognitive_acts ADD COLUMN IF NOT EXISTS execution_provenance_hash text;
ALTER TABLE cognitive_acts ADD COLUMN IF NOT EXISTS execution_provenance_version integer;
ALTER TABLE cognitive_acts ADD CONSTRAINT cognitive_acts_execution_provenance_shape CHECK (
    (execution_provenance IS NULL AND execution_provenance_hash IS NULL
     AND execution_provenance_version IS NULL)
    OR
    (jsonb_typeof(execution_provenance)='object'
     AND execution_provenance_hash ~ '^[0-9a-f]{64}$'
     AND execution_provenance_version=1)
);

ALTER TABLE cognitive_snapshots ADD COLUMN IF NOT EXISTS planned_execution_provenance jsonb;
ALTER TABLE cognitive_snapshots ADD COLUMN IF NOT EXISTS planned_execution_provenance_hash text;
ALTER TABLE cognitive_snapshots ADD CONSTRAINT cognitive_snapshots_planned_provenance_shape CHECK (
    (planned_execution_provenance IS NULL AND planned_execution_provenance_hash IS NULL)
    OR
    (jsonb_typeof(planned_execution_provenance)='object'
     AND planned_execution_provenance_hash ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS cognitive_acts_execution_provenance_idx
    ON cognitive_acts(agent_id, execution_provenance_version, execution_provenance_hash);
