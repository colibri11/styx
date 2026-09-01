-- Styx storage 0008 — causal emotional evidence and state lineage.
--
-- Additive by design: the legacy emotional_state VAD/source/metadata ABI and
-- memories.emotional_context_* scoring columns remain intact.  Existing rows
-- deliberately keep NULL provenance — unknown evidence must not be promoted
-- to false certainty during migration.

-- Source observations are distinct from the projected state they influence.
CREATE TABLE IF NOT EXISTS emotional_events (
    id              bigserial PRIMARY KEY,
    agent_id        text NOT NULL,
    occurred_at     timestamptz NOT NULL,
    observed_at     timestamptz NOT NULL DEFAULT clock_timestamp(),
    source_kind     text NOT NULL,
    source_ref      text,
    idempotency_key text,
    valence         real,
    arousal         real,
    dominance       real,
    intensity       real,
    confidence      real,
    cause_summary   text,
    cause_status    text NOT NULL DEFAULT 'unknown',
    cause_status_at timestamptz,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT emotional_events_agent_id_nonempty
        CHECK (length(btrim(agent_id)) > 0),
    CONSTRAINT emotional_events_source_kind_nonempty
        CHECK (length(btrim(source_kind)) BETWEEN 1 AND 64),
    CONSTRAINT emotional_events_source_ref_length
        CHECK (source_ref IS NULL OR length(source_ref) <= 512),
    CONSTRAINT emotional_events_idempotency_key_length
        CHECK (
            idempotency_key IS NULL
            OR length(btrim(idempotency_key)) BETWEEN 1 AND 512
        ),
    CONSTRAINT emotional_events_cause_summary_length
        CHECK (cause_summary IS NULL OR length(cause_summary) <= 1000),
    CONSTRAINT emotional_events_vad_shape
        CHECK (
            (valence IS NULL AND arousal IS NULL AND dominance IS NULL)
            OR
            (valence IS NOT NULL AND arousal IS NOT NULL AND dominance IS NOT NULL)
        ),
    CONSTRAINT emotional_events_vad_range
        CHECK (
            valence IS NULL
            OR (
                valence BETWEEN -1.0 AND 1.0
                AND arousal BETWEEN -1.0 AND 1.0
                AND dominance BETWEEN -1.0 AND 1.0
            )
        ),
    CONSTRAINT emotional_events_intensity_range
        CHECK (intensity IS NULL OR intensity BETWEEN 0.0 AND 1.0),
    CONSTRAINT emotional_events_confidence_range
        CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0),
    CONSTRAINT emotional_events_cause_status_check
        CHECK (cause_status IN ('unknown', 'active', 'resolved', 'superseded')),
    CONSTRAINT emotional_events_metadata_object
        CHECK (jsonb_typeof(metadata) = 'object'),
    UNIQUE (id, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_emotional_events_agent_occurred
    ON emotional_events(agent_id, occurred_at DESC, id DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_emotional_events_agent_idempotency
    ON emotional_events(agent_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Cause lifecycle is not mutable event metadata.  Every recognition,
-- reaffirmation, resolution or lease expiry is an ordered fact of its own.
-- ``cause_status`` on emotional_events remains the observer's immutable
-- assertion for wire/backward compatibility; this journal is authoritative
-- for current support.
CREATE TABLE IF NOT EXISTS emotional_cause_status (
    id                      bigserial PRIMARY KEY,
    agent_id                text NOT NULL,
    cause_event_id          bigint NOT NULL,
    at                      timestamptz NOT NULL DEFAULT clock_timestamp(),
    status                  text NOT NULL,
    lease_expires_at        timestamptz,
    support_valence         real,
    support_arousal         real,
    support_dominance       real,
    confidence              real,
    intensity               real,
    status_source_event_id  bigint,
    context                 jsonb NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT emotional_cause_status_status_check
        CHECK (status IN ('active', 'resolved', 'superseded', 'expired')),
    CONSTRAINT emotional_cause_status_active_lease
        CHECK (
            (status = 'active' AND lease_expires_at IS NOT NULL
             AND lease_expires_at > at)
            OR (status <> 'active' AND lease_expires_at IS NULL)
        ),
    CONSTRAINT emotional_cause_status_support_shape
        CHECK (
            (support_valence IS NULL AND support_arousal IS NULL
             AND support_dominance IS NULL)
            OR
            (support_valence IS NOT NULL AND support_arousal IS NOT NULL
             AND support_dominance IS NOT NULL)
        ),
    CONSTRAINT emotional_cause_status_support_range
        CHECK (
            support_valence IS NULL
            OR (
                support_valence BETWEEN -1.0 AND 1.0
                AND support_arousal BETWEEN -1.0 AND 1.0
                AND support_dominance BETWEEN -1.0 AND 1.0
            )
        ),
    CONSTRAINT emotional_cause_status_confidence_range
        CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0),
    CONSTRAINT emotional_cause_status_intensity_range
        CHECK (intensity IS NULL OR intensity BETWEEN 0.0 AND 1.0),
    CONSTRAINT emotional_cause_status_context_object
        CHECK (jsonb_typeof(context) = 'object'),
    UNIQUE (id, agent_id),
    CONSTRAINT emotional_cause_status_cause_same_agent
        FOREIGN KEY (cause_event_id, agent_id)
        REFERENCES emotional_events(id, agent_id)
        ON DELETE CASCADE,
    CONSTRAINT emotional_cause_status_source_same_agent
        FOREIGN KEY (status_source_event_id, agent_id)
        REFERENCES emotional_events(id, agent_id)
        ON DELETE SET NULL (status_source_event_id)
);

CREATE INDEX IF NOT EXISTS idx_emotional_cause_status_current
    ON emotional_cause_status(agent_id, cause_event_id, at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_emotional_cause_status_active_lease
    ON emotional_cause_status(agent_id, lease_expires_at)
    WHERE status = 'active';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_cause_status_cause_same_agent'
          AND conrelid = 'emotional_cause_status'::regclass
    ) THEN
        ALTER TABLE emotional_cause_status
            ADD CONSTRAINT emotional_cause_status_cause_same_agent
            FOREIGN KEY (cause_event_id, agent_id)
            REFERENCES emotional_events(id, agent_id)
            ON DELETE CASCADE;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_cause_status_source_same_agent'
          AND conrelid = 'emotional_cause_status'::regclass
    ) THEN
        ALTER TABLE emotional_cause_status
            ADD CONSTRAINT emotional_cause_status_source_same_agent
            FOREIGN KEY (status_source_event_id, agent_id)
            REFERENCES emotional_events(id, agent_id)
            ON DELETE SET NULL (status_source_event_id);
    END IF;
END$$;

-- emotional_state remains an append-only materialized projection.  New rows
-- can now say which transition produced them and which state they followed.
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS parent_state_id bigint;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS event_id bigint;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS delta_valence real;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS delta_arousal real;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS delta_dominance real;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS intensity real;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS confidence real;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS transition_confidence real;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS causal_context jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE emotional_state
    ADD COLUMN IF NOT EXISTS computation_version text;

CREATE UNIQUE INDEX IF NOT EXISTS uq_emotional_state_id_agent
    ON emotional_state(id, agent_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_parent_same_agent'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_parent_same_agent
            FOREIGN KEY (parent_state_id, agent_id)
            REFERENCES emotional_state(id, agent_id)
            ON DELETE SET NULL (parent_state_id);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_transition_confidence_range'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_transition_confidence_range
            CHECK (
                transition_confidence IS NULL
                OR transition_confidence BETWEEN 0.0 AND 1.0
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_event_same_agent'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_event_same_agent
            FOREIGN KEY (event_id, agent_id)
            REFERENCES emotional_events(id, agent_id)
            ON DELETE SET NULL (event_id);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_vad_range'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_vad_range CHECK (
                valence BETWEEN -1.0 AND 1.0
                AND arousal BETWEEN -1.0 AND 1.0
                AND dominance BETWEEN -1.0 AND 1.0
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_delta_shape'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_delta_shape CHECK (
                (delta_valence IS NULL AND delta_arousal IS NULL AND delta_dominance IS NULL)
                OR
                (delta_valence IS NOT NULL AND delta_arousal IS NOT NULL AND delta_dominance IS NOT NULL)
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_delta_range'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_delta_range CHECK (
                delta_valence IS NULL
                OR (
                    delta_valence BETWEEN -1.0 AND 1.0
                    AND delta_arousal BETWEEN -1.0 AND 1.0
                    AND delta_dominance BETWEEN -1.0 AND 1.0
                )
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_confidence_range'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_confidence_range
            CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0)
            NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_intensity_range'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_intensity_range
            CHECK (intensity IS NULL OR intensity BETWEEN 0.0 AND 1.0)
            NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_state_causal_context_array'
          AND conrelid = 'emotional_state'::regclass
    ) THEN
        ALTER TABLE emotional_state
            ADD CONSTRAINT emotional_state_causal_context_array
            CHECK (jsonb_typeof(causal_context) = 'array') NOT VALID;
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_emotional_state_parent
    ON emotional_state(parent_state_id) WHERE parent_state_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_emotional_state_event
    ON emotional_state(event_id) WHERE event_id IS NOT NULL;

-- Baseline provenance.  NULL on legacy rows means unknown, not zero evidence.
ALTER TABLE emotional_baseline
    ADD COLUMN IF NOT EXISTS source_window_from timestamptz;
ALTER TABLE emotional_baseline
    ADD COLUMN IF NOT EXISTS source_window_to timestamptz;
ALTER TABLE emotional_baseline
    ADD COLUMN IF NOT EXISTS sample_size integer;
ALTER TABLE emotional_baseline
    ADD COLUMN IF NOT EXISTS confidence real;
ALTER TABLE emotional_baseline
    ADD COLUMN IF NOT EXISTS source_state_id bigint;
ALTER TABLE emotional_baseline
    ADD COLUMN IF NOT EXISTS computation_version text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_baseline_source_state_same_agent'
          AND conrelid = 'emotional_baseline'::regclass
    ) THEN
        ALTER TABLE emotional_baseline
            ADD CONSTRAINT emotional_baseline_source_state_same_agent
            FOREIGN KEY (source_state_id, agent_id)
            REFERENCES emotional_state(id, agent_id)
            ON DELETE SET NULL (source_state_id);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_baseline_sample_size_nonnegative'
          AND conrelid = 'emotional_baseline'::regclass
    ) THEN
        ALTER TABLE emotional_baseline
            ADD CONSTRAINT emotional_baseline_sample_size_nonnegative
            CHECK (sample_size IS NULL OR sample_size >= 0) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'emotional_baseline_confidence_range'
          AND conrelid = 'emotional_baseline'::regclass
    ) THEN
        ALTER TABLE emotional_baseline
            ADD CONSTRAINT emotional_baseline_confidence_range
            CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0)
            NOT VALID;
    END IF;
END$$;

-- A memory keeps the state snapshot in which it entered the agent's line.
-- Existing VAD columns remain the scoring ABI; the new fields add provenance.
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS emotional_context_state_id bigint;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS emotional_context_at timestamptz;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS emotional_context_confidence real;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS emotional_context_causes jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_emotional_context_state_same_agent'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_emotional_context_state_same_agent
            FOREIGN KEY (emotional_context_state_id, agent_id)
            REFERENCES emotional_state(id, agent_id)
            ON DELETE SET NULL (emotional_context_state_id);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_emotional_context_shape'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_emotional_context_shape CHECK (
                (emotional_context_valence IS NULL
                 AND emotional_context_arousal IS NULL
                 AND emotional_context_dominance IS NULL)
                OR
                (emotional_context_valence IS NOT NULL
                 AND emotional_context_arousal IS NOT NULL
                 AND emotional_context_dominance IS NOT NULL)
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_emotional_context_range'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_emotional_context_range CHECK (
                emotional_context_valence IS NULL
                OR (
                    emotional_context_valence BETWEEN -1.0 AND 1.0
                    AND emotional_context_arousal BETWEEN -1.0 AND 1.0
                    AND emotional_context_dominance BETWEEN -1.0 AND 1.0
                )
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_emotional_context_confidence_range'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_emotional_context_confidence_range
            CHECK (
                emotional_context_confidence IS NULL
                OR emotional_context_confidence BETWEEN 0.0 AND 1.0
            ) NOT VALID;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_emotional_context_causes_array'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_emotional_context_causes_array CHECK (
                emotional_context_causes IS NULL
                OR jsonb_typeof(emotional_context_causes) = 'array'
            ) NOT VALID;
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_memories_emotional_context_state
    ON memories(emotional_context_state_id)
    WHERE emotional_context_state_id IS NOT NULL;

-- Terminal host hooks may be delivered concurrently or again after process
-- restart.  Each split dialogue part gets a stable per-turn coordinate; the
-- partial unique index is the durable idempotency barrier for diary capture.
CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_styx_sync_turn_part
    ON memories (
        agent_id,
        (metadata ->> 'styx_sync_turn_key'),
        (metadata ->> 'styx_sync_turn_role'),
        (metadata ->> 'styx_sync_turn_part')
    )
    WHERE metadata ? 'styx_sync_turn_key';
