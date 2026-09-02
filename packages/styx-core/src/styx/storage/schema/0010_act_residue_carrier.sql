-- Wave 38: durable reduction of terminal cognitive acts and an honest,
-- query-independent causal carrier.  This migration is additive.  In
-- particular, historical memories are retained but explicitly quarantined:
-- their presence alone cannot make a technical projection ready.

-- -------------------------------------------------------------------------
-- Subjective-line provenance

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS line_provenance text NOT NULL DEFAULT 'legacy_unknown';
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_ordinal smallint;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_reducer_version text;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_input_hash text;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_causal_role text;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_confidence real;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_evidence jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_predecessors jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_line_root_hash text;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS residue_affect jsonb NOT NULL DEFAULT '{}'::jsonb;

-- 0009 could only infer a domain from old writer conventions.  Do not turn
-- that inference into a claim that the row passed through a cognitive act.
UPDATE memories SET line_provenance = 'legacy_unknown'
WHERE line_provenance IS NULL OR line_provenance NOT IN (
    'legacy_unknown', 'validated_act_residue', 'validated_transform',
    'operator_attested'
);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_line_provenance_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_line_provenance_check
        CHECK (line_provenance IN (
            'legacy_unknown', 'validated_act_residue', 'validated_transform',
            'operator_attested'
        ));
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_residue_coordinates_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_residue_coordinates_check
        CHECK (
            line_provenance <> 'validated_act_residue'
            OR (
                memory_domain = 'subjective_trace'
                AND line_eligible = true
                AND cognitive_act_id IS NOT NULL
                AND residue_ordinal BETWEEN 0 AND 3
                AND length(residue_reducer_version) BETWEEN 1 AND 64
                AND length(residue_input_hash) = 64
                AND residue_input_hash ~ '^[0-9a-f]{64}$'
                AND residue_causal_role IN (
                    'choice', 'updated_belief', 'goal', 'constraint',
                    'unresolved_tension', 'affective_coordinate'
                )
                AND residue_confidence BETWEEN 0.0 AND 1.0
                AND jsonb_typeof(residue_evidence) = 'array'
                AND jsonb_array_length(residue_evidence) BETWEEN 1 AND 8
                AND jsonb_typeof(residue_predecessors) = 'array'
                AND jsonb_array_length(residue_predecessors) BETWEEN 0 AND 4
                AND length(residue_line_root_hash) = 64
                AND residue_line_root_hash ~ '^[0-9a-f]{64}$'
                AND jsonb_typeof(residue_affect) = 'object'
                AND (
                    (residue_causal_role = 'affective_coordinate')
                    = (residue_affect <> '{}'::jsonb)
                )
            )
        );
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS memories_validated_act_residue_uq
    ON memories(agent_id, cognitive_act_id, residue_reducer_version, residue_ordinal)
    WHERE line_provenance = 'validated_act_residue';

CREATE INDEX IF NOT EXISTS memories_line_provenance_idx
    ON memories(agent_id, line_provenance, seq)
    WHERE memory_domain = 'subjective_trace' AND line_eligible = true
      AND superseded_by IS NULL;

-- 0009 knew only the original line fields.  Rebuild its invalidation trigger
-- so every coordinate consumed by the causal carrier invalidates the cached
-- projection as well.  Capture time and embedding request a diagnostic cache
-- refresh, but do not advance the semantic line version.
CREATE OR REPLACE FUNCTION styx_mark_line_dirty() RETURNS trigger AS $$
DECLARE
    affected_agent text;
    relevant boolean := false;
    semantic_change boolean := false;
BEGIN
    affected_agent := COALESCE(NEW.agent_id, OLD.agent_id);
    IF TG_OP = 'INSERT' THEN
        relevant := NEW.memory_domain = 'subjective_trace' AND NEW.line_eligible;
        semantic_change := relevant;
    ELSIF TG_OP = 'DELETE' THEN
        relevant := OLD.memory_domain = 'subjective_trace' AND OLD.line_eligible;
        semantic_change := relevant;
    ELSE
        relevant :=
            (OLD.memory_domain = 'subjective_trace' AND OLD.line_eligible) OR
            (NEW.memory_domain = 'subjective_trace' AND NEW.line_eligible);
        semantic_change := relevant AND (
            OLD.seq IS DISTINCT FROM NEW.seq OR
            OLD.content IS DISTINCT FROM NEW.content OR
            OLD.superseded_by IS DISTINCT FROM NEW.superseded_by OR
            OLD.memory_domain IS DISTINCT FROM NEW.memory_domain OR
            OLD.line_eligible IS DISTINCT FROM NEW.line_eligible OR
            OLD.line_provenance IS DISTINCT FROM NEW.line_provenance OR
            OLD.cognitive_act_id IS DISTINCT FROM NEW.cognitive_act_id OR
            OLD.residue_ordinal IS DISTINCT FROM NEW.residue_ordinal OR
            OLD.residue_causal_role IS DISTINCT FROM NEW.residue_causal_role OR
            OLD.residue_predecessors IS DISTINCT FROM NEW.residue_predecessors OR
            OLD.residue_line_root_hash IS DISTINCT FROM NEW.residue_line_root_hash OR
            OLD.residue_affect IS DISTINCT FROM NEW.residue_affect
        );
        relevant := semantic_change OR (
            relevant AND (
                OLD.embedding IS DISTINCT FROM NEW.embedding OR
                OLD.created_at IS DISTINCT FROM NEW.created_at
            )
        );
    END IF;
    IF relevant THEN
        INSERT INTO line_state(agent_id, version, dirty, updated_at)
        VALUES (
            affected_agent,
            CASE WHEN semantic_change THEN 1 ELSE 0 END,
            true,
            clock_timestamp()
        )
        ON CONFLICT (agent_id) DO UPDATE
        SET version = line_state.version
                + CASE WHEN semantic_change THEN 1 ELSE 0 END,
            dirty = true,
            updated_at = clock_timestamp();
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_line_state_dirty ON memories;
CREATE TRIGGER memories_line_state_dirty
AFTER INSERT OR DELETE OR UPDATE OF seq,content,embedding,created_at,superseded_by,
    memory_domain,line_eligible,line_provenance,cognitive_act_id,
    residue_ordinal,residue_causal_role,residue_predecessors,
    residue_line_root_hash,residue_affect ON memories
FOR EACH ROW EXECUTE FUNCTION styx_mark_line_dirty();

-- The current retained causal structure is a technical storage coordinate.
-- It is not an ontological flag.  Historical/legacy lines start without a
-- validated root and remain distinguishable from reducer-produced ancestry.
ALTER TABLE line_state
    ADD COLUMN IF NOT EXISTS causal_root_hash text NOT NULL DEFAULT repeat('0', 64);
ALTER TABLE line_state
    ADD COLUMN IF NOT EXISTS causal_frontier jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE line_state
    ADD COLUMN IF NOT EXISTS causal_root_version bigint NOT NULL DEFAULT 0;
ALTER TABLE line_state
    ADD COLUMN IF NOT EXISTS causal_root_act_id uuid;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'line_state_causal_shape_check'
          AND conrelid = 'line_state'::regclass
    ) THEN
        ALTER TABLE line_state ADD CONSTRAINT line_state_causal_shape_check
        CHECK (
            length(causal_root_hash) = 64
            AND causal_root_hash ~ '^[0-9a-f]{64}$'
            AND jsonb_typeof(causal_frontier) = 'array'
            AND jsonb_array_length(causal_frontier) BETWEEN 0 AND 4
            AND causal_root_version BETWEEN 0 AND version
            AND (
                causal_root_act_id IS NOT NULL
                OR (
                    causal_root_hash = repeat('0', 64)
                    AND causal_frontier = '[]'::jsonb
                    AND causal_root_version = 0
                )
            )
        );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'line_state_causal_root_act_fk'
          AND conrelid = 'line_state'::regclass
    ) THEN
        ALTER TABLE line_state ADD CONSTRAINT line_state_causal_root_act_fk
        FOREIGN KEY (causal_root_act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id)
        DEFERRABLE INITIALLY DEFERRED;
    END IF;
END $$;

-- -------------------------------------------------------------------------
-- Per-act, per-reducer-version outcome ledger

CREATE TABLE IF NOT EXISTS cognitive_act_reductions (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id            text NOT NULL,
    act_id              uuid NOT NULL,
    reducer_version     text NOT NULL,
    outcome_version     integer NOT NULL DEFAULT 1,
    input_hash          text NOT NULL,
    status              text NOT NULL DEFAULT 'pending',
    task_id             uuid,
    attempt_count       integer NOT NULL DEFAULT 0,
    residue_count       smallint NOT NULL DEFAULT 0,
    output_line_version bigint,
    causal_root_hash    text,
    predecessor_frontier jsonb NOT NULL DEFAULT '[]'::jsonb,
    result_hash         text,
    last_error_code     text,
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    started_at          timestamptz,
    completed_at        timestamptz,
    UNIQUE (agent_id, act_id, reducer_version),
    UNIQUE (id, agent_id),
    CONSTRAINT cognitive_act_reductions_act_fk
        FOREIGN KEY (act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT cognitive_act_reductions_task_fk
        FOREIGN KEY (task_id) REFERENCES llm_tasks(id) ON DELETE SET NULL
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT cognitive_act_reductions_version_check
        CHECK (length(reducer_version) BETWEEN 1 AND 64 AND outcome_version >= 1),
    CONSTRAINT cognitive_act_reductions_hash_check
        CHECK (length(input_hash) = 64 AND input_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT cognitive_act_reductions_status_check
        CHECK (status IN (
            'pending', 'running', 'applied', 'no_residue',
            'retryable', 'terminal_failure'
        )),
    CONSTRAINT cognitive_act_reductions_counts_check
        CHECK (
            attempt_count >= 0 AND residue_count BETWEEN 0 AND 4
            AND (output_line_version IS NULL OR output_line_version >= 0)
            AND jsonb_typeof(predecessor_frontier) = 'array'
            AND jsonb_array_length(predecessor_frontier) BETWEEN 0 AND 4
        ),
    CONSTRAINT cognitive_act_reductions_root_hash_check
        CHECK (
            causal_root_hash IS NULL
            OR (length(causal_root_hash) = 64 AND causal_root_hash ~ '^[0-9a-f]{64}$')
        ),
    CONSTRAINT cognitive_act_reductions_result_hash_check
        CHECK (
            result_hash IS NULL
            OR (length(result_hash) = 64 AND result_hash ~ '^[0-9a-f]{64}$')
        ),
    CONSTRAINT cognitive_act_reductions_error_code_check
        CHECK (
            last_error_code IS NULL
            OR last_error_code ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
        ),
    CONSTRAINT cognitive_act_reductions_terminal_shape_check
        CHECK (
            (status IN ('applied', 'no_residue', 'terminal_failure'))
                = (completed_at IS NOT NULL)
        ),
    CONSTRAINT cognitive_act_reductions_residue_count_check
        CHECK (
            (status = 'applied' AND residue_count BETWEEN 1 AND 4)
            OR (status <> 'applied' AND residue_count = 0)
        ),
    CONSTRAINT cognitive_act_reductions_causal_outcome_check
        CHECK (
            status NOT IN ('applied', 'no_residue')
            OR (output_line_version IS NOT NULL AND causal_root_hash IS NOT NULL)
        )
);

ALTER TABLE cognitive_act_reductions
    ADD COLUMN IF NOT EXISTS output_line_version bigint;
ALTER TABLE cognitive_act_reductions
    ADD COLUMN IF NOT EXISTS causal_root_hash text;
ALTER TABLE cognitive_act_reductions
    ADD COLUMN IF NOT EXISTS predecessor_frontier jsonb NOT NULL DEFAULT '[]'::jsonb;

-- CREATE TABLE IF NOT EXISTS does not restore constraints removed by a
-- parent-table CASCADE in disposable test databases.  Re-add the two FKs
-- idempotently without touching ledger data.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cognitive_act_reductions_act_fk'
          AND conrelid = 'cognitive_act_reductions'::regclass
    ) THEN
        ALTER TABLE cognitive_act_reductions
        ADD CONSTRAINT cognitive_act_reductions_act_fk
        FOREIGN KEY (act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'cognitive_act_reductions_task_fk'
          AND conrelid = 'cognitive_act_reductions'::regclass
    ) THEN
        ALTER TABLE cognitive_act_reductions
        ADD CONSTRAINT cognitive_act_reductions_task_fk
        FOREIGN KEY (task_id) REFERENCES llm_tasks(id) ON DELETE SET NULL
        DEFERRABLE INITIALLY DEFERRED;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS cognitive_act_reductions_pending_idx
    ON cognitive_act_reductions(agent_id, created_at)
    WHERE status IN ('pending', 'running', 'retryable');

-- At most one live queue item can exist for one act/version, while completed
-- attempts remain as an immutable retry audit in llm_tasks.
CREATE UNIQUE INDEX IF NOT EXISTS llm_tasks_act_residue_active_uq
    ON llm_tasks (
        (payload ->> 'agent_id'),
        (payload ->> 'act_id'),
        (payload ->> 'reducer_version')
    )
    WHERE task_type = 'act_residue_reduction'
      AND status IN ('pending', 'running');

CREATE UNIQUE INDEX IF NOT EXISTS llm_tasks_act_residue_attempt_uq
    ON llm_tasks (
        (payload ->> 'agent_id'),
        (payload ->> 'act_id'),
        (payload ->> 'reducer_version'),
        ((payload ->> 'attempt_no')::integer)
    )
    WHERE task_type = 'act_residue_reduction';

-- The legacy queue has no agent_id column.  Enforce the same-agent binding by
-- checking its coordinate-only payload whenever a ledger row points at it.
CREATE OR REPLACE FUNCTION styx_validate_act_reduction_task()
RETURNS trigger AS $$
DECLARE
    task_row record;
BEGIN
    IF NEW.task_id IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT task_type, payload INTO task_row FROM llm_tasks WHERE id = NEW.task_id;
    IF NOT FOUND
       OR task_row.task_type <> 'act_residue_reduction'
       OR task_row.payload ->> 'agent_id' <> NEW.agent_id
       OR task_row.payload ->> 'act_id' <> NEW.act_id::text
       OR task_row.payload ->> 'reducer_version' <> NEW.reducer_version
       OR task_row.payload ->> 'input_hash' <> NEW.input_hash THEN
        RAISE EXCEPTION 'act reduction task coordinates do not match ledger';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cognitive_act_reductions_task_guard
    ON cognitive_act_reductions;
CREATE CONSTRAINT TRIGGER cognitive_act_reductions_task_guard
AFTER INSERT OR UPDATE
ON cognitive_act_reductions DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION styx_validate_act_reduction_task();

-- -------------------------------------------------------------------------
-- Honest carrier/status fields.  Existing projections are explicitly
-- provisional (or empty) and `formed=false`; only a future exact-coverage
-- carrier builder may promote them to ready.

ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS projection_status text NOT NULL DEFAULT 'provisional';
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS projection_available boolean NOT NULL DEFAULT false;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS covered_line_version bigint NOT NULL DEFAULT 0;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS coverage_count integer NOT NULL DEFAULT 0;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS coverage_hash text NOT NULL DEFAULT repeat('0', 64);
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS carrier_text text;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS carrier_payload jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS carrier_version text;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS pending_reduction_count integer NOT NULL DEFAULT 0;
ALTER TABLE will_projections
    ADD COLUMN IF NOT EXISTS carrier_built_at timestamptz;

-- Keep the additive migration compatible with pre-wave-38 writers.  Such a
-- writer may still submit formed=true while omitting carrier fields; storage
-- downgrades that row to provisional instead of either rejecting the write or
-- accepting the legacy boolean as proof of readiness.
CREATE OR REPLACE FUNCTION styx_enforce_projection_readiness()
RETURNS trigger AS $$
DECLARE
    exact_ready boolean;
BEGIN
    exact_ready :=
        NEW.projection_status = 'ready'
        AND NEW.projection_available
        AND NEW.covered_line_version = NEW.line_version
        AND NEW.coverage_count = NEW.source_count
        AND NEW.coverage_hash = NEW.source_hash
        AND NEW.carrier_text IS NOT NULL
        AND length(NEW.carrier_text) > 0
        AND NEW.carrier_version IS NOT NULL;
    NEW.formed := exact_ready;
    IF NOT NEW.projection_available
       AND NEW.projection_status NOT IN ('degraded', 'stale') THEN
        NEW.projection_status := CASE
            WHEN NEW.source_count = 0 THEN 'empty'
            ELSE 'provisional'
        END;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS will_projections_readiness_guard ON will_projections;
CREATE TRIGGER will_projections_readiness_guard
BEFORE INSERT OR UPDATE ON will_projections
FOR EACH ROW EXECUTE FUNCTION styx_enforce_projection_readiness();

UPDATE will_projections
SET projection_status = CASE WHEN source_count = 0 THEN 'empty' ELSE 'provisional' END,
    projection_available = false,
    covered_line_version = 0,
    coverage_count = 0,
    coverage_hash = repeat('0', 64),
    carrier_text = NULL,
    carrier_payload = '{}'::jsonb,
    carrier_version = NULL,
    pending_reduction_count = 0,
    carrier_built_at = NULL,
    formed = false;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'will_projections_status_check'
          AND conrelid = 'will_projections'::regclass
    ) THEN
        ALTER TABLE will_projections ADD CONSTRAINT will_projections_status_check
        CHECK (projection_status IN ('empty', 'provisional', 'ready', 'stale', 'degraded'));
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'will_projections_carrier_bounds_check'
          AND conrelid = 'will_projections'::regclass
    ) THEN
        ALTER TABLE will_projections ADD CONSTRAINT will_projections_carrier_bounds_check
        CHECK (
            covered_line_version >= 0
            AND coverage_count >= 0
            AND pending_reduction_count >= 0
            AND length(coverage_hash) = 64
            AND coverage_hash ~ '^[0-9a-f]{64}$'
            AND (carrier_text IS NULL OR length(carrier_text) <= 16000)
            AND (carrier_version IS NULL OR length(carrier_version) BETWEEN 1 AND 64)
        );
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'will_projections_ready_shape_check'
          AND conrelid = 'will_projections'::regclass
    ) THEN
        ALTER TABLE will_projections ADD CONSTRAINT will_projections_ready_shape_check
        CHECK (
            formed = (
                projection_status = 'ready'
                AND projection_available
                AND covered_line_version = line_version
                AND coverage_count = source_count
                AND coverage_hash = source_hash
                AND carrier_text IS NOT NULL
                AND length(carrier_text) > 0
                AND carrier_version IS NOT NULL
            )
            AND projection_available = (
                carrier_text IS NOT NULL AND carrier_version IS NOT NULL
            )
        );
    END IF;
END $$;

COMMENT ON COLUMN will_projections.formed IS
    'Deprecated technical-readiness alias; never a claim of will, selfhood, personality, or consciousness.';
COMMENT ON COLUMN will_projections.carrier_text IS
    'Bounded lossy technical projection of the covered eligible line, independent of recall query.';
