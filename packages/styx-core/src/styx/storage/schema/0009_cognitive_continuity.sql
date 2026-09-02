-- Wave 37: cognitive continuity is an additive, auditable projection.
--
-- The schema deliberately separates captured dialogue/external evidence from
-- the agent's subjective line.  Existing API writers remain valid, while all
-- recall/will readers can enforce the explicit boundary.

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS memory_domain text NOT NULL DEFAULT 'subjective_trace';
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS line_eligible boolean NOT NULL DEFAULT true;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS cognitive_act_id uuid;

UPDATE memories
SET memory_domain = CASE
        WHEN role IN ('user', 'assistant', 'tool') THEN 'dialogue'
        WHEN kind_src = 'experience_intake' THEN 'external_evidence'
        ELSE 'subjective_trace'
    END,
    line_eligible = CASE
        WHEN role IN ('user', 'assistant', 'tool') THEN false
        WHEN kind_src = 'experience_intake' THEN false
        ELSE true
    END;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_memory_domain_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_memory_domain_check
        CHECK (memory_domain IN ('dialogue', 'external_evidence', 'subjective_trace'));
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_line_eligibility_domain_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_line_eligibility_domain_check
        CHECK (NOT line_eligible OR memory_domain = 'subjective_trace');
    END IF;
END $$;

-- The storage boundary, not merely the Python writer, owns domain safety.
-- Coerce archive/dialogue-shaped rows before the CHECKs below so legacy and
-- direct SQL writers remain usable without ever creating a subjective trace.
CREATE OR REPLACE FUNCTION styx_enforce_memory_domain() RETURNS trigger AS $$
BEGIN
    IF NEW.role IN ('user', 'assistant', 'tool') THEN
        NEW.memory_domain := 'dialogue';
        NEW.line_eligible := false;
    ELSIF NEW.kind_src = 'experience_intake' THEN
        NEW.memory_domain := 'external_evidence';
        NEW.line_eligible := false;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_domain_guard ON memories;
CREATE TRIGGER memories_domain_guard
BEFORE INSERT OR UPDATE OF role, kind_src, memory_domain, line_eligible ON memories
FOR EACH ROW EXECUTE FUNCTION styx_enforce_memory_domain();

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_raw_dialogue_domain_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_raw_dialogue_domain_check
        CHECK (
            role NOT IN ('user', 'assistant', 'tool') OR
            (memory_domain = 'dialogue' AND line_eligible = false)
        );
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_experience_domain_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_experience_domain_check
        CHECK (
            kind_src <> 'experience_intake' OR
            (memory_domain = 'external_evidence' AND line_eligible = false)
        );
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS memories_id_agent_uq ON memories(id, agent_id);

-- Session identity is the host UUID owned by an agent.  Earlier schemas made
-- UUID globally unique, which caused bootstrap/dialogue to disagree with the
-- cognition-only UUID5 workaround.  Upgrade every session FK to the same
-- composite ownership model and retain the physical host UUID everywhere.
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_session_id_fkey;
ALTER TABLE recall_events DROP CONSTRAINT IF EXISTS recall_events_session_id_fkey;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS agent_id text;
UPDATE recall_events AS re
SET agent_id = m.agent_id
FROM memories AS m
WHERE re.memory_id = m.id AND re.agent_id IS NULL;
ALTER TABLE recall_events ALTER COLUMN agent_id SET NOT NULL;

-- Legacy/direct writers historically supplied only memory_id. Resolve the
-- owning agent at the storage boundary before NOT NULL/FK checks, and ignore a
-- contradictory caller value rather than permitting cross-agent attribution.
CREATE OR REPLACE FUNCTION styx_enforce_recall_event_agent() RETURNS trigger AS $$
BEGIN
    SELECT m.agent_id INTO NEW.agent_id
    FROM memories AS m WHERE m.id = NEW.memory_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS recall_events_agent_guard ON recall_events;
CREATE TRIGGER recall_events_agent_guard
BEFORE INSERT OR UPDATE OF memory_id, agent_id ON recall_events
FOR EACH ROW EXECUTE FUNCTION styx_enforce_recall_event_agent();

ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_pkey;
ALTER TABLE sessions ADD CONSTRAINT sessions_pkey PRIMARY KEY (id, agent_id);
ALTER TABLE memories ADD CONSTRAINT memories_session_owner_fk
    FOREIGN KEY (session_id, agent_id) REFERENCES sessions(id, agent_id)
    ON DELETE SET NULL (session_id) DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE recall_events ADD CONSTRAINT recall_events_session_owner_fk
    FOREIGN KEY (session_id, agent_id) REFERENCES sessions(id, agent_id)
    ON DELETE SET NULL (session_id) DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX IF NOT EXISTS memories_subjective_line_idx
    ON memories(agent_id, seq)
    WHERE memory_domain = 'subjective_trace'
      AND line_eligible = true AND superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS cognitive_acts (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id            text NOT NULL,
    host_key            text NOT NULL,
    session_id          uuid,
    declared_parent_key text,
    parent_act_id       uuid,
    projection_scope    text NOT NULL DEFAULT 'agent_line',
    input_line_version  bigint NOT NULL DEFAULT 0,
    input_snapshot_token text,
    status              text NOT NULL DEFAULT 'completed',
    channel_input       jsonb NOT NULL DEFAULT '{}'::jsonb,
    channel_output      jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata            jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    completed_at        timestamptz,
    UNIQUE (agent_id, host_key),
    UNIQUE (id, agent_id),
    CONSTRAINT cognitive_acts_host_key_length CHECK (length(host_key) BETWEEN 1 AND 512),
    CONSTRAINT cognitive_acts_parent_key_length CHECK (
        declared_parent_key IS NULL OR length(declared_parent_key) BETWEEN 1 AND 512
    ),
    CONSTRAINT cognitive_acts_status_check CHECK (status IN ('completed', 'failed')),
    CONSTRAINT cognitive_acts_not_self_parent_check CHECK (
        declared_parent_key IS NULL OR declared_parent_key <> host_key
    ),
    CONSTRAINT cognitive_acts_session_fk FOREIGN KEY (session_id, agent_id)
        REFERENCES sessions(id, agent_id) ON DELETE SET NULL (session_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT cognitive_acts_parent_fk FOREIGN KEY (parent_act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) DEFERRABLE INITIALLY DEFERRED
);

ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_cognitive_act_fk;
ALTER TABLE memories ADD CONSTRAINT memories_cognitive_act_fk
    FOREIGN KEY (cognitive_act_id, agent_id)
    REFERENCES cognitive_acts(id, agent_id) DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX IF NOT EXISTS cognitive_acts_parent_key_idx
    ON cognitive_acts(agent_id, declared_parent_key);

CREATE TABLE IF NOT EXISTS cognitive_snapshots (
    token        text PRIMARY KEY,
    agent_id     text NOT NULL,
    session_id   uuid,
    host_key     text,
    request_hash text,
    line_version bigint NOT NULL,
    response_payload jsonb,
    used_by_act_id uuid,
    created_at   timestamptz NOT NULL DEFAULT clock_timestamp(),
    lease_expires_at timestamptz NOT NULL DEFAULT (
        clock_timestamp() + interval '60 seconds'
    ),
    presentation_completed_at timestamptz,
    used_at      timestamptz,
    UNIQUE (token, agent_id),
    CONSTRAINT cognitive_snapshots_token_length CHECK (length(token) BETWEEN 1 AND 128),
    CONSTRAINT cognitive_snapshots_host_key_length CHECK (
        host_key IS NULL OR length(host_key) BETWEEN 1 AND 512
    ),
    CONSTRAINT cognitive_snapshots_request_hash_length CHECK (
        request_hash IS NULL OR length(request_hash) = 64
    ),
    CONSTRAINT cognitive_snapshots_line_version_check CHECK (line_version >= 0),
    CONSTRAINT cognitive_snapshots_session_fk FOREIGN KEY (session_id, agent_id)
        REFERENCES sessions(id, agent_id) ON DELETE SET NULL (session_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT cognitive_snapshots_used_fk FOREIGN KEY (used_by_act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS cognitive_snapshots_agent_created_idx
    ON cognitive_snapshots(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS cognitive_snapshots_session_created_idx
    ON cognitive_snapshots(agent_id, session_id, created_at DESC)
    WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS cognitive_snapshots_agent_host_uq
    ON cognitive_snapshots(agent_id, host_key) WHERE host_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS cognitive_actions (
    id          bigserial PRIMARY KEY,
    agent_id    text NOT NULL,
    act_id      uuid NOT NULL,
    ordinal     integer NOT NULL,
    kind        text NOT NULL,
    event_id    text NOT NULL DEFAULT '',
    name        text NOT NULL DEFAULT '',
    content     text NOT NULL DEFAULT '',
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (act_id, ordinal),
    CONSTRAINT cognitive_actions_kind_check CHECK (kind IN ('call', 'result', 'error')),
    CONSTRAINT cognitive_actions_ordinal_check CHECK (ordinal >= 0),
    CONSTRAINT cognitive_actions_content_length CHECK (length(content) <= 8000),
    CONSTRAINT cognitive_actions_act_fk FOREIGN KEY (act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS cognitive_consequences (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id                text NOT NULL,
    act_id                  uuid NOT NULL,
    ordinal                 integer NOT NULL,
    kind                    text NOT NULL,
    content                 text NOT NULL,
    metadata                jsonb NOT NULL DEFAULT '{}'::jsonb,
    status                  text NOT NULL DEFAULT 'pending',
    presented_snapshot_token text,
    presented_at            timestamptz,
    acknowledged_by_act_id  uuid,
    acknowledged_at         timestamptz,
    created_at              timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (act_id, ordinal),
    UNIQUE (id, agent_id),
    CONSTRAINT cognitive_consequences_status_check
        CHECK (status IN ('pending', 'presented', 'acknowledged')),
    CONSTRAINT cognitive_consequences_content_length CHECK (length(content) <= 8000),
    CONSTRAINT cognitive_consequences_act_fk FOREIGN KEY (act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT cognitive_consequences_ack_fk
        FOREIGN KEY (acknowledged_by_act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS cognitive_consequences_inbox_idx
    ON cognitive_consequences(agent_id, created_at, ordinal)
    WHERE status <> 'acknowledged';

-- Immutable evidence that a consequence was included in a particular
-- physical snapshot.  Expiry releases delivery to a new snapshot without
-- rebinding or deleting the old association, so a late commit cannot steal a
-- newer presentation.
CREATE TABLE IF NOT EXISTS cognitive_presentations (
    snapshot_token   text NOT NULL,
    consequence_id   uuid NOT NULL,
    agent_id          text NOT NULL,
    lease_expires_at  timestamptz NOT NULL,
    presented_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (snapshot_token, consequence_id),
    CONSTRAINT cognitive_presentations_snapshot_fk
        FOREIGN KEY (snapshot_token, agent_id)
        REFERENCES cognitive_snapshots(token, agent_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT cognitive_presentations_consequence_fk
        FOREIGN KEY (consequence_id, agent_id)
        REFERENCES cognitive_consequences(id, agent_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX IF NOT EXISTS cognitive_presentations_active_idx
    ON cognitive_presentations(agent_id, consequence_id, lease_expires_at DESC);

CREATE TABLE IF NOT EXISTS memory_lineage (
    id               bigserial PRIMARY KEY,
    agent_id         text NOT NULL,
    source_memory_id uuid,
    target_memory_id uuid,
    cognitive_act_id uuid,
    transform        text NOT NULL,
    ordinal          integer NOT NULL DEFAULT 0,
    retained_weight  real,
    source_coordinates jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT memory_lineage_transform_check CHECK (
        transform IN ('incorporated', 'consolidated', 'reinterpreted',
                      'superseded', 'forgotten_rewire', 'provenance_unknown')
    ),
    CONSTRAINT memory_lineage_source_fk FOREIGN KEY (source_memory_id, agent_id)
        REFERENCES memories(id, agent_id) ON DELETE SET NULL (source_memory_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT memory_lineage_target_fk FOREIGN KEY (target_memory_id, agent_id)
        REFERENCES memories(id, agent_id) ON DELETE SET NULL (target_memory_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT memory_lineage_act_fk FOREIGN KEY (cognitive_act_id, agent_id)
        REFERENCES cognitive_acts(id, agent_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS memory_lineage_source_idx
    ON memory_lineage(agent_id, source_memory_id, created_at);
CREATE INDEX IF NOT EXISTS memory_lineage_target_idx
    ON memory_lineage(agent_id, target_memory_id, created_at);

CREATE TABLE IF NOT EXISTS line_state (
    agent_id    text PRIMARY KEY,
    version     bigint NOT NULL DEFAULT 0,
    dirty       boolean NOT NULL DEFAULT true,
    updated_at  timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS will_projections (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id            text NOT NULL,
    line_version        bigint NOT NULL,
    formed              boolean NOT NULL,
    source_count        integer NOT NULL,
    source_hash         text NOT NULL,
    embedding           vector(768),
    supports            jsonb NOT NULL DEFAULT '[]'::jsonb,
    computation_version text NOT NULL DEFAULT 'technical_projection_v1',
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (agent_id, line_version),
    CONSTRAINT will_projections_source_count_check CHECK (source_count >= 0)
);

INSERT INTO line_state(agent_id, version, dirty)
SELECT agent_id, 1, true
FROM memories
WHERE memory_domain = 'subjective_trace' AND line_eligible = true
GROUP BY agent_id
ON CONFLICT (agent_id) DO NOTHING;

CREATE OR REPLACE FUNCTION styx_mark_line_dirty() RETURNS trigger AS $$
DECLARE
    affected_agent text;
    relevant boolean := false;
BEGIN
    affected_agent := COALESCE(NEW.agent_id, OLD.agent_id);
    IF TG_OP = 'INSERT' THEN
        relevant := NEW.memory_domain = 'subjective_trace' AND NEW.line_eligible;
    ELSIF TG_OP = 'DELETE' THEN
        relevant := OLD.memory_domain = 'subjective_trace' AND OLD.line_eligible;
    ELSE
        relevant :=
            (OLD.memory_domain = 'subjective_trace' AND OLD.line_eligible) OR
            (NEW.memory_domain = 'subjective_trace' AND NEW.line_eligible);
        relevant := relevant AND (
            OLD.content IS DISTINCT FROM NEW.content OR
            OLD.embedding IS DISTINCT FROM NEW.embedding OR
            OLD.superseded_by IS DISTINCT FROM NEW.superseded_by OR
            OLD.memory_domain IS DISTINCT FROM NEW.memory_domain OR
            OLD.line_eligible IS DISTINCT FROM NEW.line_eligible
        );
    END IF;
    IF relevant THEN
        INSERT INTO line_state(agent_id, version, dirty, updated_at)
        VALUES (affected_agent, 1, true, clock_timestamp())
        ON CONFLICT (agent_id) DO UPDATE
        SET version = line_state.version + 1,
            dirty = true,
            updated_at = clock_timestamp();
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_line_state_dirty ON memories;
CREATE TRIGGER memories_line_state_dirty
AFTER INSERT OR DELETE OR UPDATE OF content, embedding, superseded_by,
    memory_domain, line_eligible ON memories
FOR EACH ROW EXECUTE FUNCTION styx_mark_line_dirty();
