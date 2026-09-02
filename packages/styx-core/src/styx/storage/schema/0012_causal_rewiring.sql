-- Wave 40: versioned causal graph, operation ledger and tombstones.
--
-- Canonical semantic changes are append-only.  Existing legacy memories and
-- lineage remain readable, but are quarantined from the validated graph until
-- an explicit causal operation creates/attests their canonical coordinates.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE memories ADD COLUMN IF NOT EXISTS causal_node_hash text;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS causal_node_kind text NOT NULL DEFAULT 'legacy';
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS causal_payload_version text NOT NULL DEFAULT 'legacy_v0';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS causal_operation_id uuid;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS line_status text NOT NULL DEFAULT 'quarantined';

-- Existing Wave 38 residues are already reducer-validated.  Give them a
-- stable, content-addressed backfill without treating UUID/timestamp/embedding
-- as semantic material.  Pre-Wave-40 transforms remain quarantined because
-- their predecessor coverage was not atomically verified.
UPDATE memories
SET causal_node_kind = 'act_residue',
    causal_payload_version = 'causal_backfill_v1',
    causal_node_hash = encode(
        digest(
            convert_to(
                concat_ws(E'\x1f', 'causal_backfill_v1', line_provenance,
                    coalesce(residue_causal_role, ''), content),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    ),
    line_status = CASE
        WHEN memory_domain = 'subjective_trace' AND line_eligible
             AND superseded_by IS NULL THEN 'active'
        WHEN superseded_by IS NOT NULL THEN 'superseded'
        ELSE 'quarantined'
    END
WHERE line_provenance = 'validated_act_residue'
  AND causal_node_hash IS NULL;

UPDATE memories
SET causal_node_kind = 'legacy',
    causal_payload_version = 'legacy_transform_v0',
    line_status = 'quarantined'
WHERE line_provenance = 'validated_transform'
  AND causal_node_hash IS NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_causal_node_hash_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_causal_node_hash_check
        CHECK (
            causal_node_hash IS NULL OR
            (length(causal_node_hash) = 64
             AND causal_node_hash ~ '^[0-9a-f]{64}$')
        );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_causal_node_kind_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_causal_node_kind_check
        CHECK (causal_node_kind IN (
            'legacy', 'act_residue', 'consolidation', 'reinterpretation',
            'carrier_reduction'
        ));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_line_status_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_line_status_check
        CHECK (line_status IN (
            'active', 'superseded', 'forgotten', 'quarantined'
        ));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_causal_node_shape_check'
          AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_causal_node_shape_check
        CHECK (
            line_provenance NOT IN (
                'validated_act_residue', 'validated_transform'
            ) OR line_status = 'quarantined' OR (
                causal_node_hash IS NOT NULL
                AND length(causal_payload_version) BETWEEN 1 AND 64
                AND causal_node_kind <> 'legacy'
                AND (line_provenance <> 'validated_transform'
                     OR causal_operation_id IS NOT NULL)
            )
        );
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS causal_operations (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id            text NOT NULL,
    operation_key       text NOT NULL,
    operation_kind      text NOT NULL,
    input_line_version  bigint NOT NULL,
    input_root_hash     text NOT NULL,
    request_hash        text NOT NULL,
    algorithm_name      text NOT NULL,
    algorithm_version   text NOT NULL,
    status              text NOT NULL DEFAULT 'pending',
    output_line_version bigint,
    output_root_hash    text,
    source_count        integer NOT NULL DEFAULT 0,
    target_count        integer NOT NULL DEFAULT 0,
    error_code          text,
    feature_coordinates jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    applied_at          timestamptz,
    UNIQUE (agent_id, operation_key),
    UNIQUE (id, agent_id),
    CONSTRAINT causal_operations_key_check
        CHECK (length(operation_key) BETWEEN 1 AND 512),
    CONSTRAINT causal_operations_kind_check CHECK (operation_kind IN (
        'consolidate', 'reinterpret', 'forget', 'rewire', 'carrier_reduce'
    )),
    CONSTRAINT causal_operations_status_check CHECK (status IN (
        'pending', 'running', 'applied', 'noop', 'retryable',
        'terminal_failure'
    )),
    CONSTRAINT causal_operations_version_check CHECK (
        input_line_version >= 0
        AND (output_line_version IS NULL OR output_line_version >= 0)
    ),
    CONSTRAINT causal_operations_hash_check CHECK (
        length(input_root_hash) = 64
        AND input_root_hash ~ '^[0-9a-f]{64}$'
        AND length(request_hash) = 64
        AND request_hash ~ '^[0-9a-f]{64}$'
        AND (
            output_root_hash IS NULL OR
            (length(output_root_hash) = 64
             AND output_root_hash ~ '^[0-9a-f]{64}$')
        )
    ),
    CONSTRAINT causal_operations_counts_check CHECK (
        source_count >= 0 AND target_count >= 0
    ),
    CONSTRAINT causal_operations_terminal_check CHECK (
        (status IN ('applied', 'noop', 'terminal_failure'))
        = (applied_at IS NOT NULL)
    ),
    CONSTRAINT causal_operations_output_check CHECK (
        status NOT IN ('applied', 'noop') OR
        (output_line_version IS NOT NULL AND output_root_hash IS NOT NULL)
    ),
    CONSTRAINT causal_operations_error_check CHECK (
        error_code IS NULL OR error_code ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    )
);

CREATE INDEX IF NOT EXISTS causal_operations_pending_idx
    ON causal_operations(agent_id, created_at)
    WHERE status IN ('pending', 'running', 'retryable');

ALTER TABLE line_state ADD COLUMN IF NOT EXISTS causal_root_operation_id uuid;
ALTER TABLE line_state
    DROP CONSTRAINT IF EXISTS line_state_causal_root_operation_fk;
ALTER TABLE line_state ADD CONSTRAINT line_state_causal_root_operation_fk
    FOREIGN KEY (causal_root_operation_id, agent_id)
    REFERENCES causal_operations(id, agent_id)
    DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE line_state DROP CONSTRAINT IF EXISTS line_state_causal_shape_check;
ALTER TABLE line_state ADD CONSTRAINT line_state_causal_shape_check CHECK (
    length(causal_root_hash) = 64
    AND causal_root_hash ~ '^[0-9a-f]{64}$'
    AND jsonb_typeof(causal_frontier) = 'array'
    AND jsonb_array_length(causal_frontier) BETWEEN 0 AND 64
    AND causal_root_version BETWEEN 0 AND version
    AND NOT (
        causal_root_act_id IS NOT NULL
        AND causal_root_operation_id IS NOT NULL
    )
    AND (
        causal_root_act_id IS NOT NULL
        OR causal_root_operation_id IS NOT NULL
        OR (
            causal_root_hash = repeat('0', 64)
            AND causal_frontier = '[]'::jsonb
            AND causal_root_version = 0
        )
    )
);

ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_causal_operation_fk;
ALTER TABLE memories ADD CONSTRAINT memories_causal_operation_fk
    FOREIGN KEY (causal_operation_id, agent_id)
    REFERENCES causal_operations(id, agent_id)
    DEFERRABLE INITIALLY DEFERRED;

-- Branches created by late parent resolution can temporarily widen the
-- graph frontier beyond one reducer result (which is still bounded to four
-- residues).  The next act may merge that complete frontier.
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_residue_coordinates_check;
ALTER TABLE memories ADD CONSTRAINT memories_residue_coordinates_check CHECK (
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
        AND jsonb_array_length(residue_predecessors) BETWEEN 0 AND 64
        AND length(residue_line_root_hash) = 64
        AND residue_line_root_hash ~ '^[0-9a-f]{64}$'
        AND jsonb_typeof(residue_affect) = 'object'
        AND ((residue_causal_role = 'affective_coordinate')
             = (residue_affect <> '{}'::jsonb))
    )
);

ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS edge_key text;
ALTER TABLE memory_lineage
    ADD COLUMN IF NOT EXISTS edge_provenance text NOT NULL DEFAULT 'legacy_unknown';
ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS operation_id uuid;
ALTER TABLE memory_lineage
    ADD COLUMN IF NOT EXISTS relation_version integer NOT NULL DEFAULT 0;
ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS source_node_hash text;
ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS target_node_hash text;
ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS valid_from_line_version bigint;
ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS valid_to_line_version bigint;
ALTER TABLE memory_lineage ADD COLUMN IF NOT EXISTS edge_hash text;

ALTER TABLE memory_lineage
    DROP CONSTRAINT IF EXISTS memory_lineage_transform_check;
ALTER TABLE memory_lineage ADD CONSTRAINT memory_lineage_transform_check CHECK (
    transform IN ('incorporated', 'consolidated', 'reinterpreted',
                  'retained_rewire', 'superseded', 'forgotten_rewire',
                  'provenance_unknown')
);

ALTER TABLE memory_lineage DROP CONSTRAINT IF EXISTS memory_lineage_operation_fk;
ALTER TABLE memory_lineage ADD CONSTRAINT memory_lineage_operation_fk
    FOREIGN KEY (operation_id, agent_id)
    REFERENCES causal_operations(id, agent_id)
    DEFERRABLE INITIALLY DEFERRED;

-- Backfill only the predecessor edges already produced by the Wave 38
-- reducer.  NULL-source root markers remain legacy audit rows, not DAG edges.
UPDATE memory_lineage AS edge
SET edge_key = concat('backfill:', edge.id),
    edge_provenance = 'validated',
    relation_version = 1,
    source_node_hash = source.causal_node_hash,
    target_node_hash = target.causal_node_hash,
    valid_from_line_version = greatest(1, target.seq),
    edge_hash = encode(
        digest(
            convert_to(
                concat_ws(E'\x1f', 'causal_edge_v1', '1',
                    source.causal_node_hash, target.causal_node_hash,
                    edge.transform),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    )
FROM memories AS source, memories AS target
WHERE edge.agent_id = source.agent_id
  AND edge.source_memory_id = source.id
  AND edge.agent_id = target.agent_id
  AND edge.target_memory_id = target.id
  AND edge.transform = 'incorporated'
  AND source.line_provenance = 'validated_act_residue'
  AND target.line_provenance = 'validated_act_residue'
  AND edge.edge_provenance = 'legacy_unknown';

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memory_lineage_edge_provenance_check'
          AND conrelid = 'memory_lineage'::regclass
    ) THEN
        ALTER TABLE memory_lineage
        ADD CONSTRAINT memory_lineage_edge_provenance_check
        CHECK (edge_provenance IN ('validated', 'legacy_unknown'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memory_lineage_canonical_shape_check'
          AND conrelid = 'memory_lineage'::regclass
    ) THEN
        ALTER TABLE memory_lineage
        ADD CONSTRAINT memory_lineage_canonical_shape_check CHECK (
            edge_provenance <> 'validated' OR (
                length(edge_key) BETWEEN 1 AND 512
                AND relation_version >= 1
                AND length(source_node_hash) = 64
                AND source_node_hash ~ '^[0-9a-f]{64}$'
                AND length(target_node_hash) = 64
                AND target_node_hash ~ '^[0-9a-f]{64}$'
                AND valid_from_line_version IS NOT NULL
                AND valid_from_line_version >= 0
                AND (valid_to_line_version IS NULL OR
                     valid_to_line_version >= valid_from_line_version)
                AND length(edge_hash) = 64
                AND edge_hash ~ '^[0-9a-f]{64}$'
                AND (
                    valid_to_line_version IS NOT NULL
                    OR (
                        source_memory_id IS NOT NULL
                        AND target_memory_id IS NOT NULL
                        AND source_memory_id <> target_memory_id
                    )
                )
            )
        );
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS memory_lineage_edge_key_uq
    ON memory_lineage(agent_id, edge_key) WHERE edge_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS memory_lineage_active_edge_uq
    ON memory_lineage(agent_id, source_memory_id, target_memory_id, transform)
    WHERE edge_provenance = 'validated' AND valid_to_line_version IS NULL;
CREATE INDEX IF NOT EXISTS memory_lineage_active_source_idx
    ON memory_lineage(agent_id, source_memory_id, target_memory_id)
    WHERE edge_provenance = 'validated' AND valid_to_line_version IS NULL;
CREATE INDEX IF NOT EXISTS memory_lineage_active_target_idx
    ON memory_lineage(agent_id, target_memory_id, source_memory_id)
    WHERE edge_provenance = 'validated' AND valid_to_line_version IS NULL;

-- Normalize the Wave 38 root/cache coordinates to the content-addressed graph
-- algorithm introduced above.  This is deliberately reproducible in SQL:
-- semantic rows are sorted byte-for-byte before hashing and contain neither
-- database identifiers nor clocks.  Only active canonical endpoints
-- participate in the current root; superseded nodes and closed edges remain
-- available as ledger history.
WITH active_nodes AS (
    SELECT id, agent_id, causal_node_hash, cognitive_act_id, seq
    FROM memories
    WHERE line_provenance IN (
              'validated_act_residue', 'validated_transform'
          )
      AND line_status = 'active'
      AND causal_node_hash IS NOT NULL
), semantic_items AS (
    SELECT agent_id,
           concat_ws(E'\x1f', 'N', causal_node_hash, 'active') AS item
    FROM active_nodes
    UNION ALL
    SELECT edge.agent_id,
           concat_ws(E'\x1f', 'E', source.causal_node_hash,
                     target.causal_node_hash, edge.transform) AS item
    FROM memory_lineage AS edge
    JOIN active_nodes AS source
      ON source.agent_id = edge.agent_id
     AND source.id = edge.source_memory_id
    JOIN active_nodes AS target
      ON target.agent_id = edge.agent_id
     AND target.id = edge.target_memory_id
    WHERE edge.edge_provenance = 'validated'
      AND edge.valid_to_line_version IS NULL
), graph_roots AS (
    SELECT agent_id,
           encode(
               digest(
                   convert_to(
                       'causal_graph_v1' || E'\n' ||
                       string_agg(item, E'\n' ORDER BY item),
                       'UTF8'
                   ),
                   'sha256'
               ),
               'hex'
           ) AS root_hash
    FROM semantic_items
    GROUP BY agent_id
), graph_frontiers AS (
    SELECT node.agent_id,
           jsonb_agg(node.id::text ORDER BY node.id::text) AS frontier
    FROM active_nodes AS node
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_lineage AS edge
        JOIN active_nodes AS target
          ON target.agent_id = edge.agent_id
         AND target.id = edge.target_memory_id
        WHERE edge.agent_id = node.agent_id
          AND edge.source_memory_id = node.id
          AND edge.edge_provenance = 'validated'
          AND edge.valid_to_line_version IS NULL
    )
    GROUP BY node.agent_id
), latest_acts AS (
    SELECT DISTINCT ON (agent_id) agent_id, cognitive_act_id
    FROM active_nodes
    WHERE cognitive_act_id IS NOT NULL
    ORDER BY agent_id, seq DESC, id::text DESC
)
UPDATE line_state AS state
SET causal_root_hash = roots.root_hash,
    causal_frontier = frontier.frontier,
    causal_root_version = state.version,
    causal_root_act_id = coalesce(
        state.causal_root_act_id, latest.cognitive_act_id
    ),
    causal_root_operation_id = NULL,
    dirty = true,
    updated_at = clock_timestamp()
FROM graph_roots AS roots
JOIN graph_frontiers AS frontier USING (agent_id)
JOIN latest_acts AS latest USING (agent_id)
WHERE state.agent_id = roots.agent_id;

-- Carrier v1 coverage hashes are not comparable with the graph-aware v2
-- contract.  Keep the historical row, but make its non-readiness explicit;
-- the regular projection builder will replace it under the line lock.
UPDATE will_projections
SET projection_status = CASE
        WHEN source_count = 0 THEN 'empty' ELSE 'provisional' END,
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

UPDATE line_state
SET dirty = true,
    updated_at = clock_timestamp();

CREATE TABLE IF NOT EXISTS memory_tombstones (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id              text NOT NULL,
    memory_id             uuid NOT NULL,
    causal_node_hash      text NOT NULL,
    content_hash          text NOT NULL,
    operation_id          uuid NOT NULL,
    reason_code           text NOT NULL,
    removed_line_version  bigint NOT NULL,
    predecessor_hashes    jsonb NOT NULL DEFAULT '[]'::jsonb,
    successor_hashes      jsonb NOT NULL DEFAULT '[]'::jsonb,
    rewire_hash           text NOT NULL,
    created_at            timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (agent_id, memory_id),
    CONSTRAINT memory_tombstones_operation_fk
        FOREIGN KEY (operation_id, agent_id)
        REFERENCES causal_operations(id, agent_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT memory_tombstones_hash_check CHECK (
        length(causal_node_hash) = 64
        AND causal_node_hash ~ '^[0-9a-f]{64}$'
        AND length(content_hash) = 64
        AND content_hash ~ '^[0-9a-f]{64}$'
        AND length(rewire_hash) = 64
        AND rewire_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT memory_tombstones_shape_check CHECK (
        removed_line_version >= 0
        AND reason_code ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
        AND jsonb_typeof(predecessor_hashes) = 'array'
        AND jsonb_typeof(successor_hashes) = 'array'
    )
);

CREATE INDEX IF NOT EXISTS memory_tombstones_operation_idx
    ON memory_tombstones(agent_id, operation_id);

CREATE OR REPLACE FUNCTION styx_immutable_causal_node()
RETURNS trigger AS $$
BEGIN
    IF OLD.causal_node_hash IS NOT NULL
       AND OLD.line_provenance IN (
           'validated_act_residue', 'validated_transform'
       ) AND (
        NEW.agent_id IS DISTINCT FROM OLD.agent_id OR
        NEW.content IS DISTINCT FROM OLD.content OR
        NEW.role IS DISTINCT FROM OLD.role OR
        NEW.kind IS DISTINCT FROM OLD.kind OR
        NEW.kind_src IS DISTINCT FROM OLD.kind_src OR
        NEW.metadata IS DISTINCT FROM OLD.metadata OR
        NEW.memory_domain IS DISTINCT FROM OLD.memory_domain OR
        NEW.line_eligible IS DISTINCT FROM OLD.line_eligible OR
        NEW.cognitive_act_id IS DISTINCT FROM OLD.cognitive_act_id OR
        NEW.line_provenance IS DISTINCT FROM OLD.line_provenance OR
        NEW.residue_ordinal IS DISTINCT FROM OLD.residue_ordinal OR
        NEW.residue_reducer_version IS DISTINCT FROM OLD.residue_reducer_version OR
        NEW.residue_input_hash IS DISTINCT FROM OLD.residue_input_hash OR
        NEW.residue_causal_role IS DISTINCT FROM OLD.residue_causal_role OR
        NEW.residue_confidence IS DISTINCT FROM OLD.residue_confidence OR
        NEW.residue_evidence IS DISTINCT FROM OLD.residue_evidence OR
        NEW.residue_predecessors IS DISTINCT FROM OLD.residue_predecessors OR
        NEW.residue_line_root_hash IS DISTINCT FROM OLD.residue_line_root_hash OR
        NEW.residue_affect IS DISTINCT FROM OLD.residue_affect OR
        NEW.causal_node_hash IS DISTINCT FROM OLD.causal_node_hash OR
        NEW.causal_node_kind IS DISTINCT FROM OLD.causal_node_kind OR
        NEW.causal_payload_version IS DISTINCT FROM OLD.causal_payload_version OR
        NEW.causal_operation_id IS DISTINCT FROM OLD.causal_operation_id
    ) THEN
        RAISE EXCEPTION 'validated causal node semantic payload is immutable'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF OLD.causal_node_hash IS NOT NULL
       AND NEW.line_status = 'forgotten'
       AND OLD.line_status <> 'forgotten'
       AND NOT EXISTS (
           SELECT 1 FROM memory_tombstones t
           WHERE t.agent_id = OLD.agent_id AND t.memory_id = OLD.id
       ) THEN
        RAISE EXCEPTION 'causal node requires tombstone before forgetting'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_causal_node_immutable ON memories;
CREATE TRIGGER memories_causal_node_immutable
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION styx_immutable_causal_node();

CREATE OR REPLACE FUNCTION styx_guard_causal_node_delete()
RETURNS trigger AS $$
BEGIN
    IF OLD.causal_node_hash IS NOT NULL
       AND OLD.line_provenance IN (
           'validated_act_residue', 'validated_transform'
       ) AND (
           OLD.line_status <> 'forgotten'
           OR NOT EXISTS (
               SELECT 1 FROM memory_tombstones t
               WHERE t.agent_id = OLD.agent_id AND t.memory_id = OLD.id
           )
       ) THEN
        RAISE EXCEPTION 'active causal node cannot be physically deleted'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_causal_node_delete_guard ON memories;
CREATE TRIGGER memories_causal_node_delete_guard
BEFORE DELETE ON memories
FOR EACH ROW EXECUTE FUNCTION styx_guard_causal_node_delete();

-- Semantic operation code sets this transaction-local flag and advances the
-- line exactly once after all node/edge/tombstone writes.  Ordinary writers
-- retain the Wave 38 per-row invalidation behaviour.
CREATE OR REPLACE FUNCTION styx_mark_line_dirty() RETURNS trigger AS $$
DECLARE
    affected_agent text;
    relevant boolean := false;
    semantic_change boolean := false;
    operation_write boolean := false;
BEGIN
    affected_agent := COALESCE(NEW.agent_id, OLD.agent_id);
    operation_write := coalesce(
        current_setting('styx.causal_operation', true), ''
    ) = '1';
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
            OLD.residue_affect IS DISTINCT FROM NEW.residue_affect OR
            OLD.causal_node_hash IS DISTINCT FROM NEW.causal_node_hash OR
            OLD.causal_node_kind IS DISTINCT FROM NEW.causal_node_kind OR
            OLD.causal_operation_id IS DISTINCT FROM NEW.causal_operation_id OR
            OLD.line_status IS DISTINCT FROM NEW.line_status
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
            CASE WHEN semantic_change AND NOT operation_write THEN 1 ELSE 0 END,
            true,
            clock_timestamp()
        )
        ON CONFLICT (agent_id) DO UPDATE
        SET version = line_state.version + CASE
                WHEN semantic_change AND NOT operation_write THEN 1 ELSE 0 END,
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
    residue_line_root_hash,residue_affect,causal_node_hash,causal_node_kind,
    causal_operation_id,line_status ON memories
FOR EACH ROW EXECUTE FUNCTION styx_mark_line_dirty();
