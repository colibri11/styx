-- Styx storage 0002 — port схемы и scoring-инфры из memorybox.
--
-- См. .design/waves/07-long-tier-retrieval.md и .design/context/decisions.md § 17.
-- Источник: openclaw-memorybox/migrations/001..024 — миграции memorybox,
-- из которых сюда втянуты колонки и таблицы, требуемые для composite
-- recall-формулы. RLS не тащим (decisions § 5 + § 17.1) — изоляция
-- через application-level WHERE по agent_id.
--
-- Идемпотентно: каждый ALTER/CREATE обёрнут в IF NOT EXISTS либо в
-- DO $$ EXISTS-check $$. Применяется в одну транзакцию (migrate.run
-- держит per-file tx).

-- ── memories: новые колонки ───────────────────────────────────────────

ALTER TABLE memories ADD COLUMN IF NOT EXISTS visibility text NOT NULL DEFAULT 'private';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'episode';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS kind_src text NOT NULL DEFAULT 'subjective';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS archive_ref jsonb;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS superseded_by uuid REFERENCES memories(id);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS relevance double precision NOT NULL DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count integer NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed_at timestamptz;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS lifecycle text NOT NULL DEFAULT 'fresh';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS usefulness double precision NOT NULL DEFAULT 0.0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS importance_provisional real NOT NULL DEFAULT 0.5;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS importance_final real;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS unique_query_count integer NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS recall_score_sum real NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS estimated_tokens int
    GENERATED ALWAYS AS (GREATEST(1, char_length(content) / 4)) STORED;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS emotional_context_valence real;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS emotional_context_arousal real;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS emotional_context_dominance real;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_hash text;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

-- ── memories: CHECK constraints ───────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_visibility_check' AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_visibility_check
            CHECK (visibility IN ('shared', 'private'));
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_kind_check' AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_kind_check
            CHECK (kind IN ('fact', 'episode', 'decision', 'concept', 'note'));
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_kind_src_check' AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_kind_src_check
            CHECK (kind_src IN (
                'subjective',
                'subjective_tail',
                'dialogue_batch_consolidation',
                'dialogue_consolidation_daily',
                'experience_intake'
            ));
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_lifecycle_check' AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_lifecycle_check
            CHECK (lifecycle IN ('fresh', 'settled', 'dormant'));
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'memories_content_length_check' AND conrelid = 'memories'::regclass
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_content_length_check
            CHECK (length(content) <= 2400);
    END IF;
END$$;

-- ── memories: backfill importance_provisional per-kind ───────────────
-- На свежих рядах NULL'ы по kind не ожидаются (DEFAULT 'episode'),
-- но если в БД появлялись пред-2002-port значения — апдейтим.

UPDATE memories SET importance_provisional = CASE kind
    WHEN 'decision' THEN 0.85
    WHEN 'fact'     THEN 0.70
    WHEN 'concept'  THEN 0.60
    WHEN 'note'     THEN 0.45
    WHEN 'episode'  THEN 0.40
    ELSE 0.5
END
WHERE importance_provisional = 0.5;  -- только дефолтные ряды

-- ── memories: переход на vector(768) ─────────────────────────────────
-- Существующие embeddings все NULL (sync_turn до этой волны не писал
-- вектор). DROP/CREATE индекс вместе с типом колонки.

DROP INDEX IF EXISTS memories_embedding_hnsw_idx;

DO $$
DECLARE
    current_typmod integer;
BEGIN
    SELECT atttypmod INTO current_typmod
      FROM pg_attribute
     WHERE attrelid = 'memories'::regclass AND attname = 'embedding';
    IF current_typmod IS NULL OR current_typmod <> 768 THEN
        ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768);
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops);

-- ── memories: индексы из memorybox ──────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_memories_visibility ON memories(visibility);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_superseded
    ON memories(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_estimated_tokens
    ON memories(estimated_tokens);
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_memories_agent_kind_src ON memories(agent_id, kind_src);
CREATE INDEX IF NOT EXISTS idx_memories_archive_ref_kind
    ON memories ((archive_ref->>'kind'))
    WHERE archive_ref IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS memories_agent_content_hash_uniq
    ON memories(agent_id, content_hash) WHERE content_hash IS NOT NULL;

-- ── recall_events: расширение под memorybox ─────────────────────────
-- Существующее Styx-поле score (double precision) переименовываем в
-- match_score real — семантически одно и то же. RENAME безопасен,
-- старого кода, читающего score, нет (queries.py пока не SELECT'ит
-- recall_events). Тип real — точность достаточна, совпадает с memorybox.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'recall_events' AND column_name = 'score'
    ) THEN
        ALTER TABLE recall_events RENAME COLUMN score TO match_score;
    END IF;
END$$;

DO $$
DECLARE
    current_type text;
BEGIN
    SELECT data_type INTO current_type
      FROM information_schema.columns
     WHERE table_name = 'recall_events' AND column_name = 'match_score';
    IF current_type = 'double precision' THEN
        ALTER TABLE recall_events ALTER COLUMN match_score TYPE real;
    END IF;
END$$;

ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS query_hash bytea;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS used_in_output boolean NOT NULL DEFAULT false;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS classifier_run_at timestamptz;

-- UNIQUE (memory_id, query_hash) — partial для обратной совместимости
-- с z.ai-smoke данными (query_hash NULL у событий до волны 7).
-- Memorybox в волне 008 ставил жёсткий UNIQUE без partial; у нас
-- расхождение оправдано наличием старых рядов и фиксируется здесь.
CREATE UNIQUE INDEX IF NOT EXISTS idx_recall_events_unique
    ON recall_events(memory_id, query_hash)
    WHERE query_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_recall_events_unconfirmed
    ON recall_events(matched_at) WHERE used_in_output = false;

CREATE INDEX IF NOT EXISTS idx_recall_events_classifier_pending
    ON recall_events(matched_at)
    WHERE used_in_output = false AND classifier_run_at IS NULL;

-- ── relations (memorybox 001) ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS relations (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type     text NOT NULL,
    source_id       uuid NOT NULL,
    target_type     text NOT NULL,
    target_id       uuid NOT NULL,
    relation        text NOT NULL,
    weight          double precision NOT NULL DEFAULT 1.0,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_type, target_id);

-- ── llm_tasks (memorybox 008, schema-only — worker в волне 7a) ───────

CREATE TABLE IF NOT EXISTS llm_tasks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id       uuid REFERENCES memories(id) ON DELETE CASCADE,
    task_type       text NOT NULL,
    status          text NOT NULL DEFAULT 'pending',
    payload         jsonb NOT NULL DEFAULT '{}'::jsonb,
    result          jsonb,
    error           text,
    retry_count     integer NOT NULL DEFAULT 0,
    task_version    integer NOT NULL DEFAULT 1,
    created_at      timestamptz NOT NULL DEFAULT now(),
    started_at      timestamptz,
    completed_at    timestamptz
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'llm_tasks_status_check' AND conrelid = 'llm_tasks'::regclass
    ) THEN
        ALTER TABLE llm_tasks
            ADD CONSTRAINT llm_tasks_status_check
            CHECK (status IN ('pending', 'running', 'done', 'failed'));
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_llm_tasks_queue
    ON llm_tasks(task_type, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_llm_tasks_memory ON llm_tasks(memory_id);

-- Триггер: при INSERT и UPDATE OF content на memories — постановка
-- importance_scoring_from_content в очередь.
CREATE OR REPLACE FUNCTION enqueue_importance_scoring()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO llm_tasks (memory_id, task_type, payload)
    VALUES (
        NEW.id,
        'importance_scoring_from_content',
        jsonb_build_object('kind', NEW.kind, 'length', length(NEW.content))
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memory_importance_scoring ON memories;
CREATE TRIGGER memory_importance_scoring
    AFTER INSERT OR UPDATE OF content ON memories
    FOR EACH ROW EXECUTE FUNCTION enqueue_importance_scoring();

-- ── consolidation_state, sweep_runs (memorybox 009) ──────────────────

CREATE TABLE IF NOT EXISTS consolidation_state (
    key             text PRIMARY KEY,
    value           jsonb NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sweep_runs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz,
    status          text NOT NULL,
    summary         jsonb NOT NULL DEFAULT '{}'::jsonb,
    errors          jsonb NOT NULL DEFAULT '[]'::jsonb
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'sweep_runs_status_check' AND conrelid = 'sweep_runs'::regclass
    ) THEN
        ALTER TABLE sweep_runs
            ADD CONSTRAINT sweep_runs_status_check
            CHECK (status IN ('running', 'success', 'partial', 'failed'));
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_sweep_runs_started ON sweep_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sweep_runs_status ON sweep_runs(status, started_at DESC);

-- ── emotional_state (memorybox 017, без RLS — § 17.1) ────────────────

CREATE TABLE IF NOT EXISTS emotional_state (
    id              bigserial PRIMARY KEY,
    agent_id        text NOT NULL,
    at              timestamptz NOT NULL DEFAULT now(),
    valence         real NOT NULL,
    arousal         real NOT NULL,
    dominance       real NOT NULL,
    source          text,
    metadata        jsonb
);

CREATE INDEX IF NOT EXISTS idx_emotional_state_agent_at
    ON emotional_state(agent_id, at DESC);

-- ── emotional_baseline (memorybox 019, без RLS) ──────────────────────

CREATE TABLE IF NOT EXISTS emotional_baseline (
    agent_id        text PRIMARY KEY,
    valence         real NOT NULL DEFAULT 0,
    arousal         real NOT NULL DEFAULT 0,
    dominance       real NOT NULL DEFAULT 0,
    updated_at      timestamptz NOT NULL DEFAULT now(),
    mood_active     boolean NOT NULL DEFAULT false
);

-- ── memory_reinterpretations (memorybox 020, без RLS) ────────────────

CREATE TABLE IF NOT EXISTS memory_reinterpretations (
    id                          bigserial PRIMARY KEY,
    memory_id                   uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    agent_id                    text NOT NULL,
    reinterpreted_at            timestamptz NOT NULL DEFAULT now(),
    previous_text               text NOT NULL,
    new_understanding_text      text NOT NULL,
    merged_text                 text NOT NULL,
    previous_embedding          vector NOT NULL,
    merged_embedding            vector NOT NULL,
    weight_applied              real NOT NULL CHECK (weight_applied >= 0 AND weight_applied <= 1)
);

CREATE INDEX IF NOT EXISTS idx_memory_reinterpretations_memory
    ON memory_reinterpretations(memory_id, reinterpreted_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_reinterpretations_agent
    ON memory_reinterpretations(agent_id);

-- ── reinterpret_applications (memorybox 021) ─────────────────────────

CREATE TABLE IF NOT EXISTS reinterpret_applications (
    id              bigserial PRIMARY KEY,
    task_id         uuid NOT NULL REFERENCES llm_tasks(id) ON DELETE CASCADE,
    memory_id       uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    agent_id        text NOT NULL,
    status          text NOT NULL DEFAULT 'pending_sleep',
    created_at      timestamptz NOT NULL DEFAULT now(),
    applied_at      timestamptz,
    CONSTRAINT reinterpret_applications_task_unique UNIQUE (task_id),
    CONSTRAINT reinterpret_applications_status_check
        CHECK (status IN ('pending_sleep', 'applied', 'skipped')),
    CONSTRAINT reinterpret_applications_applied_at_consistent
        CHECK ((status = 'pending_sleep' AND applied_at IS NULL)
            OR (status IN ('applied', 'skipped') AND applied_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_reinterpret_applications_pending
    ON reinterpret_applications(status, created_at)
    WHERE status = 'pending_sleep';

CREATE UNIQUE INDEX IF NOT EXISTS uq_reinterpret_applications_one_pending_per_memory
    ON reinterpret_applications(memory_id) WHERE status = 'pending_sleep';

-- ── memory_consolidation_applications (memorybox 024) ────────────────

CREATE TABLE IF NOT EXISTS memory_consolidation_applications (
    id              bigserial PRIMARY KEY,
    task_id         uuid NOT NULL REFERENCES llm_tasks(id) ON DELETE CASCADE,
    agent_id        text NOT NULL,
    source_ids      uuid[] NOT NULL,
    status          text NOT NULL DEFAULT 'pending_sleep',
    new_memory_id   uuid REFERENCES memories(id),
    created_at      timestamptz NOT NULL DEFAULT now(),
    applied_at      timestamptz,
    CONSTRAINT memory_consolidation_applications_task_unique UNIQUE (task_id),
    CONSTRAINT memory_consolidation_applications_status_check
        CHECK (status IN ('pending_sleep', 'applied', 'skipped')),
    CONSTRAINT memory_consolidation_applications_status_consistent
        CHECK (
            (status = 'pending_sleep' AND applied_at IS NULL AND new_memory_id IS NULL)
         OR (status = 'applied'       AND applied_at IS NOT NULL AND new_memory_id IS NOT NULL)
         OR (status = 'skipped'       AND applied_at IS NOT NULL)
        ),
    CONSTRAINT memory_consolidation_applications_source_ids_nonempty
        CHECK (array_length(source_ids, 1) >= 2)
);

CREATE INDEX IF NOT EXISTS idx_memory_consolidation_applications_pending
    ON memory_consolidation_applications(status, created_at)
    WHERE status = 'pending_sleep';

CREATE INDEX IF NOT EXISTS idx_memory_consolidation_applications_agent
    ON memory_consolidation_applications(agent_id, status);
