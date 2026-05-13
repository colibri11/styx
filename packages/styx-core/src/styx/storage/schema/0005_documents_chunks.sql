-- 0005_documents_chunks.sql — store-routing для длинного content'а.
--
-- Волна 19. Длинный subjective write (memory_store / insert_batch_memory)
-- разделяется на chunks (через engine.chunker) и пишется в таблицы
-- documents + chunks. В memories остаётся короткая tail-memory с
-- archive_ref на document_id (поле archive_ref jsonb в memories
-- добавлено миграцией 0002).
--
-- Schema узкая (D2 в .design/waves/19-documents-chunks.md):
--   - documents без file_path/mime_type/source_ref — parser живёт в
--     OpenClaw plugin track'е, не в core.
--   - chunks с vector(768) совпадает с memories.embedding (D1).
--   - content_hash на documents nullable; UNIQUE constraint придёт с
--     волной 23 (`/ingest_experience` idempotency).
--
-- Идемпотентно через IF NOT EXISTS / pg_constraint EXISTS-check.

-- ── documents ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        text NOT NULL,
    source          text NOT NULL,                -- 'memory_store' / 'insert_batch_memory' / etc
    content_hash    text,                          -- nullable, UNIQUE приходит с волной 23
    char_count      integer NOT NULL,              -- length(content) до chunk'инга
    summary         text,                          -- nullable, копия tail-memory.content для debug
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_agent_created
    ON documents(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_source
    ON documents(source);

-- ── chunks ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position        integer NOT NULL,              -- 0..N-1 sequential
    content         text NOT NULL,
    embedding       vector(768),                   -- nullable: при partial fail оставляем без вектора
    char_start      integer NOT NULL,              -- UTF-8 byte offset в documents.content
    char_end        integer NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- UNIQUE (document_id, position) — гарантия sequential nesting и
-- защита от двойного INSERT'а одного position'а в pathological flow'ах.
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_document_position
    ON chunks(document_id, position);

-- HNSW на embedding — основной recall path (волна 20 search archive).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Position не отрицательный (защита от багов в chunker'е).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chunks_position_nonneg' AND conrelid = 'chunks'::regclass
    ) THEN
        ALTER TABLE chunks
            ADD CONSTRAINT chunks_position_nonneg
            CHECK (position >= 0);
    END IF;
END$$;

-- char_start <= char_end (защита от багов в chunker'е).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chunks_char_range_ordered' AND conrelid = 'chunks'::regclass
    ) THEN
        ALTER TABLE chunks
            ADD CONSTRAINT chunks_char_range_ordered
            CHECK (char_start <= char_end);
    END IF;
END$$;
