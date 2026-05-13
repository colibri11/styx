-- 0007_documents_pipeline.sql — structural expansion documents-таблицы под
-- file-ingest pipeline (волна 28).
--
-- До волны 28 documents — узкая (миграция 0005 / волна 19 D2): id, agent_id,
-- source, content_hash, char_count, summary, metadata, created_at. Поля под
-- file-семантику жили в metadata JSONB (migration kit 27 поместил туда
-- memorybox-extras: visibility, file_path, original_name, mime_type,
-- source_ref, size_bytes).
--
-- Волна 28:
--   1. ALTER TABLE documents — добавить 6 nullable колонок.
--   2. Backfill из metadata JSONB (memorybox-migrated ряды).
--   3. Cleanup metadata от promoted ключей.
--   4. Partial UNIQUE (agent_id, content_hash) WHERE content_hash IS NOT NULL
--      — idempotency для повторного ingest того же файла; partial чтобы не
--      мешать existing store-routed рядам (волна 19) с NULL content_hash.
--
-- Идемпотентно через IF NOT EXISTS / COALESCE на backfill / WHERE filter на
-- cleanup / IF NOT EXISTS на UNIQUE index.

-- ── ALTER TABLE documents ────────────────────────────────────────────

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS file_path     text,
    ADD COLUMN IF NOT EXISTS original_name text,
    ADD COLUMN IF NOT EXISTS mime_type     text,
    ADD COLUMN IF NOT EXISTS source_ref    text,
    ADD COLUMN IF NOT EXISTS size_bytes    bigint,
    ADD COLUMN IF NOT EXISTS visibility    text;

-- ── Backfill из metadata JSONB ───────────────────────────────────────
-- COALESCE гарантирует idempotency — вторая прогонка не перезаписывает уже
-- promoted значения.

UPDATE documents SET
    file_path     = COALESCE(file_path,     metadata->>'file_path'),
    original_name = COALESCE(original_name, metadata->>'original_name'),
    mime_type     = COALESCE(mime_type,     metadata->>'mime_type'),
    source_ref    = COALESCE(source_ref,    metadata->>'source_ref'),
    size_bytes    = COALESCE(size_bytes,
                             NULLIF(metadata->>'size_bytes', '')::bigint),
    visibility    = COALESCE(visibility,    metadata->>'visibility')
WHERE metadata ?| array['file_path','original_name','mime_type',
                        'source_ref','size_bytes','visibility'];

-- Cleanup metadata от promoted ключей (вторая прогонка — no-op, WHERE
-- больше не находит ключей).

UPDATE documents SET
    metadata = metadata
        - 'file_path' - 'original_name' - 'mime_type'
        - 'source_ref' - 'size_bytes' - 'visibility'
WHERE metadata ?| array['file_path','original_name','mime_type',
                        'source_ref','size_bytes','visibility'];

-- ── Partial UNIQUE на content_hash ───────────────────────────────────
-- Idempotency повторного ingest'а: SHA256(file_bytes) для file-ingest'а
-- (волна 28) либо явный hash для других callers. Partial WHERE NOT NULL —
-- volna 19 store-routed ряды NULL content_hash сосуществуют.

CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_agent_content_hash
    ON documents(agent_id, content_hash)
    WHERE content_hash IS NOT NULL;
