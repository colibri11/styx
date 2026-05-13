-- 0006_chunks_fts.sql — FTS index для search_archive (волна 20).
--
-- Search archive (волна 20) делает hybrid query поверх chunks:
-- ts_rank(content_tsv, plainto_tsquery('simple', $q)) + (1 - cosine).
-- Без generated tsvector + GIN — sequential scan на каждый search,
-- неприемлемо для production volume (agent-a/agent-b через миграцию данных
-- дадут tens of thousands of chunks).
--
-- TS config 'simple' — consistency с `memories.content_tsv` (миграция
-- 0002): cross-language safe (cyrillic + latin), без stemming'а.
--
-- Generated STORED column (как у memories) — без тригерров, автоматически
-- пересчитывается при INSERT/UPDATE на content. Volna 19 не делает
-- UPDATE на chunks.content (chunker'ы immutable), но на будущее
-- (reinterpret merge?) generated проще.
--
-- Идемпотентно через IF NOT EXISTS.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING gin(content_tsv);
