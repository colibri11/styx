-- Styx storage v1 — initial schema.
--
-- Таблицы: sessions, memories, recall_events.
-- Изоляция между агентами — через колонку agent_id; RLS не используется
-- (см. .design/integrations/hermes-v1.md § «Профили и хранилище»).
--
-- Embedding dim = 1024. Совместимо с mxbai-embed-large и
-- qwen3-embedding:0.6b в Ollama. Смена модели на другую размерность
-- потребует ALTER COLUMN + переиндексацию HNSW.

CREATE EXTENSION IF NOT EXISTS vector;

-- Hermes session tracking. session_id присваивается извне (Hermes),
-- мы только зеркалим жизненный цикл.
CREATE TABLE IF NOT EXISTS sessions (
    id          uuid PRIMARY KEY,
    agent_id    text NOT NULL,
    started_at  timestamptz NOT NULL DEFAULT clock_timestamp(),
    ended_at    timestamptz
);

CREATE INDEX IF NOT EXISTS sessions_agent_started_idx
    ON sessions (agent_id, started_at DESC);

-- Long-tier memory store. Каждая строка — message-level запись либо
-- производное (summary, derived). role — CHECK, не PG enum, чтобы
-- расширять без ALTER TYPE.
-- ``seq`` — монотонный bigserial, tie-breaker для ORDER BY когда
-- ``created_at`` совпадает (multiple inserts в одной транзакции внутри
-- одного microsecond — sync_turn пишет user+assistant в одной tx).
CREATE TABLE IF NOT EXISTS memories (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    seq         bigserial NOT NULL UNIQUE,
    agent_id    text NOT NULL,
    session_id  uuid REFERENCES sessions(id) ON DELETE SET NULL,
    role        text NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system', 'summary')),
    content     text NOT NULL,
    embedding   vector(1024),
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at  timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS memories_agent_seq_idx
    ON memories (agent_id, seq DESC);

CREATE INDEX IF NOT EXISTS memories_session_seq_idx
    ON memories (session_id, seq);

CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops);

-- Recall log: когда memory была поднята в active suffix.
-- agent_id сознательно отсутствует — scope через FK на memories.
CREATE TABLE IF NOT EXISTS recall_events (
    id          bigserial PRIMARY KEY,
    memory_id   uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    session_id  uuid REFERENCES sessions(id) ON DELETE SET NULL,
    matched_at  timestamptz NOT NULL DEFAULT now(),
    focus       text,
    score       double precision,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS recall_events_memory_idx
    ON recall_events (memory_id, matched_at DESC);

CREATE INDEX IF NOT EXISTS recall_events_session_idx
    ON recall_events (session_id, matched_at);
