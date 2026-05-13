-- Styx storage 0003 — Working set state persistence (волна 13).
--
-- См. .design/waves/13-working-set-persistence.md и
-- .design/context/decisions.md § 29.
--
-- Persistence module-global state'а focus_tracker (window + cached_salient
-- + epoch_id) и hot_tier (entries) между restart'ами Hermes-процесса.
-- Один agent_id == одна строка (single-agent invariant § 5; PRIMARY KEY
-- agent_id). Payload — JSONB с {version, embedding_dim, focus, hot,
-- saved_at_monotonic}; формат описан в serialize() модуля
-- styx/engine/working_set_persistence.py.
--
-- updated_at — clock_timestamp() (не now()) чтобы получать реальное
-- время каждой ON CONFLICT UPDATE-операции вне общей транзакции
-- timestamp'а save-thread'а.

CREATE TABLE IF NOT EXISTS working_set (
    agent_id    text PRIMARY KEY,
    payload     jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT clock_timestamp()
);
