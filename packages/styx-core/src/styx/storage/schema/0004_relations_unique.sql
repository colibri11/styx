-- 0004_relations_unique.sql — UNIQUE constraint для auto-link идемпотентности.
--
-- Волна 18 (auto-link при INSERT). Auto-link в subjective writers и
-- sync_turn пишет рёбра related_to с одинаковым (source, target,
-- relation) если ряд write'ится повторно (например, после reembed CLI'я
-- который обновил vector). Без UNIQUE constraint'а получим дубли;
-- INSERT ... ON CONFLICT DO NOTHING требует UNIQUE на правую часть.
--
-- Cleanup существующих дублей (если есть): оставляем самый ранний
-- ряд по id (детерминированно), остальные удаляем.

-- ── Cleanup существующих дублей ────────────────────────────────────

DELETE FROM relations
 WHERE id IN (
    SELECT id FROM (
        SELECT id,
               row_number() OVER (
                   PARTITION BY source_type, source_id,
                                target_type, target_id, relation
                   ORDER BY created_at ASC, id ASC
               ) AS rn
        FROM relations
    ) sub
    WHERE rn > 1
 );

-- ── UNIQUE constraint ──────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'relations_unique' AND conrelid = 'relations'::regclass
    ) THEN
        ALTER TABLE relations
            ADD CONSTRAINT relations_unique
            UNIQUE (source_type, source_id, target_type, target_id, relation);
    END IF;
END$$;
