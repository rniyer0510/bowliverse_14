-- Add additive deterministic expert-system provenance and selection fields.
ALTER TABLE analysis_run
    ADD COLUMN IF NOT EXISTS knowledge_pack_id TEXT,
    ADD COLUMN IF NOT EXISTS knowledge_pack_version TEXT,
    ADD COLUMN IF NOT EXISTS deterministic_diagnosis_status TEXT,
    ADD COLUMN IF NOT EXISTS deterministic_primary_mechanism_id TEXT,
    ADD COLUMN IF NOT EXISTS deterministic_archetype_id TEXT;
