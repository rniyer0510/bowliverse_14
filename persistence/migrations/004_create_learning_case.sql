CREATE TABLE IF NOT EXISTS learning_case (
    learning_case_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES analysis_run(run_id),
    player_id UUID NOT NULL REFERENCES player(player_id),
    account_id UUID NULL REFERENCES account(account_id),
    event_name TEXT NOT NULL,
    knowledge_pack_id TEXT NULL,
    knowledge_pack_version TEXT NOT NULL,
    case_type TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    suggested_gap_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    symptom_bundle_hash TEXT NOT NULL,
    chosen_mechanism_id TEXT NULL,
    renderer_mode TEXT NOT NULL,
    prescription_id TEXT NULL,
    followup_outcome TEXT NOT NULL DEFAULT 'NOT_YET_DUE',
    clip_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    detected_symptoms JSONB NOT NULL DEFAULT '[]'::jsonb,
    candidate_mechanisms JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    contradictions_triggered JSONB NOT NULL DEFAULT '[]'::jsonb,
    event_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_learning_case_type
        CHECK (case_type IN ('NO_MATCH','AMBIGUOUS_MATCH','LOW_CONFIDENCE')),
    CONSTRAINT ck_learning_case_priority
        CHECK (priority IN ('A','B','C','D','E')),
    CONSTRAINT ck_learning_case_status
        CHECK (status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')),
    CONSTRAINT ck_learning_case_followup_outcome
        CHECK (followup_outcome IN ('NOT_YET_DUE','IMPROVING','NO_CLEAR_CHANGE','WORSENING','INSUFFICIENT_DATA'))
);

CREATE INDEX IF NOT EXISTS ix_learning_case_run_id
    ON learning_case (run_id);

CREATE INDEX IF NOT EXISTS ix_learning_case_player_id
    ON learning_case (player_id);

CREATE INDEX IF NOT EXISTS ix_learning_case_status_priority
    ON learning_case (status, priority);

CREATE INDEX IF NOT EXISTS ix_learning_case_symptom_bundle_hash
    ON learning_case (symptom_bundle_hash);
