CREATE TABLE IF NOT EXISTS learning_case_cluster (
    learning_case_cluster_id UUID PRIMARY KEY,
    cluster_key_hash TEXT NOT NULL UNIQUE,
    knowledge_pack_id TEXT NULL,
    knowledge_pack_version TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'runtime_gap',
    case_type TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    suggested_gap_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    symptom_bundle_hash TEXT NOT NULL,
    renderer_mode TEXT NULL,
    chosen_mechanism_id TEXT NULL,
    prescription_id TEXT NULL,
    candidate_mechanism_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    cluster_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    case_count INTEGER NOT NULL DEFAULT 0,
    coach_flag_count INTEGER NOT NULL DEFAULT 0,
    first_run_id UUID NULL REFERENCES analysis_run(run_id),
    latest_run_id UUID NULL REFERENCES analysis_run(run_id),
    representative_learning_case_id UUID NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_learning_case_cluster_source_type
        CHECK (source_type IN ('runtime_gap','coach_feedback','prescription_followup')),
    CONSTRAINT ck_learning_case_cluster_type
        CHECK (case_type IN ('NO_MATCH','AMBIGUOUS_MATCH','LOW_CONFIDENCE','COACH_FEEDBACK','PRESCRIPTION_NON_RESPONSE')),
    CONSTRAINT ck_learning_case_cluster_priority
        CHECK (priority IN ('A','B','C','D','E')),
    CONSTRAINT ck_learning_case_cluster_status
        CHECK (status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED'))
);

CREATE INDEX IF NOT EXISTS ix_learning_case_cluster_status_priority
    ON learning_case_cluster (status, priority);

CREATE INDEX IF NOT EXISTS ix_learning_case_cluster_symptom_bundle_hash
    ON learning_case_cluster (symptom_bundle_hash);

ALTER TABLE learning_case
    ADD COLUMN IF NOT EXISTS learning_case_cluster_id UUID NULL REFERENCES learning_case_cluster(learning_case_cluster_id);

ALTER TABLE learning_case
    ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'runtime_gap';

ALTER TABLE learning_case
    DROP CONSTRAINT IF EXISTS ck_learning_case_type;

ALTER TABLE learning_case
    ADD CONSTRAINT ck_learning_case_type
        CHECK (case_type IN ('NO_MATCH','AMBIGUOUS_MATCH','LOW_CONFIDENCE','COACH_FEEDBACK','PRESCRIPTION_NON_RESPONSE'));

ALTER TABLE learning_case
    DROP CONSTRAINT IF EXISTS ck_learning_case_source_type;

ALTER TABLE learning_case
    ADD CONSTRAINT ck_learning_case_source_type
        CHECK (source_type IN ('runtime_gap','coach_feedback','prescription_followup'));

CREATE INDEX IF NOT EXISTS ix_learning_case_cluster_id
    ON learning_case (learning_case_cluster_id);

CREATE TABLE IF NOT EXISTS coach_flag (
    coach_flag_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES analysis_run(run_id),
    player_id UUID NOT NULL REFERENCES player(player_id),
    account_id UUID NOT NULL REFERENCES account(account_id),
    knowledge_pack_id TEXT NULL,
    knowledge_pack_version TEXT NULL,
    flag_type TEXT NOT NULL,
    notes TEXT NULL,
    flagged_mechanism_id TEXT NULL,
    flagged_prescription_id TEXT NULL,
    learning_case_id UUID NULL REFERENCES learning_case(learning_case_id),
    learning_case_cluster_id UUID NULL REFERENCES learning_case_cluster(learning_case_cluster_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_coach_flag_type
        CHECK (flag_type IN ('wrong_mechanism','wrong_prescription','right_mechanism_wrong_wording','renderer_story_misleading','capture_quality_bad'))
);

CREATE INDEX IF NOT EXISTS ix_coach_flag_run_id
    ON coach_flag (run_id);

CREATE INDEX IF NOT EXISTS ix_coach_flag_cluster_id
    ON coach_flag (learning_case_cluster_id);

CREATE TABLE IF NOT EXISTS prescription_followup (
    prescription_followup_id UUID PRIMARY KEY,
    prescription_assigned_at_run_id UUID NOT NULL REFERENCES analysis_run(run_id),
    player_id UUID NOT NULL REFERENCES player(player_id),
    knowledge_pack_id TEXT NULL,
    knowledge_pack_version TEXT NULL,
    prescription_id TEXT NOT NULL,
    review_window_type TEXT NOT NULL,
    followup_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
    expected_direction_of_change JSONB NOT NULL DEFAULT '{}'::jsonb,
    actual_direction_of_change JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_status TEXT NOT NULL DEFAULT 'NOT_YET_DUE',
    valid_followup_run_count INTEGER NOT NULL DEFAULT 0,
    window_closed BOOLEAN NOT NULL DEFAULT FALSE,
    latest_followup_run_id UUID NULL REFERENCES analysis_run(run_id),
    learning_case_id UUID NULL REFERENCES learning_case(learning_case_id),
    window_due_at TIMESTAMPTZ NULL,
    resolved_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_prescription_followup_review_window_type
        CHECK (review_window_type IN ('next_3_runs','next_session','next_2_weeks')),
    CONSTRAINT ck_prescription_followup_response_status
        CHECK (response_status IN ('NOT_YET_DUE','IMPROVING','NO_CLEAR_CHANGE','WORSENING','INSUFFICIENT_DATA'))
);

CREATE INDEX IF NOT EXISTS ix_prescription_followup_assigned_run
    ON prescription_followup (prescription_assigned_at_run_id);

CREATE INDEX IF NOT EXISTS ix_prescription_followup_player_status
    ON prescription_followup (player_id, response_status);
