CREATE TABLE IF NOT EXISTS knowledge_pack_release_candidate (
    knowledge_pack_release_candidate_id UUID PRIMARY KEY,
    knowledge_pack_id TEXT NOT NULL,
    base_pack_version TEXT NOT NULL,
    candidate_pack_version TEXT NOT NULL UNIQUE,
    supersedes_pack_version TEXT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('DRAFT','IN_DEV','IN_STAGING','APPROVED','PROMOTED','REJECTED','ROLLED_BACK')),
    current_environment TEXT NULL
        CHECK (current_environment IS NULL OR current_environment IN ('dev','staging','production')),
    summary TEXT NOT NULL,
    change_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    motivating_cluster_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    motivating_case_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    tests_added JSONB NOT NULL DEFAULT '[]'::jsonb,
    reinterpret_run_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    schema_validated BOOLEAN NOT NULL DEFAULT FALSE,
    referential_integrity_validated BOOLEAN NOT NULL DEFAULT FALSE,
    regression_suite_passed BOOLEAN NOT NULL DEFAULT FALSE,
    staging_evaluation_passed BOOLEAN NOT NULL DEFAULT FALSE,
    approval_granted BOOLEAN NOT NULL DEFAULT FALSE,
    created_by_account_id UUID NULL REFERENCES account(account_id),
    updated_by_account_id UUID NULL REFERENCES account(account_id),
    promoted_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_pack_release_candidate_updated_at
    ON knowledge_pack_release_candidate (updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_pack_release_event (
    knowledge_pack_release_event_id UUID PRIMARY KEY,
    knowledge_pack_release_candidate_id UUID NOT NULL REFERENCES knowledge_pack_release_candidate(knowledge_pack_release_candidate_id),
    account_id UUID NULL REFERENCES account(account_id),
    action TEXT NOT NULL
        CHECK (action IN ('create','record_schema_validation','record_referential_integrity','record_regression_pass','promote_dev','promote_staging','record_staging_evaluation','approve_production','promote_production','reject','rollback')),
    from_status TEXT NOT NULL
        CHECK (from_status IN ('NONE','DRAFT','IN_DEV','IN_STAGING','APPROVED','PROMOTED','REJECTED','ROLLED_BACK')),
    to_status TEXT NOT NULL
        CHECK (to_status IN ('DRAFT','IN_DEV','IN_STAGING','APPROVED','PROMOTED','REJECTED','ROLLED_BACK')),
    from_environment TEXT NULL
        CHECK (from_environment IS NULL OR from_environment IN ('dev','staging','production')),
    to_environment TEXT NULL
        CHECK (to_environment IS NULL OR to_environment IN ('dev','staging','production')),
    notes TEXT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_pack_release_event_candidate_id
    ON knowledge_pack_release_event (knowledge_pack_release_candidate_id, created_at DESC);
