CREATE TABLE IF NOT EXISTS knowledge_pack_regression_run (
    knowledge_pack_regression_run_id UUID PRIMARY KEY,
    knowledge_pack_release_candidate_id UUID NOT NULL REFERENCES knowledge_pack_release_candidate(knowledge_pack_release_candidate_id),
    baseline_pack_version TEXT NOT NULL,
    candidate_pack_version TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('COMPLETED','FAILED')),
    total_cases INTEGER NOT NULL DEFAULT 0,
    expected_change_cases INTEGER NOT NULL DEFAULT 0,
    stable_cases INTEGER NOT NULL DEFAULT 0,
    passed_cases INTEGER NOT NULL DEFAULT 0,
    failed_cases INTEGER NOT NULL DEFAULT 0,
    validated_regression_count INTEGER NOT NULL DEFAULT 0,
    validated_regression_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    expected_change_success_count INTEGER NOT NULL DEFAULT 0,
    expected_change_success_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by_account_id UUID NULL REFERENCES account(account_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_pack_regression_run_candidate_id
    ON knowledge_pack_regression_run (knowledge_pack_release_candidate_id, created_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_pack_regression_case_result (
    knowledge_pack_regression_case_result_id UUID PRIMARY KEY,
    knowledge_pack_regression_run_id UUID NOT NULL REFERENCES knowledge_pack_regression_run(knowledge_pack_regression_run_id),
    run_id UUID NOT NULL REFERENCES analysis_run(run_id),
    learning_case_cluster_id UUID NULL REFERENCES learning_case_cluster(learning_case_cluster_id),
    learning_case_id UUID NULL REFERENCES learning_case(learning_case_id),
    expected_behavior TEXT NOT NULL
        CHECK (expected_behavior IN ('CHANGE','PRESERVE')),
    outcome TEXT NOT NULL
        CHECK (outcome IN ('PASS','FAIL')),
    baseline_pack_version TEXT NULL,
    candidate_pack_version TEXT NOT NULL,
    baseline_diagnosis_status TEXT NULL,
    candidate_diagnosis_status TEXT NULL,
    baseline_primary_mechanism_id TEXT NULL,
    candidate_primary_mechanism_id TEXT NULL,
    baseline_renderer_mode TEXT NULL,
    candidate_renderer_mode TEXT NULL,
    reason TEXT NOT NULL,
    result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_pack_regression_case_result_run_id
    ON knowledge_pack_regression_case_result (knowledge_pack_regression_run_id, created_at ASC);
