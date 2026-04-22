CREATE TABLE IF NOT EXISTS learning_case_review_event (
    learning_case_review_event_id UUID PRIMARY KEY,
    learning_case_cluster_id UUID NOT NULL REFERENCES learning_case_cluster(learning_case_cluster_id),
    learning_case_id UUID NULL REFERENCES learning_case(learning_case_id),
    account_id UUID NULL REFERENCES account(account_id),
    action TEXT NOT NULL
        CHECK (action IN ('triage','queue','start_review','resolve','reject','reopen','supersede','expire')),
    from_status TEXT NOT NULL
        CHECK (from_status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')),
    to_status TEXT NOT NULL
        CHECK (to_status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')),
    notes TEXT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_learning_case_review_event_cluster_id
    ON learning_case_review_event (learning_case_cluster_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_learning_case_review_event_case_id
    ON learning_case_review_event (learning_case_id, created_at DESC);
