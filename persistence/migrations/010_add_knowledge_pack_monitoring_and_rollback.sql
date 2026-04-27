CREATE TABLE IF NOT EXISTS knowledge_pack_monitoring_snapshot (
    knowledge_pack_monitoring_snapshot_id UUID PRIMARY KEY,
    knowledge_pack_release_candidate_id UUID NOT NULL REFERENCES knowledge_pack_release_candidate(knowledge_pack_release_candidate_id),
    baseline_pack_version TEXT NOT NULL,
    candidate_pack_version TEXT NOT NULL,
    baseline_window_start TIMESTAMPTZ NOT NULL,
    baseline_window_end TIMESTAMPTZ NOT NULL,
    candidate_window_start TIMESTAMPTZ NOT NULL,
    candidate_window_end TIMESTAMPTZ NOT NULL,
    sufficient_data BOOLEAN NOT NULL DEFAULT FALSE,
    alert_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    rollback_recommended BOOLEAN NOT NULL DEFAULT FALSE,
    baseline_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    candidate_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    regression_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    alert_rules_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by_account_id UUID NULL REFERENCES account(account_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_pack_monitoring_snapshot_candidate_id
    ON knowledge_pack_monitoring_snapshot (knowledge_pack_release_candidate_id, created_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_pack_rollback_alert (
    knowledge_pack_rollback_alert_id UUID PRIMARY KEY,
    knowledge_pack_release_candidate_id UUID NOT NULL REFERENCES knowledge_pack_release_candidate(knowledge_pack_release_candidate_id),
    knowledge_pack_monitoring_snapshot_id UUID NOT NULL REFERENCES knowledge_pack_monitoring_snapshot(knowledge_pack_monitoring_snapshot_id),
    status TEXT NOT NULL
        CHECK (status IN ('OPEN','ACKNOWLEDGED','DISMISSED','ROLLED_BACK')),
    summary TEXT NOT NULL,
    triggered_rules_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolved_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_pack_rollback_alert_candidate_id
    ON knowledge_pack_rollback_alert (knowledge_pack_release_candidate_id, updated_at DESC);
