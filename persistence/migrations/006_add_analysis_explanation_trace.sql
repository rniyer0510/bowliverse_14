CREATE TABLE IF NOT EXISTS analysis_explanation_trace (
    run_id UUID PRIMARY KEY REFERENCES analysis_run(run_id),
    knowledge_pack_id TEXT NULL,
    knowledge_pack_version TEXT NULL,
    diagnosis_status TEXT NULL,
    primary_mechanism_id TEXT NULL,
    matched_symptom_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    candidate_mechanisms JSONB NOT NULL DEFAULT '[]'::jsonb,
    supporting_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    contradictions_triggered JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_trajectory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_prescription_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_render_story_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_history_binding_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    explanation_trace_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_analysis_explanation_trace_diagnosis_status
    ON analysis_explanation_trace (diagnosis_status);

CREATE INDEX IF NOT EXISTS ix_analysis_explanation_trace_primary_mechanism_id
    ON analysis_explanation_trace (primary_mechanism_id);
