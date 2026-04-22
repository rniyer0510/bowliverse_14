ALTER TABLE account
    DROP CONSTRAINT IF EXISTS ck_account_role;

ALTER TABLE account
    ADD CONSTRAINT ck_account_role
    CHECK (role IN ('player','coach','parent','reviewer','clinician'));

CREATE OR REPLACE FUNCTION prevent_analysis_explanation_trace_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'analysis_explanation_trace is write-once and cannot be modified';
END;
$$;

DROP TRIGGER IF EXISTS trg_analysis_explanation_trace_no_update
    ON analysis_explanation_trace;

CREATE TRIGGER trg_analysis_explanation_trace_no_update
    BEFORE UPDATE ON analysis_explanation_trace
    FOR EACH ROW
    EXECUTE FUNCTION prevent_analysis_explanation_trace_mutation();

DROP TRIGGER IF EXISTS trg_analysis_explanation_trace_no_delete
    ON analysis_explanation_trace;

CREATE TRIGGER trg_analysis_explanation_trace_no_delete
    BEFORE DELETE ON analysis_explanation_trace
    FOR EACH ROW
    EXECUTE FUNCTION prevent_analysis_explanation_trace_mutation();
