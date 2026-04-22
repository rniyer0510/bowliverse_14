import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from app.persistence.knowledge_pack_regressions import (
    evaluate_regression_case,
    run_release_candidate_regression,
)


class _FakeDb:
    def __init__(self):
        self.added = []
        self.flushed = 0

    def add(self, row):
        self.added.append(row)

    def flush(self):
        self.flushed += 1


class KnowledgePackRegressionEvaluationTests(unittest.TestCase):
    def test_change_case_passes_when_candidate_improves_status(self):
        baseline = {
            "deterministic_expert_v1": {
                "selection": {
                    "diagnosis_status": "no_match",
                    "primary_mechanism_id": None,
                },
                "render_reasoning_v1": {"renderer_mode": "event_only"},
            }
        }
        candidate = {
            "selection": {
                "diagnosis_status": "partial_match",
                "primary_mechanism_id": "soft_block_with_trunk_carry",
            },
            "render_reasoning_v1": {"renderer_mode": "partial_evidence"},
        }

        evaluation = evaluate_regression_case(
            baseline_result=baseline,
            candidate_result=candidate,
            expected_behavior="CHANGE",
        )

        self.assertEqual(evaluation["outcome"], "PASS")

    def test_preserve_case_fails_when_candidate_changes_primary_mechanism(self):
        baseline = {
            "deterministic_expert_v1": {
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "render_reasoning_v1": {"renderer_mode": "partial_evidence"},
            }
        }
        candidate = {
            "selection": {
                "diagnosis_status": "partial_match",
                "primary_mechanism_id": "gather_disorganization_before_block",
            },
            "render_reasoning_v1": {"renderer_mode": "partial_evidence"},
        }

        evaluation = evaluate_regression_case(
            baseline_result=baseline,
            candidate_result=candidate,
            expected_behavior="PRESERVE",
        )

        self.assertEqual(evaluation["outcome"], "FAIL")


class KnowledgePackRegressionRunnerTests(unittest.TestCase):
    def test_runner_marks_candidate_passed_and_records_summary(self):
        db = _FakeDb()
        candidate = SimpleNamespace(
            knowledge_pack_release_candidate_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            base_pack_version="2026-04-22.v1",
            candidate_pack_version="2026-05-01.v1",
            regression_suite_passed=False,
            updated_at=None,
            updated_by_account_id=None,
        )
        change_case = {
            "run_row": SimpleNamespace(run_id=uuid.UUID("22222222-2222-2222-2222-222222222222")),
            "raw_result": {
                "deterministic_expert_v1": {
                    "selection": {"diagnosis_status": "no_match", "primary_mechanism_id": None},
                    "render_reasoning_v1": {"renderer_mode": "event_only"},
                }
            },
            "expected_behavior": "CHANGE",
            "learning_case_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
            "learning_case_cluster_id": uuid.UUID("44444444-4444-4444-4444-444444444444"),
            "baseline_pack_version": "2026-04-22.v1",
        }
        stable_case = {
            "run_row": SimpleNamespace(run_id=uuid.UUID("55555555-5555-5555-5555-555555555555")),
            "raw_result": {
                "deterministic_expert_v1": {
                    "selection": {
                        "diagnosis_status": "partial_match",
                        "primary_mechanism_id": "soft_block_with_trunk_carry",
                    },
                    "render_reasoning_v1": {"renderer_mode": "partial_evidence"},
                }
            },
            "expected_behavior": "PRESERVE",
            "learning_case_id": uuid.UUID("66666666-6666-6666-6666-666666666666"),
            "learning_case_cluster_id": uuid.UUID("77777777-7777-7777-7777-777777777777"),
            "baseline_pack_version": "2026-04-22.v1",
        }

        reruns = [
            {
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "render_reasoning_v1": {"renderer_mode": "partial_evidence"},
            },
            {
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "render_reasoning_v1": {"renderer_mode": "partial_evidence"},
            },
        ]

        with patch(
            "app.persistence.knowledge_pack_regressions._load_regression_case_specs",
            return_value=[change_case, stable_case],
        ), patch(
            "app.persistence.knowledge_pack_regressions._load_prior_results",
            return_value=[],
        ), patch(
            "app.persistence.knowledge_pack_regressions._rerun_deterministic_payload",
            side_effect=reruns,
        ), patch(
            "app.persistence.knowledge_pack_regressions.DeterministicExpertSystem",
            return_value=SimpleNamespace(history_window_runs=4),
        ), patch(
            "app.persistence.knowledge_pack_regressions.apply_release_action",
        ) as mock_record_pass:
            regression_run = run_release_candidate_regression(
                candidate_row=candidate,
                account_id="88888888-8888-8888-8888-888888888888",
                db=db,
            )

        self.assertTrue(candidate.regression_suite_passed)
        self.assertEqual(regression_run.total_cases, 2)
        self.assertEqual(regression_run.failed_cases, 0)
        self.assertEqual(regression_run.expected_change_cases, 1)
        self.assertEqual(regression_run.stable_cases, 1)
        self.assertEqual(regression_run.expected_change_success_rate, 1.0)
        self.assertEqual(regression_run.validated_regression_rate, 0.0)
        self.assertEqual(regression_run.status, "COMPLETED")
        self.assertGreaterEqual(db.flushed, 1)
        mock_record_pass.assert_called_once()


if __name__ == "__main__":
    unittest.main()
