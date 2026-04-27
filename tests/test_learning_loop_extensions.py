import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.persistence.prescription_followups import (
    _evaluate_metric_directions,
    _resolve_followup_status,
    _select_relevant_followup_runs,
)
from app.persistence.read_api import _build_history_uncertainty_summary


class PrescriptionFollowupLogicTests(unittest.TestCase):
    def _result(self, **metric_values):
        metrics = {}
        for key, value in metric_values.items():
            metrics[key] = {"value": value, "confidence": 0.8}
        return {"deterministic_expert_v1": {"metrics": metrics}}

    def test_evaluate_metric_directions_marks_improving_for_expected_changes(self):
        assigned_result = self._result(
            trunk_drift_after_ffc=0.7,
            front_leg_support_score=0.45,
        )
        followup_runs = [
            (
                SimpleNamespace(run_id="run-2"),
                self._result(trunk_drift_after_ffc=0.52, front_leg_support_score=0.58),
            ),
            (
                SimpleNamespace(run_id="run-3"),
                self._result(trunk_drift_after_ffc=0.48, front_leg_support_score=0.61),
            ),
        ]

        summary = _evaluate_metric_directions(
            assigned_result=assigned_result,
            relevant_runs=followup_runs,
            expected_direction_of_change={
                "trunk_drift_after_ffc": "down",
                "front_leg_support_score": "up",
            },
        )

        self.assertEqual(summary["valid_followup_run_count"], 2)
        self.assertEqual(
            summary["actual_direction_of_change"]["trunk_drift_after_ffc"]["status"],
            "improving",
        )
        self.assertEqual(
            summary["actual_direction_of_change"]["front_leg_support_score"]["status"],
            "improving",
        )

    def test_resolve_followup_status_returns_worsening_when_reverse_direction_dominates(self):
        status = _resolve_followup_status(
            metrics_summary={
                "actual_direction_of_change": {
                    "terminal_impulse_score": {"status": "worsening"},
                    "dissipation_burden_score": {"status": "worsening"},
                    "approach_momentum_score": {"status": "flat"},
                }
            },
            valid_run_count=2,
            min_valid_runs=2,
            window_closed=True,
            insufficient_data_status="INSUFFICIENT_DATA",
        )

        self.assertEqual(status, "WORSENING")

    def test_select_relevant_followup_runs_for_next_session_uses_session_gap(self):
        assigned_run = SimpleNamespace(created_at=datetime(2026, 4, 22, 10, 0, 0))
        later_rows = [
            (
                SimpleNamespace(run_id="run-2", created_at=datetime(2026, 4, 22, 10, 30, 0)),
                SimpleNamespace(result_json=self._result(trunk_drift_after_ffc=0.5)),
            ),
            (
                SimpleNamespace(run_id="run-3", created_at=datetime(2026, 4, 22, 12, 5, 0)),
                SimpleNamespace(result_json=self._result(trunk_drift_after_ffc=0.45)),
            ),
        ]

        selected = _select_relevant_followup_runs(
            assigned_run=assigned_run,
            later_rows=later_rows,
            review_window_type="next_session",
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0].run_id, "run-3")


class HistoryUncertaintyTests(unittest.TestCase):
    def test_history_uncertainty_summary_trips_when_recent_unresolved_runs_cross_threshold(self):
        runs = [
            SimpleNamespace(run_id="run-1", deterministic_diagnosis_status="no_match"),
            SimpleNamespace(run_id="run-2", deterministic_diagnosis_status="partial_match"),
            SimpleNamespace(run_id="run-3", deterministic_diagnosis_status="ambiguous_match"),
            SimpleNamespace(run_id="run-4", deterministic_diagnosis_status="weak_match"),
            SimpleNamespace(run_id="run-5", deterministic_diagnosis_status="partial_match"),
            SimpleNamespace(run_id="run-6", deterministic_diagnosis_status="partial_match"),
            SimpleNamespace(run_id="run-7", deterministic_diagnosis_status="partial_match"),
            SimpleNamespace(run_id="run-8", deterministic_diagnosis_status="partial_match"),
        ]
        runtime_cases = {}

        with patch(
            "app.persistence.read_api.load_knowledge_pack",
            return_value={
                "globals": {
                    "history_uncertainty": {
                        "unresolved_min_runs": 3,
                        "unresolved_window_runs": 8,
                        "unresolved_rate_min": 0.35,
                    }
                }
            },
        ):
            summary = _build_history_uncertainty_summary(runs, runtime_cases)

        self.assertTrue(summary["pattern_still_being_understood"])
        self.assertEqual(summary["unresolved_runs"], 3)
        self.assertEqual(summary["window_runs"], 8)
        self.assertEqual(summary["unresolved_rate"], 0.375)


if __name__ == "__main__":
    unittest.main()
