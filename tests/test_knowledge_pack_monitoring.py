import unittest
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.persistence.knowledge_pack_monitoring import (
    evaluate_monitoring_snapshot,
    resolve_open_rollback_alerts_for_candidate,
    run_post_promotion_monitoring,
)


class _FakeAlertQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeDb:
    def __init__(self, alert_rows=None):
        self.added = []
        self.flushed = 0
        self.alert_rows = alert_rows or []

    def add(self, row):
        self.added.append(row)

    def flush(self):
        self.flushed += 1

    def query(self, model):
        return _FakeAlertQuery(self.alert_rows)


class KnowledgePackMonitoringEvaluationTests(unittest.TestCase):
    def test_monitoring_evaluation_triggers_alert_when_rates_breach_thresholds(self):
        evaluation = evaluate_monitoring_snapshot(
            baseline_metrics={
                "total_runs": 10,
                "no_match_rate": 0.10,
                "ambiguity_rate": 0.10,
                "coach_flag_rate": 0.10,
            },
            candidate_metrics={
                "total_runs": 10,
                "no_match_rate": 0.18,
                "ambiguity_rate": 0.20,
                "coach_flag_rate": 0.18,
            },
            regression_metrics={
                "validated_regression_rate": 0.08,
            },
        )

        self.assertTrue(evaluation["sufficient_data"])
        self.assertTrue(evaluation["alert_triggered"])
        self.assertTrue(evaluation["rollback_recommended"])
        triggered_metrics = {row["metric"] for row in evaluation["triggered_rules"]}
        self.assertIn("no_match_rate", triggered_metrics)
        self.assertIn("ambiguity_rate", triggered_metrics)
        self.assertIn("validated_regression_rate", triggered_metrics)

    def test_monitoring_evaluation_stays_quiet_when_data_is_insufficient(self):
        evaluation = evaluate_monitoring_snapshot(
            baseline_metrics={
                "total_runs": 2,
                "no_match_rate": 0.10,
                "ambiguity_rate": 0.10,
                "coach_flag_rate": 0.10,
            },
            candidate_metrics={
                "total_runs": 2,
                "no_match_rate": 0.50,
                "ambiguity_rate": 0.40,
                "coach_flag_rate": 0.30,
            },
            regression_metrics={
                "validated_regression_rate": 0.0,
            },
        )

        self.assertFalse(evaluation["sufficient_data"])
        self.assertFalse(evaluation["alert_triggered"])


class KnowledgePackMonitoringWorkflowTests(unittest.TestCase):
    def test_run_post_promotion_monitoring_creates_snapshot_and_alert(self):
        db = _FakeDb()
        candidate = SimpleNamespace(
            knowledge_pack_release_candidate_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            status="PROMOTED",
            base_pack_version="2026-04-22.v1",
            candidate_pack_version="2026-05-01.v1",
            promoted_at=datetime.utcnow() - timedelta(days=7),
        )
        with patch(
            "app.persistence.knowledge_pack_monitoring._collect_window_metrics",
            side_effect=[
                {"overall": {"total_runs": 10, "no_match_rate": 0.10, "ambiguity_rate": 0.10, "coach_flag_rate": 0.10}, "by_pack_version": {}},
                {"overall": {"total_runs": 10, "no_match_rate": 0.18, "ambiguity_rate": 0.20, "coach_flag_rate": 0.18}, "by_pack_version": {}},
            ],
        ), patch(
            "app.persistence.knowledge_pack_monitoring._latest_regression_metrics",
            return_value={"validated_regression_rate": 0.08},
        ):
            snapshot, alert = run_post_promotion_monitoring(
                candidate_row=candidate,
                account_id="22222222-2222-2222-2222-222222222222",
                db=db,
                now=datetime.utcnow(),
            )

        self.assertTrue(snapshot.alert_triggered)
        self.assertTrue(snapshot.rollback_recommended)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.status, "OPEN")
        self.assertGreaterEqual(db.flushed, 2)

    def test_resolve_open_rollback_alerts_marks_rows_resolved(self):
        alert = SimpleNamespace(
            status="OPEN",
            resolved_at=None,
            updated_at=None,
        )
        db = _FakeDb(alert_rows=[alert])

        resolved = resolve_open_rollback_alerts_for_candidate(
            candidate_id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
            db=db,
            resolution_status="ROLLED_BACK",
        )

        self.assertEqual(resolved, 1)
        self.assertEqual(alert.status, "ROLLED_BACK")
        self.assertIsNotNone(alert.resolved_at)


if __name__ == "__main__":
    unittest.main()
