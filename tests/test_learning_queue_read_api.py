import unittest
from datetime import datetime
from types import SimpleNamespace

from app.persistence.read_api import (
    _build_learning_cluster_item,
    _coverage_metrics_payload,
)


class LearningQueueReadApiTests(unittest.TestCase):
    def test_build_learning_cluster_item_rolls_up_cases_and_flags(self):
        cluster = SimpleNamespace(
            learning_case_cluster_id="cluster-1",
            knowledge_pack_id="actionlab_deterministic_expert",
            knowledge_pack_version="2026-04-22.v1",
            source_type="runtime_gap",
            case_type="AMBIGUOUS_MATCH",
            priority="B",
            status="OPEN",
            suggested_gap_type="weak_distinction_rules",
            trigger_reason="Two mechanisms stayed too close.",
            symptom_bundle_hash="abc123",
            renderer_mode="partial_evidence",
            chosen_mechanism_id="soft_block_with_trunk_carry",
            prescription_id="stack_over_landing_leg",
            candidate_mechanism_ids=["soft_block_with_trunk_carry", "gather_disorganization_before_block"],
            cluster_payload={"source_type": "runtime_gap"},
            case_count=2,
            coach_flag_count=1,
            first_run_id="run-1",
            latest_run_id="run-2",
            representative_learning_case_id="case-1",
            created_at=datetime(2026, 4, 22, 10, 0, 0),
            updated_at=datetime(2026, 4, 22, 12, 0, 0),
        )
        case_rows = [
            SimpleNamespace(
                run_id="run-1",
                player_id="player-1",
                account_id="account-1",
                status="CLUSTERED",
                followup_outcome="NOT_YET_DUE",
            ),
            SimpleNamespace(
                run_id="run-2",
                player_id="player-1",
                account_id="account-2",
                status="UNDER_REVIEW",
                followup_outcome="NO_CLEAR_CHANGE",
            ),
        ]
        coach_flags = [
            SimpleNamespace(flag_type="wrong_mechanism"),
        ]

        item = _build_learning_cluster_item(
            cluster,
            case_rows=case_rows,
            coach_flag_rows=coach_flags,
        )

        self.assertEqual(item["learning_case_cluster_id"], "cluster-1")
        self.assertEqual(item["case_count"], 2)
        self.assertEqual(item["case_statuses"]["CLUSTERED"], 1)
        self.assertEqual(item["case_statuses"]["UNDER_REVIEW"], 1)
        self.assertEqual(item["coach_flag_types"]["wrong_mechanism"], 1)
        self.assertEqual(item["player_ids"], ["player-1"])

    def test_coverage_metrics_payload_computes_rates_and_pack_breakdown(self):
        runs = [
            SimpleNamespace(
                run_id="run-1",
                deterministic_diagnosis_status="confident_match",
                knowledge_pack_version="2026-04-22.v1",
            ),
            SimpleNamespace(
                run_id="run-2",
                deterministic_diagnosis_status="no_match",
                knowledge_pack_version="2026-04-22.v1",
            ),
            SimpleNamespace(
                run_id="run-3",
                deterministic_diagnosis_status="ambiguous_match",
                knowledge_pack_version="2026-04-22.v1",
            ),
        ]
        raw_by_run_id = {
            "run-1": SimpleNamespace(result_json={"render_reasoning_v1": {"renderer_mode": "full_causal_story"}}),
            "run-2": SimpleNamespace(result_json={"render_reasoning_v1": {"renderer_mode": "event_only"}}),
            "run-3": SimpleNamespace(result_json={"render_reasoning_v1": {"renderer_mode": "partial_evidence"}}),
        }
        followups = [
            SimpleNamespace(knowledge_pack_version="2026-04-22.v1", response_status="NO_CLEAR_CHANGE"),
            SimpleNamespace(knowledge_pack_version="2026-04-22.v1", response_status="IMPROVING"),
        ]
        coach_flags = [
            SimpleNamespace(knowledge_pack_version="2026-04-22.v1"),
        ]

        payload = _coverage_metrics_payload(
            runs=runs,
            raw_by_run_id=raw_by_run_id,
            followups=followups,
            coach_flags=coach_flags,
        )

        self.assertEqual(payload["overall"]["total_runs"], 3)
        self.assertEqual(payload["overall"]["high_confidence_resolution_rate"], 0.333)
        self.assertEqual(payload["overall"]["no_match_rate"], 0.333)
        self.assertEqual(payload["overall"]["ambiguity_rate"], 0.333)
        self.assertEqual(payload["overall"]["renderer_event_only_rate"], 0.333)
        self.assertEqual(payload["overall"]["coach_flag_rate"], 0.333)
        self.assertEqual(payload["overall"]["prescription_non_response_rate"], 0.5)
        self.assertIn("2026-04-22.v1", payload["by_pack_version"])


if __name__ == "__main__":
    unittest.main()
