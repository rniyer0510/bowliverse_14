import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.persistence.learning_cases import (
    LEARNING_CASE_EVENT_NAME,
    build_learning_case_event,
    build_coach_feedback_learning_case_event,
    symptom_bundle_hash,
    write_learning_case,
    _cluster_key_hash,
)


class LearningCaseBuilderTests(unittest.TestCase):
    def _result(self, diagnosis_status="no_match"):
        return {
            "run_id": "22222222-2222-2222-2222-222222222222",
            "input": {
                "player_id": "11111111-1111-1111-1111-111111111111",
                "hand": "R",
                "age_group": "U16",
                "season": 2026,
            },
            "video": {
                "fps": 30.0,
                "total_frames": 120,
            },
            "deterministic_expert_v1": {
                "knowledge_pack_id": "actionlab_deterministic_expert",
                "knowledge_pack_version": "2026-04-22.v1",
                "capture_quality_v1": {
                    "version": "capture_quality_v1",
                    "status": "USABLE",
                    "notes": [],
                },
                "symptoms": [
                    {
                        "id": "late_trunk_drift",
                        "score": 0.66,
                        "confidence": 0.8,
                        "severity": "moderate",
                    },
                    {
                        "id": "front_leg_softening",
                        "score": 0.78,
                        "confidence": 0.84,
                        "severity": "high",
                    },
                ],
                "mechanism_hypotheses": [
                    {
                        "id": "soft_block_with_trunk_carry",
                        "overall_confidence": 0.34,
                        "support_score": 0.54,
                        "contradiction_penalty": 0.24,
                        "evidence_completeness": 0.62,
                        "contradiction_notes": ["Weak event-chain quality reduces confidence."],
                    },
                    {
                        "id": "gather_disorganization_before_block",
                        "overall_confidence": 0.32,
                        "support_score": 0.49,
                        "contradiction_penalty": 0.2,
                        "evidence_completeness": 0.59,
                        "contradiction_notes": [],
                    },
                ],
                "selection": {
                    "diagnosis_status": diagnosis_status,
                    "primary_mechanism_id": None if diagnosis_status == "no_match" else "soft_block_with_trunk_carry",
                    "primary": {
                        "id": "soft_block_with_trunk_carry",
                        "overall_confidence": 0.34,
                        "support_score": 0.54,
                        "contradiction_penalty": 0.24,
                        "evidence_completeness": 0.62,
                        "contradiction_notes": ["Weak event-chain quality reduces confidence."],
                    },
                },
                "prescription_plan_v1": {
                    "primary_prescription_id": "stack_over_landing_leg",
                },
            },
        }

    def test_build_learning_case_event_for_no_match(self):
        event = build_learning_case_event(
            result=self._result("no_match"),
            account_id="33333333-3333-3333-3333-333333333333",
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["event_name"], LEARNING_CASE_EVENT_NAME)
        self.assertEqual(event["case_type"], "NO_MATCH")
        self.assertEqual(event["priority"], "A")
        self.assertEqual(event["renderer_mode"], "event_only")
        self.assertEqual(event["source_type"], "runtime_gap")
        self.assertEqual(event["followup_outcome"], "NOT_YET_DUE")
        self.assertEqual(
            event["detected_symptoms"],
            ["late_trunk_drift", "front_leg_softening"],
        )

    def test_build_learning_case_event_for_partial_match_returns_none(self):
        event = build_learning_case_event(
            result=self._result("partial_match"),
            account_id="33333333-3333-3333-3333-333333333333",
        )

        self.assertIsNone(event)

    def test_build_learning_case_event_uses_capture_quality_gap_when_unusable(self):
        result = self._result("no_match")
        result["deterministic_expert_v1"]["capture_quality_v1"]["status"] = "UNUSABLE"

        event = build_learning_case_event(
            result=result,
            account_id="33333333-3333-3333-3333-333333333333",
        )

        self.assertEqual(event["suggested_gap_type"], "capture_quality")
        self.assertEqual(event["renderer_mode"], "event_only")

    def test_symptom_bundle_hash_is_order_independent(self):
        symptoms_a = [
            {"id": "late_trunk_drift", "score": 0.66, "confidence": 0.8, "severity": "moderate"},
            {"id": "front_leg_softening", "score": 0.78, "confidence": 0.84, "severity": "high"},
        ]
        symptoms_b = list(reversed(symptoms_a))

        self.assertEqual(symptom_bundle_hash(symptoms_a), symptom_bundle_hash(symptoms_b))

    def test_cluster_key_hash_is_order_independent_for_candidate_ids(self):
        event_a = {
            "source_type": "runtime_gap",
            "knowledge_pack_version": "2026-04-22.v1",
            "case_type": "AMBIGUOUS_MATCH",
            "suggested_gap_type": "weak_distinction_rules",
            "symptom_bundle_hash": "c7c7a0b8",
            "renderer_mode": "partial_evidence",
            "prescription_id": "stack_over_landing_leg",
            "chosen_mechanism": "soft_block_with_trunk_carry",
            "candidate_mechanisms": [
                {"id": "soft_block_with_trunk_carry"},
                {"id": "gather_disorganization_before_block"},
            ],
        }
        event_b = dict(event_a)
        event_b["candidate_mechanisms"] = list(reversed(event_a["candidate_mechanisms"]))

        self.assertEqual(_cluster_key_hash(event_a), _cluster_key_hash(event_b))

    def test_build_coach_feedback_learning_case_event(self):
        event = build_coach_feedback_learning_case_event(
            result=self._result("partial_match"),
            account_id="33333333-3333-3333-3333-333333333333",
            coach_flag_type="wrong_mechanism",
            notes="The block leak is being overstated here.",
            flagged_mechanism_id="soft_block_with_trunk_carry",
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["case_type"], "COACH_FEEDBACK")
        self.assertEqual(event["source_type"], "coach_feedback")
        self.assertEqual(event["suggested_gap_type"], "coach_wrong_mechanism")
        self.assertEqual(event["coach_flag_type"], "wrong_mechanism")


class LearningCaseWriterTests(unittest.TestCase):
    def test_write_learning_case_with_external_session_flushes_without_committing(self):
        db = MagicMock()
        event = {
            "event_name": LEARNING_CASE_EVENT_NAME,
            "event_id": "44444444-4444-4444-4444-444444444444",
            "created_at": "2026-04-22T12:00:00+00:00",
            "knowledge_pack_id": "actionlab_deterministic_expert",
            "knowledge_pack_version": "2026-04-22.v1",
            "run_id": "22222222-2222-2222-2222-222222222222",
            "player_id": "11111111-1111-1111-1111-111111111111",
            "account_id": "33333333-3333-3333-3333-333333333333",
            "source_type": "runtime_gap",
            "case_type": "NO_MATCH",
            "priority": "A",
            "status": "CLUSTERED",
            "symptom_bundle_hash": "c7c7a0b8",
            "clip_metadata": {"fps": 30.0},
            "detected_symptoms": ["front_leg_softening"],
            "candidate_mechanisms": [],
            "chosen_mechanism": None,
            "confidence_breakdown": {
                "support_score": 0.4,
                "contradiction_penalty": 0.2,
                "evidence_completeness": 0.5,
                "overall_confidence": 0.2,
            },
            "contradictions_triggered": [],
            "renderer_mode": "event_only",
            "prescription_id": None,
            "followup_outcome": "NOT_YET_DUE",
            "trigger_reason": "No mechanism exceeded the weak-match threshold.",
            "suggested_gap_type": "missing_mechanism",
        }

        cluster_id = uuid.UUID("55555555-5555-5555-5555-555555555555")
        with patch(
            "app.persistence.learning_cases._get_or_create_cluster",
            return_value=SimpleNamespace(
                learning_case_cluster_id=cluster_id,
                representative_learning_case_id=None,
                updated_at=None,
            ),
        ):
            stored = write_learning_case(event_payload=event, db=db)

        self.assertEqual(stored["learning_case_id"], "44444444-4444-4444-4444-444444444444")
        self.assertEqual(stored["learning_case_cluster_id"], str(cluster_id))
        self.assertGreaterEqual(db.flush.call_count, 1)
        db.commit.assert_not_called()


class LearningCaseOrchestratorTests(unittest.TestCase):
    def test_persist_learning_case_best_effort_swallow_errors(self):
        from app.orchestrator import orchestrator

        result = LearningCaseBuilderTests()._result("no_match")

        with patch.object(
            orchestrator,
            "write_learning_case",
            side_effect=RuntimeError("db unavailable"),
        ):
            orchestrator._persist_learning_case_best_effort(
                request_id="req-1",
                run_id="run-1",
                result=result,
                account_id="33333333-3333-3333-3333-333333333333",
            )


if __name__ == "__main__":
    unittest.main()
