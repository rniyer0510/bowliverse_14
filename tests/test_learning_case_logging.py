import unittest
from unittest.mock import MagicMock, patch

from app.persistence.learning_cases import (
    LEARNING_CASE_EVENT_NAME,
    build_learning_case_event,
    symptom_bundle_hash,
    write_learning_case,
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

    def test_symptom_bundle_hash_is_order_independent(self):
        symptoms_a = [
            {"id": "late_trunk_drift", "score": 0.66, "confidence": 0.8, "severity": "moderate"},
            {"id": "front_leg_softening", "score": 0.78, "confidence": 0.84, "severity": "high"},
        ]
        symptoms_b = list(reversed(symptoms_a))

        self.assertEqual(symptom_bundle_hash(symptoms_a), symptom_bundle_hash(symptoms_b))


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
            "case_type": "NO_MATCH",
            "priority": "A",
            "status": "OPEN",
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

        learning_case_id = write_learning_case(event_payload=event, db=db)

        self.assertEqual(learning_case_id, "44444444-4444-4444-4444-444444444444")
        db.flush.assert_called_once()
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
