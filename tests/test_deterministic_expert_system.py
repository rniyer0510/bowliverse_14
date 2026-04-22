import unittest

from app.clinician.deterministic_expert import DeterministicExpertSystem


class DeterministicExpertSystemTests(unittest.TestCase):
    def setUp(self):
        self.engine = DeterministicExpertSystem()

    def test_selects_soft_block_with_trunk_carry_for_front_leg_and_trunk_pattern(self):
        payload = self.engine.build(
            events={"event_chain": {"quality": 0.84}},
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.81},
            risks=[
                {"risk_id": "front_foot_braking_shock", "signal_strength": 0.78, "confidence": 0.88},
                {"risk_id": "knee_brace_failure", "signal_strength": 0.82, "confidence": 0.9},
                {"risk_id": "trunk_rotation_snap", "signal_strength": 0.55, "confidence": 0.79},
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.46,
                    "confidence": 0.8,
                    "debug": {"sequence_pattern": "in_sync", "sequence_delta_frames": 0},
                },
                {"risk_id": "lateral_trunk_lean", "signal_strength": 0.72, "confidence": 0.86},
                {"risk_id": "foot_line_deviation", "signal_strength": 0.41, "confidence": 0.82},
            ],
            basics={
                "knee_brace_proxy": {"status": "bad", "confidence": 0.92},
                "back_foot_stability": {"status": "ok", "confidence": 0.9},
                "front_foot_toe_alignment": {"status": "semi_open", "confidence": 1.0},
            },
            interpretation={"linear_flow": {"flow_state": "INTERRUPTED", "confidence": 0.82, "contributors": []}},
            estimated_release_speed={
                "available": True,
                "confidence": 0.76,
                "debug": {
                    "elbow_extension_velocity_deg_per_sec": 165.0,
                    "wrist_arm_ratio": 1.12,
                    "shoulder_body_ratio": 0.41,
                    "pelvis_body_ratio": 0.34,
                },
            },
        )

        selection = payload["selection"]
        self.assertEqual(selection["diagnosis_status"], "partial_match")
        self.assertEqual(selection["primary_mechanism_id"], "soft_block_with_trunk_carry")
        self.assertIn("soft_block_trunk_carry_story", selection["selected_render_story_ids"])
        history_binding_ids = [
            binding["id"]
            for binding in payload["history_plan_v1"]["history_bindings"]
        ]
        self.assertIn("transfer_block_stability", history_binding_ids)
        self.assertEqual(
            (payload["archetype_v1"] or {}).get("id"),
            "soft_block_leakage_bowler",
        )
        self.assertEqual(
            payload["mechanism_explanation_v1"]["primary_mechanism"],
            "Soft block with trunk carry",
        )
        self.assertTrue(
            payload["mechanism_explanation_v1"]["load_impact"].startswith(
                "From a load point of view, this pattern may"
            )
        )
        self.assertEqual(
            payload["prescription_plan_v1"]["primary_prescription_id"],
            "stack_over_landing_leg",
        )
        self.assertIn(
            "binding_trends",
            payload["history_plan_v1"],
        )
        self.assertIn(
            "surface_variants",
            payload["mechanism_explanation_v1"],
        )
        self.assertIn(
            "late_arm_chase",
            payload["kinetic_chain_v1"]["pace_translation"],
        )
        self.assertEqual(
            payload["render_reasoning_v1"]["renderer_mode"],
            "partial_evidence",
        )
        self.assertEqual(
            payload["render_reasoning_v1"]["selected_story_id"],
            "soft_block_trunk_carry_story",
        )
        self.assertFalse(payload["prescription_plan_v1"]["suppressed"])

    def test_returns_no_match_when_pattern_is_clean(self):
        payload = self.engine.build(
            events={"event_chain": {"quality": 0.8}},
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.82},
            risks=[
                {"risk_id": "front_foot_braking_shock", "signal_strength": 0.15, "confidence": 0.65},
                {"risk_id": "knee_brace_failure", "signal_strength": 0.15, "confidence": 0.66},
                {"risk_id": "trunk_rotation_snap", "signal_strength": 0.15, "confidence": 0.65},
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.15,
                    "confidence": 0.68,
                    "debug": {"sequence_pattern": "in_sync", "sequence_delta_frames": 0},
                },
                {"risk_id": "lateral_trunk_lean", "signal_strength": 0.15, "confidence": 0.64},
                {"risk_id": "foot_line_deviation", "signal_strength": 0.15, "confidence": 0.64},
            ],
            basics={
                "knee_brace_proxy": {"status": "ok", "confidence": 0.9},
                "back_foot_stability": {"status": "ok", "confidence": 0.88},
                "front_foot_toe_alignment": {"status": "aligned", "confidence": 1.0},
            },
            interpretation={"linear_flow": {"flow_state": "SMOOTH", "confidence": 0.9, "contributors": []}},
            estimated_release_speed={
                "available": True,
                "confidence": 0.78,
                "debug": {
                    "elbow_extension_velocity_deg_per_sec": 105.0,
                    "wrist_arm_ratio": 0.92,
                    "shoulder_body_ratio": 0.28,
                    "pelvis_body_ratio": 0.24,
                },
            },
        )

        self.assertEqual(payload["selection"]["diagnosis_status"], "no_match")
        self.assertIsNone(payload["selection"]["primary_mechanism_id"])
        self.assertEqual(
            payload["mechanism_explanation_v1"]["primary_mechanism"],
            "No confident deterministic mechanism match yet",
        )
        self.assertEqual(
            (payload["archetype_v1"] or {}).get("id"),
            "efficient_transfer_bowler",
        )
        self.assertEqual(payload["prescription_plan_v1"]["prescriptions"], [])

    def test_history_can_stabilize_archetype_and_surface_variants(self):
        payload = self.engine.build(
            events={"event_chain": {"quality": 0.82}},
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.78},
            risks=[
                {"risk_id": "front_foot_braking_shock", "signal_strength": 0.20, "confidence": 0.68},
                {"risk_id": "knee_brace_failure", "signal_strength": 0.18, "confidence": 0.66},
                {"risk_id": "trunk_rotation_snap", "signal_strength": 0.22, "confidence": 0.65},
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.18,
                    "confidence": 0.68,
                    "debug": {"sequence_pattern": "in_sync", "sequence_delta_frames": 0},
                },
                {"risk_id": "lateral_trunk_lean", "signal_strength": 0.18, "confidence": 0.64},
                {"risk_id": "foot_line_deviation", "signal_strength": 0.16, "confidence": 0.64},
            ],
            basics={
                "knee_brace_proxy": {"status": "ok", "confidence": 0.88},
                "back_foot_stability": {"status": "ok", "confidence": 0.87},
                "front_foot_toe_alignment": {"status": "aligned", "confidence": 1.0},
            },
            interpretation={"linear_flow": {"flow_state": "SMOOTH", "confidence": 0.88, "contributors": []}},
            estimated_release_speed={
                "available": True,
                "confidence": 0.74,
                "debug": {
                    "elbow_extension_velocity_deg_per_sec": 110.0,
                    "wrist_arm_ratio": 0.94,
                    "shoulder_body_ratio": 0.29,
                    "pelvis_body_ratio": 0.24,
                },
            },
            prior_results=[
                {
                    "run_id": "prior-1",
                    "result_json": {
                        "deterministic_expert_v1": {
                            "selection": {
                                "diagnosis_status": "partial_match",
                                "primary_mechanism_id": "soft_block_with_trunk_carry",
                            },
                            "archetype_v1": {"id": "soft_block_leakage_bowler"},
                            "metrics": {
                                "front_leg_support_score": {"value": 0.38},
                                "trunk_drift_after_ffc": {"value": 0.71},
                                "transfer_efficiency_score": {"value": 0.44},
                            },
                        }
                    },
                },
                {
                    "run_id": "prior-2",
                    "result_json": {
                        "deterministic_expert_v1": {
                            "selection": {
                                "diagnosis_status": "partial_match",
                                "primary_mechanism_id": "soft_block_with_trunk_carry",
                            },
                            "archetype_v1": {"id": "soft_block_leakage_bowler"},
                            "metrics": {
                                "front_leg_support_score": {"value": 0.42},
                                "trunk_drift_after_ffc": {"value": 0.67},
                                "transfer_efficiency_score": {"value": 0.48},
                            },
                        }
                    },
                },
            ],
            account_role="coach",
        )

        self.assertEqual(payload["selection"]["diagnosis_status"], "no_match")
        self.assertEqual(
            (payload["archetype_v1"] or {}).get("id"),
            "soft_block_leakage_bowler",
        )
        self.assertEqual(
            (payload["archetype_v1"] or {}).get("selection_basis"),
            "historical_consensus",
        )
        self.assertEqual(
            payload["mechanism_explanation_v1"]["selected_surface"],
            "coach",
        )
        self.assertIn(
            "coach",
            payload["mechanism_explanation_v1"]["surface_variants"],
        )
        self.assertEqual(
            payload["history_plan_v1"]["prior_run_count"],
            2,
        )
        self.assertEqual(
            payload["render_reasoning_v1"]["renderer_mode"],
            "event_only",
        )
        self.assertTrue(payload["prescription_plan_v1"]["suppressed"])

    def test_unusable_capture_quality_short_circuits_mechanism_scoring(self):
        payload = self.engine.build(
            events={
                "event_chain": {"quality": 0.12, "ordered": False},
                "release": {"frame": None, "confidence": 0.1},
            },
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.2},
            risks=[],
            basics={},
            interpretation={"linear_flow": {"flow_state": "FRAGMENTED", "confidence": 0.2, "contributors": []}},
            estimated_release_speed={"available": False, "confidence": 0.0, "debug": {}},
        )

        self.assertEqual(payload["capture_quality_v1"]["status"], "UNUSABLE")
        self.assertEqual(payload["selection"]["diagnosis_status"], "no_match")
        self.assertEqual(payload["mechanism_hypotheses"], [])
        self.assertEqual(payload["kinetic_chain_v1"]["mechanism_hypotheses"], [])
        self.assertEqual(payload["prescription_plan_v1"]["prescriptions"], [])
        self.assertEqual(
            payload["selection"]["capture_quality_status"],
            "UNUSABLE",
        )
        self.assertEqual(
            payload["render_reasoning_v1"]["renderer_mode"],
            "event_only",
        )
        self.assertTrue(payload["prescription_plan_v1"]["suppressed"])


if __name__ == "__main__":
    unittest.main()
