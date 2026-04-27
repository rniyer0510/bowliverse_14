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
        self.assertNotIn("evidence_flags", payload["metrics"])
        self.assertEqual(payload["presentation_payload_v1"]["state"], "MATCH")
        self.assertEqual(
            payload["presentation_payload_v1"]["match"]["primary_mechanism_id"],
            "soft_block_with_trunk_carry",
        )
        coach_diagnosis = payload["coach_diagnosis_v1"]
        self.assertEqual(coach_diagnosis["state"], "MATCH")
        self.assertEqual(
            coach_diagnosis["primary_mechanism"]["id"],
            "soft_block_with_trunk_carry",
        )
        self.assertEqual(
            coach_diagnosis["visible_symptom"]["id"],
            "front_leg_softening",
        )
        self.assertTrue(coach_diagnosis["upper_body_contributors"])
        self.assertTrue(coach_diagnosis["lower_body_contributors"])
        self.assertEqual(
            coach_diagnosis["renderer_bindings"]["selected_story_id"],
            "soft_block_trunk_carry_story",
        )
        self.assertIn(
            "transfer_block_stability",
            [
                binding["id"]
                for binding in coach_diagnosis["history_bindings"]["bindings"]
            ],
        )
        self.assertEqual(
            coach_diagnosis["first_priority"]["prescription_id"],
            "stack_over_landing_leg",
        )
        self.assertTrue(coach_diagnosis["do_not_change_yet"])
        self.assertEqual(
            coach_diagnosis["primary_break_point"]["phase_id"],
            "transfer_and_block",
        )
        self.assertEqual(
            coach_diagnosis["primary_break_point"]["title"],
            "Transfer and block",
        )
        self.assertEqual(
            coach_diagnosis["kinetic_chain_status"]["label"],
            "Breaking at transfer",
        )
        self.assertEqual(coach_diagnosis["root_cause"]["status"], "clear")
        self.assertEqual(
            coach_diagnosis["root_cause"]["mechanism_id"],
            "soft_block_with_trunk_carry",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["title"],
            "Soft block with trunk carry",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["where_it_starts"]["phase_id"],
            "transfer_and_block",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["primary_driver"]["id"],
            "front_leg_support_score",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["compensation"]["id"],
            "lateral_trunk_lean",
        )
        self.assertIn(
            "stable base",
            coach_diagnosis["root_cause"]["why_it_is_happening"].lower(),
        )
        self.assertTrue(coach_diagnosis["root_cause"]["chain_story"])
        self.assertEqual(
            coach_diagnosis["root_cause"]["biomechanics_basis"]["target_type"],
            "mechanism",
        )
        self.assertTrue(
            coach_diagnosis["root_cause"]["biomechanics_basis"]["primary_claim"]
        )
        self.assertTrue(
            coach_diagnosis["root_cause"]["biomechanics_basis"]["principle_claims"]
        )
        self.assertTrue(
            coach_diagnosis["root_cause"]["biomechanics_basis"]["cue_ready_proof_lines"]
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["story_id"],
            "soft_block_trunk_carry_story",
        )
        self.assertIn(
            "front_leg",
            coach_diagnosis["root_cause"]["renderer_guidance"]["focus_regions"],
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["anchor_risk_ids"]["ffc"],
            "knee_brace_failure",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["anchor_risk_ids"]["release"],
            "lateral_trunk_lean",
        )
        self.assertTrue(
            coach_diagnosis["root_cause"]["renderer_guidance"]["warning_hotspots_allowed"]
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["ffc"]["risk_id"],
            "knee_brace_failure",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["ffc"]["proof_step"]["title"],
            "Where It Starts",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["ffc"]["proof_step"]["headline"],
            "Front leg doesn't hold strong at landing.",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["ffc"]["region_priority"],
            ["knee", "shin", "groin"],
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["release"]["risk_id"],
            "lateral_trunk_lean",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["release"]["proof_step"]["title"],
            "What Happens Next",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["release"]["proof_step"]["headline"],
            "Then the body falls away through release.",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["phase_targets"]["release"]["region_priority"],
            ["side_trunk", "upper_trunk", "lumbar"],
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["load_watch_text"],
            "Front knee / leg chain\nLower back / side trunk",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["simple_symptom_text"],
            "Front leg doesn't hold strong at landing, then the body falls away.",
        )
        self.assertEqual(
            coach_diagnosis["root_cause"]["renderer_guidance"]["simple_load_watch_text"],
            "Front leg works hard.\nLower back works hard too.",
        )
        self.assertEqual(
            coach_diagnosis["easy_explanation"]["headline"],
            "Front leg does not hold, so the body spills through",
        )
        self.assertEqual(
            coach_diagnosis["easy_explanation"]["what_to_notice"],
            "Front leg doesn't hold strong at landing, then the body falls away.",
        )
        self.assertEqual(
            coach_diagnosis["easy_explanation"]["why_it_happens"],
            "The front leg does not hold strong at landing, so the body keeps moving through release.",
        )
        self.assertEqual(
            coach_diagnosis["easy_explanation"]["first_fix"],
            "Feel the chest get over the front leg sooner instead of drifting past it.",
        )
        self.assertEqual(
            coach_diagnosis["easy_explanation"]["check_next"],
            "Landing is turning into a cleaner power move.",
        )
        self.assertTrue(
            coach_diagnosis["root_cause"]["renderer_guidance"]["cue_points"]
        )
        self.assertEqual(
            coach_diagnosis["acceptance_summary"]["overall_band"],
            "problematic",
        )
        self.assertEqual(
            coach_diagnosis["key_metrics"]["front_leg_support_score"]["acceptance_band"],
            "problematic",
        )
        self.assertEqual(
            coach_diagnosis["change_strategy"]["change_size"],
            "micro",
        )
        self.assertEqual(
            coach_diagnosis["change_strategy"]["adoption_risk"],
            "low",
        )
        self.assertTrue(
            coach_diagnosis["change_strategy"]["why_smallest_useful_change"]
        )
        self.assertEqual(
            coach_diagnosis["change_strategy"]["match_pressure_risk"],
            "low",
        )
        self.assertTrue(coach_diagnosis["change_strategy"]["selection_window_safety"])
        self.assertIsNotNone(coach_diagnosis["change_reaction"])
        self.assertIsNotNone(coach_diagnosis["evidence_basis"])
        self.assertTrue(coach_diagnosis["change_reaction"]["near_term_positive"])
        self.assertTrue(coach_diagnosis["change_reaction"]["near_term_negative"])
        self.assertTrue(coach_diagnosis["change_reaction"]["long_term_positive"])
        self.assertEqual(
            coach_diagnosis["evidence_basis"]["primary_mechanism"]["canonical_concept"]["concept_id"],
            "canonical_transfer_break_story",
        )
        self.assertIn(
            "Front-foot braking load",
            [item["title"] for item in coach_diagnosis["lower_body_contributors"]],
        )
        frontend_surface = payload["frontend_surface_v1"]
        self.assertEqual(frontend_surface["state"], "MATCH")
        self.assertEqual(frontend_surface["headline"], "Leak at landing")
        self.assertEqual(len(frontend_surface["summary_lines"]), 3)
        self.assertEqual(
            frontend_surface["hero"]["kinetic_chain_status"]["label"],
            "Breaking at transfer",
        )
        self.assertEqual(
            frontend_surface["body"]["root_cause"]["mechanism_id"],
            "soft_block_with_trunk_carry",
        )
        self.assertEqual(
            frontend_surface["body"]["root_cause"]["renderer_guidance"]["story_id"],
            "soft_block_trunk_carry_story",
        )
        self.assertEqual(
            frontend_surface["easy_explanation"]["why_it_happens"],
            "The front leg does not hold strong at landing, so the body keeps moving through release.",
        )
        self.assertEqual(
            frontend_surface["guidance"]["first_priority"]["prescription_id"],
            "stack_over_landing_leg",
        )
        self.assertEqual(
            payload["presentation_payload_v1"]["match"]["root_cause"]["mechanism_id"],
            "soft_block_with_trunk_carry",
        )

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
        coach_diagnosis = payload["coach_diagnosis_v1"]
        self.assertEqual(coach_diagnosis["state"], "NO_MATCH")
        self.assertIsNone(coach_diagnosis["primary_mechanism"])
        self.assertIsNotNone(coach_diagnosis["holdback"])
        self.assertTrue(coach_diagnosis["what_is_ok"])
        self.assertEqual(
            coach_diagnosis["kinetic_chain_status"]["label"],
            "Connected",
        )
        self.assertEqual(
            coach_diagnosis["acceptance_summary"]["overall_band"],
            "acceptable",
        )
        self.assertEqual(coach_diagnosis["root_cause"]["status"], "no_clear_problem")
        self.assertIsNone(coach_diagnosis["root_cause"]["mechanism_id"])
        self.assertIsNone(coach_diagnosis["root_cause"]["renderer_guidance"])
        self.assertEqual(coach_diagnosis["what_is_not_ok"], [])
        self.assertIsNone(coach_diagnosis["visible_symptom"])
        self.assertIsNone(coach_diagnosis["primary_break_point"]["phase_id"])
        self.assertEqual(coach_diagnosis["phase_anchored_findings"], [])
        self.assertEqual(coach_diagnosis["holdback"]["top_candidates"], [])
        self.assertEqual(coach_diagnosis["change_strategy"]["change_size"], "micro")
        self.assertEqual(coach_diagnosis["change_strategy"]["adoption_risk"], "low")
        self.assertIsNone(coach_diagnosis["change_reaction"])
        frontend_surface = payload["frontend_surface_v1"]
        self.assertEqual(frontend_surface["headline"], "Action is connected")
        self.assertEqual(
            frontend_surface["hero"]["primary_break_point"]["phase_id"],
            None,
        )
        self.assertEqual(frontend_surface["body"]["what_is_not_ok"], [])
        self.assertEqual(frontend_surface["body"]["phase_anchored_findings"], [])
        self.assertEqual(
            frontend_surface["body"]["root_cause"]["status"],
            "no_clear_problem",
        )
        self.assertEqual(frontend_surface["holdback"]["top_candidates"], [])
        self.assertEqual(
            payload["presentation_payload_v1"]["no_match"]["root_cause"]["status"],
            "no_clear_problem",
        )

    def test_acceptable_and_workable_risk_window_does_not_create_a_pathology_story(self):
        payload = self.engine.build(
            events={"event_chain": {"quality": 0.82}},
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.8},
            risks=[
                {"risk_id": "front_foot_braking_shock", "signal_strength": 0.22, "confidence": 0.72},
                {"risk_id": "knee_brace_failure", "signal_strength": 0.24, "confidence": 0.73},
                {"risk_id": "trunk_rotation_snap", "signal_strength": 0.18, "confidence": 0.7},
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.32,
                    "confidence": 0.74,
                    "debug": {"sequence_pattern": "in_sync", "sequence_delta_frames": 0},
                },
                {"risk_id": "lateral_trunk_lean", "signal_strength": 0.22, "confidence": 0.72},
                {"risk_id": "foot_line_deviation", "signal_strength": 0.28, "confidence": 0.74},
            ],
            basics={
                "knee_brace_proxy": {"status": "ok", "confidence": 0.9},
                "back_foot_stability": {"status": "ok", "confidence": 0.88},
                "front_foot_toe_alignment": {"status": "aligned", "confidence": 1.0},
            },
            interpretation={"linear_flow": {"flow_state": "SMOOTH", "confidence": 0.88, "contributors": []}},
            estimated_release_speed={
                "available": True,
                "confidence": 0.76,
                "debug": {
                    "elbow_extension_velocity_deg_per_sec": 118.0,
                    "wrist_arm_ratio": 0.96,
                    "shoulder_body_ratio": 0.29,
                    "pelvis_body_ratio": 0.24,
                },
            },
        )

        self.assertEqual(payload["selection"]["diagnosis_status"], "no_match")
        coach_diagnosis = payload["coach_diagnosis_v1"]
        self.assertEqual(coach_diagnosis["state"], "NO_MATCH")
        self.assertEqual(coach_diagnosis["supporting_contributors"], [])
        self.assertEqual(
            coach_diagnosis["kinetic_chain_status"]["label"],
            "Connected",
        )
        self.assertEqual(
            coach_diagnosis["acceptance_summary"]["counts"]["problematic"],
            0,
        )
        self.assertEqual(coach_diagnosis["what_is_not_ok"], [])
        self.assertEqual(
            coach_diagnosis["easy_explanation"]["headline"],
            "Action is working well.",
        )
        self.assertEqual(payload["frontend_surface_v1"]["headline"], "Action is connected")
        self.assertEqual(
            payload["frontend_surface_v1"]["easy_explanation"]["what_to_notice"],
            "The action stays connected through landing and release.",
        )

    def test_player_surface_filters_coach_only_diagnosis_detail(self):
        payload = self.engine.build(
            events={"event_chain": {"quality": 0.84}},
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.81},
            risks=[
                {"risk_id": "front_foot_braking_shock", "signal_strength": 0.78, "confidence": 0.88},
                {"risk_id": "knee_brace_failure", "signal_strength": 0.82, "confidence": 0.9},
                {"risk_id": "lateral_trunk_lean", "signal_strength": 0.72, "confidence": 0.86},
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
            account_role="player",
        )

        coach_diagnosis = payload["coach_diagnosis_v1"]
        self.assertEqual(coach_diagnosis["detail_policy_id"], "player")
        self.assertIsNotNone(coach_diagnosis["root_cause"])
        self.assertIsNotNone(
            coach_diagnosis["root_cause"]["biomechanics_basis"]
        )
        self.assertEqual(coach_diagnosis["supporting_contributors"], [])
        self.assertEqual(coach_diagnosis["upper_body_contributors"], [])
        self.assertEqual(coach_diagnosis["lower_body_contributors"], [])
        self.assertEqual(coach_diagnosis["phase_anchored_findings"], [])
        self.assertIsNone(coach_diagnosis["change_reaction"])
        self.assertIsNone(coach_diagnosis["evidence_basis"])

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
            account_role="reviewer",
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
            "reviewer",
        )
        self.assertIn(
            "reviewer",
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
        self.assertEqual(payload["presentation_payload_v1"]["state"], "NO_MATCH")
        self.assertEqual(
            payload["presentation_payload_v1"]["selected_surface"],
            "reviewer",
        )
        self.assertEqual(
            payload["coach_diagnosis_v1"]["history_bindings"]["prior_run_count"],
            2,
        )

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
        self.assertEqual(payload["presentation_payload_v1"]["state"], "NO_MATCH")
        self.assertEqual(
            payload["presentation_payload_v1"]["capture_quality_status"],
            "UNUSABLE",
        )
        self.assertEqual(payload["coach_diagnosis_v1"]["capture_quality_status"], "UNUSABLE")
        self.assertEqual(payload["coach_diagnosis_v1"]["state"], "NO_MATCH")
        self.assertEqual(payload["coach_diagnosis_v1"]["change_strategy"]["change_size"], "hold")
        self.assertIsNone(payload["coach_diagnosis_v1"]["change_reaction"])

    def test_close_unclear_release_clip_is_weak_not_unusable(self):
        payload = self.engine.build(
            events={
                "event_chain": {"quality": 0.56, "ordered": True},
                "bfc": {"frame": 173, "confidence": 0.58},
                "ffc": {"frame": 176, "confidence": 0.63},
                "release": {
                    "frame": 190,
                    "confidence": 0.60,
                    "method": "peak_plus_offset",
                },
            },
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.74},
            risks=[],
            basics={},
            interpretation={"linear_flow": {"flow_state": "USABLE", "confidence": 0.58, "contributors": []}},
            estimated_release_speed={
                "available": True,
                "confidence": 0.70,
                "debug": {
                    "body_height_ratio": 0.40,
                    "shoulder_body_ratio": 0.98,
                },
            },
        )

        self.assertEqual(payload["capture_quality_v1"]["status"], "WEAK")
        self.assertIn("release_unclear", payload["capture_quality_v1"]["notes"])
        self.assertIn("camera_too_close", payload["capture_quality_v1"]["notes"])
        self.assertIn("framing_unusable", payload["capture_quality_v1"]["notes"])
        self.assertEqual(
            payload["presentation_payload_v1"]["capture_quality_status"],
            "WEAK",
        )
        self.assertTrue(payload["analysis_capabilities_v1"]["can_assess_structure"])
        self.assertFalse(payload["analysis_capabilities_v1"]["can_estimate_speed"])
        self.assertFalse(payload["analysis_capabilities_v1"]["can_assess_legality"])

    def test_release_missing_makes_capture_quality_unusable(self):
        payload = self.engine.build(
            events={
                "event_chain": {"quality": 0.62, "ordered": True},
                "bfc": {"frame": 173, "confidence": 0.58},
                "ffc": {"frame": 176, "confidence": 0.63},
                "release": {"frame": None, "confidence": 0.0, "method": "peak_plus_offset"},
            },
            action={"action": "SEMI_OPEN", "intent": "semi_open", "confidence": 0.74},
            risks=[],
            basics={},
            interpretation={"linear_flow": {"flow_state": "USABLE", "confidence": 0.58, "contributors": []}},
            estimated_release_speed={
                "available": False,
                "confidence": 0.0,
                "reason": "missing_release_window",
                "debug": {},
            },
        )

        self.assertEqual(payload["capture_quality_v1"]["status"], "UNUSABLE")
        self.assertIn("release_missing", payload["capture_quality_v1"]["notes"])
        self.assertFalse(payload["analysis_capabilities_v1"]["can_assess_structure"])
        self.assertFalse(payload["analysis_capabilities_v1"]["can_assess_release"])


if __name__ == "__main__":
    unittest.main()
