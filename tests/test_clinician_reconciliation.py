import unittest
from unittest.mock import patch

from app.clinician.interpreter import ClinicianInterpreter


class ClinicianReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.ci = ClinicianInterpreter()

    def test_mixed_action_without_high_confidence_risks_is_monitor_not_sound(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.20,
                    "confidence": 0.25,
                },
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.22,
                    "confidence": 0.22,
                },
            ],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.90,
                    "contributors": [],
                }
            },
            action={"action": "MIXED", "confidence": 0.72},
        )

        self.assertEqual(result["summary"]["overall_severity"], "MODERATE")
        self.assertEqual(result["summary"]["recommendation_level"], "MONITOR")
        self.assertEqual(result["summary"]["primary_pillar"], "POSTURE")
        self.assertIn("mixed", result["summary"]["overall_assessment"].lower())
        self.assertIn(
            "mixed",
            result["comprehensive_why"]["primary_cause"].lower(),
        )
        self.assertEqual(result["pillars"]["posture"]["status"], "WATCH")
        self.assertEqual(result["pillars"]["posture"]["display_label"], "Balance")
        self.assertIn("what_we_saw", result["pillars"]["posture"])
        self.assertIn("why_it_matters", result["comprehensive_why"])

    def test_unknown_action_without_high_confidence_risks_is_cautious(self):
        result = self.ci.build(
            elbow={"extension_deg": 10.0, "verdict": "LEGAL"},
            risks=[],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.85,
                    "contributors": [],
                }
            },
            action={"action": "UNKNOWN", "confidence": 0.0},
        )

        self.assertEqual(result["summary"]["overall_severity"], "MODERATE")
        self.assertEqual(result["summary"]["primary_pillar"], "POSTURE")
        self.assertIn("not clear enough", result["summary"]["overall_assessment"].lower())
        self.assertIn(
            "too unclear",
            result["comprehensive_why"]["primary_cause"].lower(),
        )
        self.assertIn("what_to_check_with_coach", result["comprehensive_why"])

    def test_rotation_cluster_surfaces_power_summary_when_flow_breaks_down(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.82,
                    "confidence": 0.91,
                },
                {
                    "risk_id": "front_foot_braking_shock",
                    "signal_strength": 0.77,
                    "confidence": 0.86,
                },
            ],
            interpretation={
                "linear_flow": {
                    "flow_state": "interrupted",
                    "confidence": 0.88,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 0.72},
        )

        self.assertEqual(result["summary"]["primary_pillar"], "POWER")
        self.assertIn("run-up", result["summary"]["overall_assessment"].lower())
        self.assertEqual(result["pillars"]["power"]["status"], "REVIEW")
        self.assertIn("flow", result["comprehensive_why"]["primary_cause"].lower())
        self.assertIn("why_flagged", result["pillars"]["power"])
        self.assertIn("what_to_check_with_coach", result["pillars"]["power"])

    def test_lateral_lean_why_it_matters_mentions_side_body_load(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.83,
                    "confidence": 0.92,
                },
            ],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.88,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 0.72},
        )

        lean_risk = result["risks"][0]
        self.assertIn("side of the body", lean_risk["why_it_matters"].lower())
        self.assertIn("side of the body", result["comprehensive_why"]["why_it_matters"].lower())

    def test_scorecard_explains_overall_benchmark_in_plain_english(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.90,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 0.78},
        )

        scorecard = result["scorecard"]
        self.assertIsInstance(scorecard["overall"]["score"], int)
        self.assertGreaterEqual(scorecard["overall"]["score"], 0)
        self.assertLessEqual(scorecard["overall"]["score"], 100)
        self.assertLess(scorecard["overall"]["score"], 100)
        self.assertGreaterEqual(scorecard["overall"]["score"], 88)
        self.assertIn("balanced", scorecard["overall"]["what_100_means"].lower())
        self.assertIn("release", scorecard["overall"]["what_100_means"].lower())
        self.assertIn("balance", scorecard["benchmark"]["headline"].lower())
        self.assertIn("overall_score", result["summary"])
        self.assertIn("confidence_score", result["summary"])
        rating_system = result["rating_system_v2"]
        self.assertIn("player_view", rating_system)
        self.assertIn("coach_view", rating_system)
        self.assertIn("expert_view", rating_system)
        self.assertIn("upper_body_alignment", rating_system["player_view"]["metrics"])
        self.assertIn("momentum_forward", rating_system["player_view"]["metrics"])
        self.assertIn("high score", rating_system["overall"]["what_high_score_means"].lower())
        self.assertNotIn("ideal action standard", rating_system["overall"]["what_it_measures"].lower())
        self.assertTrue(rating_system["coach_guidance_line"].startswith("Talk to a coach"))

    def test_scorecard_surfaces_pillar_reasons_and_confidence_separately(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.82,
                    "confidence": 0.91,
                },
                {
                    "risk_id": "foot_line_deviation",
                    "signal_strength": 0.68,
                    "confidence": 0.87,
                },
            ],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.88,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 0.76},
        )

        scorecard = result["scorecard"]
        self.assertIn("read the clip", scorecard["confidence"]["what_it_measures"].lower())
        self.assertIn("upright", scorecard["pillars"]["balance"]["what_it_measures"].lower())
        self.assertTrue(scorecard["pillars"]["balance"]["main_reasons_points_were_lost"])
        joined_reasons = " ".join(scorecard["pillars"]["balance"]["main_reasons_points_were_lost"]).lower()
        self.assertTrue(
            "lateral" in joined_reasons
            or "lean" in joined_reasons
            or "foot" in joined_reasons
        )
        rating_system = result["rating_system_v2"]
        coach_view = rating_system["coach_view"]
        self.assertIn("safety", coach_view["metrics"])
        self.assertTrue(coach_view["top_deductions"])
        self.assertTrue(coach_view["needs_to_improve"])
        expert_view = rating_system["expert_view"]
        self.assertTrue(expert_view["features"])
        self.assertIn("legacy_scorecard", expert_view)

    def test_unknown_action_caps_overall_score_and_explains_limit(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.85,
                    "contributors": [],
                }
            },
            action={"action": "UNKNOWN", "confidence": 0.0},
        )

        scorecard = result["scorecard"]
        self.assertLessEqual(scorecard["overall"]["score"], 80)
        self.assertIn("capped", scorecard["overall"]["benchmark_limit_reason"].lower())
        self.assertIn("action view", scorecard["overall"]["benchmark_limit_reason"].lower())

    def test_report_story_v1_picks_front_leg_plus_trunk_lean_story(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[
                {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                    "visual": {"image_url": "http://example.test/knee.png"},
                },
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                    "visual": {"image_url": "http://example.test/lean.png"},
                },
                {
                    "risk_id": "trunk_rotation_snap",
                    "signal_strength": 0.92,
                    "confidence": 0.85,
                },
            ],
            interpretation={
                "linear_flow": {
                    "flow_state": "unknown",
                    "confidence": 0.78,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 1.0},
        )

        story = result["report_story_v1"]
        self.assertEqual(story["theme"], "base_balance")
        self.assertIn("front leg", story["headline"].lower())
        self.assertIn("body lean away", story["why_this_matters"].lower())
        self.assertIn("back foot and front foot", story["what_to_try"].lower())
        self.assertIn("front leg", story["coach_check"].lower())
        self.assertEqual(story["hero_risk_id"], "knee_brace_failure")
        self.assertEqual(story["hero_visual"]["image_url"], "http://example.test/knee.png")
        self.assertTrue(story["key_metrics"])

    def test_report_story_v1_uses_working_pattern_story_for_high_overall_score(self):
        built_risks = self.ci.build_risks(
            [
                {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                    "visual": {"image_url": "http://example.test/knee.png"},
                },
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                },
            ]
        )

        story = self.ci._build_report_story_v1(
            rating_system_v2={
                "overall": {"score": 84},
                "player_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 78},
                        "lower_body_alignment": {"score": 78},
                        "whole_body_alignment": {"score": 82},
                        "momentum_forward": {"score": 78},
                    }
                },
                "expert_view": {
                    "features": [
                        {
                            "key": "front_leg_support",
                            "label": "Front-Leg Support",
                            "score": 35,
                            "deduction": 14,
                        }
                    ]
                },
                "top_deductions": [
                    {"key": "front_leg_support", "label": "Front-Leg Support", "deduction": 14},
                    {"key": "trunk_lean", "label": "Trunk Lean", "deduction": 13},
                ],
            },
            risks=built_risks,
            action={"action": "SEMI_OPEN", "confidence": 0.43},
        )

        self.assertEqual(story["theme"], "working_pattern")
        self.assertIn("working pattern", story["headline"].lower())
        self.assertIn("front leg holds up at landing", story["why_this_matters"].lower())
        self.assertIn("front leg keeps its shape", story["what_to_try"].lower())
        self.assertEqual(story["watch_focus"]["key"], "front_leg_support")
        self.assertEqual(
            [metric["label"] for metric in story["key_metrics"]],
            [
                "Upper Body Alignment",
                "Lower Body Alignment",
                "Whole Body Alignment",
                "Momentum Forward",
            ],
        )

    def test_report_story_v1_side_on_working_pattern_surfaces_sequence_watch_focus(self):
        built_risks = self.ci.build_risks(
            [
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.72,
                    "confidence": 0.85,
                },
                {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                    "visual": {"image_url": "http://example.test/knee.png"},
                },
            ]
        )

        story = self.ci._build_report_story_v1(
            rating_system_v2={
                "overall": {"score": 86},
                "confidence": {"score": 86},
                "player_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 80},
                        "lower_body_alignment": {"score": 80},
                        "whole_body_alignment": {"score": 84},
                        "momentum_forward": {"score": 80},
                    }
                },
                "expert_view": {
                    "features": [
                        {
                            "key": "upper_body_opening",
                            "label": "Upper Body Opening",
                            "score": 63,
                            "deduction": 8,
                        }
                    ]
                },
                "top_deductions": [
                    {"key": "upper_body_opening", "label": "Upper Body Opening", "deduction": 8},
                ],
            },
            risks=built_risks,
            action={"action": "SIDE_ON", "confidence": 0.95},
        )

        self.assertEqual(story["theme"], "working_pattern")
        self.assertIn("side-on", story["headline"].lower())
        self.assertEqual(story["watch_focus"]["key"], "upper_body_opening")
        self.assertIn("shoulders and hips", story["why_this_matters"].lower())
        self.assertEqual(story["hero_risk_id"], "hip_shoulder_mismatch")
        self.assertIsNone(story["hero_visual"])

    def test_report_story_v1_strong_clip_without_visual_support_stays_generic_when_confidence_is_low(self):
        built_risks = self.ci.build_risks(
            [
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.72,
                    "confidence": 0.85,
                },
            ]
        )

        story = self.ci._build_report_story_v1(
            rating_system_v2={
                "overall": {"score": 86},
                "confidence": {"score": 76},
                "player_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 80},
                        "lower_body_alignment": {"score": 80},
                        "whole_body_alignment": {"score": 84},
                        "momentum_forward": {"score": 80},
                    }
                },
                "expert_view": {
                    "features": [
                        {
                            "key": "upper_body_opening",
                            "label": "Upper Body Opening",
                            "score": 63,
                            "deduction": 8,
                        }
                    ]
                },
                "top_deductions": [
                    {"key": "upper_body_opening", "label": "Upper Body Opening", "deduction": 8},
                ],
            },
            risks=built_risks,
            action={"action": "SIDE_ON", "confidence": 0.95},
        )

        self.assertEqual(story["theme"], "working_pattern")
        self.assertIsNone(story["watch_focus"])
        self.assertEqual(story["why_this_matters"], "The main action shape is holding together well in this clip, even though a few measured areas are still worth watching.")
        self.assertEqual(story["what_to_try"], "Keep repeating the same simple action shape from your better clips.")
        self.assertEqual(story["coach_check"], "Ask if there is just one small thing to keep watching instead of changing the whole action.")
        self.assertIsNone(story["hero_risk_id"])
        self.assertIsNone(story["hero_visual"])

    def test_report_story_v1_strong_clip_with_close_deductions_stays_generic_when_confidence_is_soft(self):
        built_risks = self.ci.build_risks(
            [
                {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                    "visual": {"image_url": "http://example.test/knee.png"},
                },
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                },
            ]
        )

        story = self.ci._build_report_story_v1(
            rating_system_v2={
                "overall": {"score": 90},
                "confidence": {"score": 68},
                "player_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 84},
                        "lower_body_alignment": {"score": 84},
                        "whole_body_alignment": {"score": 88},
                        "momentum_forward": {"score": 84},
                    }
                },
                "expert_view": {
                    "features": [
                        {
                            "key": "front_leg_support",
                            "label": "Front-Leg Support",
                            "score": 35,
                            "deduction": 14,
                        },
                        {
                            "key": "trunk_lean",
                            "label": "Trunk Lean",
                            "score": 39,
                            "deduction": 13,
                        },
                    ]
                },
                "top_deductions": [
                    {"key": "front_leg_support", "label": "Front-Leg Support", "deduction": 14},
                    {"key": "trunk_lean", "label": "Trunk Lean", "deduction": 13},
                ],
            },
            risks=built_risks,
            action={"action": "SEMI_OPEN", "confidence": 0.43},
        )

        self.assertEqual(story["theme"], "working_pattern")
        self.assertIsNone(story["watch_focus"])
        self.assertEqual(story["why_this_matters"], "The main action shape is holding together well in this clip, even though a few measured areas are still worth watching.")
        self.assertEqual(story["what_to_try"], "Keep repeating the same simple action shape from your better clips.")
        self.assertEqual(story["coach_check"], "Ask if there is just one small thing to keep watching instead of changing the whole action.")
        self.assertIsNone(story["hero_risk_id"])
        self.assertIsNone(story["hero_visual"])

    def test_primary_pillar_follows_main_score_loss_not_tuple_order(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[],
            interpretation={
                "linear_flow": {
                    "flow_state": "unknown",
                    "confidence": 0.48,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 0.82},
        )

        self.assertEqual(result["summary"]["primary_pillar"], "POWER")
        self.assertIn(
            "carry score capped",
            result["scorecard"]["pillars"]["carry"]["benchmark_limit_reason"].lower(),
        )
        self.assertGreater(
            result["scorecard"]["pillars"]["balance"]["score"],
            result["scorecard"]["pillars"]["carry"]["score"],
        )

    def test_multiple_visible_deductions_pull_score_down_honestly(self):
        result = self.ci.build(
            elbow={"extension_deg": 12.0, "verdict": "LEGAL"},
            risks=[
                {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.79,
                    "confidence": 0.88,
                },
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": 0.73,
                    "confidence": 0.86,
                },
                {
                    "risk_id": "foot_line_deviation",
                    "signal_strength": 0.38,
                    "confidence": 0.35,
                },
                {
                    "risk_id": "trunk_rotation_snap",
                    "signal_strength": 0.67,
                    "confidence": 0.82,
                },
            ],
            interpretation={
                "linear_flow": {
                    "flow_state": "SMOOTH",
                    "confidence": 0.84,
                    "contributors": [],
                }
            },
            action={"action": "SEMI_OPEN", "confidence": 0.78},
        )

        scorecard = result["scorecard"]
        self.assertLessEqual(scorecard["overall"]["score"], 80)
        self.assertLess(scorecard["pillars"]["balance"]["score"], 80)
        self.assertLess(scorecard["pillars"]["body_load"]["score"], 82)
        joined = " ".join(scorecard["overall"]["main_reasons_points_were_lost"]).lower()
        self.assertTrue("foot" in joined or "lean" in joined or "trunk" in joined)

    def test_benchmark_style_protection_summary_softens_to_low(self):
        with patch.object(
            self.ci,
            "build_summary",
            return_value={
                "overall_assessment": "This clip needs a closer look because some parts of the body may be taking extra load.",
                "overall_severity": "VERY_HIGH",
                "recommendation_level": "CORRECT",
                "confidence": 0.7,
                "primary_pillar": "PROTECTION",
            },
        ), patch.object(
            self.ci,
            "build_scorecard",
            return_value={
                "overall": {"score": 33},
                "confidence": {"score": 68},
            },
        ), patch.object(
            self.ci,
            "build_rating_system_v2",
            return_value={
                "overall": {"score": 66},
                "confidence": {"score": 68},
            },
        ), patch.object(
            self.ci,
            "build_pillars",
            return_value={
                "posture": {"status": "REVIEW"},
                "power": {"status": "REVIEW"},
                "protection": {"status": "REVIEW"},
            },
        ), patch.object(
            self.ci,
            "build_risks",
            return_value=[],
        ), patch.object(
            self.ci,
            "build_chain",
            return_value={"confidence": 0.7},
        ), patch.object(
            self.ci,
            "build_elbow",
            return_value={"band": "OK"},
        ), patch(
            "app.clinician.interpreter.generate_comprehensive_why",
            return_value={},
        ):
            result = self.ci.build(
                elbow={"verdict": "LEGAL"},
                risks=[],
                interpretation={},
                action={"action": "SEMI_OPEN", "confidence": 0.43},
            )

        self.assertEqual(result["summary"]["overall_severity"], "LOW")
        self.assertEqual(result["summary"]["recommendation_level"], "MAINTAIN")
        self.assertEqual(result["summary"]["rating_overall_score"], 90)

    def test_benchmark_style_rating_guardrail_lifts_group_scores(self):
        with patch.object(
            self.ci,
            "build_summary",
            return_value={
                "overall_assessment": "This clip needs a closer look because some parts of the body may be taking extra load.",
                "overall_severity": "VERY_HIGH",
                "recommendation_level": "CORRECT",
                "confidence": 0.7,
                "primary_pillar": "PROTECTION",
            },
        ), patch.object(
            self.ci,
            "build_scorecard",
            return_value={
                "overall": {"score": 33},
                "confidence": {"score": 68},
            },
        ), patch.object(
            self.ci,
            "build_rating_system_v2",
            return_value={
                "overall": {
                    "score": 66,
                    "label": "Needs work",
                    "what_this_score_means": "This action has useful parts, but clear issues are now pulling it down.",
                },
                "confidence": {"score": 68},
                "player_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 66, "label": "Needs work", "what_this_score_means": "x"},
                        "lower_body_alignment": {"score": 71, "label": "Needs work", "what_this_score_means": "x"},
                        "whole_body_alignment": {"score": 74, "label": "Needs work", "what_this_score_means": "x"},
                        "momentum_forward": {"score": 66, "label": "Needs work", "what_this_score_means": "x"},
                    },
                },
                "coach_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 66, "label": "Needs work", "what_this_score_means": "x"},
                        "lower_body_alignment": {"score": 71, "label": "Needs work", "what_this_score_means": "x"},
                        "whole_body_alignment": {"score": 74, "label": "Needs work", "what_this_score_means": "x"},
                        "momentum_forward": {"score": 66, "label": "Needs work", "what_this_score_means": "x"},
                        "safety": {"score": 69, "label": "Needs work", "what_this_score_means": "x"},
                    },
                },
                "expert_view": {
                    "metrics": {
                        "upper_body_alignment": {"score": 66, "label": "Needs work", "what_this_score_means": "x"},
                        "lower_body_alignment": {"score": 71, "label": "Needs work", "what_this_score_means": "x"},
                        "whole_body_alignment": {"score": 74, "label": "Needs work", "what_this_score_means": "x"},
                        "momentum_forward": {"score": 66, "label": "Needs work", "what_this_score_means": "x"},
                        "safety": {"score": 69, "label": "Needs work", "what_this_score_means": "x"},
                        "confidence": {"score": 68},
                    },
                },
            },
        ), patch.object(
            self.ci,
            "build_pillars",
            return_value={
                "posture": {"status": "REVIEW"},
                "power": {"status": "REVIEW"},
                "protection": {"status": "REVIEW"},
            },
        ), patch.object(
            self.ci,
            "build_risks",
            return_value=[],
        ), patch.object(
            self.ci,
            "build_chain",
            return_value={"confidence": 0.7},
        ), patch.object(
            self.ci,
            "build_elbow",
            return_value={"band": "OK"},
        ), patch(
            "app.clinician.interpreter.generate_comprehensive_why",
            return_value={},
        ):
            result = self.ci.build(
                elbow={"verdict": "LEGAL"},
                risks=[],
                interpretation={},
                action={"action": "SEMI_OPEN", "confidence": 0.43},
            )

        rating = result["rating_system_v2"]
        self.assertEqual(rating["overall"]["score"], 90)
        self.assertEqual(rating["player_view"]["metrics"]["upper_body_alignment"]["score"], 84)
        self.assertEqual(rating["player_view"]["metrics"]["lower_body_alignment"]["score"], 84)
        self.assertEqual(rating["player_view"]["metrics"]["whole_body_alignment"]["score"], 88)
        self.assertEqual(rating["player_view"]["metrics"]["momentum_forward"]["score"], 84)
        self.assertEqual(rating["coach_view"]["metrics"]["safety"]["score"], 86)

    def test_side_on_benchmark_style_uses_lower_guardrail_threshold(self):
        with patch.object(
            self.ci,
            "build_summary",
            return_value={
                "overall_assessment": "This clip needs a closer look because some parts of the body may be taking extra load.",
                "overall_severity": "VERY_HIGH",
                "recommendation_level": "CORRECT",
                "confidence": 0.7,
                "primary_pillar": "PROTECTION",
            },
        ), patch.object(
            self.ci,
            "build_scorecard",
            return_value={
                "overall": {"score": 33},
                "confidence": {"score": 75},
            },
        ), patch.object(
            self.ci,
            "build_rating_system_v2",
            return_value={
                "overall": {"score": 57},
                "confidence": {"score": 75},
            },
        ), patch.object(
            self.ci,
            "build_pillars",
            return_value={
                "posture": {"status": "REVIEW"},
                "power": {"status": "REVIEW"},
                "protection": {"status": "REVIEW"},
            },
        ), patch.object(
            self.ci,
            "build_risks",
            return_value=[],
        ), patch.object(
            self.ci,
            "build_chain",
            return_value={"confidence": 0.7},
        ), patch.object(
            self.ci,
            "build_elbow",
            return_value={"band": "OK"},
        ), patch(
            "app.clinician.interpreter.generate_comprehensive_why",
            return_value={},
        ):
            result = self.ci.build(
                elbow={"verdict": "LEGAL"},
                risks=[],
                interpretation={},
                action={"action": "SIDE_ON", "confidence": 0.97},
            )

        self.assertEqual(result["summary"]["overall_severity"], "LOW")
        self.assertEqual(result["summary"]["recommendation_level"], "MAINTAIN")
        self.assertEqual(result["summary"]["rating_overall_score"], 92)

    def test_mixed_action_does_not_get_benchmark_softening(self):
        with patch.object(
            self.ci,
            "build_summary",
            return_value={
                "overall_assessment": "This clip needs a closer look because some parts of the body may be taking extra load.",
                "overall_severity": "VERY_HIGH",
                "recommendation_level": "CORRECT",
                "confidence": 0.7,
                "primary_pillar": "PROTECTION",
            },
        ), patch.object(
            self.ci,
            "build_scorecard",
            return_value={
                "overall": {"score": 17},
                "confidence": {"score": 81},
            },
        ), patch.object(
            self.ci,
            "build_rating_system_v2",
            return_value={
                "overall": {"score": 70},
                "confidence": {"score": 81},
            },
        ), patch.object(
            self.ci,
            "build_pillars",
            return_value={
                "posture": {"status": "REVIEW"},
                "power": {"status": "REVIEW"},
                "protection": {"status": "REVIEW"},
            },
        ), patch.object(
            self.ci,
            "build_risks",
            return_value=[],
        ), patch.object(
            self.ci,
            "build_chain",
            return_value={"confidence": 0.7},
        ), patch.object(
            self.ci,
            "build_elbow",
            return_value={"band": "OK"},
        ), patch(
            "app.clinician.interpreter.generate_comprehensive_why",
            return_value={},
        ):
            result = self.ci.build(
                elbow={"verdict": "LEGAL"},
                risks=[],
                interpretation={},
                action={"action": "MIXED", "confidence": 1.0},
            )

        self.assertEqual(result["summary"]["overall_severity"], "VERY_HIGH")
        self.assertEqual(result["summary"]["recommendation_level"], "CORRECT")


if __name__ == "__main__":
    unittest.main()
