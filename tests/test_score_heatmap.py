import os
import unittest
from datetime import datetime
from unittest import mock

from app.persistence.read_api import (
    _build_action_change_summary,
    _build_player_baseline_state,
    _build_rating_heatmap_v2,
    _build_score_heatmap,
    _extract_action_change_traits,
    _extract_history_plan_summary,
    _extract_kinetic_chain_summary,
    _extract_root_cause_summary,
    _extract_rating_summary_v2,
    _extract_score_summary,
    _extract_visual_walkthrough,
)


class ScoreHeatmapTests(unittest.TestCase):
    def _result_json(
        self,
        *,
        overall=77,
        upper=74,
        lower=79,
        whole=72,
        momentum=80,
        action_type="SEMI_OPEN",
        action_intent="semi_open",
        toe_status="aligned",
        toe_angle=28.0,
        front_foot_braking=0.18,
        knee_brace=0.19,
        trunk_rotation=0.20,
        hip_shoulder=0.21,
        trunk_lean=0.18,
        foot_line=0.17,
        kinetic_chain_sequence="in_sync",
        kinetic_chain_delta=0,
        event_chain_quality=0.72,
        walkthrough=None,
    ):
        result = {
            "action": {
                "action": action_type,
                "intent": action_intent,
                "confidence": 0.76,
            },
            "events": {
                "event_chain": {
                    "quality": event_chain_quality,
                }
            },
            "risks": [
                {"risk_id": "front_foot_braking_shock", "signal_strength": front_foot_braking},
                {"risk_id": "knee_brace_failure", "signal_strength": knee_brace},
                {"risk_id": "trunk_rotation_snap", "signal_strength": trunk_rotation},
                {
                    "risk_id": "hip_shoulder_mismatch",
                    "signal_strength": hip_shoulder,
                    "debug": {
                        "phase_lag": hip_shoulder,
                        "sequence_pattern": kinetic_chain_sequence,
                        "sequence_delta_frames": kinetic_chain_delta,
                    },
                },
                {"risk_id": "lateral_trunk_lean", "signal_strength": trunk_lean},
                {"risk_id": "foot_line_deviation", "signal_strength": foot_line},
            ],
            "basics": {
                "front_foot_toe_alignment": {
                    "status": toe_status,
                    "confidence": 1.0,
                    "debug": {"toe_angle_deg": toe_angle},
                }
            },
            "clinician": {
                "rating_system_v2": {
                    "overall": {"score": overall, "label": "Good base"},
                    "player_view": {
                        "metrics": {
                            "upper_body_alignment": {"score": upper},
                            "lower_body_alignment": {"score": lower},
                            "whole_body_alignment": {"score": whole},
                            "momentum_forward": {"score": momentum},
                        }
                    },
                    "coach_view": {"metrics": {"safety": {"score": 69}}},
                    "confidence": {"score": 71},
                }
            },
        }
        if walkthrough is not None:
            result["visual_walkthrough"] = walkthrough
        return result

    def test_extract_score_summary_from_clinician_scorecard(self):
        result_json = {
            "clinician": {
                "scorecard": {
                    "overall": {"score": 82, "label": "Strong base"},
                    "pillars": {
                        "balance": {"score": 79},
                        "carry": {"score": 84},
                        "body_load": {"score": 80},
                    },
                    "confidence": {"score": 68},
                }
            }
        }

        summary = _extract_score_summary(result_json)

        self.assertEqual(summary["overall"]["score"], 82)
        self.assertEqual(summary["overall"]["band"], "strong")
        self.assertEqual(summary["balance"]["score"], 79)
        self.assertEqual(summary["carry"]["band"], "strong")
        self.assertEqual(summary["confidence"]["band"], "watch")

    def test_build_score_heatmap_keeps_latest_first_cells(self):
        heatmap = _build_score_heatmap(
            [
                {
                    "run_id": "run-2",
                    "created_at": "2026-03-31T10:00:00",
                    "score_summary": {
                        "overall": {"score": 84, "band": "strong"},
                        "balance": {"score": 80, "band": "strong"},
                        "carry": {"score": 86, "band": "strong"},
                        "body_load": {"score": 82, "band": "strong"},
                        "confidence": {"score": 69, "band": "watch"},
                    },
                },
                {
                    "run_id": "run-1",
                    "created_at": "2026-03-30T10:00:00",
                    "score_summary": {
                        "overall": {"score": 74, "band": "watch"},
                        "balance": {"score": 70, "band": "watch"},
                        "carry": {"score": 76, "band": "strong"},
                        "body_load": {"score": 72, "band": "watch"},
                        "confidence": {"score": 61, "band": "watch"},
                    },
                },
            ]
        )

        self.assertEqual(heatmap["version"], "score_heatmap_v1")
        self.assertTrue(heatmap["latest_first"])
        self.assertEqual(heatmap["rows"][0]["metric"], "overall")
        self.assertEqual(heatmap["rows"][0]["cells"][0]["run_id"], "run-2")
        self.assertEqual(heatmap["rows"][0]["cells"][1]["band"], "watch")

    def test_extract_rating_summary_v2_from_clinician_payload(self):
        result_json = {
            "clinician": {
                "rating_system_v2": {
                    "overall": {"score": 77, "label": "Good base"},
                    "player_view": {
                        "metrics": {
                            "upper_body_alignment": {"score": 74},
                            "lower_body_alignment": {"score": 79},
                            "whole_body_alignment": {"score": 72},
                            "momentum_forward": {"score": 80},
                        }
                    },
                    "coach_view": {
                        "metrics": {
                            "safety": {"score": 69},
                        }
                    },
                    "confidence": {"score": 71},
                }
            }
        }

        summary = _extract_rating_summary_v2(result_json)

        self.assertEqual(summary["overall"]["score"], 77)
        self.assertEqual(summary["upper_body_alignment"]["band"], "watch")
        self.assertEqual(summary["momentum_forward"]["score"], 80)
        self.assertEqual(summary["safety"]["band"], "watch")

    def test_build_rating_heatmap_v2_keeps_latest_first_cells(self):
        heatmap = _build_rating_heatmap_v2(
            [
                {
                    "run_id": "run-2",
                    "created_at": "2026-03-31T10:00:00",
                    "rating_summary_v2": {
                        "overall": {"score": 82, "band": "strong"},
                        "upper_body_alignment": {"score": 76, "band": "strong"},
                        "lower_body_alignment": {"score": 78, "band": "strong"},
                        "whole_body_alignment": {"score": 74, "band": "watch"},
                        "momentum_forward": {"score": 84, "band": "strong"},
                        "safety": {"score": 69, "band": "watch"},
                        "confidence": {"score": 71, "band": "watch"},
                    },
                },
                {
                    "run_id": "run-1",
                    "created_at": "2026-03-30T10:00:00",
                    "rating_summary_v2": {
                        "overall": {"score": 70, "band": "watch"},
                        "upper_body_alignment": {"score": 64, "band": "watch"},
                        "lower_body_alignment": {"score": 68, "band": "watch"},
                        "whole_body_alignment": {"score": 62, "band": "watch"},
                        "momentum_forward": {"score": 73, "band": "watch"},
                        "safety": {"score": 60, "band": "watch"},
                        "confidence": {"score": 65, "band": "watch"},
                    },
                },
            ]
        )

        self.assertEqual(heatmap["version"], "rating_heatmap_v2")
        self.assertTrue(heatmap["latest_first"])
        self.assertEqual(heatmap["rows"][1]["metric"], "upper_body_alignment")
        self.assertEqual(heatmap["rows"][0]["cells"][0]["run_id"], "run-2")
        self.assertEqual(heatmap["rows"][4]["cells"][1]["band"], "watch")

    def test_extract_action_change_traits_from_result_json(self):
        traits = _extract_action_change_traits(
            self._result_json(
                overall=81,
                upper=78,
                lower=76,
                whole=80,
                momentum=82,
                action_type="MIXED",
                action_intent="semi_open",
                toe_status="semi_open",
                toe_angle=44.5,
            )
        )

        self.assertEqual(traits["numeric"]["overall"], 81)
        self.assertEqual(traits["numeric"]["front_foot_toe_angle_deg"], 44.5)
        self.assertEqual(traits["numeric"]["risk_hip_shoulder_mismatch"], 0.21)
        self.assertEqual(traits["numeric"]["kinetic_chain_delta_frames"], 0.0)
        self.assertEqual(traits["categorical"]["action_type"], "mixed")
        self.assertEqual(traits["categorical"]["front_foot_toe_alignment"], "semi_open")
        self.assertEqual(traits["categorical"]["kinetic_chain_sequence"], "in_sync")

    def test_build_action_change_summary_marks_within_range_when_latest_stays_stable(self):
        action_change = _build_action_change_summary(
            [
                {"run_id": "run-4", "result_json": self._result_json(overall=78, upper=75, lower=77, whole=74, momentum=80, toe_angle=30.0)},
                {"run_id": "run-3", "result_json": self._result_json(overall=80, upper=76, lower=78, whole=75, momentum=81, toe_angle=31.0)},
                {"run_id": "run-2", "result_json": self._result_json(overall=79, upper=74, lower=77, whole=73, momentum=79, toe_angle=29.0)},
                {"run_id": "run-1", "result_json": self._result_json(overall=81, upper=77, lower=79, whole=76, momentum=82, toe_angle=32.0)},
            ]
        )

        self.assertEqual(action_change["status"], "within_range")
        self.assertEqual(action_change["highlights"], [])
        self.assertEqual(action_change["headline"], "Your action looks like your usual pattern.")
        self.assertIn("usual pattern", action_change["summary"])
        self.assertIn("better recent clips", action_change["what_to_try"])
        self.assertIn("coach", action_change["coach_prompt"].lower())

    def test_build_action_change_summary_flags_clear_drift_in_alignment_and_toe_line(self):
        action_change = _build_action_change_summary(
            [
                {
                    "run_id": "run-5",
                    "result_json": self._result_json(
                        overall=64,
                        upper=61,
                        lower=60,
                        whole=63,
                        momentum=69,
                        action_type="FRONT_ON",
                        action_intent="front_on",
                        toe_status="open",
                        toe_angle=72.0,
                        front_foot_braking=0.34,
                        knee_brace=0.37,
                        trunk_rotation=0.48,
                        hip_shoulder=0.55,
                        trunk_lean=0.43,
                        foot_line=0.41,
                        kinetic_chain_sequence="shoulders_lead",
                        kinetic_chain_delta=-3,
                    ),
                },
                {"run_id": "run-4", "result_json": self._result_json(overall=80, upper=76, lower=79, whole=75, momentum=82, toe_angle=31.0)},
                {"run_id": "run-3", "result_json": self._result_json(overall=79, upper=75, lower=77, whole=74, momentum=80, toe_angle=29.0)},
                {"run_id": "run-2", "result_json": self._result_json(overall=81, upper=77, lower=80, whole=76, momentum=83, toe_angle=32.0)},
                {"run_id": "run-1", "result_json": self._result_json(overall=80, upper=76, lower=78, whole=75, momentum=81, toe_angle=30.0)},
            ]
        )

        self.assertEqual(action_change["status"], "clear_change")
        self.assertEqual(action_change["headline"], "Shoulders are opening earlier than usual.")
        highlight_metrics = [item["metric"] for item in action_change["highlights"]]
        self.assertIn("lower_body_alignment", highlight_metrics)
        self.assertIn("front_foot_toe_angle_deg", highlight_metrics)
        self.assertIn("action_type", highlight_metrics)
        self.assertIn("risk_hip_shoulder_mismatch", highlight_metrics)
        self.assertIn("risk_trunk_rotation_snap", highlight_metrics)
        self.assertIn("kinetic_chain_sequence", highlight_metrics)
        self.assertIn("Shoulders are opening earlier than usual", action_change["summary"])
        self.assertTrue(any(item["plain_message"] == "Shoulders are opening earlier than usual." for item in action_change["highlights"]))
        self.assertEqual(
            action_change["what_to_try"],
            "Try to let the hips lead the move into the ball instead of pulling the shoulders around early.",
        )
        self.assertEqual(
            action_change["coach_prompt"],
            "If you can, show these clips to a coach and ask whether the shoulders are opening too early.",
        )
        foot_line_item = next(item for item in action_change["highlights"] if item["metric"] == "risk_foot_line_deviation")
        self.assertEqual(
            foot_line_item["what_to_try"],
            "Try to line up the back foot and front foot so they stay balanced and in line.",
        )

    def test_build_action_change_summary_requires_recent_baseline(self):
        action_change = _build_action_change_summary(
            [
                {"run_id": "run-2", "result_json": self._result_json(overall=70, upper=68, lower=69, whole=67, momentum=72, toe_angle=34.0)},
                {"run_id": "run-1", "result_json": self._result_json(overall=79, upper=75, lower=77, whole=74, momentum=80, toe_angle=31.0)},
            ]
        )

        self.assertEqual(action_change["status"], "insufficient_history")
        self.assertEqual(action_change["comparisons"], [])
        self.assertEqual(
            action_change["headline"],
            "Action change needs a few more clips before we can judge it well.",
        )
        self.assertIn("Need at least 2 recent comparison clips", action_change["summary"])
        self.assertIn("few more clips", action_change["what_to_try"])
        self.assertIn("coach", action_change["coach_prompt"].lower())

    def test_extract_visual_walkthrough_from_result_json(self):
        walkthrough = {
            "available": True,
            "url": "/renders/run-1_walkthrough.mp4",
            "renderer_version": "coach_video_renderer_v1",
        }
        result_json = self._result_json(walkthrough=walkthrough)

        self.assertEqual(
            _extract_visual_walkthrough(result_json),
            {**walkthrough, "relative_url": "/renders/run-1_walkthrough.mp4"},
        )

    def test_extract_visual_walkthrough_normalizes_relative_url_with_public_base(self):
        walkthrough = {
            "available": True,
            "url": "/renders/run-1_walkthrough.mp4",
            "renderer_version": "coach_video_renderer_v1",
        }
        result_json = self._result_json(walkthrough=walkthrough)

        with mock.patch.dict(os.environ, {"ACTIONLAB_PUBLIC_BASE_URL": "https://api.actionlabcricket.in"}, clear=False):
            extracted = _extract_visual_walkthrough(result_json)

        self.assertEqual(extracted["relative_url"], "/renders/run-1_walkthrough.mp4")
        self.assertEqual(
            extracted["url"],
            "https://api.actionlabcricket.in/renders/run-1_walkthrough.mp4",
        )

    def test_extract_kinetic_chain_summary_from_result_json(self):
        result_json = self._result_json()
        result_json["kinetic_chain_v1"] = {
            "diagnosis_status": "partial_match",
            "confidence": 0.71,
            "archetype": {"short_label": "Soft block leakage"},
            "approach_build": {"score": 0.48, "label": "moderate"},
            "transfer": {"score": 0.42, "label": "watch"},
            "block": {"score": 0.36, "label": "weak"},
            "dissipation": {"score": 0.64, "label": "high_load_concentration"},
            "pace_translation": {
                "approach_momentum": 0.48,
                "transfer_efficiency": 0.42,
                "terminal_impulse": 0.63,
                "leakage_before_block": 0.39,
                "leakage_at_block": 0.71,
                "late_arm_chase": 0.34,
                "dissipation_burden": 0.64,
            },
        }
        result_json["mechanism_explanation_v1"] = {
            "primary_mechanism": "Soft block with trunk carry",
            "first_intervention": "Feel the chest arrive over the front leg sooner.",
        }
        result_json["prescription_plan_v1"] = {
            "prescriptions": [
                {"id": "stack_over_landing_leg", "title": "Stack over the landing leg sooner"},
            ]
        }
        result_json["coach_diagnosis_v1"] = {
            "root_cause": {
                "status": "clear",
                "mechanism_id": "soft_block_with_trunk_carry",
                "title": "Soft block with trunk carry",
                "summary": "The landing leg does not become a stable base.",
                "why_it_is_happening": "The landing leg does not become a stable base, so the trunk keeps travelling.",
                "chain_story": "It starts at transfer and block and then carries into the trunk.",
                "where_it_starts": {"phase_id": "transfer_and_block"},
                "primary_driver": {"id": "front_leg_support_score", "title": "Front-leg support"},
                "compensation": {"id": "lateral_trunk_lean", "title": "Trunk lean"},
                "renderer_guidance": {
                    "story_id": "soft_block_trunk_carry_story",
                    "cue_points": ["Show the soft base", "Show the trunk carry"],
                    "symptom_text": "Front leg softens at landing",
                    "load_watch_text": "Front knee / leg chain\nLower back / side trunk",
                },
            }
        }

        summary = _extract_kinetic_chain_summary(result_json)

        self.assertEqual(summary["diagnosis_status"], "partial_match")
        self.assertEqual(summary["archetype"], "Soft block leakage")
        self.assertEqual(summary["primary_mechanism"], "Soft block with trunk carry")
        self.assertEqual(summary["primary_prescription_title"], "Stack over the landing leg sooner")
        self.assertEqual(summary["root_cause"]["mechanism_id"], "soft_block_with_trunk_carry")
        self.assertEqual(summary["block"]["label"], "weak")
        self.assertEqual(summary["pace_translation"]["leakage_at_block"], 0.71)

    def test_extract_history_plan_summary_from_result_json(self):
        result_json = self._result_json()
        result_json["history_plan_v1"] = {
            "history_story": "Recent clips keep showing a soft-block leakage pattern at landing.",
            "coaching_priority": "Organize the body over the landing leg before coaching a harder brace.",
            "history_bindings": [
                {"id": "transfer_block_stability"},
            ],
            "binding_trends": [
                {"id": "transfer_block_stability", "status": "better"},
            ],
            "followup_checks": [
                {"id": "reduced_trunk_drift_after_ffc"},
            ],
            "render_stories": [
                {"id": "soft_block_trunk_carry_story"},
            ],
        }
        result_json["coach_diagnosis_v1"] = {
            "root_cause": {
                "status": "clear",
                "mechanism_id": "soft_block_with_trunk_carry",
                "title": "Soft block with trunk carry",
                "summary": "The landing leg does not become a stable base.",
                "why_it_is_happening": "The landing leg does not become a stable base, so the trunk keeps travelling.",
                "chain_story": "The leak begins at landing and shows up as trunk carry later.",
                "where_it_starts": {"phase_id": "transfer_and_block"},
                "primary_driver": {"id": "front_leg_support_score", "title": "Front-leg support"},
                "compensation": {"id": "lateral_trunk_lean", "title": "Trunk lean"},
                "renderer_guidance": {
                    "story_id": "soft_block_trunk_carry_story",
                    "cue_points": ["Show the soft base", "Show the trunk carry"],
                    "symptom_text": "Front leg softens at landing",
                    "load_watch_text": "Front knee / leg chain\nLower back / side trunk",
                },
            }
        }

        summary = _extract_history_plan_summary(result_json)

        self.assertEqual(
            summary["history_story"],
            "Recent clips keep showing a soft-block leakage pattern at landing.",
        )
        self.assertEqual(summary["root_cause"]["compensation_title"], "Trunk lean")
        self.assertEqual(summary["history_binding_ids"], ["transfer_block_stability"])
        self.assertEqual(
            summary["binding_trend_statuses"],
            {"transfer_block_stability": "better"},
        )
        self.assertEqual(summary["followup_check_ids"], ["reduced_trunk_drift_after_ffc"])
        self.assertEqual(summary["render_story_ids"], ["soft_block_trunk_carry_story"])

    def test_extract_root_cause_summary_from_result_json(self):
        result_json = self._result_json()
        result_json["coach_diagnosis_v1"] = {
            "root_cause": {
                "status": "clear",
                "mechanism_id": "late_arm_acceleration_due_to_chain_delay",
                "title": "Late arm acceleration due to chain delay",
                "summary": "Earlier chain segments are late enough that the arm has to chase the release window.",
                "why_it_is_happening": "The arm is rescuing a delayed chain rather than causing the leak by itself.",
                "chain_story": "The chain arrives late, then the arm speeds up to catch the ball.",
                "where_it_starts": {"phase_id": "whip_and_release"},
                "primary_driver": {"id": "shoulder_rotation_timing", "title": "Shoulder rotation timing"},
                "compensation": {"id": "trunk_rotation_snap", "title": "Shoulder rotation snap"},
                "renderer_guidance": {
                    "story_id": "delayed_chain_arm_chase_story",
                    "cue_points": ["Show the late chain", "Show the arm chase"],
                    "symptom_text": "Arm has to chase release late",
                    "load_watch_text": "Lower back / side trunk",
                },
            }
        }

        summary = _extract_root_cause_summary(result_json)

        self.assertEqual(summary["mechanism_id"], "late_arm_acceleration_due_to_chain_delay")
        self.assertEqual(summary["primary_driver_id"], "shoulder_rotation_timing")
        self.assertEqual(summary["compensation_id"], "trunk_rotation_snap")
        self.assertEqual(summary["render_story_id"], "delayed_chain_arm_chase_story")

    def test_build_player_baseline_state_marks_refresh_candidate_on_sustained_action_shift(self):
        baseline_state = _build_player_baseline_state(
            [
                {"run_id": "run-8", "created_at": datetime(2026, 4, 8), "result_json": self._result_json(action_type="MIXED", action_intent="mixed")},
                {"run_id": "run-7", "created_at": datetime(2026, 4, 7), "result_json": self._result_json(action_type="MIXED", action_intent="mixed")},
                {"run_id": "run-6", "created_at": datetime(2026, 4, 6), "result_json": self._result_json(action_type="MIXED", action_intent="mixed")},
                {"run_id": "run-5", "created_at": datetime(2026, 4, 5), "result_json": self._result_json(action_type="MIXED", action_intent="mixed")},
                {"run_id": "run-4", "created_at": datetime(2026, 3, 20), "result_json": self._result_json(action_type="SEMI_OPEN", action_intent="semi_open")},
                {"run_id": "run-3", "created_at": datetime(2026, 3, 19), "result_json": self._result_json(action_type="SEMI_OPEN", action_intent="semi_open")},
                {"run_id": "run-2", "created_at": datetime(2026, 3, 18), "result_json": self._result_json(action_type="SEMI_OPEN", action_intent="semi_open")},
                {"run_id": "run-1", "created_at": datetime(2026, 3, 17), "result_json": self._result_json(action_type="SEMI_OPEN", action_intent="semi_open")},
            ]
        )

        self.assertEqual(baseline_state["status"], "refresh_candidate")
        self.assertTrue(baseline_state["should_refresh_baseline"])
        self.assertEqual(baseline_state["recent_action_type_mode"], "mixed")
        self.assertEqual(baseline_state["trigger_reason"], "sustained_action_type_shift")


if __name__ == "__main__":
    unittest.main()
