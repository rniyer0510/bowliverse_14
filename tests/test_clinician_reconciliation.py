import unittest

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


if __name__ == "__main__":
    unittest.main()
