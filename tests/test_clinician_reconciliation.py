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
        self.assertIn("mixed action profile", result["summary"]["overall_assessment"].lower())
        self.assertIn(
            "mixed action profile worth monitoring",
            result["comprehensive_why"]["primary_cause"].lower(),
        )

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
        self.assertIn("could not be profiled", result["summary"]["overall_assessment"].lower())
        self.assertIn(
            "could not be confirmed clearly",
            result["comprehensive_why"]["primary_cause"].lower(),
        )


if __name__ == "__main__":
    unittest.main()
