import unittest

from app.orchestrator.orchestrator import (
    _gate_speed_estimate,
    _release_dependent_metrics_should_be_suppressed,
)


class SpeedDisplayPolicyTest(unittest.TestCase):
    def _base_speed(self):
        return {
            "available": True,
            "display_policy": "show",
            "value_kph": 132,
            "display": "~132 km/h",
            "confidence": 0.59,
            "method": "release_kinematics_research_v2",
            "ball_weight_oz": 5.25,
        }

    def test_borderline_chain_shows_low_confidence(self):
        result = _gate_speed_estimate(
            estimated_release_speed=self._base_speed(),
            event_chain={"ordered": True, "quality": 0.26},
            events={
                "release": {"confidence": 0.54},
                "ffc": {"method": "ultimate_fallback", "confidence": 0.15},
                "bfc": {"method": "ultimate_fallback", "confidence": 0.15},
            },
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["display_policy"], "show_low_confidence")
        self.assertEqual(result["reason"], "low_event_chain_quality")

    def test_unordered_chain_suppresses_speed(self):
        result = _gate_speed_estimate(
            estimated_release_speed=self._base_speed(),
            event_chain={"ordered": False, "quality": 0.18},
            events={
                "release": {"confidence": 0.90},
                "ffc": {"method": "ultimate_fallback", "confidence": 0.15},
                "bfc": {"method": "ultimate_fallback", "confidence": 0.15},
            },
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["display_policy"], "suppress")
        self.assertEqual(result["reason"], "event_chain_unordered")

    def test_low_release_confidence_still_suppresses(self):
        result = _gate_speed_estimate(
            estimated_release_speed=self._base_speed(),
            event_chain={"ordered": True, "quality": 0.22},
            events={
                "release": {"confidence": 0.30},
                "ffc": {"method": "ultimate_fallback", "confidence": 0.15},
                "bfc": {"method": "ultimate_fallback", "confidence": 0.15},
            },
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["display_policy"], "suppress")
        self.assertEqual(result["reason"], "low_event_chain_quality")

    def test_release_dependent_metrics_suppressed_for_close_unclear_clip(self):
        self.assertTrue(
            _release_dependent_metrics_should_be_suppressed(
                {
                    "status": "WEAK",
                    "notes": [
                        "release_unclear",
                        "camera_too_close",
                        "framing_unusable",
                    ],
                }
            )
        )

    def test_release_dependent_metrics_not_suppressed_for_clean_weak_clip(self):
        self.assertFalse(
            _release_dependent_metrics_should_be_suppressed(
                {
                    "status": "WEAK",
                    "notes": [
                        "event_chain_weak_quality",
                        "action_confidence_weak",
                    ],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
