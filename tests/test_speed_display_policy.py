import unittest

from app.orchestrator.orchestrator import _gate_speed_estimate


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
            playback_mode={"mode": "real_time_or_high_fps"},
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
            playback_mode={"mode": "real_time_or_high_fps"},
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
            playback_mode={"mode": "real_time_or_high_fps"},
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["display_policy"], "suppress")
        self.assertEqual(result["reason"], "low_event_chain_quality")

    def test_likely_slow_motion_suppresses_speed(self):
        result = _gate_speed_estimate(
            estimated_release_speed=self._base_speed(),
            event_chain={"ordered": True, "quality": 0.82},
            events={
                "release": {"confidence": 0.90},
                "ffc": {"method": "release_backward_chain_grounding", "confidence": 0.90},
                "bfc": {"method": "back_foot_support_edge", "confidence": 0.80},
            },
            playback_mode={"mode": "likely_slow_motion"},
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["display_policy"], "suppress")
        self.assertEqual(result["reason"], "slow_motion_playback_detected")


if __name__ == "__main__":
    unittest.main()
