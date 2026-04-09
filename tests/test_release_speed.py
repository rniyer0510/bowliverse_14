import unittest
from unittest.mock import patch

from app.workers.speed.release_speed import (
    _apply_low_confidence_neighbor_recovery,
    estimate_release_speed,
)


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}
        for _ in range(33)
    ]


def _set_pt(landmarks, index, x, y, vis=0.95):
    landmarks[index] = {
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "visibility": float(vis),
    }


def _frame(frame_idx, wrist_x, wrist_y, shoulder_x=0.48, shoulder_y=0.30):
    lm = _blank_landmarks()
    _set_pt(lm, 12, shoulder_x, shoulder_y)
    _set_pt(lm, 14, shoulder_x + 0.05, shoulder_y + 0.09)
    _set_pt(lm, 16, wrist_x, wrist_y)
    _set_pt(lm, 11, 0.42, 0.30)
    _set_pt(lm, 23, 0.46, 0.58)
    _set_pt(lm, 24, 0.54, 0.58)
    _set_pt(lm, 25, 0.47, 0.73)
    _set_pt(lm, 26, 0.55, 0.73)
    _set_pt(lm, 27, 0.48, 0.93)
    _set_pt(lm, 28, 0.56, 0.93)
    return {"frame": frame_idx, "landmarks": lm}


class ReleaseSpeedTests(unittest.TestCase):
    def test_returns_estimated_speed_with_tilde_display(self):
        pose_frames = [
            _frame(0, 0.40, 0.58),
            _frame(1, 0.50, 0.50),
            _frame(2, 0.63, 0.39),
            _frame(3, 0.78, 0.28),
            _frame(4, 0.88, 0.23),
            _frame(5, 0.94, 0.20),
            _frame(6, 0.98, 0.18),
        ]

        result = estimate_release_speed(
            pose_frames=pose_frames,
            events={"release": {"frame": 3}},
            video={"fps": 60.0, "width": 360, "height": 640},
            hand="R",
        )

        self.assertTrue(result["available"])
        self.assertTrue(result["display"].startswith("~"))
        self.assertGreaterEqual(result["value_kph"], 90)
        self.assertLessEqual(result["value_kph"], 145)
        self.assertGreater(result["confidence"], 0.25)
        self.assertEqual(result["method"], "release_kinematics_research_v2")

    def test_missing_release_window_returns_unavailable(self):
        pose_frames = [_frame(0, 0.40, 0.58), _frame(1, 0.50, 0.50)]

        result = estimate_release_speed(
            pose_frames=pose_frames,
            events={"release": {"frame": 0}},
            video={"fps": 25.0, "width": 360, "height": 640},
            hand="R",
        )

        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "missing_release_window")

    def test_low_release_confidence_uses_soft_recovery_metrics(self):
        pose_frames = [
            _frame(0, 0.40, 0.58),
            _frame(1, 0.50, 0.50),
            _frame(2, 0.63, 0.39),
            _frame(3, 0.78, 0.28),
            _frame(4, 0.88, 0.23),
            _frame(5, 0.94, 0.20),
            _frame(6, 0.98, 0.18),
        ]

        high_confidence = estimate_release_speed(
            pose_frames=pose_frames,
            events={"release": {"frame": 3, "confidence": 0.85}},
            video={"fps": 60.0, "width": 360, "height": 640},
            hand="R",
        )
        low_confidence = estimate_release_speed(
            pose_frames=pose_frames,
            events={"release": {"frame": 3, "confidence": 0.55}},
            video={"fps": 60.0, "width": 360, "height": 640},
            hand="R",
        )

        self.assertTrue(high_confidence["available"])
        self.assertTrue(low_confidence["available"])
        self.assertFalse(high_confidence["debug"]["soft_release_recovery_mode"])
        self.assertTrue(low_confidence["debug"]["soft_release_recovery_mode"])
        self.assertGreaterEqual(low_confidence["value_kph"], high_confidence["value_kph"])

    def test_low_confidence_neighbor_recovery_uses_stable_upper_quartile(self):
        primary = {
            "available": True,
            "display_policy": "show",
            "value_kph": 124,
            "display": "~124 km/h",
            "confidence": 0.58,
            "debug": {"release_frame": 140, "saturated": False},
        }
        recovered = _apply_low_confidence_neighbor_recovery(
            primary,
            [
                primary,
                {
                    "available": True,
                    "display_policy": "show",
                    "value_kph": 130,
                    "confidence": 0.57,
                    "debug": {"release_frame": 138, "saturated": False},
                },
                {
                    "available": True,
                    "display_policy": "show",
                    "value_kph": 122,
                    "confidence": 0.58,
                    "debug": {"release_frame": 139, "saturated": False},
                },
                {
                    "available": True,
                    "display_policy": "show",
                    "value_kph": 134,
                    "confidence": 0.55,
                    "debug": {"release_frame": 142, "saturated": False},
                },
                {
                    "available": True,
                    "display_policy": "show",
                    "value_kph": 145,
                    "confidence": 0.43,
                    "debug": {"release_frame": 137, "saturated": True},
                },
            ],
        )

        self.assertEqual(recovered["value_kph"], 131)
        self.assertEqual(recovered["display"], "~131 km/h")
        self.assertTrue(recovered["debug"]["low_confidence_neighbor_recovery"])
        self.assertEqual(recovered["debug"]["primary_value_kph"], 124)

    def test_unstable_arm_scale_returns_unavailable(self):
        pose_frames = [
            _frame(0, 0.40, 0.58),
            _frame(1, 0.52, 0.48),
            _frame(2, 0.68, 0.35),
            _frame(3, 0.91, 0.20),
            _frame(4, 0.56, 0.50),
            _frame(5, 0.92, 0.18),
            _frame(6, 0.58, 0.46),
        ]

        result = estimate_release_speed(
            pose_frames=pose_frames,
            events={"release": {"frame": 3}},
            video={"fps": 60.0, "width": 360, "height": 640},
            hand="R",
        )

        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "unstable_release_window")

    def test_estimate_release_speed_applies_neighbor_recovery_for_low_confidence_release(self):
        primary = {
            "available": True,
            "display_policy": "show",
            "value_kph": 124,
            "display": "~124 km/h",
            "confidence": 0.58,
            "method": "release_kinematics_research_v2",
            "debug": {"release_frame": 140, "saturated": False},
        }
        scripted = [
            primary,
            {
                "available": True,
                "display_policy": "show",
                "value_kph": 130,
                "display": "~130 km/h",
                "confidence": 0.57,
                "method": "release_kinematics_research_v2",
                "debug": {"release_frame": 138, "saturated": False},
            },
            {
                "available": True,
                "display_policy": "show",
                "value_kph": 122,
                "display": "~122 km/h",
                "confidence": 0.58,
                "method": "release_kinematics_research_v2",
                "debug": {"release_frame": 139, "saturated": False},
            },
            {
                "available": True,
                "display_policy": "show",
                "value_kph": 134,
                "display": "~134 km/h",
                "confidence": 0.55,
                "method": "release_kinematics_research_v2",
                "debug": {"release_frame": 141, "saturated": False},
            },
            {
                "available": True,
                "display_policy": "show",
                "value_kph": 129,
                "display": "~129 km/h",
                "confidence": 0.56,
                "method": "release_kinematics_research_v2",
                "debug": {"release_frame": 142, "saturated": False},
            },
        ]

        with patch(
            "app.workers.speed.release_speed._estimate_release_speed_pass",
            side_effect=scripted,
        ):
            result = estimate_release_speed(
                pose_frames=[{} for _ in range(171)],
                events={"release": {"frame": 140, "confidence": 0.54}},
                video={"fps": 25.0, "width": 720, "height": 1280},
                hand="R",
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["value_kph"], 130)
        self.assertTrue(result["debug"]["low_confidence_neighbor_recovery"])


if __name__ == "__main__":
    unittest.main()
