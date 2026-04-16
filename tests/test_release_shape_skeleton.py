import unittest

from app.persistence.release_shape_drift import enrich_release_shape_drift
from app.workers.release_shape import build_release_shape_skeleton


class ReleaseShapeSkeletonTests(unittest.TestCase):
    def _frames(self, elbow_x: float, elbow_y: float, wrist_x: float, wrist_y: float):
        frames = []
        for idx in range(8):
            landmarks = [{"x": 0.0, "y": 0.0, "visibility": 0.0} for _ in range(33)]
            landmarks[12] = {"x": 0.5, "y": 0.45, "visibility": 0.99}
            landmarks[14] = {"x": elbow_x, "y": elbow_y + (idx * 0.002), "visibility": 0.99}
            landmarks[16] = {"x": wrist_x, "y": wrist_y + (idx * 0.003), "visibility": 0.99}
            landmarks[24] = {"x": 0.5, "y": 0.68, "visibility": 0.99}
            landmarks[28] = {"x": 0.5, "y": 0.96, "visibility": 0.99}
            frames.append({"frame": idx, "landmarks": landmarks})
        return frames

    def test_build_release_shape_skeleton_exposes_future_contract(self):
        release_shape = build_release_shape_skeleton(
            pose_frames=self._frames(0.53, 0.35, 0.54, 0.18),
            events={
                "release": {"frame": 5, "confidence": 0.81},
                "event_chain": {"quality": 0.74},
            },
            hand="RIGHT",
            action={"action": "SEMI_OPEN"},
        )

        self.assertEqual(release_shape["version"], "release_shape_v1")
        self.assertTrue(release_shape["available"])
        self.assertEqual(
            [item["label"] for item in release_shape["supported_categories"]],
            ["Standard", "Round Arm"],
        )
        self.assertEqual(release_shape["category"]["label"], "Standard")
        self.assertIsNotNone(release_shape["release_geometry"]["height_ratio"])
        self.assertIsNotNone(release_shape["release_geometry"]["angle_deg"])
        self.assertIsNotNone(release_shape["release_geometry"]["arc_deg"])
        self.assertTrue(release_shape["trusted_fields"]["category"])
        self.assertTrue(release_shape["trusted_fields"]["arc_deg"])
        self.assertEqual(release_shape["source"]["release_frame"], 5)
        self.assertEqual(release_shape["source"]["release_confidence"], 0.81)
        self.assertEqual(release_shape["source"]["event_chain_quality"], 0.74)
        self.assertEqual(release_shape["source"]["hand"], "right")
        self.assertEqual(release_shape["source"]["action_type"], "semi_open")
        self.assertEqual(release_shape["source"]["pose_frame_count"], 8)
        self.assertEqual(release_shape["reason"], "computed_from_torso_relative_release_geometry")

    def test_build_release_shape_marks_round_arm_when_release_angle_is_wide(self):
        release_shape = build_release_shape_skeleton(
            pose_frames=self._frames(0.69, 0.47, 0.86, 0.56),
            events={"release": {"frame": 5, "confidence": 0.74}, "event_chain": {"quality": 0.72}},
            hand="RIGHT",
            action={"action": "MIXED"},
        )

        self.assertEqual(release_shape["category"]["key"], "round_arm")
        self.assertGreaterEqual(release_shape["release_geometry"]["angle_deg"], 60.0)

    def test_build_release_shape_hides_arc_when_event_chain_is_weak(self):
        release_shape = build_release_shape_skeleton(
            pose_frames=self._frames(0.53, 0.35, 0.54, 0.18),
            events={
                "release": {"frame": 5, "confidence": 0.81},
                "event_chain": {"quality": 0.22},
            },
            hand="RIGHT",
            action={"action": "SEMI_OPEN"},
        )

        self.assertIsNotNone(release_shape["release_geometry"]["angle_deg"])
        self.assertIsNone(release_shape["release_geometry"]["arc_deg"])
        self.assertFalse(release_shape["trusted_fields"]["arc_deg"])

    def test_enrich_release_shape_drift_compares_against_recent_trusted_baseline(self):
        def entry(run_id: str, angle: float, category_key: str):
            return {
                "run_id": run_id,
                "trusted": True,
                "result_json": {
                    "release_shape": {
                        "available": True,
                        "category": {
                            "key": category_key,
                            "label": "Round Arm" if category_key == "round_arm" else "Standard",
                        },
                        "release_geometry": {"angle_deg": angle},
                    }
                },
            }

        drift = enrich_release_shape_drift(
            [
                entry("run-4", 62.0, "round_arm"),
                entry("run-3", 33.0, "standard"),
                entry("run-2", 35.0, "standard"),
                entry("run-1", 34.0, "standard"),
            ]
        )

        self.assertEqual(drift["run-4"]["status"], "clear_change")
        self.assertEqual(drift["run-4"]["baseline_category_key"], "standard")
        self.assertEqual(drift["run-4"]["delta_deg"], 28.0)

    def test_enrich_release_shape_drift_keeps_small_angle_variation_within_range(self):
        def entry(run_id: str, angle: float):
            return {
                "run_id": run_id,
                "trusted": True,
                "result_json": {
                    "release_shape": {
                        "available": True,
                        "category": {"key": "standard", "label": "Standard"},
                        "release_geometry": {"angle_deg": angle},
                    }
                },
            }

        drift = enrich_release_shape_drift(
            [
                entry("run-4", 39.0),
                entry("run-3", 33.0),
                entry("run-2", 35.0),
                entry("run-1", 34.0),
            ]
        )

        self.assertEqual(drift["run-4"]["status"], "within_range")
        self.assertEqual(drift["run-4"]["allowed_delta_deg"], 6.0)


if __name__ == "__main__":
    unittest.main()
