import unittest
from copy import deepcopy

from app.clinician.interpreter import ClinicianInterpreter
from app.workers.elbow.elbow_legality import evaluate_elbow_legality


class ElbowLegalityFallbackTests(unittest.TestCase):
    def setUp(self):
        self.ci = ClinicianInterpreter()

    @staticmethod
    def _low_vis_pose_frames(n=8):
        frames = []
        for i in range(n):
            landmarks = [
                {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
                for _ in range(33)
            ]
            landmarks[12] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.95}
            landmarks[14] = {"x": 1.0, "y": 0.0, "z": 0.0, "visibility": 0.20}
            landmarks[16] = {"x": 2.0, "y": 0.0, "z": 0.0, "visibility": 0.20}
            frames.append({"frame": i, "landmarks": landmarks})
        return frames

    def test_sparse_extension_uses_smooth_flow_fallback(self):
        elbow_signal = [
            {"frame": 10, "angle_deg": 82.0, "valid": True},
            {"frame": 11, "angle_deg": 84.0, "valid": True},
            {"frame": 12, "angle_deg": 85.0, "valid": True},
            {"frame": 13, "angle_deg": 87.0, "valid": True},
            {"frame": 14, "angle_deg": 88.0, "valid": True},
            {"frame": 15, "angle_deg": 89.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 10},
            "uah": {"frame": 28},
            "release": {"frame": 30},
        }

        result = evaluate_elbow_legality(elbow_signal=elbow_signal, events=events)

        self.assertEqual(result["verdict"], "LEGAL")
        self.assertIsNone(result["extension_deg"])
        self.assertEqual(result["reason"], "flow_consistent_fallback")

    def test_sparse_extension_uses_irregular_flow_fallback(self):
        elbow_signal = [
            {"frame": 10, "angle_deg": 80.0, "valid": True},
            {"frame": 11, "angle_deg": 100.0, "valid": True},
            {"frame": 12, "angle_deg": 75.0, "valid": True},
            {"frame": 13, "angle_deg": 110.0, "valid": True},
            {"frame": 14, "angle_deg": 70.0, "valid": True},
            {"frame": 15, "angle_deg": 120.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 10},
            "uah": {"frame": 28},
            "release": {"frame": 30},
        }

        result = evaluate_elbow_legality(elbow_signal=elbow_signal, events=events)

        self.assertEqual(result["verdict"], "ILLEGAL")
        self.assertIsNone(result["extension_deg"])
        self.assertEqual(result["reason"], "flow_irregular_fallback")

    def test_release_anchored_window_measures_extension_when_uah_is_too_late(self):
        elbow_signal = [
            {"frame": 10, "angle_deg": 100.0, "valid": True},
            {"frame": 11, "angle_deg": 102.0, "valid": True},
            {"frame": 12, "angle_deg": 104.0, "valid": True},
            {"frame": 13, "angle_deg": 108.0, "valid": True},
            {"frame": 14, "angle_deg": 112.0, "valid": True},
            {"frame": 15, "angle_deg": 118.0, "valid": True},
            {"frame": 16, "angle_deg": 124.0, "valid": True},
            {"frame": 17, "angle_deg": 128.0, "valid": True},
            {"frame": 18, "angle_deg": 131.0, "valid": True},
            {"frame": 19, "angle_deg": 134.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 8},
            "uah": {"frame": 18},
            "release": {"frame": 20},
        }

        result = evaluate_elbow_legality(elbow_signal=elbow_signal, events=events)

        self.assertIn(result["verdict"], {"BORDERLINE", "ILLEGAL"})
        self.assertGreater(result["extension_deg"], 15.0)
        self.assertEqual(result["debug"]["window_mode"], "release_anchored")

    def test_release_grace_rescues_sparse_primary_window(self):
        elbow_signal = [
            {"frame": 19, "angle_deg": 110.0, "valid": True},
            {"frame": 20, "angle_deg": 114.0, "valid": True},
            {"frame": 21, "angle_deg": 118.0, "valid": True},
            {"frame": 22, "angle_deg": 121.0, "valid": True},
            {"frame": 23, "angle_deg": 124.0, "valid": True},
            {"frame": 24, "angle_deg": 128.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 12},
            "uah": {"frame": 21},
            "release": {"frame": 22},
        }

        result = evaluate_elbow_legality(elbow_signal=elbow_signal, events=events)

        self.assertEqual(result["debug"]["window_mode"], "release_grace_rescue")
        self.assertIsNotNone(result["extension_deg"])

    def test_weak_legal_window_with_clear_margin_stays_legal(self):
        elbow_signal = [
            {"frame": 19, "angle_deg": 110.0, "valid": True},
            {"frame": 20, "angle_deg": 111.0, "valid": True},
            {"frame": 21, "angle_deg": 112.0, "valid": True},
            {"frame": 22, "angle_deg": 113.0, "valid": True},
            {"frame": 23, "angle_deg": 114.0, "valid": True},
            {"frame": 24, "angle_deg": 115.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 12},
            "uah": {"frame": 21},
            "release": {"frame": 22},
        }

        result = evaluate_elbow_legality(
            elbow_signal=elbow_signal,
            events=events,
            pose_frames=self._low_vis_pose_frames(30),
            hand="R",
        )

        self.assertEqual(result["verdict"], "LEGAL")
        self.assertEqual(result["reason"], "weak_window_but_clear_margin")

    def test_weak_legal_window_near_threshold_becomes_suspect(self):
        elbow_signal = [
            {"frame": 19, "angle_deg": 110.0, "valid": True},
            {"frame": 20, "angle_deg": 112.0, "valid": True},
            {"frame": 21, "angle_deg": 115.0, "valid": True},
            {"frame": 22, "angle_deg": 137.0, "valid": True},
            {"frame": 23, "angle_deg": 160.0, "valid": True},
            {"frame": 24, "angle_deg": 183.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 12},
            "uah": {"frame": 21},
            "release": {"frame": 22},
        }

        result = evaluate_elbow_legality(
            elbow_signal=elbow_signal,
            events=events,
            pose_frames=self._low_vis_pose_frames(30),
            hand="R",
        )

        self.assertEqual(result["verdict"], "SUSPECT")
        self.assertEqual(result["reason"], "weak_window_conflicted")

    def test_low_visibility_rescue_degrades_to_suspect_when_only_grace_window_supports_it(self):
        pose_frames = self._low_vis_pose_frames(8)
        elbow_signal = [{"frame": i, "angle_deg": None, "valid": False} for i in range(8)]
        events = {
            "ffc": {"frame": 0},
            "uah": {"frame": 3},
            "release": {"frame": 4},
        }

        result = evaluate_elbow_legality(
            elbow_signal=elbow_signal,
            events=events,
            pose_frames=pose_frames,
            hand="R",
        )

        self.assertEqual(result["verdict"], "SUSPECT")
        self.assertEqual(result["reason"], "low_visibility_rescue_inconclusive")

    def test_interpreter_handles_null_extension_unknown(self):
        result = self.ci.build_elbow({"verdict": "UNKNOWN", "extension_deg": None})

        self.assertEqual(result["band"], "REVIEW")
        self.assertIn("could not be confirmed", result["player_text"].lower())

    def test_interpreter_handles_flow_based_illegal_message(self):
        result = self.ci.build_elbow(
            {"verdict": "ILLEGAL", "extension_deg": None, "reason": "flow_irregular_fallback"}
        )

        self.assertEqual(result["band"], "ILLEGAL")
        self.assertIn("flow appears abrupt", result["player_text"].lower())

    def test_interpreter_handles_suspect_message(self):
        result = self.ci.build_elbow({"verdict": "SUSPECT", "extension_deg": None})

        self.assertEqual(result["band"], "REVIEW")
        self.assertIn("should be reviewed", result["player_text"].lower())

    def test_primary_elbow_evaluation_is_idempotent(self):
        elbow_signal = [
            {"frame": 10, "angle_deg": 100.0, "valid": True},
            {"frame": 11, "angle_deg": 102.0, "valid": True},
            {"frame": 12, "angle_deg": 104.0, "valid": True},
            {"frame": 13, "angle_deg": 108.0, "valid": True},
            {"frame": 14, "angle_deg": 112.0, "valid": True},
            {"frame": 15, "angle_deg": 118.0, "valid": True},
            {"frame": 16, "angle_deg": 124.0, "valid": True},
            {"frame": 17, "angle_deg": 128.0, "valid": True},
            {"frame": 18, "angle_deg": 131.0, "valid": True},
            {"frame": 19, "angle_deg": 134.0, "valid": True},
        ]
        events = {
            "ffc": {"frame": 8},
            "uah": {"frame": 18},
            "release": {"frame": 20},
        }

        first = evaluate_elbow_legality(
            elbow_signal=deepcopy(elbow_signal),
            events=deepcopy(events),
        )
        second = evaluate_elbow_legality(
            elbow_signal=deepcopy(elbow_signal),
            events=deepcopy(events),
        )

        self.assertEqual(first, second)

    def test_low_visibility_rescue_is_idempotent(self):
        def frame(i):
            landmarks = [
                {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
                for _ in range(33)
            ]
            landmarks[12] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.95}
            landmarks[14] = {"x": 1.0, "y": 0.0, "z": 0.0, "visibility": 0.20}
            landmarks[16] = {"x": 2.0, "y": 0.0, "z": 0.0, "visibility": 0.20}
            return {"frame": i, "landmarks": landmarks}

        pose_frames = [frame(i) for i in range(8)]
        elbow_signal = [{"frame": i, "angle_deg": None, "valid": False} for i in range(8)]
        events = {
            "ffc": {"frame": 0},
            "uah": {"frame": 3},
            "release": {"frame": 4},
        }

        first = evaluate_elbow_legality(
            elbow_signal=deepcopy(elbow_signal),
            events=deepcopy(events),
            pose_frames=deepcopy(pose_frames),
            hand="R",
        )
        second = evaluate_elbow_legality(
            elbow_signal=deepcopy(elbow_signal),
            events=deepcopy(events),
            pose_frames=deepcopy(pose_frames),
            hand="R",
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
