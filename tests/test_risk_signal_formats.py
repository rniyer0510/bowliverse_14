import unittest

from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean
from app.workers.risk.trunk_rotation_snap import compute_trunk_rotation_snap


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95}
        for _ in range(33)
    ]


class RiskSignalFormatTests(unittest.TestCase):
    def _pose_frames(self):
        frames = []
        for i in range(20):
            lm = _blank_landmarks()
            curve = ((i - 10) ** 2) / 300.0
            lm[11] = {"x": 0.35 + (i * 0.004), "y": 0.30 + (curve * 0.6), "visibility": 0.95}
            lm[12] = {"x": 0.63 - (i * 0.006), "y": 0.34 + (i * 0.005) - (curve * 0.4), "visibility": 0.95}
            lm[23] = {"x": 0.42 + (i * 0.002), "y": 0.58 + (i * 0.002), "visibility": 0.95}
            lm[24] = {"x": 0.58 + (i * 0.007), "y": 0.61 + (i * 0.004), "visibility": 0.95}
            lm[27] = {"x": 0.40, "y": 0.78 + (i * 0.004), "visibility": 0.95}
            lm[28] = {"x": 0.60, "y": 0.76 + (i * 0.010), "visibility": 0.95}
            frames.append({"frame": i, "landmarks": lm})
        return frames

    def test_risk_modules_read_list_landmarks(self):
        pose_frames = self._pose_frames()
        self.assertGreater(compute_lateral_trunk_lean(pose_frames, 8, 10, 12, 60.0, {})['signal_strength'], 0.15)
        self.assertGreater(compute_hip_shoulder_mismatch(pose_frames, 10, 12, 60.0, {})['signal_strength'], 0.15)
        self.assertGreater(compute_trunk_rotation_snap(pose_frames, 10, 12, 60.0, {})['signal_strength'], 0.15)
        self.assertGreater(compute_front_foot_braking_shock(pose_frames, 10, 60.0, {}, action={})['signal_strength'], 0.15)
        self.assertGreater(compute_knee_brace_failure(pose_frames, 10, 60.0, {})['signal_strength'], 0.15)


if __name__ == '__main__':
    unittest.main()
