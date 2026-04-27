import unittest
import numpy as np

from app.workers.risk.foot_line_deviation import compute_foot_line_deviation
from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean
from app.workers.risk.neck_tilt_left_bfc import compute_neck_tilt_left_bfc
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

    def _pose_frames_with_shoulder_lead(self):
        frames = []
        shoulder_angles = [0, 2, 5, 9, 14, 20, 26, 30, 33, 35, 36, 36]
        hip_angles = [0, 0, 1, 2, 4, 7, 11, 16, 22, 27, 31, 34]
        shoulder_half = 0.12
        hip_half = 0.10
        for i in range(20):
            lm = _blank_landmarks()
            local_i = max(0, min(len(shoulder_angles) - 1, i - 2))
            srad = np.radians(shoulder_angles[local_i])
            hrad = np.radians(hip_angles[local_i])
            sdx = np.cos(srad) * shoulder_half
            sdy = np.sin(srad) * shoulder_half * 0.25
            hdx = np.cos(hrad) * hip_half
            hdy = np.sin(hrad) * hip_half * 0.25
            lm[11] = {"x": 0.50 - sdx, "y": 0.30 - sdy, "visibility": 0.95}
            lm[12] = {"x": 0.50 + sdx, "y": 0.30 + sdy, "visibility": 0.95}
            lm[23] = {"x": 0.50 - hdx, "y": 0.60 - hdy, "visibility": 0.95}
            lm[24] = {"x": 0.50 + hdx, "y": 0.60 + hdy, "visibility": 0.95}
            frames.append({"frame": i, "landmarks": lm})
        return frames

    def test_risk_modules_read_list_landmarks(self):
        pose_frames = self._pose_frames()
        self.assertGreaterEqual(compute_lateral_trunk_lean(pose_frames, 8, 10, 12, 60.0, {})['signal_strength'], 0.15)
        self.assertGreater(compute_hip_shoulder_mismatch(pose_frames, 10, 12, 60.0, {})['signal_strength'], 0.15)
        self.assertGreaterEqual(compute_trunk_rotation_snap(pose_frames, 10, 12, 60.0, {})['signal_strength'], 0.15)
        braking = compute_front_foot_braking_shock(pose_frames, 10, 60.0, {}, action={})
        self.assertGreaterEqual(braking['signal_strength'], 0.15)
        self.assertGreater(compute_knee_brace_failure(pose_frames, 10, 60.0, {})['signal_strength'], 0.15)

    def test_lateral_trunk_lean_uses_release_window_bowling_side_angle(self):
        upright_frames = self._pose_frames()
        leaning_frames = self._pose_frames()

        for i, frame in enumerate(upright_frames):
            lm = frame['landmarks']
            if i >= 6:
                lm[12] = {'x': 0.60, 'y': 0.30, 'visibility': 0.95}
                lm[24] = {'x': 0.58, 'y': 0.60, 'visibility': 0.95}

        for i, frame in enumerate(leaning_frames):
            lm = frame['landmarks']
            if i >= 6:
                shift = (i - 10) * 0.03
                lm[12] = {'x': 0.60 + shift, 'y': 0.30, 'visibility': 0.95}
                lm[24] = {'x': 0.58, 'y': 0.60, 'visibility': 0.95}

        upright = compute_lateral_trunk_lean(upright_frames, 8, 10, 12, 60.0, {'hand': 'R'})
        leaning = compute_lateral_trunk_lean(leaning_frames, 8, 10, 12, 60.0, {'hand': 'R'})

        self.assertGreater(leaning['debug']['late_angle_deg'], upright['debug']['late_angle_deg'])
        self.assertIn('late_angle_deg', leaning['debug'])

    def test_trunk_rotation_snap_unwraps_and_penalizes_late_abrupt_rotation(self):
        smooth_frames = []
        abrupt_frames = []
        smooth_angles = [170, 175, 179, -179, -175, -170, -166, -162]
        abrupt_angles = [170, 171, 172, 174, 176, -170, -130, -100]
        for i, (smooth_ang, abrupt_ang) in enumerate(zip(smooth_angles, abrupt_angles)):
            smooth_lm = _blank_landmarks()
            abrupt_lm = _blank_landmarks()
            srad = np.radians(smooth_ang)
            arad = np.radians(abrupt_ang)
            sdx = np.cos(srad) * 0.12
            sdy = np.sin(srad) * 0.02
            adx = np.cos(arad) * 0.12
            ady = np.sin(arad) * 0.02
            smooth_lm[11] = {'x': 0.5 - sdx, 'y': 0.3 - sdy, 'visibility': 0.95}
            smooth_lm[12] = {'x': 0.5 + sdx, 'y': 0.3 + sdy, 'visibility': 0.95}
            abrupt_lm[11] = {'x': 0.5 - adx, 'y': 0.3 - ady, 'visibility': 0.95}
            abrupt_lm[12] = {'x': 0.5 + adx, 'y': 0.3 + ady, 'visibility': 0.95}
            smooth_frames.append({'frame': i, 'landmarks': smooth_lm})
            abrupt_frames.append({'frame': i, 'landmarks': abrupt_lm})

        smooth = compute_trunk_rotation_snap(smooth_frames, 3, 5, 60.0, {})
        abrupt = compute_trunk_rotation_snap(abrupt_frames, 3, 5, 60.0, {})

        self.assertLess(smooth['signal_strength'], abrupt['signal_strength'])
        self.assertIn('snap_index', abrupt['debug'])
        self.assertIn('timing_ratio', abrupt['debug'])

    def test_knee_brace_uses_front_knee_support_not_pelvis_drop_alone(self):
        braced_frames = self._pose_frames()
        collapsing_frames = self._pose_frames()

        for i, frame in enumerate(braced_frames):
            lm = frame['landmarks']
            if i >= 10:
                lm[23] = {'x': 0.44, 'y': 0.58, 'visibility': 0.95}
                lm[25] = {'x': 0.46, 'y': 0.72, 'visibility': 0.95}
                lm[27] = {'x': 0.47, 'y': 0.90, 'visibility': 0.95}

        for i, frame in enumerate(collapsing_frames):
            lm = frame['landmarks']
            if i >= 10:
                sink = (i - 10) * 0.01
                lm[23] = {'x': 0.44, 'y': 0.58 + (sink * 0.4), 'visibility': 0.95}
                lm[25] = {'x': 0.46 + (sink * 0.8), 'y': 0.72 + sink, 'visibility': 0.95}
                lm[27] = {'x': 0.47, 'y': 0.90, 'visibility': 0.95}

        braced = compute_knee_brace_failure(braced_frames, 10, 60.0, {'hand': 'R'})
        collapsing = compute_knee_brace_failure(collapsing_frames, 10, 60.0, {'hand': 'R'})

        self.assertGreater(collapsing['signal_strength'], braced['signal_strength'])
        self.assertIn('collapse_deg', collapsing['debug'])
        self.assertIn('release_angle', collapsing['debug'])

    def test_foot_line_deviation_downweights_outward_toe_out_without_plant_cross(self):
        outward_frames = self._pose_frames()
        inward_frames = self._pose_frames()

        outward_frames[8]['landmarks'][32] = {'x': 0.62, 'y': 0.88, 'visibility': 0.95}
        outward_frames[10]['landmarks'][31] = {'x': 0.38, 'y': 0.88, 'visibility': 0.95}
        outward_frames[10]['landmarks'][29] = {'x': 0.60, 'y': 0.90, 'visibility': 0.95}
        outward_frames[10]['landmarks'][27] = {'x': 0.59, 'y': 0.88, 'visibility': 0.95}
        outward_frames[10]['landmarks'][25] = {'x': 0.58, 'y': 0.72, 'visibility': 0.95}
        outward_frames[10]['landmarks'][23] = {'x': 0.44, 'y': 0.58, 'visibility': 0.95}
        outward_frames[10]['landmarks'][24] = {'x': 0.56, 'y': 0.58, 'visibility': 0.95}

        inward_frames[8]['landmarks'][32] = {'x': 0.62, 'y': 0.88, 'visibility': 0.95}
        inward_frames[10]['landmarks'][31] = {'x': 0.38, 'y': 0.88, 'visibility': 0.95}
        inward_frames[10]['landmarks'][29] = {'x': 0.41, 'y': 0.90, 'visibility': 0.95}
        inward_frames[10]['landmarks'][27] = {'x': 0.42, 'y': 0.88, 'visibility': 0.95}
        inward_frames[10]['landmarks'][25] = {'x': 0.39, 'y': 0.72, 'visibility': 0.95}
        inward_frames[10]['landmarks'][23] = {'x': 0.44, 'y': 0.58, 'visibility': 0.95}
        inward_frames[10]['landmarks'][24] = {'x': 0.56, 'y': 0.58, 'visibility': 0.95}

        outward = compute_foot_line_deviation(outward_frames, 8, 10, 60.0, {}, action={'hand': 'R'})
        inward = compute_foot_line_deviation(inward_frames, 8, 10, 60.0, {}, action={'hand': 'R'})

        self.assertEqual(outward['mode'], 'OUTWARD_STEP')
        self.assertEqual(inward['mode'], 'OUTWARD_STEP')
        self.assertLess(outward['signal_strength'], inward['signal_strength'])
        self.assertIn('support_line_norm', outward['metrics'])

    def test_front_foot_braking_prefers_coherent_rigid_foot_motion(self):
        stable_frames = self._pose_frames()
        noisy_frames = self._pose_frames()

        for i, frame in enumerate(stable_frames):
            lm = frame['landmarks']
            if i >= 10:
                drift = (i - 10) * 0.006
                lm[29] = {'x': 0.44 - drift, 'y': 0.90 - (drift * 0.5), 'visibility': 0.95}
                lm[31] = {'x': 0.50 - drift, 'y': 0.90 - (drift * 0.4), 'visibility': 0.95}
                lm[23] = {'x': 0.44, 'y': 0.58, 'visibility': 0.95}
                lm[24] = {'x': 0.56, 'y': 0.58, 'visibility': 0.95}

        for i, frame in enumerate(noisy_frames):
            lm = frame['landmarks']
            if i >= 10:
                drift = (i - 10) * 0.006
                lm[29] = {'x': 0.44 - (drift * 2.0), 'y': 0.90 - (drift * 0.3), 'visibility': 0.55}
                lm[31] = {'x': 0.50 + (drift * 0.1), 'y': 0.90 + (drift * 0.8), 'visibility': 0.55}
                lm[23] = {'x': 0.44, 'y': 0.58, 'visibility': 0.95}
                lm[24] = {'x': 0.56, 'y': 0.58, 'visibility': 0.95}

        stable = compute_front_foot_braking_shock(stable_frames, 10, 60.0, {}, action={})
        noisy = compute_front_foot_braking_shock(noisy_frames, 10, 60.0, {}, action={})

        self.assertGreater(stable['signal_strength'], noisy['signal_strength'])
        self.assertGreater(stable['confidence'], noisy['confidence'])
        self.assertIn('coherent_motion', stable['debug'])

    def test_neck_tilt_left_bfc_detects_head_offset_from_shoulder_line(self):
        pose_frames = self._pose_frames()
        for frame in pose_frames[8:11]:
            lm = frame["landmarks"]
            lm[0] = {"x": 0.40, "y": 0.20, "visibility": 0.95}
            lm[11] = {"x": 0.44, "y": 0.30, "visibility": 0.95}
            lm[12] = {"x": 0.64, "y": 0.30, "visibility": 0.95}

        result = compute_neck_tilt_left_bfc(pose_frames, 9, 60.0, {})

        self.assertGreater(result["signal_strength"], 0.35)
        self.assertGreater(result["confidence"], 0.5)
        self.assertIn("median_left_offset", result["debug"])

    def test_hip_shoulder_mismatch_exposes_sequence_direction(self):
        result = compute_hip_shoulder_mismatch(
            self._pose_frames_with_shoulder_lead(),
            10,
            12,
            60.0,
            {},
        )

        self.assertEqual(result["debug"]["sequence_pattern"], "shoulders_lead")
        self.assertLess(result["debug"]["sequence_delta_frames"], 0)
        self.assertIn("phase_lag_deg", result["debug"])


if __name__ == '__main__':
    unittest.main()
