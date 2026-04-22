import unittest
from unittest.mock import patch

from app.workers.risk import risk_worker


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "visibility": 0.0}
        for _ in range(33)
    ]


def _front_frame():
    lm = _blank_landmarks()

    def set_pt(idx, x, y, vis=0.95):
        lm[idx] = {"x": x, "y": y, "visibility": vis}

    set_pt(11, 0.40, 0.24)
    set_pt(12, 0.60, 0.24)
    set_pt(23, 0.44, 0.55)
    set_pt(24, 0.56, 0.55)
    set_pt(25, 0.46, 0.72)
    set_pt(26, 0.54, 0.72)
    set_pt(27, 0.47, 0.92)
    set_pt(28, 0.53, 0.92)
    set_pt(29, 0.46, 0.92)
    set_pt(30, 0.54, 0.92)
    set_pt(31, 0.48, 0.92)
    set_pt(32, 0.52, 0.92)
    return {"frame": 0, "landmarks": lm}


class RiskWorkerNoVisualsTests(unittest.TestCase):
    def test_run_risk_worker_handles_flagged_ffc_without_crashing(self):
        pose_frames = [_front_frame() for _ in range(6)]
        video = {"path": "/tmp/fake.mp4", "fps": 60}
        events = {
            "ffc": {
                "frame": 2,
                "confidence": 0.2,
                "timing_flag": "early_relative_to_release",
            },
            "bfc": {"frame": 1, "confidence": 0.4},
            "uah": {"frame": 4, "confidence": 0.8},
            "release": {"frame": 5, "confidence": 0.8, "method": "velocity_drop_20pct"},
        }

        with patch(
            "app.workers.risk.risk_worker.compute_front_foot_braking_shock",
            return_value={"signal_strength": 0.2, "confidence": 0.5},
        ) as braking_mock, patch(
            "app.workers.risk.risk_worker.compute_knee_brace_failure",
            return_value={"signal_strength": 0.2, "confidence": 0.5},
        ) as knee_mock, patch(
            "app.workers.risk.risk_worker.compute_trunk_rotation_snap",
            return_value={"signal_strength": 0.2, "confidence": 0.5},
        ), patch(
            "app.workers.risk.risk_worker.compute_hip_shoulder_mismatch",
            return_value={"signal_strength": 0.2, "confidence": 0.5},
        ), patch(
            "app.workers.risk.risk_worker.compute_lateral_trunk_lean",
            return_value={"signal_strength": 0.2, "confidence": 0.5},
        ), patch(
            "app.workers.risk.risk_worker.compute_foot_line_deviation",
            return_value={"signal_strength": 0.2, "confidence": 0.5},
        ) as foot_line_mock, patch(
            "app.workers.risk.risk_worker.attach_deviation_and_impact",
            side_effect=lambda risk, **_: risk,
        ):
            out = risk_worker.run_risk_worker(
                pose_frames=pose_frames,
                video=video,
                events=events,
                action={},
                run_id="run-1",
            )

        self.assertEqual(len(out), 6)
        self.assertEqual(braking_mock.call_args.args[1], 2)
        self.assertEqual(knee_mock.call_args.args[1], 2)
        self.assertEqual(foot_line_mock.call_args.args[1], 1)
        self.assertEqual(foot_line_mock.call_args.args[2], 2)

    def test_run_risk_worker_does_not_emit_visual_fields(self):
        pose_frames = [_front_frame() for _ in range(4)]
        events = {
            "bfc": {"frame": 1, "confidence": 0.7},
            "ffc": {"frame": 2, "confidence": 0.8},
            "uah": {"frame": 3, "confidence": 0.8},
            "release": {"frame": 3, "confidence": 0.8, "method": "velocity_drop_20pct"},
        }

        with patch(
            "app.workers.risk.risk_worker.compute_front_foot_braking_shock",
            return_value={"signal_strength": 0.3, "confidence": 0.6},
        ), patch(
            "app.workers.risk.risk_worker.compute_knee_brace_failure",
            return_value={"signal_strength": 0.3, "confidence": 0.6},
        ), patch(
            "app.workers.risk.risk_worker.compute_trunk_rotation_snap",
            return_value={"signal_strength": 0.3, "confidence": 0.6},
        ), patch(
            "app.workers.risk.risk_worker.compute_hip_shoulder_mismatch",
            return_value={"signal_strength": 0.3, "confidence": 0.6},
        ), patch(
            "app.workers.risk.risk_worker.compute_lateral_trunk_lean",
            return_value={"signal_strength": 0.3, "confidence": 0.6},
        ), patch(
            "app.workers.risk.risk_worker.compute_foot_line_deviation",
            return_value={"signal_strength": 0.3, "confidence": 0.6},
        ), patch(
            "app.workers.risk.risk_worker.attach_deviation_and_impact",
            side_effect=lambda risk, **_: {
                **risk,
                "deviation": {"band": 2},
                "impact": {"primary": ["lower back"]},
            },
        ):
            out = risk_worker.run_risk_worker(
                pose_frames=pose_frames,
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events=events,
                action={},
                run_id="run-2",
            )

        self.assertEqual(len(out), 6)
        for risk in out:
            self.assertNotIn("visual", risk)
            self.assertNotIn("visual_window", risk)
            self.assertNotIn("visual_unavailable_reason", risk)
            self.assertNotIn("capture_feedback", risk)
            self.assertIn("deviation", risk)
            self.assertIn("impact", risk)


if __name__ == "__main__":
    unittest.main()
