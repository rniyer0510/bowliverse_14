import unittest
from unittest.mock import patch

from app.workers.risk import risk_worker

NOSE = 0
LS = 11
RS = 12
LH = 23
RH = 24
LK = 25
RK = 26
LA = 27
RA = 28
LHEEL = 29
RHEEL = 30
LFOOT = 31
RFOOT = 32


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "visibility": 0.0}
        for _ in range(33)
    ]


def _front_frame(*, head_y=0.08, left_x=0.40, right_x=0.60, foot_y=0.93):
    lm = _blank_landmarks()

    def set_pt(idx, x, y, vis=0.95):
        lm[idx] = {"x": x, "y": y, "visibility": vis}

    set_pt(NOSE, 0.50, head_y)
    set_pt(LS, left_x, 0.25)
    set_pt(RS, right_x, 0.25)
    set_pt(LH, 0.44, 0.55)
    set_pt(RH, 0.56, 0.55)
    set_pt(LK, 0.46, 0.73)
    set_pt(RK, 0.54, 0.73)
    set_pt(LA, 0.47, foot_y)
    set_pt(RA, 0.53, foot_y)
    set_pt(LHEEL, 0.46, foot_y)
    set_pt(RHEEL, 0.54, foot_y)
    set_pt(LFOOT, 0.48, foot_y)
    set_pt(RFOOT, 0.52, foot_y)
    return {"frame": 0, "landmarks": lm}


class RiskVisualGateTests(unittest.TestCase):
    def test_run_risk_worker_handles_low_confidence_ffc_without_crashing(self):
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
        ), patch(
            "app.workers.risk.risk_worker._attach_visual",
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

    def test_rear_view_keeps_visual_suppressed_with_guidance(self):
        risk = {"risk_id": "front_foot_braking_shock", "signal_strength": 0.7}

        out = risk_worker._attach_visual(
            risk,
            pose_frames=[_front_frame() for _ in range(5)],
            video={"path": "/tmp/fake.mp4", "fps": 30},
            events={
                "ffc": {"frame": 2, "confidence": 0.9},
                "release": {"frame": 4, "method": "velocity_drop_20pct"},
            },
            run_id="run-1",
            rear_view_only=True,
        )

        self.assertNotIn("visual", out)
        self.assertEqual(
            out["visual_unavailable_reason"],
            risk_worker.FULL_BODY_GUIDANCE_MESSAGE,
        )
        self.assertEqual(out["capture_feedback"]["view"], "rear")

    def test_cropped_head_alone_does_not_block_visual_when_body_is_visible(self):
        risk = {"risk_id": "front_foot_braking_shock", "signal_strength": 0.7}
        visual_payload = {
            "frame": 2,
            "anchor": "event",
            "visual_confidence": "HIGH",
            "image_url": "http://example.test/visual.png",
        }
        cropped_frames = [
            _front_frame(head_y=0.01)
            for _ in range(5)
        ]

        with patch(
            "app.workers.risk.risk_worker.draw_and_save_visual",
            return_value=visual_payload,
        ) as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=cropped_frames,
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events={
                    "ffc": {"frame": 2, "confidence": 0.9},
                    "release": {"frame": 4, "method": "velocity_drop_20pct"},
                },
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(out["visual"], visual_payload)
        self.assertNotIn("visual_unavailable_reason", out)

    def test_supported_front_view_keeps_visual_generation_enabled(self):
        risk = {"risk_id": "front_foot_braking_shock", "signal_strength": 0.7}
        visual_payload = {
            "frame": 2,
            "anchor": "event",
            "visual_confidence": "HIGH",
            "image_url": "http://example.test/visual.png",
        }

        with patch(
            "app.workers.risk.risk_worker.draw_and_save_visual",
            return_value=visual_payload,
        ) as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=[_front_frame() for _ in range(5)],
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events={
                    "ffc": {"frame": 2, "confidence": 0.9},
                    "release": {"frame": 4, "method": "velocity_drop_20pct"},
                },
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(out["visual"], visual_payload)
        self.assertNotIn("visual_unavailable_reason", out)

    def test_foot_line_visual_prefers_uah_for_clearer_view(self):
        risk = {"risk_id": "foot_line_deviation", "signal_strength": 0.42}
        visual_payload = {
            "frame": 4,
            "anchor": "event",
            "visual_confidence": "HIGH",
            "image_url": "http://example.test/visual.png",
        }

        with patch(
            "app.workers.risk.risk_worker.draw_and_save_visual",
            return_value=visual_payload,
        ) as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=[_front_frame() for _ in range(6)],
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events={
                    "ffc": {
                        "frame": 2,
                        "method": "pelvis_then_geometry_relaxed",
                        "confidence": 0.9,
                    },
                    "uah": {"frame": 4, "method": "shoulder_peak", "confidence": 0.72},
                    "release": {"frame": 6, "method": "velocity_drop_20pct"},
                },
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(draw_mock.call_args.kwargs["frame_idx"], 4)
        self.assertEqual(out["visual"], visual_payload)

    def test_foot_line_visual_falls_back_to_ffc_plus_one_when_uah_is_weak(self):
        risk = {"risk_id": "foot_line_deviation", "signal_strength": 0.42}
        visual_payload = {
            "frame": 3,
            "anchor": "event",
            "visual_confidence": "HIGH",
            "image_url": "http://example.test/visual.png",
        }

        with patch(
            "app.workers.risk.risk_worker.draw_and_save_visual",
            return_value=visual_payload,
        ) as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=[_front_frame() for _ in range(6)],
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events={
                    "ffc": {
                        "frame": 2,
                        "method": "pelvis_then_geometry_relaxed",
                        "confidence": 0.9,
                    },
                    "uah": {
                        "frame": 4,
                        "method": "release_minus_one_fallback",
                        "confidence": 0.2,
                    },
                    "release": {"frame": 5, "method": "velocity_drop_20pct"},
                },
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(draw_mock.call_args.kwargs["frame_idx"], 3)
        self.assertEqual(out["visual"], visual_payload)

    def test_ffc_dependent_visual_can_still_render_with_flagged_ffc(self):
        risk = {"risk_id": "knee_brace_failure", "signal_strength": 0.72}
        visual_payload = {
            "frame": 2,
            "anchor": "event",
            "visual_confidence": "HIGH",
            "image_url": "http://example.test/visual.png",
        }

        with patch(
            "app.workers.risk.risk_worker.draw_and_save_visual",
            return_value=visual_payload,
        ) as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=[_front_frame() for _ in range(6)],
                video={"path": "/tmp/fake.mp4", "fps": 60},
                events={
                    "ffc": {
                        "frame": 2,
                        "confidence": 0.2,
                        "timing_flag": "early_relative_to_release",
                    },
                    "release": {"frame": 5, "method": "velocity_drop_20pct"},
                },
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(draw_mock.call_args.kwargs["frame_idx"], 2)
        self.assertEqual(out["visual"], visual_payload)


if __name__ == "__main__":
    unittest.main()
