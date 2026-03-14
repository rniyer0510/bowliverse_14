import unittest
from unittest.mock import patch

from app.workers.risk import risk_worker


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "visibility": 0.0}
        for _ in range(33)
    ]


def _front_frame(*, head_y=0.08, left_x=0.40, right_x=0.60, foot_y=0.93):
    lm = _blank_landmarks()

    def set_pt(idx, x, y, vis=0.95):
        lm[idx] = {"x": x, "y": y, "visibility": vis}

    set_pt(risk_worker.NOSE, 0.50, head_y)
    set_pt(risk_worker.LS, left_x, 0.25)
    set_pt(risk_worker.RS, right_x, 0.25)
    set_pt(risk_worker.LH, 0.44, 0.55)
    set_pt(risk_worker.RH, 0.56, 0.55)
    set_pt(risk_worker.LK, 0.46, 0.73)
    set_pt(risk_worker.RK, 0.54, 0.73)
    set_pt(risk_worker.LA, 0.47, foot_y)
    set_pt(risk_worker.RA, 0.53, foot_y)
    set_pt(risk_worker.LHEEL, 0.46, foot_y)
    set_pt(risk_worker.RHEEL, 0.54, foot_y)
    set_pt(risk_worker.LFOOT, 0.48, foot_y)
    set_pt(risk_worker.RFOOT, 0.52, foot_y)
    return {"frame": 0, "landmarks": lm}


class RiskVisualGateTests(unittest.TestCase):
    def test_rear_view_keeps_visual_suppressed_with_guidance(self):
        risk = {"risk_id": "front_foot_braking_shock", "signal_strength": 0.7}

        out = risk_worker._attach_visual(
            risk,
            pose_frames=[_front_frame() for _ in range(5)],
            video={"path": "/tmp/fake.mp4", "fps": 30},
            events={"ffc": {"frame": 2}},
            run_id="run-1",
            rear_view_only=True,
        )

        self.assertNotIn("visual", out)
        self.assertEqual(
            out["visual_unavailable_reason"],
            risk_worker.FULL_BODY_GUIDANCE_MESSAGE,
        )
        self.assertEqual(out["capture_feedback"]["view"], "rear")

    def test_cropped_front_view_suppresses_visual(self):
        risk = {"risk_id": "front_foot_braking_shock", "signal_strength": 0.7}
        cropped_frames = [
            _front_frame(head_y=0.01)
            for _ in range(5)
        ]

        with patch("app.workers.risk.risk_worker.draw_and_save_visual") as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=cropped_frames,
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events={"ffc": {"frame": 2}},
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_not_called()
        self.assertNotIn("visual", out)
        self.assertEqual(
            out["visual_unavailable_reason"],
            risk_worker.FULL_BODY_GUIDANCE_MESSAGE,
        )
        self.assertEqual(out["capture_feedback"]["issue"], "cropped_or_incomplete")

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
                events={"ffc": {"frame": 2}},
                run_id="run-1",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(out["visual"], visual_payload)
        self.assertNotIn("visual_unavailable_reason", out)


if __name__ == "__main__":
    unittest.main()
