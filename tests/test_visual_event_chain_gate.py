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

    set_pt(risk_worker.LS, 0.40, 0.24)
    set_pt(risk_worker.RS, 0.60, 0.24)
    set_pt(risk_worker.LH, 0.44, 0.55)
    set_pt(risk_worker.RH, 0.56, 0.55)
    set_pt(risk_worker.LW, 0.38, 0.38)
    set_pt(risk_worker.RW, 0.62, 0.38)
    set_pt(25, 0.46, 0.72)
    set_pt(26, 0.54, 0.72)
    set_pt(27, 0.47, 0.92)
    set_pt(28, 0.53, 0.92)
    set_pt(29, 0.46, 0.92)
    set_pt(30, 0.54, 0.92)
    set_pt(31, 0.48, 0.92)
    set_pt(32, 0.52, 0.92)
    return {"frame": 0, "landmarks": lm}


class VisualEventChainGateTests(unittest.TestCase):
    def test_trunk_visual_suppressed_for_weak_release_chain(self):
        risk = {"risk_id": "trunk_rotation_snap", "signal_strength": 0.7}
        events = {
            "release": {"frame": 4, "method": "peak_plus_offset", "confidence": 0.60},
            "uah": {"frame": 3, "method": "release_minus_one_fallback", "confidence": 0.20},
        }

        with patch("app.workers.risk.risk_worker.draw_and_save_visual") as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=[_front_frame() for _ in range(8)],
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events=events,
                run_id="run-weak",
                rear_view_only=False,
            )

        draw_mock.assert_not_called()
        self.assertNotIn("visual", out)
        self.assertEqual(out["visual_unavailable_reason"], risk_worker.EVENT_CHAIN_GUIDANCE_MESSAGE)
        self.assertEqual(out["capture_feedback"]["issue"], "weak_event_chain")

    def test_ffc_visual_suppressed_only_for_jointly_weak_chain(self):
        risk = {"risk_id": "front_foot_braking_shock", "signal_strength": 0.7}
        events = {
            "release": {"frame": 6, "method": "peak_plus_offset", "confidence": 0.60},
            "ffc": {"frame": 5, "method": "ultimate_fallback", "confidence": 0.15},
        }

        with patch("app.workers.risk.risk_worker.draw_and_save_visual") as draw_mock:
            out = risk_worker._attach_visual(
                risk,
                pose_frames=[_front_frame() for _ in range(8)],
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events=events,
                run_id="run-ffc-weak",
                rear_view_only=False,
            )

        draw_mock.assert_not_called()
        self.assertEqual(out["visual_unavailable_reason"], risk_worker.EVENT_CHAIN_GUIDANCE_MESSAGE)

    def test_supported_chain_still_renders_visual(self):
        risk = {"risk_id": "hip_shoulder_mismatch", "signal_strength": 0.7}
        events = {
            "release": {"frame": 5, "method": "wrist_at_shoulder", "confidence": 0.85},
            "uah": {"frame": 4, "method": "upper_arm_horizontal", "confidence": 0.82},
        }
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
                pose_frames=[_front_frame() for _ in range(8)],
                video={"path": "/tmp/fake.mp4", "fps": 30},
                events=events,
                run_id="run-strong",
                rear_view_only=False,
            )

        draw_mock.assert_called_once()
        self.assertEqual(out["visual"], visual_payload)
        self.assertNotIn("visual_unavailable_reason", out)


if __name__ == "__main__":
    unittest.main()
