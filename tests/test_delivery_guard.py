import os
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import HTTPException

from app.workers.events.delivery_guard import detect_delivery_candidates

os.environ.setdefault("ACTIONLAB_AUTO_CREATE_SCHEMA", "false")


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}
        for _ in range(33)
    ]


def _make_pose_frames(peaks, *, frames=150, hand="R"):
    wrist_velocity = np.zeros(frames, dtype=float)
    idx = np.arange(frames)
    for peak in peaks:
        if isinstance(peak, tuple):
            center, scale = peak
        else:
            center, scale = peak, 1.0
        wrist_velocity += float(scale) * np.exp(
            -0.5 * ((idx - float(center)) / 4.0) ** 2
        )

    wrist_x = 0.30 + np.cumsum(wrist_velocity) * 0.01
    pelvis_x = np.linspace(0.35, 0.60, frames)
    hand = (hand or "R").upper()
    shoulder_idx = 12 if hand == "R" else 11
    wrist_idx = 16 if hand == "R" else 15
    nb_elbow_idx = 13 if hand == "R" else 14

    out = []
    for i in range(frames):
        lm = _blank_landmarks()

        def set_pt(index, x, y, vis=0.95):
            lm[index] = {
                "x": float(x),
                "y": float(y),
                "z": 0.0,
                "visibility": float(vis),
            }

        set_pt(shoulder_idx, 0.48, 0.34)
        set_pt(wrist_idx, wrist_x[i], 0.26)
        set_pt(23, pelvis_x[i] - 0.04, 0.62)  # left hip
        set_pt(24, pelvis_x[i] + 0.04, 0.62)  # right hip
        set_pt(nb_elbow_idx, 0.42, 0.35)

        out.append({"frame": i, "landmarks": lm})

    return out


def _query_chain(first_return=None):
    query = MagicMock()
    query.filter.return_value = query
    query.first.return_value = first_return
    return query


class DeliveryGuardTests(unittest.TestCase):
    def test_single_delivery_clip_is_not_flagged(self):
        pose_frames = _make_pose_frames([70])

        result = detect_delivery_candidates(
            pose_frames=pose_frames,
            hand="R",
            fps=30.0,
        )

        self.assertLessEqual(result["delivery_count"], 1)

    def test_multiple_deliveries_are_flagged(self):
        pose_frames = _make_pose_frames([45, 105])

        result = detect_delivery_candidates(
            pose_frames=pose_frames,
            hand="R",
            fps=30.0,
        )

        self.assertGreaterEqual(result["delivery_count"], 2)
        self.assertEqual(result["method"], "wrist_velocity")
        self.assertGreaterEqual(len(result["candidate_frames"]), 2)

    def test_early_weaker_peak_is_not_counted_as_second_delivery(self):
        pose_frames = _make_pose_frames([(32, 0.65), (90, 1.0)], frames=150)

        result = detect_delivery_candidates(
            pose_frames=pose_frames,
            hand="R",
            fps=30.0,
        )

        self.assertLessEqual(result["delivery_count"], 1)
        self.assertEqual(result["method"], "wrist_velocity")

    def test_close_late_peak_is_not_counted_as_second_delivery(self):
        pose_frames = _make_pose_frames(
            [(223, 0.76), (264, 1.0)],
            frames=407,
            hand="L",
        )

        result = detect_delivery_candidates(
            pose_frames=pose_frames,
            hand="L",
            fps=30.0,
        )

        self.assertLessEqual(result["delivery_count"], 1)
        self.assertEqual(result["method"], "wrist_velocity")

    def test_early_and_close_late_peaks_collapse_to_single_delivery(self):
        pose_frames = _make_pose_frames(
            [(135, 0.42), (219, 0.78), (279, 1.0)],
            frames=407,
            hand="L",
        )

        result = detect_delivery_candidates(
            pose_frames=pose_frames,
            hand="L",
            fps=30.0,
        )

        self.assertLessEqual(result["delivery_count"], 1)
        self.assertEqual(result["method"], "wrist_velocity")

    def test_analyze_rejects_multi_delivery_video(self):
        from app.orchestrator.orchestrator import analyze

        request = SimpleNamespace(state=SimpleNamespace(request_id="req-1"))
        background_tasks = MagicMock()
        upload = SimpleNamespace(
            content_type="video/mp4",
            file=BytesIO(b"fake-video"),
        )
        current_account = SimpleNamespace(account_id="acc-1", role="COACH")
        db = MagicMock()
        db.query.side_effect = [
            _query_chain(first_return=SimpleNamespace()),
            _query_chain(first_return=SimpleNamespace(handedness="R", age_group="U16", season=2026)),
        ]

        with patch("app.orchestrator.orchestrator.SessionLocal", return_value=db), patch(
            "app.orchestrator.orchestrator.load_video",
            return_value=({"path": "/tmp/fake.mp4", "fps": 30.0, "total_frames": 150}, [], {}),
        ), patch(
            "app.orchestrator.orchestrator.run_preanalysis_screen",
            return_value={
                "passed": False,
                "blocking_issues": [
                    {
                        "code": "multiple_deliveries",
                        "detail": "Please upload a video with only one bowling delivery.",
                    }
                ],
            },
        ):
            with self.assertRaises(HTTPException) as context:
                analyze(
                    request=request,
                    background_tasks=background_tasks,
                    file=upload,
                    player_id="player-1",
                    bowler_type="pace",
                    age_group="U16",
                    season=2026,
                    actor=None,
                    current_account=current_account,
                )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(
            context.exception.detail,
            {
                "code": "multiple_deliveries",
                "message": "Please upload a video with only one bowling delivery.",
            },
        )


if __name__ == "__main__":
    unittest.main()
