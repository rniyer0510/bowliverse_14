import os
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import HTTPException

from app.workers.events.delivery_guard import detect_delivery_candidates

os.environ.setdefault("ACTIONLAB_AUTO_CREATE_SCHEMA", "false")
from app.orchestrator.orchestrator import analyze


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}
        for _ in range(33)
    ]


def _make_pose_frames(peaks, *, frames=150):
    wrist_velocity = np.zeros(frames, dtype=float)
    idx = np.arange(frames)
    for peak in peaks:
        wrist_velocity += np.exp(-0.5 * ((idx - peak) / 4.0) ** 2)

    wrist_x = 0.30 + np.cumsum(wrist_velocity) * 0.01
    pelvis_x = np.linspace(0.35, 0.60, frames)

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

        set_pt(12, 0.48, 0.34)  # right shoulder
        set_pt(16, wrist_x[i], 0.26)  # right wrist
        set_pt(23, pelvis_x[i] - 0.04, 0.62)  # left hip
        set_pt(24, pelvis_x[i] + 0.04, 0.62)  # right hip
        set_pt(13, 0.42, 0.35)  # left elbow (non-bowling elbow for R hand)

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

    def test_analyze_rejects_multi_delivery_video(self):
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
