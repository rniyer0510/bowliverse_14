import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.io.loader import load_video


class _FakeCap:
    def __init__(self, frames):
        self.frames = frames
        self.pos = 0
        self.set_calls = []

    def isOpened(self):
        return True

    def get(self, _prop):
        return 0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        self.pos = int(value)
        return True

    def read(self):
        if self.pos >= len(self.frames):
            return False, None
        frame = self.frames[self.pos]
        self.pos += 1
        return True, frame

    def release(self):
        return None


class _FakeTracker:
    def __init__(self):
        point = SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.95)
        self._result = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=[point for _ in range(33)])
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def process(self, _rgb):
        return self._result


class LoaderPhase2Tests(unittest.TestCase):
    def test_load_video_only_runs_pose_inside_detected_window(self):
        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(120)]
        fake_cap = _FakeCap(frames)
        video = {
            "path": "/tmp/fake.mp4",
            "fps": 30.0,
            "total_frames": 120,
            "width": 24,
            "height": 24,
        }

        with patch(
            "app.io.loader._save_upload_to_temp_path",
            return_value="/tmp/fake.mp4",
        ), patch(
            "app.io.loader._read_video_metadata",
            return_value=(fake_cap, dict(video)),
        ), patch(
            "app.io.loader.detect_delivery_window",
            return_value={
                "available": True,
                "confidence": 0.8,
                "analysis_start": 30,
                "analysis_end": 90,
                "delivery_count": 1,
                "peak_frames": [62],
            },
        ), patch(
            "app.io.loader._create_pose_tracker",
            return_value=_FakeTracker(),
        ):
            loaded_video, pose_frames, events = load_video(SimpleNamespace(file=None))

        self.assertEqual(loaded_video["pose_window"]["start"], 30)
        self.assertEqual(loaded_video["pose_window"]["end"], 90)
        self.assertEqual(len(pose_frames), 120)
        self.assertIsNone(pose_frames[29]["landmarks"])
        self.assertIsInstance(pose_frames[30]["landmarks"], list)
        self.assertIsInstance(pose_frames[90]["landmarks"], list)
        self.assertIsNone(pose_frames[91]["landmarks"])
        self.assertEqual(events, {})

    def test_load_video_falls_back_to_full_clip_when_window_unavailable(self):
        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(4)]
        fake_cap = _FakeCap(frames)
        video = {
            "path": "/tmp/fake.mp4",
            "fps": 30.0,
            "total_frames": 4,
            "width": 24,
            "height": 24,
        }

        with patch(
            "app.io.loader._save_upload_to_temp_path",
            return_value="/tmp/fake.mp4",
        ), patch(
            "app.io.loader._read_video_metadata",
            return_value=(fake_cap, dict(video)),
        ), patch(
            "app.io.loader.detect_delivery_window",
            return_value={
                "available": False,
                "reason": "low_motion_signal",
                "release_hint": 3,
                "analysis_start": 0,
                "analysis_end": 3,
            },
        ), patch(
            "app.io.loader._create_pose_tracker",
            return_value=_FakeTracker(),
        ):
            loaded_video, pose_frames, _ = load_video(SimpleNamespace(file=None))

        self.assertEqual(loaded_video["pose_window"]["start"], 0)
        self.assertEqual(loaded_video["pose_window"]["end"], 3)
        self.assertEqual(loaded_video["pose_window"]["source"], "late_window_fallback")
        self.assertTrue(all(isinstance(item["landmarks"], list) for item in pose_frames))

    def test_load_video_uses_late_clip_fallback_without_release_hint(self):
        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(120)]
        fake_cap = _FakeCap(frames)
        video = {
            "path": "/tmp/fake.mp4",
            "fps": 30.0,
            "total_frames": 120,
            "width": 24,
            "height": 24,
        }

        with patch(
            "app.io.loader._save_upload_to_temp_path",
            return_value="/tmp/fake.mp4",
        ), patch(
            "app.io.loader._read_video_metadata",
            return_value=(fake_cap, dict(video)),
        ), patch(
            "app.io.loader.detect_delivery_window",
            return_value={"available": False, "reason": "low_motion_signal"},
        ), patch(
            "app.io.loader._create_pose_tracker",
            return_value=_FakeTracker(),
        ):
            loaded_video, pose_frames, _ = load_video(SimpleNamespace(file=None))

        self.assertEqual(loaded_video["pose_window"]["source"], "late_window_fallback")
        self.assertGreaterEqual(loaded_video["pose_window"]["start"], 40)
        self.assertEqual(loaded_video["pose_window"]["end"], 119)
        self.assertIsNone(pose_frames[loaded_video["pose_window"]["start"] - 1]["landmarks"])
        self.assertIsInstance(pose_frames[loaded_video["pose_window"]["start"]]["landmarks"], list)
