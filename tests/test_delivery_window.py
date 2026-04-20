import os
import tempfile
import unittest

import cv2
import numpy as np

from app.workers.windowing.delivery_window import detect_delivery_window


class DeliveryWindowTests(unittest.TestCase):
    def test_missing_video_metadata_returns_unavailable(self):
        result = detect_delivery_window({"fps": 30.0, "total_frames": 90})
        self.assertFalse(result["available"])

    def test_detects_motion_burst_window(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        path = tmp.name

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (160, 120),
        )
        for idx in range(180):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            x = 40 if idx < 80 else min(120, 40 + ((idx - 80) * 3))
            if idx > 120:
                x = 120
            cv2.rectangle(frame, (x, 30), (x + 18, 95), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        try:
            result = detect_delivery_window(
                {"path": path, "fps": 30.0, "total_frames": 180}
            )
        finally:
            os.remove(path)

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["release_hint"], 80)
        self.assertLessEqual(result["analysis_start"], result["release_hint"])
        self.assertGreaterEqual(result["analysis_end"], result["release_hint"])
        self.assertEqual(result["delivery_count"], 1)
        self.assertEqual(len(result["peak_frames"]), 1)
        self.assertEqual(result["method"], "subject_local_motion_scan")

    def test_reports_multiple_motion_bursts(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        path = tmp.name

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (160, 120),
        )
        for idx in range(240):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            if 50 <= idx <= 80:
                x = 30 + ((idx - 50) * 3)
                cv2.rectangle(frame, (x, 28), (x + 16, 96), (255, 255, 255), -1)
            if 150 <= idx <= 182:
                x = 20 + ((idx - 150) * 4)
                cv2.rectangle(frame, (x, 24), (x + 20, 98), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        try:
            result = detect_delivery_window(
                {"path": path, "fps": 30.0, "total_frames": 240}
            )
        finally:
            os.remove(path)

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["delivery_count"], 2)
        self.assertGreaterEqual(len(result["peak_frames"]), 2)

    def test_prefers_later_peak_as_release_hint_when_bursts_are_similar(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        path = tmp.name

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (160, 120),
        )
        for idx in range(240):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            if 45 <= idx <= 72:
                x = 28 + ((idx - 45) * 3)
                cv2.rectangle(frame, (x, 26), (x + 18, 98), (255, 255, 255), -1)
            if 155 <= idx <= 182:
                x = 24 + ((idx - 155) * 3)
                cv2.rectangle(frame, (x, 26), (x + 18, 98), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        try:
            result = detect_delivery_window(
                {"path": path, "fps": 30.0, "total_frames": 240}
            )
        finally:
            os.remove(path)

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["release_hint"], 150)
        self.assertTrue(any(frame <= 80 for frame in result["peak_frames"]))
        self.assertTrue(any(frame >= 160 for frame in result["peak_frames"]))
        self.assertGreaterEqual(len(result["peak_frames"]), 2)

    def test_filters_weaker_early_bursts_when_late_peak_is_dominant(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        path = tmp.name

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (160, 120),
        )
        for idx in range(240):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            if 20 <= idx <= 42:
                x = 22 + ((idx - 20) * 2)
                cv2.rectangle(frame, (x, 32), (x + 10, 86), (180, 180, 180), -1)
            if 90 <= idx <= 114:
                x = 26 + ((idx - 90) * 2)
                cv2.rectangle(frame, (x, 32), (x + 10, 86), (180, 180, 180), -1)
            if 172 <= idx <= 214:
                x = 18 + ((idx - 172) * 4)
                cv2.rectangle(frame, (x, 24), (x + 22, 102), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        try:
            result = detect_delivery_window(
                {"path": path, "fps": 30.0, "total_frames": 240}
            )
        finally:
            os.remove(path)

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["release_hint"], 170)
        self.assertEqual(result["delivery_count"], 1)
        self.assertEqual(len(result["peak_frames"]), 1)
        self.assertGreaterEqual(len(result["raw_candidate_peak_frames"]), 2)


if __name__ == "__main__":
    unittest.main()
