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


if __name__ == "__main__":
    unittest.main()
