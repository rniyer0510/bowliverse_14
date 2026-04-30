import unittest

from app.workers.action.foot_orientation import compute_foot_intent


def _landmark(x, y, visibility=0.99):
    return {
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "visibility": float(visibility),
    }


def _blank_landmarks():
    return [_landmark(0.0, 0.0, 0.0) for _ in range(33)]


class ActionClassifierThresholdTests(unittest.TestCase):
    def test_compute_foot_intent_keeps_moderate_opening_in_semi_open_band(self):
        lm = _blank_landmarks()
        lm[27] = _landmark(0.50, 0.50)  # left ankle
        lm[29] = _landmark(0.70, 0.54)  # left heel
        lm[31] = _landmark(0.74, 0.56)  # left toe
        pose_frames = [{"frame": 0, "landmarks": lm}]

        result = compute_foot_intent(
            pose_frames=pose_frames,
            hand="R",
            bfc_frame=0,
            axis=(0.0, 1.0),
        )

        self.assertIsNotNone(result)
        self.assertGreater(result["angle"], 65.0)
        self.assertLess(result["angle"], 80.0)
        self.assertEqual(result["intent"], "SEMI_OPEN")


if __name__ == "__main__":
    unittest.main()
