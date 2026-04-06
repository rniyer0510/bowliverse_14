import unittest

from app.workers.risk.foot_line_deviation import compute_foot_line_deviation


LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT = 31
RIGHT_FOOT = 32


def _blank_landmarks():
    return [{"x": 0.5, "y": 0.5, "visibility": 0.0} for _ in range(33)]


def _set_point(landmarks, idx, x, y, vis=0.95):
    landmarks[idx] = {"x": x, "y": y, "visibility": vis}


def _frame(
    *,
    left_hip=(0.44, 0.55),
    right_hip=(0.56, 0.55),
    left_knee=(0.46, 0.73),
    right_knee=(0.54, 0.73),
    left_ankle=(0.47, 0.92),
    right_ankle=(0.53, 0.92),
    left_heel=(0.46, 0.93),
    right_heel=(0.54, 0.93),
    left_foot=(0.48, 0.94),
    right_foot=(0.52, 0.94),
):
    lm = _blank_landmarks()
    _set_point(lm, LEFT_HIP, *left_hip)
    _set_point(lm, RIGHT_HIP, *right_hip)
    _set_point(lm, LEFT_KNEE, *left_knee)
    _set_point(lm, RIGHT_KNEE, *right_knee)
    _set_point(lm, LEFT_ANKLE, *left_ankle)
    _set_point(lm, RIGHT_ANKLE, *right_ankle)
    _set_point(lm, LEFT_HEEL, *left_heel)
    _set_point(lm, RIGHT_HEEL, *right_heel)
    _set_point(lm, LEFT_FOOT, *left_foot)
    _set_point(lm, RIGHT_FOOT, *right_foot)
    return {"landmarks": lm}


class FootLineDeviationTests(unittest.TestCase):
    def test_visibly_wide_outward_step_scores_as_amber_not_green(self):
        pose_frames = [
            _frame(
                left_hip=(0.44, 0.52),
                right_hip=(0.56, 0.52),
                left_foot=(0.48, 0.90),
                right_foot=(0.52, 0.90),
                left_heel=(0.47, 0.90),
                right_heel=(0.51, 0.90),
                left_ankle=(0.47, 0.88),
                right_ankle=(0.51, 0.88),
                left_knee=(0.46, 0.72),
                right_knee=(0.54, 0.72),
            ),
            _frame(
                left_hip=(0.44, 0.56),
                right_hip=(0.56, 0.56),
                left_foot=(0.68, 0.95),
                right_foot=(0.52, 0.95),
                left_heel=(0.66, 0.95),
                right_heel=(0.51, 0.95),
                left_ankle=(0.66, 0.92),
                right_ankle=(0.51, 0.92),
                left_knee=(0.64, 0.76),
                right_knee=(0.54, 0.76),
            ),
        ]

        result = compute_foot_line_deviation(
            pose_frames,
            bfc_frame=0,
            ffc_frame=1,
            fps=30.0,
            cfg={},
            action={"hand": "R"},
        )

        self.assertEqual(result["mode"], "OUTWARD_STEP")
        self.assertGreaterEqual(result["signal_strength"], 0.35)
        self.assertLess(result["signal_strength"], 0.60)


if __name__ == "__main__":
    unittest.main()
