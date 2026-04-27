import unittest

from app.workers.events.ffc_bfc import detect_ffc_bfc
from app.workers.events.release_uah import detect_release_uah, _nb_elbow_peak_is_plausible


def _landmark(x, y, visibility=0.99):
    return {
        "x": None if x is None else float(x),
        "y": None if y is None else float(y),
        "z": 0.0,
        "visibility": float(visibility),
    }


def _blank_landmarks():
    return [_landmark(0.0, 0.0, 0.0) for _ in range(33)]


def _frame(idx: int):
    lm = _blank_landmarks()

    # hips
    lm[23] = _landmark(0.45 + 0.001 * idx, 0.63)
    lm[24] = _landmark(0.55 + 0.001 * idx, 0.63)

    # bowling side right arm
    shoulder_y = 0.42
    elbow_y = max(0.18, 0.48 - 0.0006 * idx)
    wrist_y = max(0.18, 0.56 - 0.00075 * idx)
    lm[12] = _landmark(0.56, shoulder_y)
    lm[14] = _landmark(0.63, elbow_y)
    lm[16] = _landmark(0.71 + 0.0015 * idx, wrist_y)

    # non-bowling elbow rises then falls around release
    nb_peak = 24
    if idx <= nb_peak:
        nb_y = 0.58 - (0.008 * idx)
    else:
        nb_y = 0.39 + (0.010 * (idx - nb_peak))
    lm[11] = _landmark(0.42, 0.44)
    lm[13] = _landmark(0.34, nb_y)

    # feet: right foot lands first, left foot stabilizes into ffc
    lm[27] = _landmark(0.43, 0.93)
    lm[31] = _landmark(0.46, 0.96)

    if idx < 20:
        left_ankle_y = 0.86 - 0.01 * (20 - idx)
        left_toe_y = 0.89 - 0.01 * (20 - idx)
    else:
        left_ankle_y = 0.93
        left_toe_y = 0.96
    lm[28] = _landmark(0.62, left_ankle_y)
    lm[32] = _landmark(0.65, left_toe_y)

    return {"frame": idx, "landmarks": lm}


def _frame_no_feet(idx: int):
    frame = _frame(idx)
    lm = frame["landmarks"]
    for landmark_idx in (27, 28, 31, 32):
        lm[landmark_idx] = _landmark(None, None, 0.0)
    return frame


def _frame_chaotic_feet(idx: int):
    frame = _frame(idx)
    lm = frame["landmarks"]

    # Alternate foot heights aggressively so grounding heuristics never get a
    # stable hold window, forcing the detector into its conservative fallback.
    right_cycle = (0.95, 0.84, 0.90)
    left_cycle = (0.84, 0.95, 0.89)
    right_phase = right_cycle[idx % 3]
    left_phase = left_cycle[idx % 3]
    lm[27] = _landmark(0.43, right_phase)
    lm[31] = _landmark(0.46, right_phase + 0.025)
    lm[28] = _landmark(0.62, left_phase)
    lm[32] = _landmark(0.65, left_phase + 0.025)
    return frame


class EventDetectionContractTest(unittest.TestCase):
    def test_early_non_bowling_elbow_peak_is_not_treated_as_plausible_release_anchor(self):
        self.assertFalse(
            _nb_elbow_peak_is_plausible(
                nb_elbow_peak_i=14,
                wrist_peak_i=143,
                total_frames=196,
                fps=30.0,
            )
        )
        self.assertFalse(
            _nb_elbow_peak_is_plausible(
                nb_elbow_peak_i=0,
                wrist_peak_i=276,
                total_frames=394,
                fps=30.0,
            )
        )
        self.assertTrue(
            _nb_elbow_peak_is_plausible(
                nb_elbow_peak_i=248,
                wrist_peak_i=248,
                total_frames=403,
                fps=30.0,
            )
        )

    def test_release_uah_exposes_candidates_and_window(self):
        pose_frames = [_frame(i) for i in range(40)]

        events = detect_release_uah(
            pose_frames=pose_frames,
            hand="R",
            fps=60.0,
        )

        self.assertIn("release", events)
        self.assertIn("uah", events)
        self.assertIn("delivery_window", events)
        self.assertIn("candidates", events["release"])
        self.assertTrue(events["release"]["candidates"])
        self.assertIn("window", events["release"])
        self.assertLessEqual(events["uah"]["frame"], events["release"]["frame"])

    def test_ffc_bfc_obeys_release_window(self):
        pose_frames = [_frame(i) for i in range(40)]
        release_events = detect_release_uah(
            pose_frames=pose_frames,
            hand="R",
            fps=60.0,
        )

        result = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand="R",
            release_frame=release_events["release"]["frame"],
            delivery_window=tuple(release_events["delivery_window"]),
            fps=60.0,
        )

        self.assertIn("ffc", result)
        self.assertIn("bfc", result)
        self.assertIn("candidates", result["ffc"])
        self.assertTrue(result["ffc"]["candidates"])
        self.assertLessEqual(result["bfc"]["frame"], result["ffc"]["frame"])
        self.assertLessEqual(result["ffc"]["frame"], release_events["release"]["frame"])
        self.assertEqual(result["ffc"]["method"], "release_backward_chain_grounding")
        self.assertIn(result["bfc"]["method"], {"simple_grounded_bfc", "context_pre_ffc", "no_ground_confirmed"})

    def test_ffc_bfc_returns_empty_for_too_few_frames(self):
        result = detect_ffc_bfc(
            pose_frames=[_frame(i) for i in range(8)],
            hand="R",
            release_frame=5,
            delivery_window=(0, 7),
            fps=60.0,
        )

        self.assertEqual(result, {})

    def test_ffc_bfc_returns_empty_when_release_is_missing(self):
        result = detect_ffc_bfc(
            pose_frames=[_frame(i) for i in range(40)],
            hand="R",
            release_frame=None,
            delivery_window=(0, 39),
            fps=60.0,
        )

        self.assertEqual(result, {})

    def test_ffc_bfc_uses_no_foot_data_fallback_when_feet_are_occluded(self):
        pose_frames = [_frame_no_feet(i) for i in range(40)]
        release_events = detect_release_uah(
            pose_frames=pose_frames,
            hand="R",
            fps=60.0,
        )

        result = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand="R",
            release_frame=release_events["release"]["frame"],
            delivery_window=tuple(release_events["delivery_window"]),
            fps=60.0,
        )

        self.assertEqual(result["ffc"]["method"], "no_foot_data_fallback")
        self.assertEqual(result["bfc"]["method"], "no_foot_data_fallback")
        self.assertLessEqual(result["bfc"]["frame"], result["ffc"]["frame"])
        self.assertLessEqual(result["ffc"]["frame"], release_events["release"]["frame"])

    def test_ffc_bfc_uses_ultimate_fallback_when_grounding_is_never_stable(self):
        pose_frames = [_frame_chaotic_feet(i) for i in range(40)]
        release_events = detect_release_uah(
            pose_frames=pose_frames,
            hand="R",
            fps=60.0,
        )

        result = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand="R",
            release_frame=release_events["release"]["frame"],
            delivery_window=tuple(release_events["delivery_window"]),
            fps=60.0,
        )

        self.assertEqual(result["ffc"]["method"], "ultimate_fallback")
        self.assertEqual(result["bfc"]["method"], "ultimate_fallback")
        self.assertIn("candidates", result["ffc"])
        self.assertTrue(result["ffc"]["candidates"])
        self.assertLessEqual(result["ffc"]["frame"], release_events["release"]["frame"])

    def test_ffc_bfc_keeps_ordering_on_lower_fps(self):
        pose_frames = [_frame(i) for i in range(40)]
        release_events = detect_release_uah(
            pose_frames=pose_frames,
            hand="R",
            fps=24.0,
        )

        result = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand="R",
            release_frame=release_events["release"]["frame"],
            delivery_window=tuple(release_events["delivery_window"]),
            fps=24.0,
        )

        self.assertIn(result["ffc"]["method"], {"release_backward_chain_grounding", "single_foot_fallback", "ultimate_fallback"})
        self.assertLessEqual(result["bfc"]["frame"], result["ffc"]["frame"])
        self.assertLessEqual(result["ffc"]["frame"], release_events["release"]["frame"])
        self.assertGreater(result["ffc"]["confidence"], 0.0)

    def test_single_foot_fallback_prefers_latest_plausible_frame(self):
        pose_frames = [_frame(i) for i in range(40)]

        result = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand="R",
            release_frame=8,
            delivery_window=(0, 39),
            fps=24.0,
        )

        self.assertEqual(result["ffc"]["method"], "single_foot_fallback")
        self.assertEqual(result["ffc"]["frame"], 3)
        self.assertLessEqual(result["bfc"]["frame"], result["ffc"]["frame"])


if __name__ == "__main__":
    unittest.main()
