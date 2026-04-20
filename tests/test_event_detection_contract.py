import unittest

import numpy as np

from app.workers.events.ffc_bfc import (
    _apply_ffc_timing_guard,
    _hold_frames,
    _interp_nans_limited,
    _pelvis_activity_onset,
    detect_ffc_bfc,
)
from app.workers.events.release_uah import detect_release_uah, _nb_elbow_peak_is_plausible


def _landmark(x, y, visibility=0.99):
    return {
        "x": float(x),
        "y": float(y),
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


class EventDetectionContractTest(unittest.TestCase):
    def test_ffc_bfc_helper_enforces_min_hold_of_five_frames(self):
        self.assertEqual(_hold_frames(30.0), 5)
        self.assertEqual(_hold_frames(60.0), 5)
        self.assertEqual(_hold_frames(120.0), 6)

    def test_pelvis_activity_onset_prefers_stronger_later_segment(self):
        R = np.array([0.1, 0.9, 0.8, 0.2, 0.1, 1.2, 1.1, 0.3], dtype=float)
        vis_ok = np.array([True] * len(R), dtype=bool)

        onset = _pelvis_activity_onset(
            R=R,
            vis_ok=vis_ok,
            win_start=0,
            win_end=len(R) - 1,
            threshold=0.7,
        )

        self.assertEqual(onset, 5)

    def test_limited_interp_preserves_long_occlusion_gap(self):
        series = np.array([1.0, np.nan, np.nan, np.nan, 5.0], dtype=float)

        out = _interp_nans_limited(series, max_gap=2)

        self.assertTrue(np.isnan(out[1]))
        self.assertTrue(np.isnan(out[2]))
        self.assertTrue(np.isnan(out[3]))

    def test_ffc_timing_guard_marks_overearly_contact_as_untrusted(self):
        guarded = _apply_ffc_timing_guard(
            {
                "ffc": {"frame": 444, "confidence": 0.78, "method": "pelvis_then_geometry"},
                "bfc": {"frame": 438, "confidence": 0.72},
            },
            release_frame=492,
            fps=59.93,
        )

        self.assertEqual(guarded["ffc"]["timing_flag"], "early_relative_to_release")
        self.assertEqual(guarded["ffc"]["release_gap_frames"], 48)
        self.assertLessEqual(float(guarded["ffc"]["confidence"]), 0.20)

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


if __name__ == "__main__":
    unittest.main()
