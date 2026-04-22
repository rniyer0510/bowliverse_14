import unittest

import numpy as np

from app.workers.events.ffc_bfc import (
    _pick_bfc_backward_from_ffc,
    _pick_ffc_backward_from_release,
    _sanitize_bfc_frame,
)


class FfcBfcRegressionTest(unittest.TestCase):
    def test_ffc_search_can_reach_valid_contact_before_late_pelvis_on(self):
        n = 24
        hold = 3
        win_start = 0
        win_end = 18
        pelvis_on = 16
        search_start = 11
        dt = 1.0 / 25.0

        y_front_ank = np.array(
            [0.70] * 10
            + [0.76, 0.84, 0.90, 0.91, 0.79, 0.77, 0.76, 0.75, 0.75]
            + [0.75] * (n - 19),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03
        y_back_ank = np.array(
            [0.74] * 10
            + [0.88, 0.91, 0.92, 0.80, 0.76, 0.74, 0.73, 0.72, 0.71]
            + [0.71] * (n - 19),
            dtype=float,
        )
        y_back_toe = y_back_ank + 0.03

        frame, front_side, candidates, confidence = _pick_ffc_backward_from_release(
            search_start=search_start,
            pelvis_on=pelvis_on,
            preferred_front_side="left",
            win_end=win_end,
            hold=hold,
            win_start=win_start,
            dt=dt,
            back_recent=2,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
        )

        self.assertEqual(front_side, "left")
        self.assertIsNotNone(frame)
        self.assertLess(frame, pelvis_on)
        self.assertGreaterEqual(frame, search_start)
        self.assertTrue(candidates)
        self.assertGreater(confidence, 0.0)

    def test_bfc_prefers_latest_stable_support_before_ffc_when_visible(self):
        n = 30
        hold = 3
        win_start = 0
        win_end = 20
        ffc = 12
        dt = 1.0 / 25.0

        # Right side is the back foot when front_side == "left".
        # Several frames are plausibly grounded, but the selector should pick
        # the latest stable support before the front-foot transfer.
        y_back_ank = np.array(
            [
                0.70,
                0.71,
                0.72,
                0.88,
                0.89,
                0.90,
                0.91,
                0.915,
                0.92,
                0.92,
                0.92,
                0.92,
                0.84,
                0.80,
                0.78,
            ]
            + [0.78] * (n - 15),
            dtype=float,
        )
        y_back_toe = y_back_ank + 0.03

        # Front foot settles later so the best transfer frame sits close to FFC.
        y_front_ank = np.array(
            [
                0.74,
                0.75,
                0.76,
                0.77,
                0.78,
                0.79,
                0.80,
                0.84,
                0.88,
                0.91,
                0.93,
                0.93,
                0.93,
                0.93,
                0.93,
            ]
            + [0.93] * (n - 15),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03

        frame, confidence, method = _pick_bfc_backward_from_ffc(
            ffc=ffc,
            front_side="left",
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            fps=25.0,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
            vis_LA=np.full(n, 0.99, dtype=float),
            vis_RA=np.full(n, 0.99, dtype=float),
            vis_LFI=np.full(n, 0.99, dtype=float),
            vis_RFI=np.full(n, 0.99, dtype=float),
            approach_speed=np.linspace(0.2, 1.0, n, dtype=float),
        )

        self.assertEqual(method, "simple_grounded_bfc")
        self.assertEqual(frame, 10)
        self.assertEqual(confidence, 0.0)

    def test_bfc_correction_rejects_frame_when_front_foot_is_already_grounded(self):
        n = 30
        hold = 3
        win_start = 0
        win_end = 20
        ffc = 12
        dt = 1.0 / 25.0

        y_back_ank = np.array(
            [
                0.70,
                0.71,
                0.72,
                0.88,
                0.89,
                0.90,
                0.91,
                0.915,
                0.92,
                0.92,
                0.92,
                0.92,
                0.84,
                0.80,
                0.78,
            ]
            + [0.78] * (n - 15),
            dtype=float,
        )
        y_back_toe = y_back_ank + 0.03
        y_front_ank = np.array(
            [
                0.74,
                0.75,
                0.76,
                0.77,
                0.78,
                0.79,
                0.80,
                0.84,
                0.88,
                0.91,
                0.93,
                0.93,
                0.93,
                0.93,
                0.93,
            ]
            + [0.93] * (n - 15),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03

        corrected, changed = _sanitize_bfc_frame(
            bfc=10,
            ffc=ffc,
            front_side="left",
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
        )

        self.assertTrue(changed)
        self.assertEqual(corrected, 7)

    def test_bfc_uses_runup_speed_prior_when_back_foot_is_occluded(self):
        n = 30
        hold = 3
        win_start = 0
        win_end = 20
        ffc = 12
        dt = 1.0 / 25.0

        y_back_ank = np.array(
            [
                0.70,
                0.71,
                0.72,
                0.88,
                0.89,
                0.90,
                0.91,
                0.915,
                0.92,
                0.92,
                0.92,
                0.92,
                0.84,
                0.80,
                0.78,
            ]
            + [0.78] * (n - 15),
            dtype=float,
        )
        y_back_toe = y_back_ank + 0.03
        y_front_ank = np.array(
            [
                0.74,
                0.75,
                0.76,
                0.77,
                0.78,
                0.79,
                0.80,
                0.84,
                0.88,
                0.91,
                0.93,
                0.93,
                0.93,
                0.93,
                0.93,
            ]
            + [0.93] * (n - 15),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03

        frame, confidence, method = _pick_bfc_backward_from_ffc(
            ffc=ffc,
            front_side="left",
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            fps=25.0,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
            vis_LA=np.full(n, 0.99, dtype=float),
            vis_RA=np.full(n, 0.2, dtype=float),
            vis_LFI=np.full(n, 0.99, dtype=float),
            vis_RFI=np.full(n, 0.2, dtype=float),
            approach_speed=np.linspace(0.2, 1.0, n, dtype=float),
        )

        self.assertEqual(method, "simple_grounded_bfc")
        self.assertEqual(frame, 10)
        self.assertEqual(confidence, 0.0)

    def test_bfc_uses_runup_speed_prior_when_visibility_is_moderate_and_direct_pick_is_early(self):
        n = 30
        hold = 3
        win_start = 0
        win_end = 20
        ffc = 12
        dt = 1.0 / 25.0

        y_back_ank = np.array(
            [
                0.70,
                0.71,
                0.72,
                0.88,
                0.89,
                0.90,
                0.91,
                0.915,
                0.86,
                0.92,
                0.86,
                0.92,
                0.84,
                0.80,
                0.78,
            ]
            + [0.78] * (n - 15),
            dtype=float,
        )
        y_back_toe = y_back_ank + 0.03
        y_front_ank = np.array(
            [
                0.74,
                0.75,
                0.76,
                0.77,
                0.78,
                0.79,
                0.80,
                0.84,
                0.88,
                0.91,
                0.93,
                0.93,
                0.93,
                0.93,
                0.93,
            ]
            + [0.93] * (n - 15),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03

        frame, confidence, method = _pick_bfc_backward_from_ffc(
            ffc=ffc,
            front_side="left",
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            fps=25.0,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
            vis_LA=np.full(n, 0.99, dtype=float),
            vis_RA=np.full(n, 0.55, dtype=float),
            vis_LFI=np.full(n, 0.99, dtype=float),
            vis_RFI=np.full(n, 0.55, dtype=float),
            approach_speed=np.linspace(0.2, 1.0, n, dtype=float),
        )

        self.assertEqual(method, "simple_grounded_bfc")
        self.assertEqual(frame, 9)
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
