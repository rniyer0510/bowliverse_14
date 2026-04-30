import unittest

import numpy as np

from app.workers.events.ffc_bfc import (
    _ffc_search_start,
    _pick_bfc_backward_from_ffc,
    _pick_ffc_backward_from_release,
    _sanitize_bfc_frame,
)
from app.workers.events.ffc_bfc_support import (
    _adaptive_settling_frames,
    _refine_to_established_support,
)


class FfcBfcRegressionTest(unittest.TestCase):
    def test_adaptive_settling_window_shrinks_for_faster_approach(self):
        dt = 1.0 / 30.0
        hold = 3
        speed = np.linspace(0.10, 1.00, 40, dtype=float)

        slow_frames = _adaptive_settling_frames(
            speed,
            frame=8,
            hold=hold,
            start=0,
            end=30,
            dt=dt,
        )
        fast_frames = _adaptive_settling_frames(
            speed,
            frame=28,
            hold=hold,
            start=0,
            end=39,
            dt=dt,
        )

        self.assertGreater(slow_frames, fast_frames)
        self.assertGreaterEqual(slow_frames, 4)
        self.assertLessEqual(fast_frames, 3)

    def test_ffc_search_start_keeps_pre_pelvis_lead_band_when_pelvis_on_is_late(self):
        start = _ffc_search_start(
            win_start=141,
            win_end=166,
            pelvis_on=159,
            fps=29.89,
            hold=3,
        )

        self.assertLess(start, 159)
        self.assertEqual(start, 146)

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

    def test_ffc_refine_prefers_fully_established_plant_over_first_touch(self):
        n = 24
        hold = 3
        win_start = 0
        win_end = 18
        dt = 1.0 / 60.0

        y_front_ank = np.array(
            [0.70] * 10
            + [0.74, 0.82, 0.90, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92]
            + [0.92] * (n - 19),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03

        refined = _refine_to_established_support(
            y_front_ank,
            y_front_toe,
            frame=11,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            approach_speed=np.linspace(0.2, 1.0, n, dtype=float),
        )

        self.assertEqual(refined, 13)

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

        self.assertEqual(method, "back_foot_support_edge")
        self.assertEqual(frame, 9)
        self.assertGreater(confidence, 0.0)

    def test_bfc_prefers_recent_support_edge_when_close_to_support_block(self):
        n = 40
        hold = 3
        win_start = 0
        win_end = 30
        ffc = 18
        dt = 1.0 / 30.0

        y_back_ank = np.array(
            [
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.74,
                0.82,
                0.91,
                0.92,
                0.92,
                0.91,
                0.82,
                0.78,
            ]
            + [0.78] * (n - 20),
            dtype=float,
        )
        y_back_toe = y_back_ank + 0.03
        y_front_ank = np.array(
            [
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.72,
                0.73,
                0.75,
                0.78,
                0.83,
                0.89,
                0.92,
                0.93,
                0.93,
            ]
            + [0.93] * (n - 20),
            dtype=float,
        )
        y_front_toe = y_front_ank + 0.03
        pelvis_jerk = np.zeros(n, dtype=float)
        pelvis_jerk[14] = 2.0
        pelvis_jerk[15] = 1.8

        frame, confidence, method = _pick_bfc_backward_from_ffc(
            ffc=ffc,
            front_side="left",
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            fps=30.0,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
            vis_LA=np.full(n, 0.99, dtype=float),
            vis_RA=np.full(n, 0.99, dtype=float),
            vis_LFI=np.full(n, 0.99, dtype=float),
            vis_RFI=np.full(n, 0.99, dtype=float),
            approach_speed=np.linspace(0.2, 1.0, n, dtype=float),
            pelvis_jerk=pelvis_jerk,
        )

        self.assertEqual(method, "back_foot_support_edge")
        self.assertEqual(frame, 14)
        self.assertGreater(confidence, 0.0)

    def test_bfc_grounding_stays_local_to_ffc_band_even_when_release_window_runs_late(self):
        hold = 3
        dt = 1.0 / 59.93
        ffc = 13

        y_back_ank = np.array(
            [
                0.5984,
                0.5984,
                0.5984,
                0.5984,
                0.5984,
                0.5978,
                0.5984,
                0.5831,
                0.5930,
                0.5928,
                0.5980,
                0.5953,
                0.5965,
                0.5939,
                0.5929,
                0.5921,
                0.5969,
                0.5960,
                0.5995,
                0.6011,
                0.6022,
                0.6052,
            ],
            dtype=float,
        )
        y_back_toe = np.array(
            [
                0.6170,
                0.6170,
                0.6170,
                0.6170,
                0.6170,
                0.6097,
                0.6154,
                0.5959,
                0.6131,
                0.6144,
                0.6186,
                0.6149,
                0.6189,
                0.6188,
                0.6153,
                0.6195,
                0.6239,
                0.6242,
                0.6270,
                0.6306,
                0.6321,
                0.6400,
            ],
            dtype=float,
        )
        y_front_ank = np.array(
            [
                0.5707,
                0.5707,
                0.5707,
                0.5707,
                0.5707,
                0.5794,
                0.5908,
                0.6054,
                0.6009,
                0.5981,
                0.6030,
                0.6138,
                0.6245,
                0.6357,
                0.6406,
                0.6458,
                0.6467,
                0.6440,
                0.6441,
                0.6381,
                0.6356,
                0.6364,
            ],
            dtype=float,
        )
        y_front_toe = np.array(
            [
                0.5891,
                0.5891,
                0.5891,
                0.5891,
                0.5891,
                0.5752,
                0.5914,
                0.6004,
                0.6088,
                0.5907,
                0.5970,
                0.6106,
                0.6372,
                0.6441,
                0.6586,
                0.6674,
                0.6660,
                0.6645,
                0.6610,
                0.6574,
                0.6569,
                0.6529,
            ],
            dtype=float,
        )
        vis = np.full(len(y_back_ank), 0.99, dtype=float)

        frame, confidence, method = _pick_bfc_backward_from_ffc(
            ffc=ffc,
            front_side="left",
            hold=hold,
            win_start=0,
            win_end=21,
            dt=dt,
            fps=59.93,
            y_LA=y_front_ank,
            y_RA=y_back_ank,
            y_LFI=y_front_toe,
            y_RFI=y_back_toe,
            vis_LA=vis,
            vis_RA=vis,
            vis_LFI=vis,
            vis_RFI=vis,
        )

        self.assertEqual(method, "back_foot_support_edge")
        self.assertEqual(frame, 10)
        self.assertGreater(confidence, 0.0)

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

        self.assertEqual(method, "back_foot_support_edge")
        self.assertEqual(frame, 9)
        self.assertGreater(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
