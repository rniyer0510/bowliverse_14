import unittest

import numpy as np

from app.workers.events.release_uah_support import _late_release_cap, release_consensus


class ReleaseConsensusTests(unittest.TestCase):
    def test_pelvis_jerk_is_used_as_gate_not_early_locator(self):
        n = 12
        vis = np.ones(n, dtype=float)
        weights = np.ones(n, dtype=float)

        nb_elbow = np.full(n, 0.9, dtype=float)
        nb_elbow[7] = 0.1

        shoulder = np.zeros(n, dtype=float)
        shoulder[7] = 1.0

        wrist = np.zeros(n, dtype=float)
        wrist[7] = 1.0

        pelvis_jerk = np.zeros(n, dtype=float)
        pelvis_jerk[3] = 1.0

        signals = {
            "nb_elbow_y": nb_elbow,
            "nb_elbow_vis_raw": vis,
            "nb_elbow_vis_weight": weights,
            "shoulder_ang_vel": shoulder,
            "shoulder_vis_raw": vis,
            "shoulder_vis_weight": weights,
            "wrist_fwd_vel": wrist,
            "wrist_vis_raw": vis,
            "wrist_vis_weight": weights,
            "pelvis_jerk": pelvis_jerk,
            "pelvis_vis_raw": vis,
            "pelvis_vis_weight": weights,
            "pelvis_fwd_vel": np.full(n, 0.05, dtype=float),
            "wrist_height_rel": np.zeros(n, dtype=float),
        }

        frame, confidence, used = release_consensus(
            signals,
            0,
            n,
            {"vote_sigma": 1.0, "post_release_rel": 0.06},
        )

        self.assertEqual(frame, 7)
        self.assertGreater(confidence, 0.0)
        self.assertFalse(any(item.get("name") == "pelvis_jerk" and item.get("used") for item in used))
        self.assertTrue(any(item.get("name") == "pelvis_jerk_gate" and item.get("used") for item in used))

    def test_late_release_cap_prefers_earlier_wrist_shoulder_cluster_when_elbow_trails(self):
        n = 40
        vis = np.ones(n, dtype=float)
        weights = np.ones(n, dtype=float)

        nb_elbow = np.full(n, 1.0, dtype=float)
        nb_elbow[30] = 0.0

        shoulder = np.zeros(n, dtype=float)
        shoulder[14] = 1.0

        wrist = np.zeros(n, dtype=float)
        wrist[12] = 1.0

        signals = {
            "nb_elbow_y": nb_elbow,
            "nb_elbow_vis_raw": vis,
            "nb_elbow_vis_weight": weights,
            "shoulder_ang_vel": shoulder,
            "wrist_fwd_vel": wrist,
        }

        end = _late_release_cap(
            signals,
            0,
            n - 1,
            {"fps": 120.0},
        )

        self.assertEqual(end, 23)

    def test_late_release_cap_keeps_elbow_tail_when_signals_agree(self):
        n = 40
        vis = np.ones(n, dtype=float)
        weights = np.ones(n, dtype=float)

        nb_elbow = np.full(n, 1.0, dtype=float)
        nb_elbow[18] = 0.0

        shoulder = np.zeros(n, dtype=float)
        shoulder[17] = 1.0

        wrist = np.zeros(n, dtype=float)
        wrist[16] = 1.0

        signals = {
            "nb_elbow_y": nb_elbow,
            "nb_elbow_vis_raw": vis,
            "nb_elbow_vis_weight": weights,
            "shoulder_ang_vel": shoulder,
            "wrist_fwd_vel": wrist,
        }

        end = _late_release_cap(
            signals,
            0,
            n - 1,
            {"fps": 120.0},
        )

        self.assertEqual(end, 39)

    def test_late_release_cap_ignores_low_visibility_late_wrist_spike(self):
        n = 40
        nb_vis = np.ones(n, dtype=float)
        weights = np.ones(n, dtype=float)
        wrist_vis = np.ones(n, dtype=float)
        shoulder_vis = np.ones(n, dtype=float)

        nb_elbow = np.full(n, 1.0, dtype=float)
        nb_elbow[30] = 0.0

        shoulder = np.zeros(n, dtype=float)
        shoulder[14] = 1.0

        wrist = np.zeros(n, dtype=float)
        wrist[12] = 1.0
        wrist[30] = 3.0
        wrist_vis[30] = 0.05

        signals = {
            "nb_elbow_y": nb_elbow,
            "nb_elbow_vis_raw": nb_vis,
            "nb_elbow_vis_weight": weights,
            "shoulder_ang_vel": shoulder,
            "shoulder_vis_raw": shoulder_vis,
            "wrist_fwd_vel": wrist,
            "wrist_vis_raw": wrist_vis,
        }

        end = _late_release_cap(
            signals,
            0,
            n - 1,
            {"fps": 120.0},
        )

        self.assertEqual(end, 23)
