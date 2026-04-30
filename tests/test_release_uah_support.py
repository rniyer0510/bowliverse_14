import unittest

import numpy as np

from app.workers.events.release_uah_support import release_consensus


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

