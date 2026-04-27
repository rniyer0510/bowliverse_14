import unittest

from app.workers.elbow.compute_elbow_signal import compute_elbow_signal
from app.workers.elbow.elbow_legality import THRESH_LEGAL, evaluate_elbow_legality
from app.workers.events.ffc_bfc import detect_ffc_bfc
from app.workers.events.release_uah import detect_release_uah
from app.workers.speed.release_speed import estimate_release_speed


def _landmark(x, y, visibility=0.99):
    return {
        "x": None if x is None else float(x),
        "y": None if y is None else float(y),
        "z": 0.0,
        "visibility": float(visibility),
    }


def _blank_landmarks():
    return [_landmark(0.0, 0.0, 0.0) for _ in range(33)]


def _reference_frame(idx: int):
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
    lm[18] = _landmark(0.73 + 0.0014 * idx, wrist_y + 0.01)
    lm[20] = _landmark(0.75 + 0.0013 * idx, wrist_y + 0.005)
    lm[22] = _landmark(0.72 + 0.0012 * idx, wrist_y + 0.015)

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


class ReferenceDeliveryRegressionTest(unittest.TestCase):
    def test_reference_chain_preserves_events_elbow_and_speed(self):
        pose_frames = [_reference_frame(i) for i in range(40)]

        release_events = detect_release_uah(
            pose_frames=pose_frames,
            hand="R",
            fps=60.0,
        )
        foot_events = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand="R",
            release_frame=release_events["release"]["frame"],
            delivery_window=tuple(release_events["delivery_window"]),
            fps=60.0,
        )
        events = {**release_events, **foot_events}

        elbow_signal = compute_elbow_signal(pose_frames=pose_frames, hand="R")
        elbow = evaluate_elbow_legality(
            elbow_signal=elbow_signal,
            events=events,
            fps=60.0,
            pose_frames=pose_frames,
            hand="R",
        )
        speed = estimate_release_speed(
            pose_frames=pose_frames,
            events=events,
            video={"fps": 60.0, "width": 360, "height": 640},
            hand="R",
        )

        self.assertEqual(release_events["release"]["frame"], 12)
        self.assertEqual(release_events["uah"]["frame"], 11)
        self.assertEqual(foot_events["ffc"]["frame"], 7)
        self.assertEqual(foot_events["bfc"]["frame"], 6)
        self.assertTrue((foot_events.get("event_chain") or {}).get("ordered"))

        self.assertEqual(elbow["verdict"], "LEGAL")
        self.assertLess(elbow["extension_deg"], THRESH_LEGAL)
        self.assertEqual((elbow.get("debug") or {}).get("window_mode"), "release_anchored")

        self.assertTrue(speed["available"])
        self.assertEqual(speed["display_policy"], "show")
        self.assertGreaterEqual(speed["value_kph"], 65)
        self.assertLessEqual(speed["value_kph"], 85)
        self.assertGreaterEqual(speed["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
