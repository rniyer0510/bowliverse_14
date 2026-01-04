# app/common/timebase.py

import numpy as np
from app.utils.angles import angle, angle_between


# ---------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------

def _get_side_keys(hand: str):
    if hand.upper() == "R":
        return {
            "SHOULDER": "RIGHT_SHOULDER",
            "ELBOW": "RIGHT_ELBOW",
            "WRIST": "RIGHT_WRIST",
            "HIP": "RIGHT_HIP",
        }
    else:
        return {
            "SHOULDER": "LEFT_SHOULDER",
            "ELBOW": "LEFT_ELBOW",
            "WRIST": "LEFT_WRIST",
            "HIP": "LEFT_HIP",
        }


def _vec(a, b):
    return np.array([
        a["x"] - b["x"],
        a["y"] - b["y"],
        a.get("z", 0.0) - b.get("z", 0.0),
    ])


# ---------------------------------------------------------
# RELEASE (reverse-first)
# ---------------------------------------------------------

def detect_release_frame(pose_frames, hand: str):
    """
    Detect RELEASE frame first (reverse philosophy).

    Strategy:
    - scan forward
    - find max elbow angle
    - stop when upper arm starts descending
    """

    K = _get_side_keys(hand)

    max_angle = -1.0
    release_frame = None
    prev_upper_arm_angle = None

    for f in pose_frames:
        lm = f.get("landmarks", {})
        if not all(k in lm for k in K.values()):
            continue

        shoulder = lm[K["SHOULDER"]]
        elbow = lm[K["ELBOW"]]
        wrist = lm[K["WRIST"]]
        hip = lm[K["HIP"]]

        # elbow inner angle
        elbow_angle = angle(
            np.array([shoulder["x"], shoulder["y"], shoulder.get("z", 0.0)]),
            np.array([elbow["x"], elbow["y"], elbow.get("z", 0.0)]),
            np.array([wrist["x"], wrist["y"], wrist.get("z", 0.0)]),
        )

        # upper-arm vs torso angle
        upper_arm = _vec(elbow, shoulder)
        torso = _vec(shoulder, hip)
        ua_angle = angle_between(upper_arm, torso)

        # stop when arm begins to descend
        if prev_upper_arm_angle is not None:
            if ua_angle > prev_upper_arm_angle:
                break

        prev_upper_arm_angle = ua_angle

        if elbow_angle > max_angle:
            max_angle = elbow_angle
            release_frame = f["frame"]

    if release_frame is None:
        return None

    return {
        "frame": release_frame,
        "angle_deg": round(max_angle, 2),
    }


# ---------------------------------------------------------
# UAH (must be BEFORE release)
# ---------------------------------------------------------

def detect_uah_frame(pose_frames, hand: str, release_frame: int):
    """
    Detect UAH strictly BEFORE release.
    """

    K = _get_side_keys(hand)

    min_ua_angle = 1e9
    uah_frame = None

    for f in pose_frames:
        if f["frame"] >= release_frame:
            break

        lm = f.get("landmarks", {})
        if not all(k in lm for k in K.values()):
            continue

        shoulder = lm[K["SHOULDER"]]
        elbow = lm[K["ELBOW"]]
        hip = lm[K["HIP"]]

        upper_arm = _vec(elbow, shoulder)
        torso = _vec(shoulder, hip)

        ua_angle = angle_between(upper_arm, torso)

        if ua_angle < min_ua_angle:
            min_ua_angle = ua_angle
            uah_frame = f["frame"]

    if uah_frame is None:
        return None

    return {
        "frame": uah_frame,
        "angle_deg": round(min_ua_angle, 2),
    }


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def extract_events(pose_frames, hand: str):
    """
    Reverse-time event extraction:
    Release → UAH → (later FFC → BFC)
    """

    events = {}

    release = detect_release_frame(pose_frames, hand)
    if release:
        events["release"] = release

        uah = detect_uah_frame(
            pose_frames,
            hand,
            release_frame=release["frame"],
        )
        if uah:
            events["uah"] = uah

    return events

