import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)

# MediaPipe hips
LEFT_HIP = 23
RIGHT_HIP = 24

VIS_THR = 0.50


def compute_front_foot_braking_shock(pose_frames, ffc_frame, fps, config):
    """
    Cricket-native, baseball-style braking signal.

    Uses pelvis midpoint (LEFT_HIP/RIGHT_HIP) as a COM proxy.
    Risk is driven by forward deceleration *jerk* near FFC.
    """

    if ffc_frame is None or ffc_frame <= 0 or fps <= 0:
        return None

    pre = int(config["pre_window"])
    post = int(config["post_window"])

    start = max(0, int(ffc_frame) - pre)
    end = min(len(pose_frames) - 1, int(ffc_frame) + post)

    if end - start < 3:
        return None

    xs = []
    used_frames = 0

    for i in range(start, end + 1):
        item = pose_frames[i]
        lms = item.get("landmarks")
        if not lms:
            continue

        lh = lms[LEFT_HIP]
        rh = lms[RIGHT_HIP]
        if (
            lh.get("visibility", 0.0) < VIS_THR or
            rh.get("visibility", 0.0) < VIS_THR
        ):
            continue

        xs.append((lh["x"] + rh["x"]) / 2.0)
        used_frames += 1

    if len(xs) < 4:
        return None

    xs = np.array(xs, dtype=float)

    # velocity, acceleration, jerk (frame-time mapped via fps)
    vel = np.gradient(xs) * fps
    acc = np.gradient(vel) * fps
    jerk = np.gradient(acc) * fps

    peak_jerk = float(np.max(np.abs(jerk)))

    jerk_ref = float(config["jerk_reference"])
    signal_strength = min(peak_jerk / jerk_ref, 1.0)

    confidence = min(used_frames / max(1, (pre + post + 1)), 1.0)

    return {
        "risk_id": "front_foot_braking_shock",
        "signal_strength": round(signal_strength, 3),
        "confidence": round(confidence, 3),
        "window": {"start_frame": start, "end_frame": end},
        "debug": {
            "peak_jerk": round(peak_jerk, 3),
            "jerk_ref": jerk_ref,
            "samples": used_frames,
        }
    }
