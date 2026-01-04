# app/workers/risk/knee_brace_failure.py

import math
import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)


def _angle_deg(a, b, c):
    """
    Angle at point b formed by a-b-c
    """
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosang = np.clip(cosang, -1.0, 1.0)
        return math.degrees(math.acos(cosang))
    except Exception:
        return None


def compute_knee_brace_failure(
    pose_frames,
    ffc_frame: int,
    fps: float,
    config: dict,
):
    """
    Knee brace failure = collapse (flexion) of the FRONT knee after FFC.

    Front leg is inferred geometrically:
    - ankle closer to ground at FFC (or nearest reliable frame)

    No handedness, no pace/spin.
    """

    if ffc_frame is None or ffc_frame < 0:
        return {
            "risk_id": "knee_brace_failure",
            "signal_strength": 0.0,
            "confidence": 0.0,
            "note": "FFC not available",
        }

    post_window = int(config.get("post_window", 12))
    min_vis = float(config.get("min_visibility", 0.5))

    # -------------------------------------------------
    # Find best knee frame near FFC (robust to occlusion)
    # -------------------------------------------------
    search_offsets = [-2, -1, 0, 1, 2, 3, 4]
    best_frame = None
    best_vis = 0.0

    for off in search_offsets:
        idx = ffc_frame + off
        if idx < 0 or idx >= len(pose_frames):
            continue

        lm = pose_frames[idx].get("landmarks", {})
        required = [
            "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
            "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE",
        ]
        if not all(k in lm for k in required):
            continue

        vis = min(
            lm["LEFT_HIP"]["v"], lm["LEFT_KNEE"]["v"], lm["LEFT_ANKLE"]["v"],
            lm["RIGHT_HIP"]["v"], lm["RIGHT_KNEE"]["v"], lm["RIGHT_ANKLE"]["v"],
        )

        if vis > best_vis:
            best_vis = vis
            best_frame = idx

    if best_frame is None:
        return {
            "risk_id": "knee_brace_failure",
            "signal_strength": 0.0,
            "confidence": 0.0,
            "note": "Knee landmarks occluded around FFC",
        }

    lm0 = pose_frames[best_frame]["landmarks"]

    # -------------------------------------------------
    # Determine front leg by ankle height (lower = front)
    # -------------------------------------------------
    left_ankle_y = lm0["LEFT_ANKLE"]["y"]
    right_ankle_y = lm0["RIGHT_ANKLE"]["y"]

    if left_ankle_y > right_ankle_y:
        hip_key, knee_key, ankle_key = "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"
    else:
        hip_key, knee_key, ankle_key = "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"

    # -------------------------------------------------
    # Collect knee angles post-FFC
    # -------------------------------------------------
    angles = []
    vis_list = []

    end_idx = min(len(pose_frames), best_frame + post_window)

    for i in range(best_frame, end_idx):
        lm = pose_frames[i].get("landmarks", {})
        if hip_key not in lm or knee_key not in lm or ankle_key not in lm:
            continue

        hip = lm[hip_key]
        knee = lm[knee_key]
        ankle = lm[ankle_key]

        if min(hip.get("v", 0), knee.get("v", 0), ankle.get("v", 0)) < min_vis:
            continue

        angle = _angle_deg(
            (hip["x"], hip["y"]),
            (knee["x"], knee["y"]),
            (ankle["x"], ankle["y"]),
        )

        if angle is not None:
            angles.append(angle)
            vis_list.append(min(hip["v"], knee["v"], ankle["v"]))

    if len(angles) < 3:
        return {
            "risk_id": "knee_brace_failure",
            "signal_strength": 0.0,
            "confidence": 0.0,
            "note": "Insufficient knee samples",
        }

    # -------------------------------------------------
    # Collapse = flexion after FFC
    # -------------------------------------------------
    angle_at_ffc = angles[0]
    min_angle = min(angles)

    collapse_deg = max(0.0, angle_at_ffc - min_angle)

    # Normalise: 0–15° → 0–1
    signal = min(1.0, collapse_deg / 15.0)
    confidence = float(np.mean(vis_list))

    return {
        "risk_id": "knee_brace_failure",
        "signal_strength": round(signal, 3),
        "confidence": round(confidence, 3),
        "debug": {
            "collapse_deg": round(collapse_deg, 2),
            "front_leg": knee_key,
            "ffc_frame_used": best_frame,
        },
    }

