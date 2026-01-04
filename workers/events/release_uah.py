"""
Release + UAH detector for ActionLab V14.

Principles:
- Reverse detection: Release → UAH
- Camera-agnostic (pure pose geometry)
- Deterministic, conservative logic

Fixes:
- Release prefers last "arm-up" frame (wrist above shoulder) to avoid follow-through drift.
- UAH search is bounded to a lookback window before Release to avoid early-runup minima.
"""

from typing import Dict, List, Any, Optional
import math

# MediaPipe Pose indices
POSE = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
}

# Config-like constants (centralized here; no magic numbers inside loops)
VIS_THR = 0.60
UAH_LOOKBACK_SEC = 0.80   # bounded search window before release
MAX_ARM_TILT_DEG = 90.0   # discard anomalous angles


def _angle(v1, v2) -> Optional[float]:
    """Angle between two vectors in degrees."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return None
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def detect_release_uah(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: Optional[float] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Detect Release and UAH frames.

    Returns:
        {
          "release": {"frame": int},
          "uah": {"frame": int}
        }
    """

    if not pose_frames:
        return {}

    # Select bowling arm
    if hand.upper() == "R":
        S, E, W = POSE["RIGHT_SHOULDER"], POSE["RIGHT_ELBOW"], POSE["RIGHT_WRIST"]
    else:
        S, E, W = POSE["LEFT_SHOULDER"], POSE["LEFT_ELBOW"], POSE["LEFT_WRIST"]

    # -------------------------------------------------
    # Step 1: RELEASE (reverse scan)
    # Prefer last frame where wrist is still above shoulder (arm-up zone),
    # to avoid drifting into follow-through.
    # -------------------------------------------------
    release_frame = None

    # Pass A: arm-up preference
    for item in reversed(pose_frames):
        lms = item.get("landmarks")
        if not lms:
            continue

        if (
            lms[E]["visibility"] >= VIS_THR and
            lms[W]["visibility"] >= VIS_THR and
            lms[S]["visibility"] >= VIS_THR
        ):
            # y axis: smaller y is higher on screen
            if lms[W]["y"] < lms[S]["y"]:
                release_frame = item["frame"]
                break

    # Pass B: fallback to last stable visibility
    if release_frame is None:
        for item in reversed(pose_frames):
            lms = item.get("landmarks")
            if not lms:
                continue
            if lms[E]["visibility"] >= VIS_THR and lms[W]["visibility"] >= VIS_THR:
                release_frame = item["frame"]
                break

    if release_frame is None:
        return {}

    # -------------------------------------------------
    # Step 2: UAH (bounded search backward from Release)
    # UAH = minimum upper-arm tilt in the lookback window.
    # -------------------------------------------------
    if fps and fps > 0:
        lookback = max(10, int(round(UAH_LOOKBACK_SEC * fps)))
    else:
        # safe fallback if fps not provided
        lookback = 60

    start_frame = max(0, release_frame - lookback)
    end_frame = release_frame - 1

    min_angle = None
    uah_frame = None

    for item in pose_frames:
        f = item.get("frame")
        if f is None:
            continue
        if f < start_frame:
            continue
        if f > end_frame:
            break

        lms = item.get("landmarks")
        if not lms:
            continue

        if (
            lms[S]["visibility"] < VIS_THR or
            lms[E]["visibility"] < VIS_THR
        ):
            continue

        # Upper-arm vector (shoulder → elbow)
        ux = lms[E]["x"] - lms[S]["x"]
        uy = lms[E]["y"] - lms[S]["y"]

        # Vertical reference (image space)
        vertical = (0.0, -1.0)

        ang = _angle((ux, uy), vertical)
        if ang is None:
            continue

        # Discard anomalies
        if ang > MAX_ARM_TILT_DEG:
            continue

        if min_angle is None or ang < min_angle:
            min_angle = ang
            uah_frame = f

    if uah_frame is None:
        # If bounded search fails (occlusion), fallback to closest valid frame before Release
        uah_frame = max(0, release_frame - 1)

    return {
        "uah": {"frame": int(uah_frame)},
        "release": {"frame": int(release_frame)},
    }
