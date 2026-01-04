"""
Event worker (V14).

Detects:
- RELEASE: peak wrist speed near delivery
- UAH: maximum elbow height above shoulder (Gaussian-smoothed)

Definitions are biomechanics-first and ICC-aligned.
"""

import math
import numpy as np
from scipy.ndimage import gaussian_filter1d

# MediaPipe landmark indices
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _mag(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def detect_events(pose_frames, hand: str):
    """
    Returns:
      {
        "uah": { "frame": int },
        "release": { "frame": int }
      }
    """

    if hand == "L":
        s_idx, e_idx, w_idx = LS, LE, LW
    else:
        s_idx, e_idx, w_idx = RS, RE, RW

    n = len(pose_frames)
    if n == 0:
        return {"uah": {"frame": 0}, "release": {"frame": 0}}

    # -------------------------------------------------
    # 1. Detect RELEASE (peak wrist speed, late phase)
    # -------------------------------------------------
    start_search = int(n * 0.60)
    start_search = max(2, start_search)

    best_speed = -1.0
    release_frame = pose_frames[-1]["frame"]

    prev_wrist = None
    for i in range(start_search, n):
        lm = pose_frames[i]["landmarks"]
        if lm is None:
            continue

        wrist = lm[w_idx]
        if prev_wrist is not None:
            v = _mag(_vec(prev_wrist, wrist))
            if v > best_speed:
                best_speed = v
                release_frame = pose_frames[i]["frame"]
        prev_wrist = wrist

    # -------------------------------------------------
    # 2. Build elbow-above-shoulder signal (delta Y)
    # -------------------------------------------------
    frames = []
    delta_y = []

    for item in pose_frames:
        f = item["frame"]
        lm = item["landmarks"]
        if lm is None:
            continue

        shoulder_y = lm[s_idx][1]
        elbow_y = lm[e_idx][1]

        frames.append(f)
        delta_y.append(shoulder_y - elbow_y)  # +ve = elbow above shoulder

    if len(delta_y) < 5:
        return {
            "uah": {"frame": max(0, release_frame - 5)},
            "release": {"frame": release_frame}
        }

    # -------------------------------------------------
    # 3. Gaussian smoothing (critical step)
    # -------------------------------------------------
    smoothed = gaussian_filter1d(np.array(delta_y), sigma=2)

    # -------------------------------------------------
    # 4. UAH = peak elbow height before release
    # -------------------------------------------------
    valid_indices = [
        i for i, f in enumerate(frames)
        if f < release_frame - 2  # ensure UAH precedes release
    ]

    if not valid_indices:
        uah_frame = max(0, release_frame - 5)
    else:
        peak_i = max(valid_indices, key=lambda i: smoothed[i])
        uah_frame = frames[peak_i]

    # Final guard: ensure temporal separation
    if uah_frame >= release_frame:
        uah_frame = max(0, release_frame - 4)

    return {
        "uah": {"frame": int(uah_frame)},
        "release": {"frame": int(release_frame)}
    }

