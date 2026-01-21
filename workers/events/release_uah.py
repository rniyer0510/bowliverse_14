"""
ActionLab V14 — Release + UAH detector (clean rewrite, physics-correct)

LOCKED PRINCIPLES:
- Peak wrist forward velocity is the ONLY temporal anchor
- Release occurs shortly AFTER peak, during torque transfer cessation
- Wrist velocity does NOT need to dampen to zero
- Frames where wrist is below shoulder are POST-release and discarded
- UAH is geometric and must occur BEFORE Release
- Follow-through is structurally unreachable by design

Contracts preserved:
- detect_release_uah(...)
- returns {"release":{"frame":...}, "uah":{"frame":...}}
"""

from typing import Any, Dict, List, Optional
import math
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.common.logger import get_logger

logger = get_logger(__name__)

# Pose landmarks
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24

MIN_VIS = 0.25
DEBUG = True

DEBUG_DIR = "/tmp/actionlab_debug/release_uah"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _xyz(pt):
    try:
        return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
    except Exception:
        return None

def _mag(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _unit(v):
    m = _mag(v) or 1.0
    return (v[0]/m, v[1]/m, v[2]/m)

def _dump(video_path, idx, tag):
    if not DEBUG or not video_path or idx is None or idx < 0:
        return
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{tag}_{idx}.png"), frame)

def _upper_arm_angle(s, e, forward):
    if not (s and e):
        return None
    v = (e[0]-s[0], e[1]-s[1], e[2]-s[2])
    m = _mag(v)
    if m < 1e-6:
        return None
    u = (v[0]/m, v[1]/m, v[2]/m)
    d = max(-1.0, min(1.0, _dot(u, forward)))
    return math.degrees(math.acos(d))

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def detect_release_uah(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
    video_path: Optional[str] = None
) -> Dict[str, Any]:

    fps = float(fps or 25.0)
    dt = 1.0 / fps
    h = (hand or "R").upper()

    s_idx, e_idx, w_idx = (RS, RE, RW) if h == "R" else (LS, LE, LW)

    frames = []
    wrist_xyz, shoulder_xyz, elbow_xyz, pelvis_xyz = [], [], [], []

    # ------------------------------------------------------------
    # Collect landmarks
    # ------------------------------------------------------------
    for it in pose_frames:
        frames.append(int(it.get("frame", len(frames))))
        lm = it.get("landmarks") or []

        def _vis(i):
            if i >= len(lm) or lm[i].get("visibility", 0) < MIN_VIS:
                return None
            return lm[i]

        w = _vis(w_idx)
        s = _vis(s_idx)
        e = _vis(e_idx)
        hl = _vis(LH)
        hr = _vis(RH)

        wrist_xyz.append(_xyz(w) if w else None)
        shoulder_xyz.append(_xyz(s) if s else None)
        elbow_xyz.append(_xyz(e) if e else None)

        if hl and hr:
            pelvis_xyz.append(((hl["x"]+hr["x"])/2, (hl["y"]+hr["y"])/2, 0))
        else:
            pelvis_xyz.append(None)

    n = len(frames)
    if n < 10:
        logger.error("[Release/UAH] Too few frames")
        return {}

    # ------------------------------------------------------------
    # Forward direction from pelvis drift
    # ------------------------------------------------------------
    vels = []
    for i in range(1, n):
        if pelvis_xyz[i] and pelvis_xyz[i-1]:
            vels.append((
                pelvis_xyz[i][0]-pelvis_xyz[i-1][0],
                pelvis_xyz[i][1]-pelvis_xyz[i-1][1],
                0.0
            ))
    forward = _unit(tuple(map(sum, zip(*vels)))) if vels else (1, 0, 0)

    # ------------------------------------------------------------
    # Wrist forward velocity (smoothed)
    # ------------------------------------------------------------
    wrist_fwd = np.zeros(n, dtype=float)
    for i in range(1, n):
        if wrist_xyz[i] and wrist_xyz[i-1]:
            dv = (
                wrist_xyz[i][0]-wrist_xyz[i-1][0],
                wrist_xyz[i][1]-wrist_xyz[i-1][1],
                wrist_xyz[i][2]-wrist_xyz[i-1][2],
            )
            wrist_fwd[i] = _dot(dv, forward) / dt

    wrist_fwd_s = gaussian_filter1d(wrist_fwd, sigma=max(1.0, 0.03 * fps))
    peak_i = int(np.argmax(wrist_fwd_s))
    vmax = float(max(1e-6, np.max(np.abs(wrist_fwd_s))))

    # ------------------------------------------------------------
    # Wrist-above-shoulder HARD FILTER
    # ------------------------------------------------------------
    def wrist_above_shoulder(i):
        w, s = wrist_xyz[i], shoulder_xyz[i]
        if not (w and s):
            return False
        return w[1] <= s[1] + 0.04  # small tolerance

    # ------------------------------------------------------------
    # RELEASE WINDOW (POST-PEAK, VERY NARROW)
    # ------------------------------------------------------------
    # At 25fps → ~3 frames
    # At 60fps → ~6 frames
    min_after_peak = int(max(2, round(0.05 * fps)))
    max_after_peak = int(max(4, round(0.10 * fps)))

    start = min(n - 1, peak_i + min_after_peak)
    end   = min(n - 1, peak_i + max_after_peak)

    # ------------------------------------------------------------
    # RELEASE SELECTION
    #
    # Release = earliest frame in window where:
    # - wrist velocity stops increasing (local max / plateau)
    # - wrist is still ABOVE shoulder
    # ------------------------------------------------------------
    release_i = None
    for i in range(start, end + 1):
        if not wrist_above_shoulder(i):
            continue
        if wrist_fwd_s[i] <= wrist_fwd_s[i - 1]:
            release_i = i
            break

    if release_i is None:
        # Deterministic fallback: first valid frame after peak
        for i in range(start, end + 1):
            if wrist_above_shoulder(i):
                release_i = i
                break

    if release_i is None:
        # Absolute fallback (should never happen)
        release_i = start

    # ------------------------------------------------------------
    # UAH (geometric, BEFORE release)
    # ------------------------------------------------------------
    theta = np.zeros(n, dtype=float)
    for i in range(n):
        ang = _upper_arm_angle(shoulder_xyz[i], elbow_xyz[i], forward)
        theta[i] = float(ang) if ang is not None else 0.0
    theta_s = gaussian_filter1d(theta, sigma=max(1.2, 0.03 * fps))

    # Search backward from release for best horizontal arm
    uah_i = max(0, release_i - 1)
    best_score = 1e9

    for i in range(release_i - 1, max(0, release_i - int(0.30 * fps)), -1):
        if not wrist_above_shoulder(i):
            continue
        if 70.0 <= theta_s[i] <= 110.0:
            score = abs(theta_s[i] - 90.0)
            if score < best_score:
                best_score = score
                uah_i = i

    if uah_i >= release_i:
        uah_i = max(0, release_i - 1)

    # ------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------
    if DEBUG:
        logger.info(
            f"[Release/UAH] n={n} fps={fps:.2f} "
            f"peak={peak_i} "
            f"window=[{start}..{end}] "
            f"release={release_i} "
            f"uah={uah_i} "
            f"vmax={vmax:.4f}"
        )

        _dump(video_path, peak_i, "peak_wrist")
        _dump(video_path, start, "window_start")
        _dump(video_path, end, "window_end")
        _dump(video_path, release_i, "release")
        _dump(video_path, uah_i, "uah")

    return {
        "release": {"frame": int(frames[release_i])},
        "uah": {"frame": int(frames[uah_i])},
    }

