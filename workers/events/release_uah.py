"""
Release & UAH detection — ActionLab V14 (REVIEWED & HARDENED)

Biomechanics-first, object-aware, wrist-safe.

Priority order for Release detection:
1. Ball separation (future-ready, optional)
2. Elbow angular velocity peak (PRIMARY fallback)
3. Wrist-speed decay (LAST resort)

UAH detection remains unchanged and is anchored backward from release.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

# MediaPipe indices (LOCKED)
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16

MIN_VIS = 0.5
SMOOTH_WIN = 3

# Release detection tuning
ELBOW_PEAK_SEARCH_RATIO = 0.45     # search after this fraction of clip
ELBOW_PEAK_MIN_SAMPLES = 3
WRIST_FALLBACK_RATIO = 0.85

# UAH constraints
UAH_LOOKBACK_SEC = 0.9
UAH_MIN_SEP_SEC = 0.15

CARRY_DECAY = 0.92


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _xyz(pt):
    if pt is None:
        return None
    try:
        return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
    except Exception:
        return None


def _dist(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def _smooth(vals: List[Optional[float]], k: int) -> List[float]:
    """
    Smooth while ignoring None values.
    During occlusion, decay last valid value.
    """
    out: List[float] = []
    last_valid = 0.0

    for i in range(len(vals)):
        lo = max(0, i - k // 2)
        hi = min(len(vals), i + k // 2 + 1)
        window = [v for v in vals[lo:hi] if v is not None]

        if window:
            val = sum(window) / len(window)
            last_valid = val
            out.append(val)
        else:
            last_valid *= CARRY_DECAY
            out.append(last_valid)

    return out


def _angle(a, b, c) -> Optional[float]:
    """
    Internal elbow angle at b: Shoulder–Elbow–Wrist
    """
    bax, bay, baz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    bcx, bcy, bcz = c[0] - b[0], c[1] - b[1], c[2] - b[2]

    mag_ba = math.sqrt(bax*bax + bay*bay + baz*baz)
    mag_bc = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None

    dot = bax*bcx + bay*bcy + baz*bcz
    cosv = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosv))


def _upper_arm_tilt(s, e):
    """
    Angle between shoulder→elbow and vertical.
    """
    vx, vy, vz = e[0] - s[0], e[1] - s[1], e[2] - s[2]
    mag = math.sqrt(vx * vx + vy * vy + vz * vz)
    if mag < 1e-6:
        return None
    cosv = abs(vy) / mag
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def detect_release_uah(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
) -> Dict[str, Dict[str, int]]:

    if hand == "R":
        s_idx, e_idx, w_idx = RS, RE, RW
    else:
        s_idx, e_idx, w_idx = LS, LE, LW

    frames: List[int] = []
    elbow_angles: List[Optional[float]] = []
    wrist_xyz: List[Optional[Tuple[float, float, float]]] = []
    shoulder_xyz: List[Optional[Tuple[float, float, float]]] = []
    elbow_xyz: List[Optional[Tuple[float, float, float]]] = []

    # -----------------------------
    # Extract signals
    # -----------------------------
    for it in pose_frames:
        f = int(it["frame"])
        lm = it.get("landmarks")

        frames.append(f)

        if not lm or len(lm) <= max(s_idx, e_idx, w_idx):
            elbow_angles.append(None)
            wrist_xyz.append(None)
            shoulder_xyz.append(None)
            elbow_xyz.append(None)
            continue

        try:
            if (
                lm[s_idx].get("visibility", 1.0) < MIN_VIS or
                lm[e_idx].get("visibility", 1.0) < MIN_VIS or
                lm[w_idx].get("visibility", 1.0) < MIN_VIS
            ):
                elbow_angles.append(None)
                wrist_xyz.append(None)
                shoulder_xyz.append(None)
                elbow_xyz.append(None)
                continue
        except Exception:
            elbow_angles.append(None)
            wrist_xyz.append(None)
            shoulder_xyz.append(None)
            elbow_xyz.append(None)
            continue

        s = _xyz(lm[s_idx])
        e = _xyz(lm[e_idx])
        w = _xyz(lm[w_idx])

        shoulder_xyz.append(s)
        elbow_xyz.append(e)
        wrist_xyz.append(w)

        ang = _angle(s, e, w) if s and e and w else None
        elbow_angles.append(ang)

    # -----------------------------
    # Elbow angular velocity (PRIMARY)
    # -----------------------------
    ang_vel: List[Optional[float]] = [None]
    for i in range(1, len(elbow_angles)):
        if elbow_angles[i] is not None and elbow_angles[i - 1] is not None:
            ang_vel.append(abs(elbow_angles[i] - elbow_angles[i - 1]))
        else:
            ang_vel.append(None)

    ang_vel_s = _smooth(ang_vel, SMOOTH_WIN)

    start_i = int(len(ang_vel_s) * ELBOW_PEAK_SEARCH_RATIO)
    peak_elbow_i = None

    if sum(v is not None for v in ang_vel_s[start_i:]) >= ELBOW_PEAK_MIN_SAMPLES:
        peak_elbow_i = max(
            range(start_i, len(ang_vel_s)),
            key=lambda i: ang_vel_s[i] if ang_vel_s[i] is not None else -1
        )

    # -----------------------------
    # Wrist-speed fallback (LAST)
    # -----------------------------
    wrist_speed: List[Optional[float]] = [None]
    for i in range(1, len(frames)):
        if wrist_xyz[i] and wrist_xyz[i - 1]:
            wrist_speed.append(_dist(wrist_xyz[i], wrist_xyz[i - 1]))
        else:
            wrist_speed.append(None)

    wrist_speed_s = _smooth(wrist_speed, SMOOTH_WIN)
    peak_wrist_i = max(range(len(wrist_speed_s)), key=lambda i: wrist_speed_s[i])

    # -----------------------------
    # Final release selection
    # -----------------------------
    if peak_elbow_i is not None:
        release_i = peak_elbow_i
    else:
        release_i = peak_wrist_i

    release_frame = frames[release_i]

    # -----------------------------
    # UAH (unchanged logic)
    # -----------------------------
    best_tilt = None
    uah_i = None

    fps_eff = fps or 25.0
    lookback = max(8, int(UAH_LOOKBACK_SEC * fps_eff))
    min_sep = max(2, int(UAH_MIN_SEP_SEC * fps_eff))

    uah_end = max(0, release_i - min_sep)
    uah_start = max(0, uah_end - lookback)

    for i in range(uah_end, uah_start - 1, -1):
        if shoulder_xyz[i] and elbow_xyz[i]:
            tilt = _upper_arm_tilt(shoulder_xyz[i], elbow_xyz[i])
            if tilt is None:
                continue
            if best_tilt is None or tilt < best_tilt:
                best_tilt = tilt
                uah_i = i

    if uah_i is None:
        uah_i = max(0, release_i - min_sep)

    uah_frame = frames[uah_i]

    # Safety
    if release_frame <= uah_frame:
        release_frame = frames[min(len(frames) - 1, uah_i + 1)]

    return {
        "uah": {"frame": int(uah_frame)},
        "release": {"frame": int(release_frame)},
    }

