"""
ActionLab V14 — Release + UAH detector (reverse, deterministic, guardrail-based)

Final refinement: Release anchored to peak (+0 or +1), UAH tightened to ~140 when release ~143
- max_back_sec = 0.12 for pace (~3 frames back from release)
- Added lateness bonus to prefer later acceleration segments
- Hard guardrail: UAH cannot be >5 frames before release
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

# MediaPipe indices
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24

MIN_VIS = 0.25
DEBUG = True

DEBUG_DIR = "/tmp/actionlab_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------

def _xyz(pt) -> Optional[Tuple[float, float, float]]:
    try:
        return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
    except Exception:
        return None


def _mag(v) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _dot(a, b) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _upper_arm_angle(s, e, forward) -> Optional[float]:
    if not (s and e):
        return None
    vx, vy, vz = e[0]-s[0], e[1]-s[1], e[2]-s[2]
    mag = _mag((vx, vy, vz))
    if mag < 1e-6:
        return None
    ux, uy, uz = vx/mag, vy/mag, vz/mag
    dot = max(-1.0, min(1.0, ux*forward[0] + uy*forward[1] + uz*forward[2]))
    return math.degrees(math.acos(dot))


def _elbow_extension_deg(s, e, w) -> Optional[float]:
    if not (s and e and w):
        return None
    se = (s[0]-e[0], s[1]-e[1], s[2]-e[2])
    ew = (w[0]-e[0], w[1]-e[1], w[2]-e[2])
    mag = _mag(se) * _mag(ew)
    if mag < 1e-6:
        return None
    dot = max(-1.0, min(1.0, _dot(se, ew) / mag))
    return math.degrees(math.acos(dot))


def _safe_lm(lm: Any, idx: int) -> Optional[Dict[str, Any]]:
    if not lm or not isinstance(lm, list):
        return None
    if idx < 0 or idx >= len(lm):
        return None
    pt = lm[idx]
    return pt if isinstance(pt, dict) else None


def _vis_ok(pt: Optional[Dict[str, Any]], thr: float = MIN_VIS) -> bool:
    return bool(pt) and float(pt.get("visibility", 0.0)) >= thr


def _forearm_below_shoulder(s, e, w, y_margin: float) -> Optional[bool]:
    if not (s and e and w):
        return None
    forearm_y = min(e[1], w[1])
    return forearm_y > (s[1] + y_margin)


def _unit(v):
    m = _mag(v) or 1.0
    return (v[0]/m, v[1]/m, v[2]/m)


def _robust_tail_window(n: int, wrist_present: List[bool]) -> Tuple[int, int]:
    if n < 20:
        return (0, n-1)

    candidates = [0.20, 0.25, 0.33, 0.40, 0.50]
    best = None

    for frac in candidates:
        start = int((1.0 - frac) * n)
        end = n - 1
        present = sum(1 for i in range(start, end+1) if wrist_present[i])
        density = present / max(1, (end - start + 1))
        score = density + 0.15 * frac
        if best is None or score > best[0]:
            best = (score, start, end, density)

    _, start, end, density = best
    if density < 0.08:
        start = int(0.40 * n)

    return (start, end)


def _dump_frame(video_path: Optional[str], frame_idx: int, tag: str):
    if not DEBUG or not video_path or not os.path.exists(video_path):
        return
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{tag}_{frame_idx}.png"), frame)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def detect_release_uah(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
    video_path: Optional[str] = None,
):
    if not pose_frames:
        return {"uah": {"frame": 0}, "release": {"frame": 0}}

    fps = float(fps or 25.0)
    dt = 1.0 / fps

    s_idx, e_idx, w_idx = (RS, RE, RW) if hand == "R" else (LS, LE, LW)

    frames = []
    wrist_xyz = []
    shoulder_xyz = []
    elbow_xyz = []
    pelvis_xyz = []
    wrist_present = []

    for it in pose_frames:
        frames.append(int(it.get("frame", len(frames))))
        lm = it.get("landmarks")

        w = _safe_lm(lm, w_idx)
        s = _safe_lm(lm, s_idx)
        e = _safe_lm(lm, e_idx)
        hl = _safe_lm(lm, LH)
        hr = _safe_lm(lm, RH)

        w_xyz = _xyz(w) if _vis_ok(w) else None
        s_xyz = _xyz(s) if _vis_ok(s) else None
        e_xyz = _xyz(e) if _vis_ok(e) else None

        wrist_xyz.append(w_xyz)
        shoulder_xyz.append(s_xyz)
        elbow_xyz.append(e_xyz)
        wrist_present.append(w_xyz is not None)

        if _vis_ok(hl) and _vis_ok(hr):
            pelvis_xyz.append((
                (float(hl["x"]) + float(hr["x"])) / 2.0,
                (float(hl["y"]) + float(hr["y"])) / 2.0,
                (float(hl.get("z", 0.0)) + float(hr.get("z", 0.0))) / 2.0,
            ))
        else:
            pelvis_xyz.append(None)

    n = len(frames)

    vels = []
    for i in range(1, n):
        if pelvis_xyz[i] and pelvis_xyz[i-1]:
            vels.append((
                pelvis_xyz[i][0]-pelvis_xyz[i-1][0],
                pelvis_xyz[i][1]-pelvis_xyz[i-1][1],
                pelvis_xyz[i][2]-pelvis_xyz[i-1][2]
            ))

    forward = _unit(tuple(map(sum, zip(*vels)))) if vels else (1.0, 0.0, 0.0)

    wrist_speed = np.zeros(n)
    for i in range(1, n):
        if wrist_xyz[i] and wrist_xyz[i-1]:
            wrist_speed[i] = _mag((
                wrist_xyz[i][0]-wrist_xyz[i-1][0],
                wrist_xyz[i][1]-wrist_xyz[i-1][1],
                wrist_xyz[i][2]-wrist_xyz[i-1][2],
            )) / dt

    wrist_speed_s = gaussian_filter1d(wrist_speed, sigma=max(1.0, 0.02 * fps))

    SEARCH_START, SEARCH_END = _robust_tail_window(n, wrist_present)
    SEARCH_END = min(SEARCH_END, n - 1)

    candidates = [(i, wrist_speed[i]) for i in range(SEARCH_START, SEARCH_END + 1) if wrist_present[i]]

    top = sorted(candidates, key=lambda x: x[1], reverse=True)[:14]

    clusters = []
    for idx, _ in top:
        for c in clusters:
            if abs(np.mean(c) - idx) <= max(3, int(0.06 * fps)):
                c.append(idx)
                break
        else:
            clusters.append([idx])

    best_cluster = max(clusters, key=lambda c: len(c) * np.mean([wrist_speed[i] for i in c]) if c else 0)
    peak_idx = int(np.median(best_cluster))
    peak_speed_raw = wrist_speed[peak_idx] or 1.0

    tail_vals = np.array([v for _, v in candidates])
    p90 = np.percentile(tail_vals, 90) if len(tail_vals) > 0 else 0
    p50 = np.percentile(tail_vals, 50) if len(tail_vals) > 0 else 0
    pace_like = peak_speed_raw > max(1.8 * p90, 3.0 * p50)

    # Spinner-safe conditional parameters
    if pace_like:
        FOREARM_Y_MARGIN = 0.05
        max_back_sec = 0.12  # tightened to ~3 frames back from release
        FORWARD_DIST_THR = 0.92
        WRIST_DROP_THR = 0.48
        WRIST_DECAY_FRAC = 0.07
        MAX_ELBOW_EXT = 80.0
        min_after_peak_sec = 0.04  # extremely soft (~1 frame)
    else:
        FOREARM_Y_MARGIN = 0.10
        max_back_sec = 0.40
        FORWARD_DIST_THR = 1.00
        WRIST_DROP_THR = 0.54
        WRIST_DECAY_FRAC = 0.08
        MAX_ELBOW_EXT = 160.0
        min_after_peak_sec = 0.18

    MIN_ELBOW_EXT = 10.0

    pre_sec = 0.10
    post_sec = 0.16

    delivery_start = max(SEARCH_START, peak_idx - int(pre_sec * fps))
    delivery_end = min(SEARCH_END, peak_idx + int(post_sec * fps))

    # RELEASE detection - anchor to peak
    release_i = peak_idx  # default = peak itself
    considered_count = 0
    exit_reason = "anchored to peak wrist frame"

    decay_count = 0

    for i in range(delivery_end, delivery_start - 1, -1):
        considered_count += 1
        if not wrist_present[i]:
            continue

        s = shoulder_xyz[i]
        e = elbow_xyz[i]
        w = wrist_xyz[i]

        # Decay check
        if wrist_speed_s[i] < WRIST_DECAY_FRAC * peak_speed_raw:
            exit_reason = f"speed decay at {i}"
            release_i = i + 1
            break

        if not (s and e and w):
            continue

        fb = _forearm_below_shoulder(s, e, w, FOREARM_Y_MARGIN)
        if pace_like and fb is True:
            continue

        ext = _elbow_extension_deg(s, e, w)
        if ext is not None:
            if ext > MAX_ELBOW_EXT:
                exit_reason = f"elbow extension > max at {i}"
                release_i = i + 1
                break
            if ext < MIN_ELBOW_EXT:
                continue

        ws = (w[0] - s[0], w[1] - s[1], w[2] - s[2])
        fwd_dist = _dot(ws, forward)
        if fwd_dist > FORWARD_DIST_THR:
            exit_reason = f"forward dist > thr at {i}"
            release_i = i + 1
            break

        if w[1] > s[1] + WRIST_DROP_THR:
            exit_reason = f"wrist drop at {i}"
            release_i = i + 1
            break

        # Per-frame debug around peak
        if DEBUG and peak_idx - 10 <= i <= peak_idx + 10:
            print(f"  Frame {i:3d} | speed_s={wrist_speed_s[i]:.2f} | decay_thr={WRIST_DECAY_FRAC*peak_speed_raw:.2f} | fwd_dist={fwd_dist:.2f} | ext={ext if ext else 'N/A'} | fb={fb}")

    # Post-adjust forearm safety
    if pace_like:
        for k in range(0, max(2, int(0.05 * fps))):
            j = min(n - 1, release_i + k)
            fb = _forearm_below_shoulder(shoulder_xyz[j], elbow_xyz[j], wrist_xyz[j], FOREARM_Y_MARGIN)
            if fb is False or fb is None:
                release_i = j
                break

    # Soft min delay (almost none)
    min_release_i = peak_idx + int(min_after_peak_sec * fps)
    if release_i < min_release_i:
        release_i = min_release_i
        exit_reason += " (soft min delay adjustment)"

    # Hard cap: max +1 after peak
    if release_i > peak_idx + 1:
        release_i = peak_idx + 1
        exit_reason += " (capped to peak + 1)"

    release_i = int(max(delivery_start, min(release_i, SEARCH_END)))

    # UAH detection - tightened and biased to later frames
    theta = np.zeros(n, dtype=float)
    theta_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        ang = _upper_arm_angle(shoulder_xyz[i], elbow_xyz[i], forward)
        if ang is not None and shoulder_xyz[i] and elbow_xyz[i]:
            theta[i] = float(ang)
            theta_mask[i] = True

    theta_s = gaussian_filter1d(theta, sigma=max(1.2, 0.03 * fps))
    dtheta = np.gradient(theta_s, dt)
    ddtheta = np.gradient(dtheta, dt)

    uah_lo = max(SEARCH_START, release_i - int(max_back_sec * fps))
    uah_hi = max(uah_lo, release_i - int(0.08 if pace_like else 0.10 * fps))

    persist = max(3, int(0.10 * fps))

    best_i = uah_lo
    best_score = -1e9

    for i in range(uah_hi, uah_lo - 1, -1):
        if not theta_mask[i]:
            continue
        if pace_like and _forearm_below_shoulder(shoulder_xyz[i], elbow_xyz[i], wrist_xyz[i], FOREARM_Y_MARGIN) is True:
            continue

        seg_v = dtheta[i:i+persist]
        seg_a = ddtheta[i:i+persist]
        if len(seg_v) < persist or len(seg_a) < persist:
            continue

        if np.all(seg_v > 0.0) and np.all(seg_a > 0.0):
            # Add lateness bonus to prefer later frames
            lateness_bonus = 0.3 * (i - uah_lo) / max(1, uah_hi - uah_lo)
            score = float(np.mean(seg_a)) + 0.15 * float(np.mean(seg_v)) + lateness_bonus
            if score > best_score:
                best_score = score
                best_i = i

    if best_score < -1e8:
        cand = [i for i in range(uah_lo, uah_hi + 1) if theta_mask[i] and (not pace_like or _forearm_below_shoulder(shoulder_xyz[i], elbow_xyz[i], wrist_xyz[i], FOREARM_Y_MARGIN) is not True)]
        if cand:
            best_i = max(cand, key=lambda j: ddtheta[j] + 0.3 * (j - uah_lo))  # lateness bonus here too

    uah_i = int(max(0, min(best_i, release_i - 1)))

    # Hard guardrail: UAH cannot be more than 5 frames before release
    min_uah = release_i - 5
    if uah_i < min_uah:
        uah_i = min_uah

    # Debug
    if DEBUG:
        print("\n=== RELEASE / UAH DEBUG ===")
        print(f"Peak wrist frame: {peak_idx}")
        print(f"Release frame: {release_i}")
        print(f"UAH frame: {uah_i}")
        print(f"Frames considered in reverse loop: {considered_count}")
        print(f"Reverse loop stopped because: {exit_reason}")
        print("===========================\n")

        _dump_frame(video_path, peak_idx, "peak")
        _dump_frame(video_path, release_i, "release")
        _dump_frame(video_path, uah_i, "uah")

    return {
        "uah": {"frame": int(frames[uah_i])},
        "release": {"frame": int(frames[release_i])},
    }
