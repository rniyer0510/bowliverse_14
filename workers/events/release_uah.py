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
    
    # Non-bowling arm indices (opposite of bowling hand)
    nb_s_idx, nb_e_idx = (LS, LE) if h == "R" else (RS, RE)

    frames = []
    wrist_xyz, shoulder_xyz, elbow_xyz, pelvis_xyz = [], [], [], []
    nb_elbow_xyz = []  # Non-bowling elbow

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
        nb_e = _vis(nb_e_idx)  # Non-bowling elbow

        wrist_xyz.append(_xyz(w) if w else None)
        shoulder_xyz.append(_xyz(s) if s else None)
        elbow_xyz.append(_xyz(e) if e else None)
        nb_elbow_xyz.append(_xyz(nb_e) if nb_e else None)

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
    wrist_visible_count = 0
    for i in range(1, n):
        if wrist_xyz[i] and wrist_xyz[i-1]:
            dv = (
                wrist_xyz[i][0]-wrist_xyz[i-1][0],
                wrist_xyz[i][1]-wrist_xyz[i-1][1],
                wrist_xyz[i][2]-wrist_xyz[i-1][2],
            )
            wrist_fwd[i] = _dot(dv, forward) / dt
            wrist_visible_count += 1

    wrist_fwd_s = gaussian_filter1d(wrist_fwd, sigma=max(1.0, 0.03 * fps))
    peak_i = int(np.argmax(wrist_fwd_s))
    vmax = float(max(1e-6, np.max(np.abs(wrist_fwd_s))))
    
    # Check wrist visibility - if low (camera behind bowler), we'll relax geometric constraints
    wrist_visibility_ratio = wrist_visible_count / max(1, n - 1)
    low_wrist_visibility = wrist_visibility_ratio < 0.3  # Less than 30% frames have wrist
    
    # ------------------------------------------------------------
    # Non-bowling elbow vertical peak (BEST indicator from behind!)
    # ------------------------------------------------------------
    # The non-bowling arm sweeps UP before release, reaches highest point AT release,
    # then pulls DOWN into follow-through. This is very visible from behind.
    nb_elbow_y = np.full(n, np.nan, dtype=float)
    nb_elbow_visible_count = 0
    for i in range(n):
        if nb_elbow_xyz[i] and nb_elbow_xyz[i][1] is not None:
            nb_elbow_y[i] = nb_elbow_xyz[i][1]  # Y coordinate (negative = higher)
            nb_elbow_visible_count += 1
    
    # Smooth and find minimum Y (highest point, since Y increases downward)
    if nb_elbow_visible_count > max(5, int(0.1 * fps)):
        nb_elbow_y_valid = nb_elbow_y[np.isfinite(nb_elbow_y)]
        if len(nb_elbow_y_valid) > 0:
            # Interpolate gaps
            idx = np.arange(n)
            good = np.isfinite(nb_elbow_y)
            if good.sum() >= 2:
                nb_elbow_y[~good] = np.interp(idx[~good], idx[good], nb_elbow_y[good])
            
            nb_elbow_y_s = gaussian_filter1d(nb_elbow_y, sigma=max(1.0, 0.03 * fps))
            nb_elbow_peak_i = int(np.argmin(nb_elbow_y_s))  # Minimum Y = highest point
            
            nb_elbow_available = True
            logger.info(f"[Release/UAH] Non-bowling elbow peak found at frame {nb_elbow_peak_i} (vis={nb_elbow_visible_count}/{n})")
        else:
            nb_elbow_available = False
            nb_elbow_peak_i = None
    else:
        nb_elbow_available = False
        nb_elbow_peak_i = None

    # ------------------------------------------------------------
    # Post-release detection (multiple strategies in priority order)
    # ------------------------------------------------------------
    def is_pre_followthrough(i):
        """
        Detect if frame is BEFORE follow-through (i.e., valid for release window).
        
        Strategy 0 (BEST for behind): Non-bowling elbow hasn't peaked yet
        Strategy 1 (best for side): Wrist above shoulder
        Strategy 2 (fallback): Bowling elbow above shoulder
        Strategy 3 (fallback): Upper arm angle hasn't collapsed yet
        """
        
        # Strategy 0: Non-bowling elbow hasn't reached peak yet
        # This is THE BEST indicator from behind the bowler!
        # The front arm sweeps up, peaks at release, then pulls down
        if nb_elbow_available and nb_elbow_peak_i is not None:
            # We're pre-release if we haven't passed the elbow peak yet
            # Allow small window around peak since it plateaus briefly
            return i <= nb_elbow_peak_i + int(0.02 * fps)  # ~20ms tolerance
        
        w, s = wrist_xyz[i], shoulder_xyz[i]
        e = elbow_xyz[i]
        
        # Strategy 1: Wrist above shoulder (best when wrist visible)
        if w and s:
            return w[1] <= s[1] + 0.04
        
        # Strategy 2: Bowling elbow above shoulder (works from behind)
        if e and s:
            return e[1] <= s[1] + 0.10
        
        # Strategy 3: Upper arm angle (works from behind if shoulder/elbow visible)
        if s and e:
            angle = _upper_arm_angle(s, e, forward)
            if angle is not None:
                return angle <= 120.0
        
        # Conservative fallback
        return low_wrist_visibility

    # ------------------------------------------------------------
    # RELEASE WINDOW (POST-PEAK, VERY NARROW)
    # ------------------------------------------------------------
    # If non-bowling elbow peak is available and reliable, use it as anchor
    # Otherwise use wrist velocity peak
    if nb_elbow_available and nb_elbow_peak_i is not None and low_wrist_visibility:
        # Camera from behind - use non-bowling elbow peak as primary anchor!
        anchor_i = nb_elbow_peak_i
        anchor_type = "nb_elbow_peak"
        # Tighter window since elbow peak IS the release
        min_offset = int(max(1, round(0.02 * fps)))  # ~20ms before
        max_offset = int(max(2, round(0.05 * fps)))  # ~50ms after
        logger.info(f"[Release/UAH] Using non-bowling elbow peak as anchor (camera from behind)")
    else:
        # Normal case - use wrist velocity peak
        anchor_i = peak_i
        anchor_type = "wrist_velocity_peak"
        # Standard window
        min_offset = int(max(2, round(0.05 * fps)))
        max_offset = int(max(4, round(0.10 * fps)))
    
    start = max(0, min(n - 1, anchor_i - min_offset))
    end   = min(n - 1, anchor_i + max_offset)

    # ------------------------------------------------------------
    # RELEASE SELECTION (ROBUST 3-TIER APPROACH)
    # Priority: nb_elbow > wrist_geometry > velocity_heuristic
    # ------------------------------------------------------------
    release_i = None
    selection_method = None
    selection_confidence = 0.0

    # ============================================================
    # TIER 1: Use most reliable signal available
    # ============================================================

    # TIER 1A: Non-bowling elbow peak (best for behind-camera)
    if nb_elbow_available and nb_elbow_peak_i is not None:
        if start <= nb_elbow_peak_i <= end:
            release_i = nb_elbow_peak_i
            selection_method = "nb_elbow_peak"
            selection_confidence = 0.90
            logger.info(f"[Release/UAH] Tier 1A: nb_elbow peak at frame {release_i}")

    # TIER 1B: Wrist geometry (best for side-camera with good visibility)
    if release_i is None:
        wrist_visible_in_window = [i for i in range(start, end + 1) if wrist_xyz[i] is not None]
        
        if len(wrist_visible_in_window) >= 3:
            # Sub-strategy 1B.1: Wrist at shoulder height (geometric release indicator)
            for i in wrist_visible_in_window:
                w, s = wrist_xyz[i], shoulder_xyz[i]
                if w and s:
                    dy = w[1] - s[1]
                    # Wrist within ±2cm of shoulder AND velocity still high
                    if -0.02 <= dy <= 0.02 and wrist_fwd_s[i] >= 0.70 * vmax:
                        release_i = i
                        selection_method = "wrist_at_shoulder"
                        selection_confidence = 0.85
                        logger.info(f"[Release/UAH] Tier 1B.1: wrist at shoulder, frame {release_i}")
                        break
            
            # Sub-strategy 1B.2: Significant velocity drop (not just noise)
            if release_i is None:
                for i in wrist_visible_in_window:
                    if i > start:
                        # Must drop at least 20% from peak
                        if wrist_fwd_s[i] <= 0.80 * vmax and wrist_fwd_s[i-1] > 0.80 * vmax:
                            # Verify still pre-follow-through if possible
                            w, s = wrist_xyz[i], shoulder_xyz[i]
                            if w and s:
                                if w[1] <= s[1] + 0.10:  # Wrist not far below shoulder
                                    release_i = i
                                    selection_method = "velocity_drop_20pct"
                                    selection_confidence = 0.75
                                    logger.info(f"[Release/UAH] Tier 1B.2: velocity drop at frame {release_i}")
                                    break

    # ============================================================
    # TIER 2: Wrist occluded - use alternative signals
    # ============================================================

    if release_i is None:
        # Check if nb_elbow peak is nearby (even if not exactly in window)
        if nb_elbow_available and nb_elbow_peak_i is not None:
            dist = abs(nb_elbow_peak_i - anchor_i)
            if dist <= int(0.10 * fps):  # Within 100ms of anchor
                release_i = nb_elbow_peak_i
                selection_method = "nb_elbow_nearby"
                selection_confidence = 0.80
                logger.info(f"[Release/UAH] Tier 2A: nb_elbow nearby at frame {release_i}")
        
        # Use velocity peak + empirical offset
        if release_i is None:
            # Empirical: release is typically 1-3 frames after velocity peak
            offset_frames = max(1, int(round(2.0 * fps / 30.0)))
            release_i = min(end, peak_i + offset_frames)
            selection_method = "peak_plus_offset"
            selection_confidence = 0.60
            logger.info(f"[Release/UAH] Tier 2B: velocity peak + {offset_frames} = frame {release_i}")

    # ============================================================
    # TIER 3: Absolute fallback
    # ============================================================

    if release_i is None:
        release_i = start
        selection_method = "window_start"
        selection_confidence = 0.40
        logger.warning(f"[Release/UAH] Tier 3: fallback to window start {release_i}")

    # ============================================================
    # VALIDATION: Sanity check the selection
    # ============================================================

    if release_i is not None and wrist_xyz[release_i] and shoulder_xyz[release_i]:
        w, s = wrist_xyz[release_i], shoulder_xyz[release_i]
        
        # If wrist is way below shoulder, we're in follow-through!
        if w[1] > s[1] + 0.15:
            logger.warning(f"[Release/UAH] Frame {release_i} appears to be follow-through (wrist >> shoulder)")
            
            # Try to find earlier frame that's better
            for i in range(max(start, release_i - 5), release_i):
                if wrist_xyz[i] and shoulder_xyz[i]:
                    wi, si = wrist_xyz[i], shoulder_xyz[i]
                    if wi[1] <= si[1] + 0.05:
                        release_i = i
                        selection_method = f"{selection_method}_corrected"
                        selection_confidence *= 0.9
                        logger.info(f"[Release/UAH] Corrected to earlier frame {release_i}")
                        break

    logger.info(
        f"[Release/UAH] FINAL: frame={release_i} method={selection_method} "
        f"confidence={selection_confidence:.2f}"
    )

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
        if not is_pre_followthrough(i):
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
        # Determine which detection strategy was used
        detection_strategy = "unknown"
        if release_i and release_i < len(wrist_xyz):
            if nb_elbow_available:
                detection_strategy = "nb_elbow_peak"
            else:
                w, s, e = wrist_xyz[release_i], shoulder_xyz[release_i], elbow_xyz[release_i]
                if w and s:
                    detection_strategy = "wrist_above_shoulder"
                elif e and s:
                    detection_strategy = "bowling_elbow_above_shoulder"
                elif s and e:
                    detection_strategy = "upper_arm_angle"
                else:
                    detection_strategy = "fallback"
        
        logger.info(
            f"[Release/UAH] n={n} fps={fps:.2f} "
            f"anchor={anchor_type} anchor_i={anchor_i} "
            f"wrist_peak={peak_i} "
            f"nb_elbow_peak={nb_elbow_peak_i if nb_elbow_available else 'N/A'} "
            f"window=[{start}..{end}] "
            f"release={release_i} "
            f"uah={uah_i} "
            f"vmax={vmax:.4f} "
            f"wrist_vis={wrist_visibility_ratio:.2%} "
            f"nb_elbow_vis={nb_elbow_visible_count}/{n} "
            f"low_vis_mode={low_wrist_visibility} "
            f"strategy={detection_strategy}"
        )

        _dump(video_path, peak_i, "peak_wrist")
        _dump(video_path, start, "window_start")
        _dump(video_path, end, "window_end")
        _dump(video_path, release_i, "release")
        _dump(video_path, uah_i, "uah")

    # ------------------------------------------------------------
    # RETURN (contract preserved + non-breaking additions)
    # ------------------------------------------------------------
    return {
        "release": {"frame": int(frames[release_i])},
        "uah": {"frame": int(frames[uah_i])},

        # Non-breaking additions:
        # delivery_window is the exact window used for release selection,
        # exposed so downstream components (FFC/BFC visuals) never invent timing.
        "delivery_window": [int(frames[start]), int(frames[end])],

        # Useful for debugging / consumers (optional)
        "peak": {"frame": int(frames[peak_i])},
    }
