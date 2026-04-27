"""
ActionLab V14 — Release + UAH detector

CURRENT PRINCIPLES:
- Peak wrist forward velocity remains the preferred distal anchor
- Release is inferred from multiple kinematic and geometric candidates
- Low-distal-visibility clips can fall back to proximal consensus
- Frames that look like follow-through are penalized or rejected
- UAH is geometric and must occur BEFORE Release

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
from app.workers.events.event_confidence import build_candidate, chain_quality, compact_candidates

logger = get_logger(__name__)

# Pose landmarks
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24

MIN_VIS = 0.25
DEBUG = os.getenv("ACTIONLAB_RELEASE_UAH_DEBUG", "").strip().lower() == "true"
DEBUG_DIR = os.getenv(
    "ACTIONLAB_RELEASE_UAH_DEBUG_DIR",
    "/tmp/actionlab_debug/release_uah",
)

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
    os.makedirs(DEBUG_DIR, exist_ok=True)
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


def _nb_elbow_peak_is_plausible(
    *,
    nb_elbow_peak_i: Optional[int],
    wrist_peak_i: int,
    total_frames: int,
    fps: float,
) -> bool:
    if nb_elbow_peak_i is None or total_frames <= 0:
        return False

    # Reject spurious front-loaded peaks from long run-ups or clipped starts.
    min_delivery_frame = max(5, int(0.20 * total_frames))
    max_edge_frame = max(0, total_frames - max(3, int(0.02 * total_frames)))
    close_to_wrist_peak = abs(int(nb_elbow_peak_i) - int(wrist_peak_i)) <= int(max(2, round(0.25 * fps)))

    if nb_elbow_peak_i < min_delivery_frame and not close_to_wrist_peak:
        return False
    if nb_elbow_peak_i >= max_edge_frame and not close_to_wrist_peak:
        return False
    return True


def _build_release_assessment(
    *,
    release_i: Optional[int],
    peak_i: int,
    selection_method: Optional[str],
    selection_confidence: float,
    wrist_xyz: List[Optional[tuple]],
    shoulder_xyz: List[Optional[tuple]],
    low_wrist_visibility: bool,
    corrected_followthrough: bool,
) -> Dict[str, Any]:
    if release_i is None:
        return {
            "state": "MISSING",
            "score": 0.0,
            "notes": ["release_missing"],
        }

    weak_methods = {"peak_plus_offset", "window_start"}
    method = str(selection_method or "").strip().lower()
    notes: List[str] = []
    score = float(selection_confidence)

    if method in weak_methods:
        notes.append("heuristic_release_anchor")
        score -= 0.18
    if low_wrist_visibility:
        notes.append("low_distal_visibility")
        score -= 0.10
    if corrected_followthrough:
        notes.append("followthrough_correction_applied")
        score -= 0.08

    wrist = wrist_xyz[release_i] if 0 <= release_i < len(wrist_xyz) else None
    shoulder = shoulder_xyz[release_i] if 0 <= release_i < len(shoulder_xyz) else None
    if wrist and shoulder:
        if wrist[1] > shoulder[1] + 0.15:
            notes.append("late_followthrough_frame")
            return {
                "state": "INVALID",
                "score": round(max(0.0, score - 0.35), 3),
                "notes": notes,
            }
        if wrist[1] > shoulder[1] + 0.05:
            notes.append("release_geometry_soft")
            score -= 0.10

    if release_i > peak_i + 3:
        notes.append("release_far_past_peak")
        score -= 0.08

    state = "CONFIDENT"
    if score < 0.45:
        state = "INVALID"
    elif score < 0.72 or notes:
        state = "WEAK"

    return {
        "state": state,
        "score": round(max(0.0, min(1.0, score)), 3),
        "notes": notes,
    }


def _is_local_extremum(values: np.ndarray, idx: int, *, window: int, mode: str) -> bool:
    start = max(0, idx - window)
    end = min(len(values), idx + window + 1)
    segment = values[start:end]
    if segment.size == 0:
        return False
    center = float(values[idx])
    if mode == "max":
        return center >= float(np.max(segment))
    return center <= float(np.min(segment))


def _score_release_candidate(
    *,
    i: int,
    n: int,
    fps: float,
    peak_i: int,
    consensus_start: int,
    consensus_end: int,
    wrist_fwd_s: np.ndarray,
    vmax: float,
    wrist_xyz: List[Optional[tuple]],
    shoulder_xyz: List[Optional[tuple]],
    elbow_xyz: List[Optional[tuple]],
    nb_elbow_peak_i: Optional[int],
    nb_elbow_available: bool,
    shoulder_peak_i: Optional[int],
    shoulder_y_s: np.ndarray,
    is_pre_followthrough,
    low_wrist_visibility: bool,
    forward: tuple,
) -> Dict[str, Any]:
    score = 0.0
    reasons: List[str] = []

    if _is_local_extremum(wrist_fwd_s, i, window=3, mode="max"):
        score += 2.6
        reasons.append("wrist_vel_peak")
    else:
        vel_ratio = float(wrist_fwd_s[i] / max(vmax, 1e-6))
        if vel_ratio >= 0.80:
            score += 1.8
            reasons.append("wrist_vel_high")
        elif vel_ratio >= 0.60:
            score += 1.0
            reasons.append("wrist_vel_usable")

    if is_pre_followthrough(i):
        score += 1.8
        reasons.append("pre_followthrough")
    else:
        score -= 3.5
        reasons.append("penalty_followthrough")

    w = wrist_xyz[i] if 0 <= i < len(wrist_xyz) else None
    s = shoulder_xyz[i] if 0 <= i < len(shoulder_xyz) else None
    e = elbow_xyz[i] if 0 <= i < len(elbow_xyz) else None

    if not low_wrist_visibility and w and s:
        dy = w[1] - s[1]
        if -0.04 <= dy <= 0.03:
            score += 2.0
            reasons.append("wrist_at_shoulder")
        elif dy <= 0.08:
            score += 1.0
            reasons.append("wrist_near_shoulder")
        elif dy > 0.15:
            score -= 2.5
            reasons.append("penalty_wrist_low")

    if w and s and e:
        if w[1] > s[1] + 0.10 and e[1] > s[1] + 0.10:
            score -= 2.8
            reasons.append("penalty_recovery_posture")

    if s and e:
        angle = _upper_arm_angle(s, e, forward)
        if angle is not None:
            if 72.0 <= angle <= 110.0:
                score += 1.2
                reasons.append("upper_arm_horizontal")
            elif 60.0 <= angle <= 122.0:
                score += 0.5
                reasons.append("upper_arm_plausible")

    if 0 <= i < len(shoulder_y_s) and _is_local_extremum(shoulder_y_s, i, window=4, mode="min"):
        score += 1.1
        reasons.append("shoulder_peak")

    if shoulder_peak_i is not None:
        shoulder_delta = i - shoulder_peak_i
        if abs(shoulder_delta) <= 2:
            score += 0.8
            reasons.append("near_shoulder_peak")
        elif shoulder_delta > max(2, int(round(0.10 * fps))):
            score -= 1.4
            reasons.append("penalty_post_shoulder_peak")

    if nb_elbow_available and nb_elbow_peak_i is not None:
        dist = abs(i - nb_elbow_peak_i)
        signed_dist = i - nb_elbow_peak_i
        if dist <= 1:
            score += 2.4
            reasons.append("nb_elbow_peak")
        elif dist <= 3:
            score += 1.4
            reasons.append("nb_elbow_near")
        elif signed_dist > max(1, int(round(0.08 * fps))):
            score -= 1.8
            reasons.append("penalty_post_nb_elbow_peak")

    window_span = max(1, consensus_end - consensus_start)
    window_position = float(i - consensus_start) / float(window_span)
    if 0.10 <= window_position <= 0.72:
        score += 0.9
        reasons.append("window_phase_plausible")
    elif window_position > 0.85:
        score -= 2.2
        reasons.append("penalty_window_tail")

    relative_position = float(i) / max(1.0, float(n))
    if 0.35 <= relative_position <= 0.80:
        score += 0.8
        reasons.append("temporal_plausible")
    elif relative_position > 0.85:
        score -= 3.0
        reasons.append("penalty_late_frame")

    if i > peak_i + int(max(2, round(0.08 * fps))):
        score -= 1.2
        reasons.append("penalty_far_past_peak")

    return {
        "frame_idx": i,
        "release_score": score,
        "reasons": reasons,
    }


def _consensus_score_to_confidence(score: float, *, low_wrist_visibility: bool) -> float:
    base = 0.30 + (0.06 * float(score))
    if low_wrist_visibility:
        base += 0.03
    return max(0.35, min(0.92, base))


def _release_geometry_is_late(
    *,
    i: Optional[int],
    wrist_xyz: List[Optional[tuple]],
    shoulder_xyz: List[Optional[tuple]],
) -> bool:
    if i is None or i < 0 or i >= len(wrist_xyz) or i >= len(shoulder_xyz):
        return False
    wrist = wrist_xyz[i]
    shoulder = shoulder_xyz[i]
    if not wrist or not shoulder:
        return False
    return wrist[1] > shoulder[1] + 0.12


def _select_release_by_consensus(
    *,
    n: int,
    start: int,
    end: int,
    fps: float,
    peak_i: int,
    wrist_fwd_s: np.ndarray,
    vmax: float,
    wrist_xyz: List[Optional[tuple]],
    shoulder_xyz: List[Optional[tuple]],
    elbow_xyz: List[Optional[tuple]],
    nb_elbow_peak_i: Optional[int],
    nb_elbow_available: bool,
    low_wrist_visibility: bool,
    forward: tuple,
    is_pre_followthrough,
) -> Dict[str, Any]:
    shoulder_y = np.full(n, np.nan, dtype=float)
    for idx in range(n):
        s = shoulder_xyz[idx]
        if s is not None:
            shoulder_y[idx] = float(s[1])
    if np.isfinite(shoulder_y).sum() >= 2:
        good = np.isfinite(shoulder_y)
        ids = np.arange(n)
        shoulder_y[~good] = np.interp(ids[~good], ids[good], shoulder_y[good])
        shoulder_y_s = gaussian_filter1d(shoulder_y, sigma=max(1.0, 0.03))
    else:
        shoulder_y_s = np.zeros(n, dtype=float)

    candidate_ids = set(range(max(0, start), min(n - 1, end) + 1))
    candidate_ids.update(range(max(0, peak_i - 3), min(n - 1, peak_i + 4)))
    if nb_elbow_available and nb_elbow_peak_i is not None:
        candidate_ids.update(range(max(0, nb_elbow_peak_i - 2), min(n - 1, nb_elbow_peak_i + 3)))
    shoulder_peak_i = None
    if end >= start and len(shoulder_y_s) == n:
        shoulder_segment = shoulder_y_s[start : end + 1]
        if shoulder_segment.size:
            shoulder_peak_i = int(start + np.argmin(shoulder_segment))

    scored = [
        _score_release_candidate(
            i=i,
            n=n,
            fps=fps,
            peak_i=peak_i,
            consensus_start=start,
            consensus_end=end,
            wrist_fwd_s=wrist_fwd_s,
            vmax=vmax,
            wrist_xyz=wrist_xyz,
            shoulder_xyz=shoulder_xyz,
            elbow_xyz=elbow_xyz,
            nb_elbow_peak_i=nb_elbow_peak_i,
            nb_elbow_available=nb_elbow_available,
            shoulder_peak_i=shoulder_peak_i,
            shoulder_y_s=shoulder_y_s,
            is_pre_followthrough=is_pre_followthrough,
            low_wrist_visibility=low_wrist_visibility,
            forward=forward,
        )
        for i in sorted(candidate_ids)
    ]
    if not scored:
        return {"frame_idx": peak_i, "release_score": 0.0, "reasons": ["no_candidates"]}
    return max(scored, key=lambda item: (item["release_score"], -abs(item["frame_idx"] - peak_i)))


def _detect_uah_for_release_frame(
    *,
    release_i: int,
    fps: float,
    theta_s: np.ndarray,
    is_pre_followthrough,
) -> Dict[str, Any]:
    uah_i = max(0, release_i - 1)
    best_score = 1e9
    uah_method = "release_minus_one_fallback"
    uah_confidence = 0.20
    uah_candidates: List[Dict[str, Any]] = []

    for i in range(release_i - 1, max(0, release_i - int(0.30 * fps)), -1):
        if not is_pre_followthrough(i):
            continue
        if 70.0 <= theta_s[i] <= 110.0:
            score = abs(theta_s[i] - 90.0)
            if score < best_score:
                best_score = score
                uah_i = i
                candidate = build_candidate(
                    frame=i,
                    method="upper_arm_horizontal",
                    confidence=max(0.35, 0.90 - min(score, 25.0) / 40.0),
                    score=max(0.0, 1.0 - (score / 25.0)),
                )
                if candidate is not None:
                    uah_candidates.append(candidate)

    if uah_i >= release_i:
        uah_i = max(0, release_i - 1)
    elif best_score < 1e9:
        uah_method = "upper_arm_horizontal"
        uah_confidence = max(0.35, 0.90 - min(best_score, 25.0) / 40.0)

    fallback_uah = build_candidate(
        frame=uah_i,
        method=uah_method,
        confidence=uah_confidence,
        score=0.0 if uah_method == "release_minus_one_fallback" else None,
    )
    if fallback_uah is not None:
        uah_candidates.append(fallback_uah)

    return {
        "frame": int(uah_i),
        "method": uah_method,
        "confidence": round(float(uah_confidence), 2),
        "candidates": compact_candidates(uah_candidates),
    }

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
        nb_peak_plausible = _nb_elbow_peak_is_plausible(
            nb_elbow_peak_i=nb_elbow_peak_i,
            wrist_peak_i=peak_i,
            total_frames=n,
            fps=fps,
        )
        if not nb_peak_plausible:
            logger.info(
                f"[Release/UAH] Ignoring early/edge non-bowling elbow peak at frame {nb_elbow_peak_i} "
                f"(wrist_peak={peak_i}, n={n})"
            )
    else:
        nb_peak_plausible = False

    if nb_elbow_available and nb_elbow_peak_i is not None and low_wrist_visibility and nb_peak_plausible:
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
    release_candidates: List[Dict[str, Any]] = []
    corrected_followthrough = False

    # ============================================================
    # TIER 1: Use most reliable signal available
    # ============================================================

    # TIER 1A: Non-bowling elbow peak (best for behind-camera)
    if nb_elbow_available and nb_elbow_peak_i is not None and (
        not low_wrist_visibility or nb_peak_plausible
    ):
        if start <= nb_elbow_peak_i <= end:
            candidate = build_candidate(
                frame=nb_elbow_peak_i,
                method="nb_elbow_peak",
                confidence=0.90,
                score=1.0,
            )
            if candidate is not None:
                release_candidates.append(candidate)
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
                        candidate = build_candidate(
                            frame=i,
                            method="wrist_at_shoulder",
                            confidence=0.85,
                            score=float(wrist_fwd_s[i] / max(vmax, 1e-6)),
                        )
                        if candidate is not None:
                            release_candidates.append(candidate)
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
                                    candidate = build_candidate(
                                        frame=i,
                                        method="velocity_drop_20pct",
                                        confidence=0.75,
                                        score=float(wrist_fwd_s[i] / max(vmax, 1e-6)),
                                    )
                                    if candidate is not None:
                                        release_candidates.append(candidate)
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
        if nb_elbow_available and nb_elbow_peak_i is not None and (
            not low_wrist_visibility or nb_peak_plausible
        ):
            dist = abs(nb_elbow_peak_i - anchor_i)
            if dist <= int(0.10 * fps):  # Within 100ms of anchor
                candidate = build_candidate(
                    frame=nb_elbow_peak_i,
                    method="nb_elbow_nearby",
                    confidence=0.80,
                    score=max(0.0, 1.0 - (dist / max(1.0, 0.10 * fps))),
                )
                if candidate is not None:
                    release_candidates.append(candidate)
                    release_i = nb_elbow_peak_i
                    selection_method = "nb_elbow_nearby"
                    selection_confidence = 0.80
                    logger.info(f"[Release/UAH] Tier 2A: nb_elbow nearby at frame {release_i}")
        
        # Use velocity peak + empirical offset
        if release_i is None:
            # Empirical: release is typically 1-3 frames after velocity peak
            offset_frames = max(1, int(round(2.0 * fps / 30.0)))
            fallback_frame = min(end, peak_i + offset_frames)
            candidate = build_candidate(
                frame=fallback_frame,
                method="peak_plus_offset",
                confidence=0.60,
                score=float(wrist_fwd_s[min(fallback_frame, n - 1)] / max(vmax, 1e-6)),
            )
            if candidate is not None:
                release_candidates.append(candidate)
                release_i = fallback_frame
                selection_method = "peak_plus_offset"
                selection_confidence = 0.60
                logger.info(f"[Release/UAH] Tier 2B: velocity peak + {offset_frames} = frame {release_i}")

    # ============================================================
    # TIER 3: Absolute fallback
    # ============================================================

    if release_i is None:
        candidate = build_candidate(
            frame=start,
            method="window_start",
            confidence=0.40,
            score=0.0,
        )
        if candidate is not None:
            release_candidates.append(candidate)
            release_i = start
            selection_method = "window_start"
            selection_confidence = 0.40
            logger.warning(f"[Release/UAH] Tier 3: fallback to window start {release_i}")

    # ============================================================
    # SECOND SLICE: consensus rescoring and proximal fallback
    # ============================================================

    weak_methods = {"peak_plus_offset", "window_start"}
    consensus_start = max(
        0,
        min(
            start,
            peak_i - int(max(2, round(0.12 * fps))),
            (nb_elbow_peak_i - int(max(1, round(0.04 * fps)))) if nb_elbow_peak_i is not None else start,
        ),
    )
    consensus_end = min(
        n - 1,
        max(
            end,
            peak_i + int(max(2, round(0.06 * fps))),
            (nb_elbow_peak_i + int(max(1, round(0.04 * fps)))) if nb_elbow_peak_i is not None else end,
        ),
    )
    current_pick = (
        _score_release_candidate(
            i=release_i,
            n=n,
            fps=fps,
            peak_i=peak_i,
            consensus_start=consensus_start,
            consensus_end=consensus_end,
            wrist_fwd_s=wrist_fwd_s,
            vmax=vmax,
            wrist_xyz=wrist_xyz,
            shoulder_xyz=shoulder_xyz,
            elbow_xyz=elbow_xyz,
            nb_elbow_peak_i=nb_elbow_peak_i,
            nb_elbow_available=nb_elbow_available,
            shoulder_peak_i=None,
            shoulder_y_s=np.zeros(n, dtype=float),
            is_pre_followthrough=is_pre_followthrough,
            low_wrist_visibility=low_wrist_visibility,
            forward=forward,
        )
        if release_i is not None
        else {"frame_idx": anchor_i, "release_score": -999.0, "reasons": ["no_initial_release"]}
    )
    consensus_pick = _select_release_by_consensus(
        n=n,
        start=consensus_start,
        end=consensus_end,
        fps=fps,
        peak_i=peak_i,
        wrist_fwd_s=wrist_fwd_s,
        vmax=vmax,
        wrist_xyz=wrist_xyz,
        shoulder_xyz=shoulder_xyz,
        elbow_xyz=elbow_xyz,
        nb_elbow_peak_i=nb_elbow_peak_i,
        nb_elbow_available=nb_elbow_available,
        low_wrist_visibility=low_wrist_visibility,
        forward=forward,
        is_pre_followthrough=is_pre_followthrough,
    )
    consensus_confidence = _consensus_score_to_confidence(
        consensus_pick["release_score"],
        low_wrist_visibility=low_wrist_visibility,
    )
    consensus_method = (
        "proximal_consensus"
        if low_wrist_visibility or "nb_elbow_peak" in consensus_pick["reasons"] or "shoulder_peak" in consensus_pick["reasons"]
        else "consensus_rescore"
    )
    consensus_candidate = build_candidate(
        frame=consensus_pick["frame_idx"],
        method=consensus_method,
        confidence=consensus_confidence,
        score=float(consensus_pick["release_score"]),
        reason=",".join(consensus_pick["reasons"][:3]),
    )
    if consensus_candidate is not None:
        release_candidates.append(consensus_candidate)

    current_method = str(selection_method or "").strip().lower()
    should_rescore = (
        release_i is None
        or current_method in weak_methods
        or low_wrist_visibility
        or _release_geometry_is_late(i=release_i, wrist_xyz=wrist_xyz, shoulder_xyz=shoulder_xyz)
        or release_i > peak_i + 2
        or current_pick["release_score"] < 4.5
    )
    consensus_is_better = (
        consensus_pick["release_score"] >= current_pick["release_score"] + 0.75
        or (
            consensus_pick["release_score"] >= current_pick["release_score"] + 0.10
            and consensus_pick["frame_idx"] <= release_i
        )
    )
    if (
        not consensus_is_better
        and current_method in weak_methods
        and selection_confidence <= 0.60
        and consensus_pick["frame_idx"] != release_i
        and consensus_pick["release_score"] >= 5.0
    ):
        consensus_is_better = True
    if should_rescore and consensus_is_better:
        release_i = int(consensus_pick["frame_idx"])
        selection_method = consensus_method
        selection_confidence = consensus_confidence
        logger.info(
            "[Release/UAH] Consensus rescore adopted frame %s score=%.2f reasons=%s",
            release_i,
            consensus_pick["release_score"],
            ",".join(consensus_pick["reasons"]) or "-",
        )

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
                        candidate = build_candidate(
                            frame=i,
                            method=selection_method,
                            confidence=selection_confidence,
                            score=0.5,
                            reason="followthrough_correction",
                        )
                        if candidate is not None:
                            release_candidates.append(candidate)
                        logger.info(f"[Release/UAH] Corrected to earlier frame {release_i}")
                        corrected_followthrough = True
                        break

    release_assessment = _build_release_assessment(
        release_i=release_i,
        peak_i=peak_i,
        selection_method=selection_method,
        selection_confidence=selection_confidence,
        wrist_xyz=wrist_xyz,
        shoulder_xyz=shoulder_xyz,
        low_wrist_visibility=low_wrist_visibility,
        corrected_followthrough=corrected_followthrough,
    )

    # ------------------------------------------------------------
    # UAH (geometric, BEFORE release)
    # ------------------------------------------------------------
    theta = np.zeros(n, dtype=float)
    for i in range(n):
        ang = _upper_arm_angle(shoulder_xyz[i], elbow_xyz[i], forward)
        theta[i] = float(ang) if ang is not None else 0.0
    theta_s = gaussian_filter1d(theta, sigma=max(1.2, 0.03 * fps))
    uah_result = _detect_uah_for_release_frame(
        release_i=release_i,
        fps=fps,
        theta_s=theta_s,
        is_pre_followthrough=is_pre_followthrough,
    )
    uah_i = int(uah_result["frame"])
    uah_method = str(uah_result["method"])
    uah_confidence = float(uah_result["confidence"])
    uah_candidates = list(uah_result["candidates"])

    # ------------------------------------------------------------
    # THIRD SLICE: joint event-chain inference
    # ------------------------------------------------------------
    selected_foot_events: Dict[str, Any] = {}
    selected_chain = chain_quality(
        bfc_frame=None,
        ffc_frame=None,
        uah_frame=int(frames[uah_i]),
        release_frame=int(frames[release_i]),
        uah_confidence=uah_confidence,
        release_confidence=selection_confidence,
    )
    try:
        from app.workers.events.ffc_bfc import detect_ffc_bfc

        best_joint_score = -1e9
        best_joint_payload: Optional[Dict[str, Any]] = None
        for candidate in compact_candidates(release_candidates, limit=4):
            candidate_release_i = int(candidate.get("frame"))
            candidate_method = str(candidate.get("method") or selection_method or "candidate")
            candidate_conf = float(candidate.get("confidence") or selection_confidence or 0.0)
            candidate_assessment = _build_release_assessment(
                release_i=candidate_release_i,
                peak_i=peak_i,
                selection_method=candidate_method,
                selection_confidence=candidate_conf,
                wrist_xyz=wrist_xyz,
                shoulder_xyz=shoulder_xyz,
                low_wrist_visibility=low_wrist_visibility,
                corrected_followthrough=corrected_followthrough and candidate_release_i == release_i,
            )
            candidate_uah = _detect_uah_for_release_frame(
                release_i=candidate_release_i,
                fps=fps,
                theta_s=theta_s,
                is_pre_followthrough=is_pre_followthrough,
            )
            foot_events = detect_ffc_bfc(
                pose_frames=pose_frames,
                hand=hand,
                release_frame=int(frames[candidate_release_i]),
                delivery_window=(int(frames[start]), int(frames[end])),
                fps=fps,
            ) or {}
            ffc_evt = foot_events.get("ffc") or {}
            bfc_evt = foot_events.get("bfc") or {}
            candidate_chain = chain_quality(
                bfc_frame=bfc_evt.get("frame"),
                ffc_frame=ffc_evt.get("frame"),
                uah_frame=candidate_uah.get("frame"),
                release_frame=int(frames[candidate_release_i]),
                bfc_confidence=float(bfc_evt.get("confidence") or 0.0),
                ffc_confidence=float(ffc_evt.get("confidence") or 0.0),
                uah_confidence=float(candidate_uah.get("confidence") or 0.0),
                release_confidence=candidate_conf,
            )
            joint_score = (
                0.42 * float(candidate_assessment.get("score") or 0.0)
                + 0.28 * float(candidate_chain.get("quality") or 0.0)
                + 0.15 * float(candidate_uah.get("confidence") or 0.0)
                + 0.15 * min(
                    float(ffc_evt.get("confidence") or 0.0),
                    float(bfc_evt.get("confidence") or 0.0),
                )
            )
            if str(candidate_assessment.get("state")) == "INVALID":
                joint_score -= 0.30
            if not bool(candidate_chain.get("ordered")):
                joint_score -= 0.35
            if str(ffc_evt.get("method") or "").endswith("fallback"):
                joint_score -= 0.04
            if str(bfc_evt.get("method") or "").endswith("fallback"):
                joint_score -= 0.04
            if int(candidate_uah.get("frame") or 0) >= int(frames[candidate_release_i]):
                joint_score -= 0.20

            if joint_score > best_joint_score:
                best_joint_score = joint_score
                best_joint_payload = {
                    "release_i": candidate_release_i,
                    "release_method": candidate_method,
                    "release_confidence": candidate_conf,
                    "release_assessment": candidate_assessment,
                    "uah": candidate_uah,
                    "foot_events": foot_events,
                    "chain": candidate_chain,
                    "score": joint_score,
                }

        if best_joint_payload is not None:
            current_joint_score = (
                0.42 * float(release_assessment.get("score") or 0.0)
                + 0.28 * float(selected_chain.get("quality") or 0.0)
                + 0.15 * float(uah_confidence or 0.0)
            )
            if best_joint_payload["score"] >= current_joint_score + 0.05:
                release_i = int(best_joint_payload["release_i"])
                selection_method = str(best_joint_payload["release_method"])
                selection_confidence = float(best_joint_payload["release_confidence"])
                release_assessment = dict(best_joint_payload["release_assessment"])
                uah_i = int(best_joint_payload["uah"]["frame"])
                uah_method = str(best_joint_payload["uah"]["method"])
                uah_confidence = float(best_joint_payload["uah"]["confidence"])
                uah_candidates = list(best_joint_payload["uah"]["candidates"])
                selected_foot_events = dict(best_joint_payload["foot_events"])
                selected_chain = dict(best_joint_payload["chain"])
                logger.info(
                    "[Release/UAH] Joint chain selected frame %s method=%s joint_score=%.2f",
                    release_i,
                    selection_method,
                    best_joint_payload["score"],
                )
    except Exception as exc:
        logger.warning("[Release/UAH] joint_chain_inference_failed error=%s", exc)

    logger.info(
        f"[Release/UAH] FINAL: frame={release_i} method={selection_method} "
        f"confidence={selection_confidence:.2f} state={release_assessment['state']} "
        f"score={release_assessment['score']:.2f} notes={','.join(release_assessment['notes']) or '-'}"
    )

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
    result = {
        "release": {
            "frame": int(frames[release_i]),
            "method": selection_method,
            "confidence": round(float(selection_confidence), 2),
            "candidates": compact_candidates(release_candidates),
            "window": [int(frames[start]), int(frames[end])],
            "state": release_assessment["state"],
            "score": release_assessment["score"],
            "notes": release_assessment["notes"],
        },
        "uah": {
            "frame": int(frames[uah_i]),
            "method": uah_method,
            "confidence": round(float(uah_confidence), 2),
            "candidates": compact_candidates(uah_candidates),
            "window": [int(frames[max(0, release_i - int(0.30 * fps))]), int(frames[max(0, release_i - 1)])],
        },

        # Non-breaking additions:
        # delivery_window is the exact window used for release selection,
        # exposed so downstream components (FFC/BFC visuals) never invent timing.
        "delivery_window": [int(frames[start]), int(frames[end])],

        # Useful for debugging / consumers (optional)
        "peak": {"frame": int(frames[peak_i])},
    }
    if selected_foot_events:
        result.update(selected_foot_events)
        result["event_chain"] = selected_chain
    return result
