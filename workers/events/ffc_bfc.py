# app/workers/events/ffc_bfc.py
"""
PHASE-1 INVARIANTS:

1. Pelvis kinematics define WHEN braking begins.
2. Geometry defines WHEN contact actually occurs.
3. FFC = first frame >= pelvis_on where BOTH feet are grounded.
4. Kinematics and geometry never compete on the same frame.
5. If uncertain, return a conservative fallback with low confidence.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from app.common.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# MediaPipe landmark indices (LOCKED)
# ------------------------------------------------------------
LS, RS = 11, 12
LH, RH = 23, 24
LA, RA = 27, 28
LFI, RFI = 31, 32

MIN_VIS = 0.25
EPS = 1e-9

# ------------------------------------------------------------
# Named intent weights (NOT magic numbers)
# ------------------------------------------------------------
# Hierarchy: rotation exists > braking begins > dominance refines confidence
W_ROT_VEL   = 1.0
W_ROT_DECEL = 0.7
W_DOM_RATIO = 0.4

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs for numerical stability."""
    x = x.astype(float)
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 3:
        return x
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, w, mode="same")


def _robust_percentile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, p))


def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1e9
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) + 1e-9)


def _vis(lm: list, idx: int) -> float:
    try:
        if idx >= len(lm) or lm[idx] is None:
            return 0.0
        return float(lm[idx].get("visibility", 0.0))
    except Exception:
        return 0.0


def _xy(lm: list, idx: int):
    try:
        if idx >= len(lm) or lm[idx] is None:
            return None
        x = lm[idx].get("x", None)
        y = lm[idx].get("y", None)
        if x is None or y is None:
            return None
        return float(x), float(y)
    except Exception:
        return None


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_of_vec(v):
    return math.atan2(v[1], v[0])


def _safe_angle(lm: list, i1: int, i2: int) -> float:
    p1 = _xy(lm, i1)
    p2 = _xy(lm, i2)
    if p1 is None or p2 is None:
        return float("nan")
    return _angle_of_vec((p2[0] - p1[0], p2[1] - p1[1]))


def _pelvis_xy(lm: list):
    return _midpoint(_xy(lm, LH), _xy(lm, RH))


def _series_y(pose_frames: List[Dict], idx: int) -> np.ndarray:
    y = np.full(len(pose_frames), np.nan, dtype=float)
    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list):
            continue
        p = _xy(lm, idx)
        if p is not None and np.isfinite(p[1]):
            y[i] = p[1]
    return _interp_nans(y)


def _is_foot_grounded(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
) -> bool:
    """
    Geometry-only groundedness gate.
    BOTH stability + near-low tests must pass.
    """
    n = len(y_ank)
    if t < 0 or t + hold >= n:
        return False

    a_seg = y_ank[t : t + hold]
    t_seg = y_toe[t : t + hold]
    if not np.all(np.isfinite(a_seg)) or not np.all(np.isfinite(t_seg)):
        return False

    w0 = max(win0, t - max(hold * 3, 6))
    w1 = min(win1, t + max(hold, 3))
    hist_a = y_ank[w0 : w1 + 1]
    hist_t = y_toe[w0 : w1 + 1]
    if not np.any(np.isfinite(hist_a)) or not np.any(np.isfinite(hist_t)):
        return False

    # Near-low (y increases downward)
    low_a = _robust_percentile(hist_a, 85)
    low_t = _robust_percentile(hist_t, 85)
    near_low = (np.median(a_seg) >= low_a - 0.01) and (np.median(t_seg) >= low_t - 0.01)

    # Stability
    rng_a = max(_robust_percentile(hist_a, 90) - _robust_percentile(hist_a, 10), 1e-6)
    rng_t = max(_robust_percentile(hist_t, 90) - _robust_percentile(hist_t, 10), 1e-6)

    dy_a = np.diff(a_seg) / max(dt, 1e-6)
    dy_t = np.diff(t_seg) / max(dt, 1e-6)

    v_ok = (np.median(np.abs(dy_a)) <= 0.15 * (rng_a / max(dt, 1e-6))) and \
           (np.median(np.abs(dy_t)) <= 0.15 * (rng_t / max(dt, 1e-6)))

    jit_ok = (_robust_mad(a_seg) <= 0.12 * rng_a) and (_robust_mad(t_seg) <= 0.12 * rng_t)

    return bool(near_low and v_ok and jit_ok)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def detect_ffc_bfc(
    pose_frames: List[Dict],
    hand: str,
    release_frame: int,
    delivery_window: Tuple[int, int],  # ignored for FFC windowing
    fps: Optional[float] = None,
    **_ignored,
) -> Dict[str, Dict]:

    n = len(pose_frames)
    if n < 10:
        logger.warning("[FFC/BFC] Too few frames")
        return {}

    rel = _as_int(release_frame)
    if rel is None:
        logger.error("[FFC/BFC] Missing release frame")
        return {}

    dt = 1.0 / float(fps) if (fps and float(fps) > 1e-3) else 1.0

    # ------------------------------------------------------------
    # Window upstream of release
    # ------------------------------------------------------------
    if fps and fps > 1.0:
        lookback = int(round(0.9 * fps))      # ~900 ms
        hold = max(3, int(round(0.05 * fps))) # ~50 ms
        smooth_k = max(3, int(round(0.02 * fps)))
    else:
        lookback = int(round(0.25 * n))
        hold = 4
        smooth_k = 5

    win_start = _clamp(rel - lookback, 0, n - 1)
    win_end   = _clamp(rel - 2, win_start, n - 1)

    logger.info(
        f"[FFC/BFC][WINDOW] derived=[{win_start}..{win_end}] release={rel} fps={fps}"
    )

    if win_end <= win_start + 2:
        logger.warning("[FFC/BFC] Degenerate window")
        return {}

    # ------------------------------------------------------------
    # Pelvis signals (interp → smooth → diff)
    # ------------------------------------------------------------
    px = np.full(n, np.nan)
    py = np.full(n, np.nan)
    hip_ang = np.full(n, np.nan)
    vis_ok = np.zeros(n, dtype=bool)

    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list):
            continue
        pel = _pelvis_xy(lm)
        if pel is not None:
            px[i], py[i] = pel
        hip_ang[i] = _safe_angle(lm, LH, RH)
        if _vis(lm, LH) >= MIN_VIS and _vis(lm, RH) >= MIN_VIS:
            vis_ok[i] = True

    px = _interp_nans(px)
    py = _interp_nans(py)
    hip_ang = _interp_nans(hip_ang)

    v_lin = np.sqrt(np.diff(px, prepend=px[0])**2 + np.diff(py, prepend=py[0])**2) / max(dt, 1e-6)
    hip_u = np.unwrap(hip_ang)
    w_rot = np.abs(np.diff(hip_u, prepend=hip_u[0])) / max(dt, 1e-6)

    v_lin = _moving_average(v_lin, smooth_k)
    w_rot = _moving_average(w_rot, smooth_k)

    a_lin = np.diff(v_lin, prepend=v_lin[0]) / max(dt, 1e-6)
    a_rot = np.diff(w_rot, prepend=w_rot[0]) / max(dt, 1e-6)

    R = w_rot / (v_lin + EPS)

    # ------------------------------------------------------------
    # Pelvis activity onset (kinematics)
    # ------------------------------------------------------------
    R_on = _robust_percentile(R[win_start:win_end+1], 70)
    pelvis_on = None

    for i in range(win_end, win_start, -1):
        if vis_ok[i] and R[i] > R_on:
            pelvis_on = i
        elif pelvis_on is not None:
            break

    logger.info(f"[FFC/BFC][PELVIS_ON] idx={pelvis_on}")

    if pelvis_on is None:
        logger.warning("[FFC/BFC] Pelvis never activated")
        pelvis_on = win_start

    # ------------------------------------------------------------
    # Geometry forward lock: FIRST frame where BOTH feet grounded
    # ------------------------------------------------------------
    y_LA  = _series_y(pose_frames, LA)
    y_RA  = _series_y(pose_frames, RA)
    y_LFI = _series_y(pose_frames, LFI)
    y_RFI = _series_y(pose_frames, RFI)

    ffc = None
    for i in range(pelvis_on, win_end):
        left  = _is_foot_grounded(y_LA,  y_LFI,  i, hold, win_start, win_end, dt)
        right = _is_foot_grounded(y_RA,  y_RFI,  i, hold, win_start, win_end, dt)
        if left and right:
            ffc = i
            break

    # ------------------------------------------------------------
    # Fallback ladder (never return empty)
    # ------------------------------------------------------------
    if ffc is None:
        # 1) geometry-only fallback
        for i in range(win_start, win_end):
            left  = _is_foot_grounded(y_LA,  y_LFI,  i, hold, win_start, win_end, dt)
            right = _is_foot_grounded(y_RA,  y_RFI,  i, hold, win_start, win_end, dt)
            if left and right:
                ffc = i
                logger.warning(f"[FFC/BFC][FALLBACK] geometry_only frame={ffc}")
                return {
                    "ffc": {"frame": int(ffc), "confidence": 0.25, "method": "geometry_only_fallback"}
                }

        # 2) late-window fallback
        ffc = win_start + int(0.75 * (win_end - win_start))
        logger.warning(f"[FFC/BFC][FALLBACK] late_window frame={ffc}")
        return {
            "ffc": {"frame": int(ffc), "confidence": 0.20, "method": "late_window_fallback"}
        }

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------
    logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")

    bfc = _clamp(ffc - max(3, hold), win_start, ffc)

    return {
        "ffc": {"frame": int(ffc), "confidence": 0.62, "method": "pelvis_then_geometry"},
        "bfc": {"frame": int(bfc), "confidence": 0.35, "method": "context_pre_ffc"},
    }

