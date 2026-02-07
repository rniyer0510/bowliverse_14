# app/workers/events/ffc_bfc.py
"""
PHASE-1 INVARIANTS (UPDATED FOR REAL POSE DATA):

1. Pelvis kinematics define WHEN braking begins.
2. Geometry defines WHEN contact actually occurs.
3. FFC = first frame >= pelvis_on where FRONT foot is grounded AND back foot is grounded OR recently grounded.
   (Back foot can unload during braking; strict BOTH-feet-flat is too brittle with pose jitter.)
4. Kinematics and geometry never compete on the same frame.
5. If uncertain, return a conservative fallback with low confidence (never return {}).
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
    if good.sum() < 2:  # Need at least 2 points for interp
        return np.full_like(x, np.nan)
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


# ------------------------------------------------------------
# Geometry: grounded scoring (robust to pose noise)
# ------------------------------------------------------------
def _foot_ground_score(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
) -> int:
    """
    Return grounded score in {0,1,2,3} using three cues:
      1) near-low position
      2) low vertical velocity
      3) low jitter (MAD)
    We accept grounded if score >= 2 (NOT all 3), to tolerate pose noise.
    """
    n = len(y_ank)
    if t < 0 or t + hold >= n:
        return 0

    a_seg = y_ank[t : t + hold]
    t_seg = y_toe[t : t + hold]
    if not np.all(np.isfinite(a_seg)) or not np.all(np.isfinite(t_seg)):
        return 0

    w0 = max(win0, t - max(hold * 3, 6))
    w1 = min(win1, t + max(hold, 3))
    hist_a = y_ank[w0 : w1 + 1]
    hist_t = y_toe[w0 : w1 + 1]
    if not np.any(np.isfinite(hist_a)) or not np.any(np.isfinite(hist_t)):
        return 0

    # 1) Near-low (y increases downward)
    # Use 85th percentile as "near-ground" reference; small slack for normalization noise.
    low_a = _robust_percentile(hist_a, 85)
    low_t = _robust_percentile(hist_t, 85)
    near_low = (np.median(a_seg) >= low_a - 0.01) and (np.median(t_seg) >= low_t - 0.01)

    # 2) Low vertical velocity (but avoid over-tight gating)
    rng_a = max(_robust_percentile(hist_a, 90) - _robust_percentile(hist_a, 10), 1e-6)
    rng_t = max(_robust_percentile(hist_t, 90) - _robust_percentile(hist_t, 10), 1e-6)

    dy_a = np.diff(a_seg) / max(dt, 1e-6)
    dy_t = np.diff(t_seg) / max(dt, 1e-6)

    v_ok = (np.median(np.abs(dy_a)) <= 0.18 * (rng_a / max(dt, 1e-6))) and \
           (np.median(np.abs(dy_t)) <= 0.18 * (rng_t / max(dt, 1e-6)))

    # 3) Low jitter (MAD)
    jit_ok = (_robust_mad(a_seg) <= 0.15 * rng_a) and (_robust_mad(t_seg) <= 0.15 * rng_t)

    score = 0
    score += 1 if near_low else 0
    score += 1 if v_ok else 0
    score += 1 if jit_ok else 0
    return score


def _is_grounded(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
) -> bool:
    return _foot_ground_score(y_ank, y_toe, t, hold, win0, win1, dt) >= 2


def _recently_grounded(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
    lookback: int,
) -> bool:
    """
    True if foot is grounded in any of the previous `lookback` frames.
    Used to allow natural unloading of back foot during front-foot braking.
    """
    j0 = max(win0, t - lookback)
    for j in range(t - 1, j0 - 1, -1):
        if _is_grounded(y_ank, y_toe, j, hold, win0, win1, dt):
            return True
    return False


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

    # Robust FPS handling
    try:
        fps_f = float(fps) if fps else 0.0
    except Exception:
        fps_f = 0.0
    if fps_f <= 1e-3:
        fps_f = 30.0
    fps_f = max(1.0, fps_f)
    dt = 1.0 / fps_f

    # ------------------------------------------------------------
    # Window upstream of release
    # ------------------------------------------------------------
    lookback = int(round(0.9 * fps_f))       # ~900 ms
    hold = max(3, int(round(0.05 * fps_f)))  # ~50 ms
    smooth_k = max(3, int(round(0.02 * fps_f)))

    win_start = _clamp(rel - lookback, 0, n - 1)
    win_end   = _clamp(rel - 2, win_start, n - 1)

    logger.info(f"[FFC/BFC][WINDOW] derived=[{win_start}..{win_end}] release={rel} fps={fps_f:.2f}")

    if win_end <= win_start + 2:
        logger.warning("[FFC/BFC] Degenerate window")
        return {"ffc": {"frame": win_start, "confidence": 0.15, "method": "degenerate_window"}}

    # ------------------------------------------------------------
    # Pelvis signals (interp → smooth → diff)
    # ------------------------------------------------------------
    px = np.full(n, np.nan)
    py = np.full(n, np.nan)
    hip_ang = np.full(n, np.nan)
    vis_ok = np.zeros(n, dtype=bool)

    valid_pelvis_count = 0
    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list) or len(lm) == 0:
            continue
        pel = _pelvis_xy(lm)
        if pel is not None and np.isfinite(pel[0]) and np.isfinite(pel[1]):
            px[i], py[i] = pel
            valid_pelvis_count += 1
        hip_ang[i] = _safe_angle(lm, LH, RH)
        if _vis(lm, LH) >= MIN_VIS and _vis(lm, RH) >= MIN_VIS:
            vis_ok[i] = True

    if valid_pelvis_count < max(hold, 6):
        logger.warning("[FFC/BFC] Insufficient valid pelvis landmarks")
        return {"ffc": {"frame": win_start, "confidence": 0.10, "method": "insufficient_landmarks"}}

    px = _interp_nans(px)
    py = _interp_nans(py)
    hip_ang = _interp_nans(hip_ang)

    # speeds
    v_lin = np.sqrt(np.diff(px, prepend=px[0])**2 + np.diff(py, prepend=py[0])**2) / max(dt, 1e-6)
    hip_u = np.unwrap(hip_ang)
    w_rot = np.abs(np.diff(hip_u, prepend=hip_u[0])) / max(dt, 1e-6)

    v_lin = _moving_average(v_lin, smooth_k)
    w_rot = _moving_average(w_rot, smooth_k)

    R = w_rot / (v_lin + EPS)

    # ------------------------------------------------------------
    # Pelvis activity onset (kinematics)
    # ------------------------------------------------------------
    R_on = _robust_percentile(R[win_start:win_end + 1], 70)
    pelvis_on = None

    # Backward scan: find last region where R exceeds threshold and hips readable
    for i in range(win_end, win_start, -1):
        if vis_ok[i] and R[i] > R_on:
            pelvis_on = i
        elif pelvis_on is not None:
            break

    if pelvis_on is None:
        logger.warning("[FFC/BFC] Pelvis never activated; using win_start")
        pelvis_on = win_start

    logger.info(f"[FFC/BFC][PELVIS_ON] idx={pelvis_on}")

    # ------------------------------------------------------------
    # Geometry forward lock (RELAXED): front grounded + back grounded OR recently grounded
    # ------------------------------------------------------------
    y_LA  = _series_y(pose_frames, LA)
    y_RA  = _series_y(pose_frames, RA)
    y_LFI = _series_y(pose_frames, LFI)
    y_RFI = _series_y(pose_frames, RFI)

    # If ankle data is missing, fall back conservatively to pelvis_on (low conf)
    if (not np.any(np.isfinite(y_LA)) and not np.any(np.isfinite(y_RA))) or (not np.any(np.isfinite(y_LFI)) and not np.any(np.isfinite(y_RFI))):
        logger.warning("[FFC/BFC] No valid foot landmarks; pelvis fallback")
        ffc = pelvis_on
        bfc = _clamp(ffc - max(3, hold), win_start, ffc)
        logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
        return {
            "ffc": {"frame": int(ffc), "confidence": 0.20, "method": "no_foot_data_fallback"},
            "bfc": {"frame": int(bfc), "confidence": 0.20, "method": "no_foot_data_fallback"},
        }

    # Determine which side is front-foot at FFC? We can't know from hand reliably here (camera dependent).
    # So we test both possibilities and pick the earliest frame that satisfies either:
    #   (L is front and R is back) OR (R is front and L is back)
    back_recent = max(2, int(round(0.03 * fps_f)))  # ~30ms lookback
    ffc = None

    for i in range(pelvis_on, win_end - hold):
        Lg = _is_grounded(y_LA, y_LFI, i, hold, win_start, win_end, dt)
        Rg = _is_grounded(y_RA, y_RFI, i, hold, win_start, win_end, dt)

        # Case A: Left is front, Right is back (allow recent grounding for back)
        okA = Lg and (Rg or _recently_grounded(y_RA, y_RFI, i, hold, win_start, win_end, dt, back_recent))

        # Case B: Right is front, Left is back
        okB = Rg and (Lg or _recently_grounded(y_LA, y_LFI, i, hold, win_start, win_end, dt, back_recent))

        if okA or okB:
            ffc = i
            break

    # ------------------------------------------------------------
    # Fallback ladder (never return empty)
    # ------------------------------------------------------------
    if ffc is None:
        # 1) Relax further: any single grounded foot after pelvis_on
        for i in range(pelvis_on, win_end - hold):
            if _is_grounded(y_LA, y_LFI, i, hold, win_start, win_end, dt) or _is_grounded(y_RA, y_RFI, i, hold, win_start, win_end, dt):
                ffc = i
                logger.warning(f"[FFC/BFC][FALLBACK] single_foot frame={ffc}")
                bfc = _clamp(ffc - max(3, hold), win_start, ffc)
                logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
                return {
                    "ffc": {"frame": int(ffc), "confidence": 0.22, "method": "single_foot_fallback"},
                    "bfc": {"frame": int(bfc), "confidence": 0.22, "method": "single_foot_fallback"},
                }

        # 2) Ultimate fallback: 3/4 into pelvis->release window (must obey win_end fence)
        ffc = pelvis_on + int(0.75 * (win_end - pelvis_on))
        ffc = _clamp(ffc, pelvis_on, win_end)
        logger.warning(f"[FFC/BFC][FALLBACK] ultimate_3quarter frame={ffc}")

        bfc = _clamp(ffc - max(3, hold), win_start, ffc)
        logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
        return {
            "ffc": {"frame": int(ffc), "confidence": 0.15, "method": "ultimate_fallback"},
            "bfc": {"frame": int(bfc), "confidence": 0.15, "method": "ultimate_fallback"},
        }

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------
    logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")

    bfc = _clamp(ffc - max(3, hold), win_start, ffc)

    return {
        "ffc": {"frame": int(ffc), "confidence": 0.62, "method": "pelvis_then_geometry_relaxed"},
        "bfc": {"frame": int(bfc), "confidence": 0.35, "method": "context_pre_ffc"},
    }

