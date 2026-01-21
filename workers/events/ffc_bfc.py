from typing import Dict, List, Optional, Tuple
import numpy as np

from app.common.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# MediaPipe landmark indices (LOCKED)
# ------------------------------------------------------------
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ------------------------------------------------------------
# Tunable biomechanical constants (TIME-BASED)
# ------------------------------------------------------------
FFC_LOOKBACK_MS = 300        # how far BEFORE release to search
FFC_LOOKAHEAD_MS = 30        # stop slightly before release
STABILIZATION_MS = 100       # foot must remain stable this long

VERTICAL_JITTER_EPS = 0.015
HORIZONTAL_JITTER_EPS = 0.020


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def ms_to_frames(ms: float, fps: float) -> int:
    return max(1, int(round((ms / 1000.0) * fps)))


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def extract_front_foot_series(
    pose_frames: List[Dict],
    hand: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract normalized x/y series for the FRONT foot
    using MediaPipe landmark indices.
    """

    xs, ys = [], []

    foot_idx = LEFT_FOOT_INDEX if hand == "R" else RIGHT_FOOT_INDEX

    for f in pose_frames:
        lms = f.get("landmarks")

        if not isinstance(lms, list) or foot_idx >= len(lms):
            xs.append(np.nan)
            ys.append(np.nan)
            continue

        pt = lms[foot_idx]

        if pt is None:
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            xs.append(pt.get("x", np.nan))
            ys.append(pt.get("y", np.nan))

    return np.array(xs), np.array(ys)


def is_stable(
    xs: np.ndarray,
    ys: np.ndarray,
    start: int,
    end: int,
) -> bool:
    """
    Foot considered stable if positional jitter is low.
    """
    seg_x = xs[start:end + 1]
    seg_y = ys[start:end + 1]

    if np.isnan(seg_x).any() or np.isnan(seg_y).any():
        return False

    dx = np.max(seg_x) - np.min(seg_x)
    dy = np.max(seg_y) - np.min(seg_y)

    return dx < HORIZONTAL_JITTER_EPS and dy < VERTICAL_JITTER_EPS


# ------------------------------------------------------------
# Core detector
# ------------------------------------------------------------

def detect_ffc_bfc(
    pose_frames: List[Dict],
    hand: str,
    uah_frame: Optional[int] = None,      # retained for compatibility
    release_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Detect Front-Foot Contact (FFC) and Back-Foot Contact (BFC).

    RULES:
    - Release is the ONLY anchor
    - Time-based windows (FPS-aware)
    - Earliest stabilized contact wins
    """

    total_frames = len(pose_frames)

    logger.warning(
        f"[FFC/BFC][ENTRY] release_frame={release_frame}, "
        f"uah_frame={uah_frame}, total_frames={total_frames}"
    )

    if release_frame is None:
        raise ValueError(
            "detect_ffc_bfc(): release_frame must be supplied by orchestrator"
        )

    # ------------------------------------------------------------
    # FPS handling
    # ------------------------------------------------------------

    if fps is None or fps <= 0:
        fps = 25.0

    xs, ys = extract_front_foot_series(pose_frames, hand)

    # ------------------------------------------------------------
    # Define FFC window (anchored on RELEASE)
    # ------------------------------------------------------------

    lookback = ms_to_frames(FFC_LOOKBACK_MS, fps)
    lookahead = ms_to_frames(FFC_LOOKAHEAD_MS, fps)
    stab_frames = ms_to_frames(STABILIZATION_MS, fps)

    win_start = clamp(release_frame - lookback, 0, total_frames - 1)
    win_end = clamp(release_frame - lookahead, 0, total_frames - 1)

    logger.info(
        f"[FFC/BFC][WINDOW] release={release_frame} "
        f"search=[{win_start}..{win_end}] "
        f"stab={stab_frames}"
    )

    # ------------------------------------------------------------
    # STEP 1: landing candidates (downward Y motion)
    # ------------------------------------------------------------

    candidates: List[int] = []

    for f in range(win_start + 1, win_end + 1):
        if np.isnan(ys[f]) or np.isnan(ys[f - 1]):
            continue

        if ys[f] >= ys[f - 1]:
            candidates.append(f)

    # ------------------------------------------------------------
    # STEP 2: stabilization check
    # ------------------------------------------------------------

    ffc_frame = None
    ffc_conf = 0.0
    ffc_method = None

    for f in candidates:
        stab_end = clamp(f + stab_frames, 0, total_frames - 1)
        if is_stable(xs, ys, f, stab_end):
            ffc_frame = f
            ffc_conf = 0.6
            ffc_method = "stabilized_contact"
            break

    # ------------------------------------------------------------
    # STEP 3: midpoint fallback (LOW confidence)
    # ------------------------------------------------------------

    if ffc_frame is None:
        ffc_frame = (win_start + win_end) // 2
        ffc_conf = 0.0
        ffc_method = "window_midpoint_fallback"

        logger.warning(
            f"[FFC/BFC] No stabilized contact — midpoint fallback {ffc_frame}"
        )

    # ------------------------------------------------------------
    # BFC: relative earlier stabilized foot
    # ------------------------------------------------------------

    bfc_search_start = clamp(ffc_frame - lookback, 0, total_frames - 1)
    bfc_search_end = clamp(ffc_frame - 2, 0, total_frames - 1)

    bfc_frame = None
    bfc_conf = 0.0
    bfc_method = None

    for f in range(bfc_search_start, bfc_search_end + 1):
        if np.isnan(ys[f]):
            continue

        stab_end = clamp(f + stab_frames, 0, total_frames - 1)
        if is_stable(xs, ys, f, stab_end):
            bfc_frame = f
            bfc_conf = 0.5
            bfc_method = "stabilized_contact"
            break

    if bfc_frame is None:
        bfc_frame = clamp(ffc_frame - stab_frames, 0, total_frames - 1)
        bfc_conf = 0.25
        bfc_method = "relative_fallback"

    logger.info(
        f"[FFC/BFC][RESULT] anchor={release_frame}(release) "
        f"ffc={ffc_frame} ({ffc_method}, conf={ffc_conf:.2f}) "
        f"bfc={bfc_frame} ({bfc_method}, conf={bfc_conf:.2f})"
    )

    return {
        "ffc": {
            "frame": int(ffc_frame),
            "confidence": float(ffc_conf),
            "method": ffc_method,
        },
        "bfc": {
            "frame": int(bfc_frame),
            "confidence": float(bfc_conf),
            "method": bfc_method,
        },
    }

