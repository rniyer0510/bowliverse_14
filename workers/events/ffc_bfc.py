"""
ActionLab V14 — FFC + BFC detector (rigorous, contract-safe)

Contracts preserved:
- Reverse event identification
- FFC is found relative to Release (reverse search)
- BFC is anchored to FFC (NOT Release)
- Typical priors:
    FFC ≈ Release - (4..5) frames
    BFC ≈ FFC - (3..4) frames
- No "not found" allowed — always return a frame with confidence + method
- Debuggable: logs + frame dumps
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import cv2
import math

from app.common.logger import get_logger
from app.workers.pose.landmarks import get_lm, vis_ok

logger = get_logger(__name__)

DEBUG = True
DEBUG_DIR = "/tmp/actionlab_debug/ffc_bfc"
os.makedirs(DEBUG_DIR, exist_ok=True)

def _dump(video_path: Optional[str], frame_idx: int, tag: str):
    if not DEBUG or not video_path:
        return
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{tag}_{int(frame_idx)}.png"), frame)
    except Exception as e:
        logger.warning(f"[FFC/BFC] frame dump failed: {e}")

def _xy(pf, name: str) -> Optional[Tuple[float, float]]:
    lm = get_lm(pf, name)
    if not vis_ok(lm):
        return None
    try:
        return (float(lm["x"]), float(lm["y"]))
    except Exception:
        return None

def _visible_score(pf, a: str, b: str) -> int:
    return int(vis_ok(get_lm(pf, a))) + int(vis_ok(get_lm(pf, b)))

def _foot_com(pf, heel: str, toe: str) -> Optional[Tuple[float, float]]:
    h = _xy(pf, heel)
    t = _xy(pf, toe)
    if h and t:
        return ((h[0] + t[0]) * 0.5, (h[1] + t[1]) * 0.5)
    return h or t

def _stability_score(pose_frames: List[Dict[str, Any]], i: int, heel: str, toe: str, K: int) -> float:
    """
    Simple stability proxy: how little the foot COM moves over the next K frames.
    Smaller movement => more "planted".
    """
    n = len(pose_frames)
    c0 = _foot_com(pose_frames[i], heel, toe)
    if not c0:
        return 0.0
    acc = 0.0
    cnt = 0
    for j in range(i+1, min(n, i+1+K)):
        c1 = _foot_com(pose_frames[j], heel, toe)
        if not c1:
            continue
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        acc += math.sqrt(dx*dx + dy*dy)
        cnt += 1
        c0 = c1
    if cnt == 0:
        return 0.0
    # invert: lower movement => higher score
    mean_move = acc / cnt
    return float(max(0.0, 1.0 - 40.0 * mean_move))  # scale for normalized coords

def _pick_best_in_band(
    pose_frames: List[Dict[str, Any]],
    lo: int,
    hi: int,
    heel: str,
    toe: str,
    K: int
) -> Tuple[int, float, str]:
    """
    Return (idx, confidence, method).
    Confidence is explicit and degrades if we are forced outside priors.
    """
    lo = max(0, lo)
    hi = min(len(pose_frames)-1, hi)
    if hi < lo:
        lo, hi = hi, lo

    best_i = lo
    best_score = -1.0

    for i in range(lo, hi+1):
        vis = _visible_score(pose_frames[i], heel, toe)  # 0..2
        stab = _stability_score(pose_frames, i, heel, toe, K)  # 0..1
        score = (1.6 * vis) + (1.0 * stab)
        if score > best_score:
            best_score = score
            best_i = i

    # Confidence mapping
    vis_best = _visible_score(pose_frames[best_i], heel, toe)
    conf = 0.35 + 0.25 * (vis_best / 2.0) + 0.35 * min(1.0, best_score / 3.2)
    conf = float(min(0.90, max(0.25, conf)))
    method = "band_best_score"
    return best_i, conf, method

def detect_ffc_bfc(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    release_frame: Optional[int] = None,
    uah_frame: Optional[int] = None,
    fps: float = 25.0,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    VISUAL FOOT EVENTS — rigorous, contract-safe.

    Contracts:
    - reverse-detected: Release known (or fallback to UAH if needed)
    - FFC is BEFORE Release in time, but detected by searching backward relative to Release
    - BFC anchored to FFC
    - Must always return both frames with confidence + method
    """
    n = len(pose_frames)
    if n == 0:
        return {}

    fps = float(fps or 25.0)
    K = max(3, int(0.12 * fps))

    # Anchor (prefer Release)
    if release_frame is not None:
        rel = int(release_frame)
        anchor_method = "release"
    elif uah_frame is not None:
        rel = int(uah_frame)
        anchor_method = "uah_fallback"
    else:
        rel = n - 1
        anchor_method = "tail_fallback"

    rel = max(0, min(rel, n-1))

    h = (hand or "R").upper()
    if h == "R":
        front_heel, front_toe = "RIGHT_HEEL", "RIGHT_FOOT_INDEX"
        back_heel, back_toe   = "LEFT_HEEL",  "LEFT_FOOT_INDEX"
    else:
        front_heel, front_toe = "LEFT_HEEL",  "LEFT_FOOT_INDEX"
        back_heel, back_toe   = "RIGHT_HEEL", "RIGHT_FOOT_INDEX"

    # Priors (fps-adaptive)
    min_gap = max(3, int(0.10 * fps))           # "FFC must be at least 3 frames behind Release"
    d_ffc   = max(min_gap + 1, int(round(0.18 * fps)))  # ~4-5 @25fps
    d_bfc   = max(3, int(round(0.14 * fps)))            # ~3-4 @25fps
    buf     = max(3, int(round(0.12 * fps)))            # search buffer

    # -------------------------
    # FFC: search backward around Release - d_ffc
    # enforce ffc <= release - min_gap
    # -------------------------
    center_ffc = rel - d_ffc
    band_lo = max(0, center_ffc - buf)
    band_hi = min(rel - min_gap, center_ffc + buf)

    if band_hi < band_lo:
        # Ensure a valid band that respects min_gap
        band_hi = max(0, rel - min_gap)
        band_lo = max(0, band_hi - (2 * buf))

    ffc_i, ffc_conf, ffc_method = _pick_best_in_band(pose_frames, band_lo, band_hi, front_heel, front_toe, K)

    # If still violates min gap due to short clips, clamp (rigorous invariant)
    if ffc_i > rel - min_gap:
        ffc_i = max(0, rel - min_gap)
        ffc_conf = min(ffc_conf, 0.40)
        ffc_method = "clamped_to_release_min_gap"

    # If visibility is terrible in the band, widen once (still no 'not found')
    if _visible_score(pose_frames[ffc_i], front_heel, front_toe) == 0:
        wide_lo = max(0, rel - int(0.60 * fps))
        wide_hi = max(0, rel - min_gap)
        ffc_i2, c2, m2 = _pick_best_in_band(pose_frames, wide_lo, wide_hi, front_heel, front_toe, K)
        # prefer if better visibility
        if _visible_score(pose_frames[ffc_i2], front_heel, front_toe) > 0:
            ffc_i, ffc_conf, ffc_method = ffc_i2, min(0.70, c2), "widened_band_best"
        else:
            ffc_conf = min(ffc_conf, 0.45)
            ffc_method = "forced_low_signal"

    # -------------------------
    # BFC: anchored to FFC, search backward around FFC - d_bfc
    # -------------------------
    center_bfc = ffc_i - d_bfc
    b_lo = max(0, center_bfc - buf)
    b_hi = min(ffc_i - 1, center_bfc + buf)

    if b_hi < b_lo:
        b_hi = max(0, ffc_i - 1)
        b_lo = max(0, b_hi - (2 * buf))

    bfc_i, bfc_conf, bfc_method = _pick_best_in_band(pose_frames, b_lo, b_hi, back_heel, back_toe, K)

    # If adjacent due to short window, widen once
    if bfc_i >= ffc_i:
        bfc_i = max(0, ffc_i - 1)
        bfc_conf = min(bfc_conf, 0.40)
        bfc_method = "clamped_before_ffc"

    if _visible_score(pose_frames[bfc_i], back_heel, back_toe) == 0:
        wide_lo = max(0, ffc_i - int(0.60 * fps))
        wide_hi = max(0, ffc_i - 1)
        bfc_i2, c2, m2 = _pick_best_in_band(pose_frames, wide_lo, wide_hi, back_heel, back_toe, K)
        if _visible_score(pose_frames[bfc_i2], back_heel, back_toe) > 0:
            bfc_i, bfc_conf, bfc_method = bfc_i2, min(0.70, c2), "widened_band_best"
        else:
            bfc_conf = min(bfc_conf, 0.45)
            bfc_method = "forced_low_signal"

    # -------------------------
    # Final invariant checks (rigour)
    # -------------------------
    # BFC < FFC < Release (in time)
    if not (bfc_i < ffc_i):
        # Force ordering, degrade confidence explicitly
        bfc_i = max(0, ffc_i - 1)
        bfc_conf = min(bfc_conf, 0.35)
        bfc_method = "forced_ordering_bfc_before_ffc"

    if not (ffc_i < rel):
        # Force ordering, degrade explicitly
        ffc_i = max(0, rel - min_gap)
        ffc_conf = min(ffc_conf, 0.35)
        ffc_method = "forced_ordering_ffc_before_release"

    # Debug / frame dumps
    if DEBUG:
        logger.info(
            f"[FFC/BFC] anchor={rel}({anchor_method}) "
            f"priors: min_gap={min_gap} d_ffc={d_ffc} d_bfc={d_bfc} buf={buf}"
        )
        logger.info(
            f"[FFC/BFC] ffc={ffc_i}({ffc_method}, conf={ffc_conf:.2f}) "
            f"bfc={bfc_i}({bfc_method}, conf={bfc_conf:.2f})"
        )
        _dump(video_path, ffc_i, "ffc")
        _dump(video_path, bfc_i, "bfc")

    return {
        "ffc": {"frame": int(ffc_i), "confidence": float(ffc_conf), "method": ffc_method},
        "bfc": {"frame": int(bfc_i), "confidence": float(bfc_conf), "method": bfc_method},
    }
