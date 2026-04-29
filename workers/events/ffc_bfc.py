# app/workers/events/ffc_bfc.py
"""
PHASE-1 INVARIANTS (UPDATED FOR REAL POSE DATA):

1. Pelvis kinematics define WHEN braking begins.
2. Geometry defines WHEN contact actually occurs.
3. FFC = latest plausible landing frame before release, found by searching backward from release
   and confirmed with front-foot grounding plus back-foot support/unloading heuristics.
4. Kinematics and geometry never compete on the same frame.
5. If uncertain, return a conservative fallback with low confidence (never return {}).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from app.common.logger import get_logger
from app.workers.events.event_confidence import build_candidate, chain_quality, compact_candidates
from app.workers.events.ffc_bfc_support import (
    MIN_VIS,
    annotated_result,
    as_int,
    clamp,
    detection_context,
    ffc_search_start,
    is_grounded,
    moving_average,
    pick_bfc_backward_from_ffc,
    pick_ffc_backward_from_release,
    robust_percentile,
    sanitize_bfc_frame,
    series_vis,
    series_y,
)
from app.workers.events.signal_cache import build_signal_cache
from app.workers.events.timing_constants import foot_contact_timing

logger = get_logger(__name__)

LS, RS = 11, 12
LH, RH = 23, 24
LA, RA = 27, 28
LFI, RFI = 31, 32

EPS = 1e-9

_ffc_search_start = ffc_search_start
_pick_ffc_backward_from_release = pick_ffc_backward_from_release
_pick_bfc_backward_from_ffc = pick_bfc_backward_from_ffc
_sanitize_bfc_frame = sanitize_bfc_frame


def detect_ffc_bfc(
    pose_frames: List[Dict],
    hand: str,
    release_frame: int,
    delivery_window: Tuple[int, int],
    fps: Optional[float] = None,
    **_ignored,
) -> Dict[str, Dict]:
    del delivery_window

    n = len(pose_frames)
    if n < 10:
        logger.warning("[FFC/BFC] Too few frames")
        return {}

    rel = as_int(release_frame)
    if rel is None:
        logger.error("[FFC/BFC] Missing release frame")
        return {}

    try:
        fps_f = float(fps) if fps else 0.0
    except Exception:
        fps_f = 0.0
    fps_was_defaulted = False
    if fps_f <= 1e-3:
        fps_f = 30.0
        fps_was_defaulted = True
    fps_f = max(1.0, fps_f)
    timing = foot_contact_timing(fps_f)
    fps_f = float(timing["fps"])
    dt = float(timing["dt"])

    lookback = int(timing["lookback"])
    hold = int(timing["hold"])
    smooth_k = int(timing["smooth_k"])

    unclamped_start = rel - lookback
    unclamped_end = rel - 2
    win_start = clamp(unclamped_start, 0, n - 1)
    win_end = clamp(unclamped_end, win_start, n - 1)
    window_was_clipped = win_start != unclamped_start or win_end != unclamped_end
    detection = detection_context(
        fps=fps_f,
        fps_was_defaulted=fps_was_defaulted,
        window_was_clipped=window_was_clipped,
        win_start=win_start,
        win_end=win_end,
    )

    logger.info(f"[FFC/BFC][WINDOW] derived=[{win_start}..{win_end}] release={rel} fps={fps_f:.2f}")

    if win_end <= win_start + 2:
        logger.warning("[FFC/BFC] Degenerate window")
        return annotated_result({
            "ffc": {"frame": win_start, "confidence": 0.15, "method": "degenerate_window"},
            "detection_context": detection,
        })

    cache = build_signal_cache(
        pose_frames=pose_frames,
        hand=hand,
        fps=fps_f,
    )

    pelvis_xy = cache["pelvis_centre_xy"]
    px = pelvis_xy[:, 0]
    vis_ok = np.asarray(cache["hips_vis_raw"] >= MIN_VIS, dtype=bool)
    valid_pelvis_count = int(np.sum(np.isfinite(px)))

    if valid_pelvis_count < max(hold, 6):
        logger.warning("[FFC/BFC] Insufficient valid pelvis landmarks")
        return annotated_result({
            "ffc": {"frame": win_start, "confidence": 0.10, "method": "insufficient_landmarks"},
            "detection_context": detection,
        })

    v_lin = moving_average(
        np.nan_to_num(cache["pelvis_linear_speed"], nan=0.0),
        smooth_k,
    )
    w_rot = moving_average(
        np.nan_to_num(cache["hip_line_angular_velocity"], nan=0.0),
        smooth_k,
    )
    pelvis_jerk = moving_average(
        np.nan_to_num(cache["pelvis_jerk"], nan=0.0),
        smooth_k,
    )

    ratio = w_rot / (v_lin + EPS)
    threshold = robust_percentile(ratio[win_start:win_end + 1], 70)
    pelvis_on = None
    for i in range(win_end, win_start, -1):
        if vis_ok[i] and ratio[i] > threshold:
            pelvis_on = i
        elif pelvis_on is not None:
            break
    if pelvis_on is None:
        logger.warning("[FFC/BFC] Pelvis never activated; using win_start")
        pelvis_on = win_start

    logger.info(f"[FFC/BFC][PELVIS_ON] idx={pelvis_on}")

    y_LA = series_y(pose_frames, LA)
    y_RA = series_y(pose_frames, RA)
    y_LFI = series_y(pose_frames, LFI)
    y_RFI = series_y(pose_frames, RFI)
    vis_LA = series_vis(pose_frames, LA)
    vis_RA = series_vis(pose_frames, RA)
    vis_LFI = series_vis(pose_frames, LFI)
    vis_RFI = series_vis(pose_frames, RFI)

    if (not np.any(np.isfinite(y_LA)) and not np.any(np.isfinite(y_RA))) or (not np.any(np.isfinite(y_LFI)) and not np.any(np.isfinite(y_RFI))):
        logger.warning("[FFC/BFC] No valid foot landmarks; pelvis fallback")
        ffc = pelvis_on
        bfc = clamp(ffc - max(3, hold), win_start, ffc)
        logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
        return annotated_result({
            "ffc": {"frame": int(ffc), "confidence": 0.20, "method": "no_foot_data_fallback"},
            "bfc": {"frame": int(bfc), "confidence": 0.20, "method": "no_foot_data_fallback"},
            "detection_context": detection,
        })

    back_recent = int(timing["back_recent"])
    preferred_front_side = "left" if str(hand or "").upper().startswith("R") else "right"
    search_start = ffc_search_start(
        win_start=win_start,
        win_end=win_end,
        pelvis_on=pelvis_on,
        fps=fps_f,
        hold=hold,
    )
    if search_start < pelvis_on:
        logger.info(
            "[FFC/BFC][FFC_SEARCH] widening lower bound from pelvis_on=%s to search_start=%s",
            pelvis_on,
            search_start,
        )

    skip_chain_grounding = (win_end - search_start) <= max(hold * 2, 6)

    if skip_chain_grounding:
        ffc, front_side, ffc_candidates, ffc_confidence = None, None, [], 0.0
    else:
        ffc, front_side, ffc_candidates, ffc_confidence = pick_ffc_backward_from_release(
            search_start=search_start,
            pelvis_on=pelvis_on,
            preferred_front_side=preferred_front_side,
            win_end=win_end,
            hold=hold,
            win_start=win_start,
            dt=dt,
            back_recent=back_recent,
            y_LA=y_LA,
            y_RA=y_RA,
            y_LFI=y_LFI,
            y_RFI=y_RFI,
            fps=fps_f,
            approach_speed=v_lin,
            pelvis_jerk=pelvis_jerk,
        )

    if ffc is None:
        for i in range(win_end - hold, search_start - 1, -1):
            if is_grounded(y_LA, y_LFI, i, hold, win_start, win_end, dt) or is_grounded(y_RA, y_RFI, i, hold, win_start, win_end, dt):
                ffc = i
                candidate = build_candidate(
                    frame=i,
                    method="single_foot_fallback",
                    confidence=0.22,
                    score=1.0,
                )
                if candidate is not None:
                    ffc_candidates.append(candidate)
                logger.warning(f"[FFC/BFC][FALLBACK] single_foot frame={ffc}")
                bfc = clamp(ffc - max(3, hold), win_start, ffc)
                corrected_bfc, corrected = sanitize_bfc_frame(
                    bfc=bfc,
                    ffc=ffc,
                    front_side=preferred_front_side,
                    hold=hold,
                    win_start=win_start,
                    win_end=win_end,
                    dt=dt,
                    y_LA=y_LA,
                    y_RA=y_RA,
                    y_LFI=y_LFI,
                    y_RFI=y_RFI,
                )
                if corrected:
                    logger.info("[FFC/BFC][BFC_CORRECT] single_foot %s -> %s", bfc, corrected_bfc)
                    bfc = corrected_bfc
                chain = chain_quality(
                    bfc_frame=bfc,
                    ffc_frame=ffc,
                    uah_frame=None,
                    release_frame=rel,
                    bfc_confidence=0.22,
                    ffc_confidence=0.22,
                )
                logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
                return annotated_result({
                    "ffc": {
                        "frame": int(ffc),
                        "confidence": 0.22,
                        "method": "single_foot_fallback",
                        "candidates": compact_candidates(ffc_candidates),
                        "window": [int(win_start), int(win_end)],
                    },
                    "bfc": {
                        "frame": int(bfc),
                        "confidence": 0.22,
                        "method": "single_foot_fallback",
                    },
                    "event_chain": chain,
                    "detection_context": detection,
                })

        ffc = pelvis_on + int(0.75 * (win_end - pelvis_on))
        ffc = clamp(ffc, pelvis_on, win_end)
        candidate = build_candidate(
            frame=ffc,
            method="ultimate_fallback",
            confidence=0.15,
            score=0.0,
        )
        if candidate is not None:
            ffc_candidates.append(candidate)
        logger.warning(f"[FFC/BFC][FALLBACK] ultimate_3quarter frame={ffc}")

        bfc = clamp(ffc - max(3, hold), win_start, ffc)
        corrected_bfc, corrected = sanitize_bfc_frame(
            bfc=bfc,
            ffc=ffc,
            front_side=preferred_front_side,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            y_LA=y_LA,
            y_RA=y_RA,
            y_LFI=y_LFI,
            y_RFI=y_RFI,
        )
        if corrected:
            logger.info("[FFC/BFC][BFC_CORRECT] ultimate_fallback %s -> %s", bfc, corrected_bfc)
            bfc = corrected_bfc
        chain = chain_quality(
            bfc_frame=bfc,
            ffc_frame=ffc,
            uah_frame=None,
            release_frame=rel,
            bfc_confidence=0.15,
            ffc_confidence=0.15,
        )
        logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
        return annotated_result({
            "ffc": {
                "frame": int(ffc),
                "confidence": 0.15,
                "method": "ultimate_fallback",
                "candidates": compact_candidates(ffc_candidates),
                "window": [int(win_start), int(win_end)],
            },
            "bfc": {"frame": int(bfc), "confidence": 0.15, "method": "ultimate_fallback"},
            "event_chain": chain,
            "detection_context": detection,
        })

    logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")

    bfc, bfc_confidence, bfc_method = pick_bfc_backward_from_ffc(
        ffc=ffc,
        front_side=front_side,
        hold=hold,
        win_start=win_start,
        win_end=win_end,
        dt=dt,
        fps=fps_f,
        y_LA=y_LA,
        y_RA=y_RA,
        y_LFI=y_LFI,
        y_RFI=y_RFI,
        vis_LA=vis_LA,
        vis_RA=vis_RA,
        vis_LFI=vis_LFI,
        vis_RFI=vis_RFI,
        approach_speed=v_lin,
        pelvis_jerk=pelvis_jerk,
    )
    corrected_bfc, corrected = sanitize_bfc_frame(
        bfc=bfc,
        ffc=ffc,
        front_side=front_side or preferred_front_side,
        hold=hold,
        win_start=win_start,
        win_end=win_end,
        dt=dt,
        y_LA=y_LA,
        y_RA=y_RA,
        y_LFI=y_LFI,
        y_RFI=y_RFI,
    )
    if corrected:
        logger.info("[FFC/BFC][BFC_CORRECT] %s %s -> %s", bfc_method, bfc, corrected_bfc)
        bfc = corrected_bfc
        bfc_method = f"{bfc_method}_front_contact_corrected"
        bfc_confidence = max(0.0, float(bfc_confidence) - 0.05)
    ffc_confidence = max(0.45, ffc_confidence or 0.62)
    chain = chain_quality(
        bfc_frame=bfc,
        ffc_frame=ffc,
        uah_frame=None,
        release_frame=rel,
        bfc_confidence=bfc_confidence,
        ffc_confidence=ffc_confidence,
    )

    return annotated_result({
        "ffc": {
            "frame": int(ffc),
            "confidence": ffc_confidence,
            "method": "release_backward_chain_grounding",
            "candidates": compact_candidates(ffc_candidates),
            "window": [int(win_start), int(win_end)],
        },
        "bfc": {"frame": int(bfc), "confidence": bfc_confidence, "method": bfc_method},
        "event_chain": chain,
        "detection_context": detection,
    })
