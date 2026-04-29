"""
ActionLab V15 — Release + UAH detector

Architecture:
- One signal-extraction pass builds the kinematic truth inputs.
- Release is detected by visibility-weighted multi-signal consensus.
- UAH is detected strictly backward from Release.
- Timing is defined in milliseconds and converted to frames at runtime.
- The public contract stays backward-compatible for downstream consumers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

import numpy as np

from app.common.logger import get_logger
from app.workers.events.event_confidence import build_candidate, compact_candidates, annotate_detection_contract
from app.workers.events.release_uah_support import (
    build_release_candidates,
    detect_uah,
    extract_signals,
    refine_release_frame,
    release_consensus,
    release_search_window,
)
from app.workers.events.timing_constants import release_uah_timing

logger = get_logger(__name__)

DEBUG = os.getenv("ACTIONLAB_RELEASE_UAH_DEBUG", "").strip().lower() == "true"
DEBUG_DIR = os.getenv(
    "ACTIONLAB_RELEASE_UAH_DEBUG_DIR",
    "/tmp/actionlab_debug/release_uah",
)


def _tc(fps: float, n_frames: int) -> Dict[str, float]:
    return release_uah_timing(fps, n_frames)


def _nb_elbow_peak_is_plausible(
    *,
    nb_elbow_peak_i: Optional[int],
    wrist_peak_i: int,
    total_frames: int,
    fps: float,
) -> bool:
    if nb_elbow_peak_i is None or total_frames <= 0:
        return False
    min_delivery_frame = max(5, int(0.20 * total_frames))
    max_edge_frame = max(0, total_frames - max(3, int(0.02 * total_frames)))
    close_to_wrist_peak = abs(int(nb_elbow_peak_i) - int(wrist_peak_i)) <= int(max(2, round(0.25 * fps)))
    if nb_elbow_peak_i < min_delivery_frame and not close_to_wrist_peak:
        return False
    if nb_elbow_peak_i >= max_edge_frame and not close_to_wrist_peak:
        return False
    return True


def _dump(video_path: Optional[str], idx: Optional[int], tag: str) -> None:
    if not DEBUG or not video_path or idx is None or idx < 0:
        return
    try:
        import cv2  # local debug dependency only
    except Exception:
        logger.warning("[Release/UAH] Debug frame dump skipped because cv2 is unavailable")
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{tag}_{idx}.png"), frame)


def detect_release_uah(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
    delivery_window: Optional[Dict[str, Any]] = None,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    frames = [int(item.get("frame", idx)) for idx, item in enumerate(pose_frames)]
    n_frames = len(frames)
    fps = float(fps or 25.0)

    if n_frames < 10:
        logger.error("[Release/UAH] Too few frames")
        return {}

    tc = _tc(fps, n_frames)
    signals = extract_signals(pose_frames, hand, tc)
    search_start, search_end = release_search_window(signals, n_frames, tc, delivery_window)
    wrist_peak = int(np.clip(signals["wrist_peak"], 0, n_frames - 1))

    release_frame, release_confidence, signals_used = release_consensus(
        signals,
        search_start,
        search_end,
        tc,
    )

    if release_frame is None:
        release_frame = max(search_start, min(search_end, search_start + int(round(0.80 * (search_end - search_start)))))
        release_confidence = 0.20
        release_method = "hard_fallback"
    else:
        release_method = "multi_signal_consensus"

    refined_release = refine_release_frame(signals, release_frame, tc, n_frames)
    if refined_release is not None and int(refined_release) != int(release_frame):
        release_frame = int(refined_release)

    release_candidates = build_release_candidates(
        consensus_frame=release_frame,
        confidence=release_confidence,
        start=search_start,
        end=search_end,
        signals=signals,
    )

    uah_frame, uah_confidence, uah_method, uah_candidates, uah_window = detect_uah(
        signals,
        release_frame,
        tc,
    )

    if uah_frame >= release_frame:
        uah_frame = max(0, release_frame - 1)
        uah_method = "release_minus_one_guard"
        uah_confidence = min(uah_confidence, 0.25)
        uah_candidates = compact_candidates(
            uah_candidates
            + [
                build_candidate(
                    frame=uah_frame,
                    method="release_minus_one_guard",
                    confidence=uah_confidence,
                    score=0.0,
                )
            ]
        )

    if DEBUG:
        logger.info(
            "[Release/UAH] n=%d fps=%.2f window=[%d..%d] release=%d uah=%d wrist_peak=%d "
            "signals=%s",
            n_frames,
            fps,
            search_start,
            search_end,
            release_frame,
            uah_frame,
            wrist_peak,
            [(item["name"], item.get("effective_weight")) for item in signals_used if item.get("used")],
        )
        _dump(video_path, search_start, "release_window_start")
        _dump(video_path, search_end, "release_window_end")
        _dump(video_path, wrist_peak, "wrist_peak")
        _dump(video_path, release_frame, "release")
        _dump(video_path, uah_frame, "uah")

    return annotate_detection_contract({
        "release": {
            "frame": int(frames[release_frame]),
            "method": release_method,
            "confidence": round(float(release_confidence), 2),
            "candidates": release_candidates,
            "window": [int(frames[search_start]), int(frames[search_end])],
            "signals_used": signals_used,
        },
        "uah": {
            "frame": int(frames[uah_frame]),
            "method": uah_method,
            "confidence": round(float(uah_confidence), 2),
            "candidates": uah_candidates,
            "window": [int(frames[uah_window[0]]), int(frames[max(uah_window[0], uah_window[1])])],
        },
        "delivery_window": [int(frames[search_start]), int(frames[search_end])],
        "peak": {"frame": int(frames[wrist_peak])},
    })
