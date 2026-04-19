from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

TARGET_SCAN_FPS = 15.0
MIN_SAMPLES = 12
PAD_BEFORE_SEC = 1.1
PAD_AFTER_SEC = 0.7
MIN_PEAK_SCORE = 2.5
LOW_MOTION_REASON = "low_motion_signal"


def _moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    if values.size < width:
        return values
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def detect_delivery_window(video: Dict[str, Any]) -> Dict[str, Any]:
    video_path = str(video.get("path") or "")
    total_frames = int(video.get("total_frames") or 0)
    fps = float(video.get("fps") or 0.0)
    if not video_path or total_frames <= 0 or fps <= 0.0:
        return {"available": False, "reason": "missing_video_metadata"}

    step = max(1, int(round(fps / TARGET_SCAN_FPS)))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"available": False, "reason": "video_unavailable"}

    sample_frames = []
    scores = []
    prev = None
    frame_idx = 0

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (96, 54))
            score = 0.0 if prev is None else float(np.mean(cv2.absdiff(small, prev)))
            sample_frames.append(frame_idx)
            scores.append(score)
            prev = small
            frame_idx += 1
    finally:
        cap.release()

    if len(scores) < MIN_SAMPLES:
        return {"available": False, "reason": "insufficient_samples"}

    signal = _moving_average(np.asarray(scores, dtype=float))
    peak_idx = int(np.argmax(signal))
    peak = float(signal[peak_idx])
    baseline = float(np.median(signal))
    if peak < MIN_PEAK_SCORE or peak <= max(MIN_PEAK_SCORE, baseline * 1.8):
        return {"available": False, "reason": LOW_MOTION_REASON}

    cutoff = baseline + (peak - baseline) * 0.35
    left = peak_idx
    right = peak_idx
    while left > 0 and signal[left - 1] >= cutoff:
        left -= 1
    while right < len(signal) - 1 and signal[right + 1] >= cutoff:
        right += 1

    coarse_start = int(sample_frames[left])
    coarse_end = int(sample_frames[right])
    analysis_start = max(0, coarse_start - int(round(PAD_BEFORE_SEC * fps)))
    analysis_end = min(total_frames - 1, coarse_end + int(round(PAD_AFTER_SEC * fps)))
    width = max(1, analysis_end - analysis_start + 1)
    contrast = max(0.0, peak - baseline) / max(peak, 1.0)
    confidence = 0.45 + min(0.45, contrast * 0.6)
    if width < int(round(1.2 * fps)) or width > int(round(5.5 * fps)):
        confidence = max(0.25, confidence - 0.15)

    return {
        "available": True,
        "method": "coarse_motion_scan",
        "confidence": round(confidence, 3),
        "sample_step": step,
        "sample_count": len(sample_frames),
        "release_hint": int(sample_frames[peak_idx]),
        "coarse_start": coarse_start,
        "coarse_end": coarse_end,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
    }
