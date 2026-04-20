from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

TARGET_SAMPLE_MS = 66.0
MIN_SAMPLES = 12
PAD_BEFORE_SEC = 1.1
PAD_AFTER_SEC = 0.7
MIN_PEAK_SCORE = 2.5
LOW_MOTION_REASON = "low_motion_signal"
MULTI_DELIVERY_GAP_SEC = 0.85
MOTION_DIFF_THRESHOLD = 16
MIN_COMPONENT_PIXELS = 24
LATE_BIAS_FLOOR = 0.78
LATE_BIAS_GAIN = 0.42


def _sample_step_for_fps(fps: float) -> int:
    return max(1, int(round((float(fps or 0.0) * TARGET_SAMPLE_MS) / 1000.0)))


def _moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    if values.size < width:
        return values
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def _window_from_hint(
    *,
    total_frames: int,
    fps: float,
    hint_frame: Optional[int],
) -> Tuple[int, int]:
    if total_frames <= 0:
        return 0, 0
    if hint_frame is None:
        width = min(
            total_frames,
            max(
                24,
                int(round(max(2.6 * fps, total_frames * 0.38))),
            ),
        )
        start = max(0, total_frames - width)
        end = total_frames - 1
        return start, end

    center = max(0, min(total_frames - 1, int(hint_frame)))
    start = max(0, center - int(round(PAD_BEFORE_SEC * fps)))
    end = min(total_frames - 1, center + int(round(PAD_AFTER_SEC * fps)))
    return start, max(start, end)


def _motion_box_score(
    current_small: np.ndarray,
    prev_small: Optional[np.ndarray],
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    if prev_small is None:
        return 0.0, None

    diff = cv2.absdiff(current_small, prev_small)
    _, mask = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    active_pixels = int(np.count_nonzero(mask))
    if active_pixels < MIN_COMPONENT_PIXELS:
        return float(np.mean(diff)) * 0.20, None

    mask = cv2.medianBlur(mask, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float(np.mean(diff)) * 0.35, None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area = float(max(1, w * h))
    area_frac = area / float(diff.size)
    roi = diff[y : y + h, x : x + w]
    if roi.size == 0:
        return float(np.mean(diff)) * 0.35, None

    roi_strength = float(np.percentile(roi, 88))
    component_gain = 0.55 + min(1.45, np.sqrt(max(area_frac, 1e-6)) * 8.0)
    score = roi_strength * component_gain
    return score, (int(x), int(y), int(w), int(h))


def _peak_candidate_indices(
    *,
    signal: np.ndarray,
    baseline: float,
) -> List[int]:
    if signal.size == 0:
        return []

    min_height = max(MIN_PEAK_SCORE * 0.55, baseline * 1.22)
    min_distance = max(4, int(round((1000.0 / TARGET_SAMPLE_MS) * MULTI_DELIVERY_GAP_SEC)))
    prominence = max(0.35, (float(np.max(signal)) - baseline) * 0.12)
    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence,
    )
    if len(peaks) == 0:
        return []
    return [int(idx) for idx in peaks if float(signal[int(idx)]) >= min_height]


def _select_peak_index(
    *,
    signal: np.ndarray,
    sample_frames: List[int],
    candidate_indices: List[int],
    total_frames: int,
) -> Optional[int]:
    if signal.size == 0 or not sample_frames:
        return None

    ranked_indices = candidate_indices or [int(np.argmax(signal))]
    best_idx: Optional[int] = None
    best_score = float("-inf")
    frame_denom = max(1.0, float(max(0, total_frames - 1)))
    for idx in ranked_indices:
        idx = int(idx)
        frame_ratio = float(sample_frames[idx]) / frame_denom
        late_bias = LATE_BIAS_FLOOR + (frame_ratio * LATE_BIAS_GAIN)
        score = float(signal[idx]) * late_bias
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _filter_reported_peak_indices(
    *,
    signal: np.ndarray,
    sample_frames: List[int],
    candidate_indices: List[int],
    selected_idx: Optional[int],
    total_frames: int,
) -> List[int]:
    if not candidate_indices:
        return []
    if selected_idx is None or selected_idx not in candidate_indices:
        return candidate_indices

    selected_frame = int(sample_frames[selected_idx])
    selected_value = float(signal[selected_idx])
    if selected_value <= 1e-6 or total_frames <= 0:
        return candidate_indices

    late_anchor_start = int(round(total_frames * 0.65))
    early_cutoff = int(round(total_frames * 0.45))
    keep_threshold = selected_value * 0.72

    filtered: List[int] = []
    for idx in candidate_indices:
        idx = int(idx)
        if idx == selected_idx:
            filtered.append(idx)
            continue

        frame = int(sample_frames[idx])
        value = float(signal[idx])
        remove_as_weak_early_burst = (
            selected_frame >= late_anchor_start
            and frame < late_anchor_start
            and value < keep_threshold
        )
        remove_as_minor_early_noise = (
            frame < early_cutoff
            and value < (selected_value * 0.80)
        )
        if remove_as_weak_early_burst or remove_as_minor_early_noise:
            continue
        filtered.append(idx)

    return filtered or [selected_idx]


def detect_delivery_window(video: Dict[str, Any]) -> Dict[str, Any]:
    video_path = str(video.get("path") or "")
    total_frames = int(video.get("total_frames") or 0)
    fps = float(video.get("fps") or 0.0)
    if not video_path or total_frames <= 0 or fps <= 0.0:
        return {"available": False, "reason": "missing_video_metadata"}

    step = _sample_step_for_fps(fps)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"available": False, "reason": "video_unavailable"}

    sample_frames: List[int] = []
    scores: List[float] = []
    motion_boxes: List[Optional[Tuple[int, int, int, int]]] = []
    prev: Optional[np.ndarray] = None
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
            score, motion_box = _motion_box_score(small, prev)
            sample_frames.append(frame_idx)
            scores.append(score)
            motion_boxes.append(motion_box)
            prev = small
            frame_idx += 1
    finally:
        cap.release()

    if len(scores) < MIN_SAMPLES:
        return {"available": False, "reason": "insufficient_samples"}

    signal = _moving_average(np.asarray(scores, dtype=float))
    baseline = float(np.median(signal))
    candidate_indices = _peak_candidate_indices(signal=signal, baseline=baseline)
    selected_idx = _select_peak_index(
        signal=signal,
        sample_frames=sample_frames,
        candidate_indices=candidate_indices,
        total_frames=total_frames,
    )
    if selected_idx is None:
        return {"available": False, "reason": LOW_MOTION_REASON, "delivery_count": 0, "peak_frames": []}

    peak = float(signal[selected_idx])
    raw_peak_frames = [int(sample_frames[idx]) for idx in candidate_indices]
    reported_peak_indices = _filter_reported_peak_indices(
        signal=signal,
        sample_frames=sample_frames,
        candidate_indices=candidate_indices,
        selected_idx=selected_idx,
        total_frames=total_frames,
    )
    peak_frames = [int(sample_frames[idx]) for idx in reported_peak_indices]
    release_hint = int(sample_frames[selected_idx])
    analysis_start, analysis_end = _window_from_hint(
        total_frames=total_frames,
        fps=fps,
        hint_frame=release_hint,
    )

    if peak < (MIN_PEAK_SCORE * 0.75) or peak <= max(MIN_PEAK_SCORE * 0.75, baseline * 1.35):
        return {
            "available": False,
            "reason": LOW_MOTION_REASON,
            "sample_step": step,
            "sample_count": len(sample_frames),
            "delivery_count": len(peak_frames),
            "peak_frames": peak_frames,
            "candidate_peak_frames": peak_frames or [release_hint],
            "raw_candidate_peak_frames": raw_peak_frames or [release_hint],
            "release_hint": release_hint,
            "analysis_start": analysis_start,
            "analysis_end": analysis_end,
        }

    cutoff = baseline + (peak - baseline) * 0.32
    left = selected_idx
    right = selected_idx
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
    confidence = 0.42 + min(0.48, contrast * 0.72)
    if width < int(round(1.2 * fps)) or width > int(round(5.5 * fps)):
        confidence = max(0.28, confidence - 0.12)

    return {
        "available": True,
        "method": "subject_local_motion_scan",
        "confidence": round(confidence, 3),
        "sample_step": step,
        "sample_count": len(sample_frames),
        "delivery_count": max(1, len(peak_frames)),
        "peak_frames": peak_frames or [release_hint],
        "candidate_peak_frames": peak_frames or [release_hint],
        "raw_candidate_peak_frames": raw_peak_frames or [release_hint],
        "release_hint": release_hint,
        "selected_motion_box": motion_boxes[selected_idx] if 0 <= selected_idx < len(motion_boxes) else None,
        "coarse_start": coarse_start,
        "coarse_end": coarse_end,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
    }
