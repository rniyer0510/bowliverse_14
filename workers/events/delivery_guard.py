from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from app.workers.events.timing_constants import signal_cache_timing

# MediaPipe pose indices
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24

MIN_VIS = 0.25


def _xyz(pt: Optional[Dict[str, Any]]) -> Optional[tuple[float, float, float]]:
    try:
        return (
            float(pt["x"]),
            float(pt["y"]),
            float(pt.get("z", 0.0)),
        )
    except Exception:
        return None


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = float(np.linalg.norm(v))
    if mag <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _strong_peak_frames(
    signal: np.ndarray,
    *,
    fps: float,
    min_height_ratio: float,
    min_prominence_ratio: float,
) -> List[int]:
    if signal.size < 10:
        return []

    global_max = float(np.max(signal))
    if global_max <= 1e-6:
        return []

    distance = max(18, int(round(max(1.0, fps) * 1.2)))
    prominence = max(1e-3, global_max * min_prominence_ratio)
    peaks, props = find_peaks(
        signal,
        distance=distance,
        prominence=prominence,
    )
    if len(peaks) == 0:
        return []

    heights = props.get("peak_heights")
    if heights is None:
        heights = signal[peaks]

    strong = [
        int(peaks[i])
        for i, height in enumerate(heights)
        if float(height) >= global_max * min_height_ratio
    ]
    return strong


def _filter_delivery_candidates(
    peaks: List[int],
    signal: np.ndarray,
    *,
    total_frames: int,
) -> List[int]:
    if len(peaks) < 2 or total_frames <= 0:
        return peaks

    strongest_peak = max(peaks, key=lambda peak: float(signal[peak]))
    strongest_value = float(signal[strongest_peak])
    latest_peak = max(peaks)
    early_cutoff = int(round(total_frames * 0.35))
    close_window = max(24, int(round(total_frames * 0.18)))
    late_phase_start = int(round(total_frames * 0.45))

    filtered: List[int] = []
    for peak in peaks:
        peak_value = float(signal[peak])
        remove_early_peak = (
            peak != strongest_peak
            and strongest_peak > peak
            and strongest_value > 1e-6
            and peak < early_cutoff
            and peak_value < strongest_value * 0.8
        )
        remove_late_cluster_peak = (
            peak != latest_peak
            and peak >= late_phase_start
            and latest_peak > peak
            and (latest_peak - peak) <= close_window
        )
        if remove_early_peak or remove_late_cluster_peak:
            continue
        filtered.append(peak)
    return filtered or peaks


def detect_delivery_candidates(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
) -> Dict[str, Any]:
    timing = signal_cache_timing(fps or 25.0)
    fps = float(timing["fps"])
    dt = float(timing["dt"])
    h = (hand or "R").upper()

    s_idx, w_idx = (RS, RW) if h == "R" else (LS, LW)
    nb_e_idx = LE if h == "R" else RE

    wrist_xyz: List[Optional[tuple[float, float, float]]] = []
    pelvis_xyz: List[Optional[tuple[float, float, float]]] = []
    nb_elbow_y = []
    wrist_visible = 0
    nb_elbow_visible = 0

    for item in pose_frames or []:
        landmarks = (item or {}).get("landmarks") or []

        def _visible(idx: int) -> Optional[Dict[str, Any]]:
            if not isinstance(landmarks, list) or idx >= len(landmarks):
                return None
            point = landmarks[idx]
            if not isinstance(point, dict):
                return None
            if float(point.get("visibility", 0.0)) < MIN_VIS:
                return None
            return point

        wrist = _visible(w_idx)
        shoulder = _visible(s_idx)
        left_hip = _visible(LH)
        right_hip = _visible(RH)
        nb_elbow = _visible(nb_e_idx)

        wrist_xyz.append(_xyz(wrist) if wrist and shoulder else None)
        if wrist and shoulder:
            wrist_visible += 1

        if left_hip and right_hip:
            pelvis_xyz.append(
                (
                    (float(left_hip["x"]) + float(right_hip["x"])) / 2.0,
                    (float(left_hip["y"]) + float(right_hip["y"])) / 2.0,
                    0.0,
                )
            )
        else:
            pelvis_xyz.append(None)

        if nb_elbow:
            nb_elbow_y.append(float(nb_elbow["y"]))
            nb_elbow_visible += 1
        else:
            nb_elbow_y.append(np.nan)

    n = len(wrist_xyz)
    if n < 10:
        return {
            "delivery_count": 0,
            "method": "insufficient_frames",
            "candidate_frames": [],
        }

    pelvis_vels = []
    for i in range(1, n):
        if pelvis_xyz[i] and pelvis_xyz[i - 1]:
            pelvis_vels.append(
                (
                    pelvis_xyz[i][0] - pelvis_xyz[i - 1][0],
                    pelvis_xyz[i][1] - pelvis_xyz[i - 1][1],
                    0.0,
                )
            )
    if pelvis_vels:
        steps = np.asarray(pelvis_vels, dtype=float)
        med_dx = float(np.median(steps[:, 0]))
        med_dy = float(np.median(steps[:, 1]))
        med_dz = float(np.median(steps[:, 2]))
        forward = _unit((med_dx, med_dy, med_dz))
    else:
        forward = (1.0, 0.0, 0.0)

    wrist_signal = np.zeros(n, dtype=float)
    for i in range(1, n):
        if wrist_xyz[i] and wrist_xyz[i - 1]:
            dv = (
                wrist_xyz[i][0] - wrist_xyz[i - 1][0],
                wrist_xyz[i][1] - wrist_xyz[i - 1][1],
                wrist_xyz[i][2] - wrist_xyz[i - 1][2],
            )
            wrist_signal[i] = float(np.dot(np.asarray(dv), np.asarray(forward))) / dt

    sigma = float(timing["smooth_sigma"])
    wrist_signal = gaussian_filter1d(wrist_signal, sigma=sigma)
    wrist_peaks = _strong_peak_frames(
        wrist_signal,
        fps=fps,
        min_height_ratio=0.55,
        min_prominence_ratio=0.18,
    )
    wrist_peaks = _filter_delivery_candidates(
        wrist_peaks,
        wrist_signal,
        total_frames=n,
    )

    wrist_visibility_ratio = wrist_visible / float(max(1, n))
    if len(wrist_peaks) >= 2 and wrist_visibility_ratio >= 0.30:
        return {
            "delivery_count": len(wrist_peaks),
            "method": "wrist_velocity",
            "candidate_frames": wrist_peaks,
        }

    good = np.isfinite(nb_elbow_y)
    if good.sum() >= max(6, int(0.15 * fps)):
        idx = np.arange(n)
        interp = np.array(nb_elbow_y, dtype=float)
        if good.sum() >= 2:
            interp[~good] = np.interp(idx[~good], idx[good], interp[good])
        nb_signal = gaussian_filter1d(-interp, sigma=sigma)
        nb_peaks = _strong_peak_frames(
            nb_signal,
            fps=fps,
            min_height_ratio=0.65,
            min_prominence_ratio=0.16,
        )
        nb_peaks = _filter_delivery_candidates(
            nb_peaks,
            nb_signal,
            total_frames=n,
        )
        if len(nb_peaks) >= 2:
            return {
                "delivery_count": len(nb_peaks),
                "method": "non_bowling_elbow",
                "candidate_frames": nb_peaks,
            }

    method = "wrist_velocity" if wrist_visibility_ratio >= 0.30 else "non_bowling_elbow"
    candidate_frames = wrist_peaks if wrist_visibility_ratio >= 0.30 else []
    return {
        "delivery_count": max(1, len(candidate_frames)) if n else 0,
        "method": method,
        "candidate_frames": candidate_frames,
    }
