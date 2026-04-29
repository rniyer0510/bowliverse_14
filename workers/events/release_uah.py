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

from typing import Any, Dict, List, Optional, Tuple
import math
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.common.logger import get_logger
from app.workers.events.event_confidence import build_candidate, compact_candidates
from app.workers.events.signal_cache import build_signal_cache
from app.workers.events.timing_constants import release_uah_timing

logger = get_logger(__name__)

# MediaPipe pose landmarks
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24

MIN_VIS_HARD = 0.20
MIN_VIS_SOFT = 0.35
MIN_VIS_FULL = 0.80

DEBUG = os.getenv("ACTIONLAB_RELEASE_UAH_DEBUG", "").strip().lower() == "true"
DEBUG_DIR = os.getenv(
    "ACTIONLAB_RELEASE_UAH_DEBUG_DIR",
    "/tmp/actionlab_debug/release_uah",
)


def _tc(fps: float, n_frames: int) -> Dict[str, float]:
    return release_uah_timing(fps, n_frames)


def _xyz(pt: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if isinstance(pt, dict):
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                return None
            return float(x), float(y), float(pt.get("z", 0.0))
        x = getattr(pt, "x", None)
        y = getattr(pt, "y", None)
        if x is None or y is None:
            return None
        return float(x), float(y), float(getattr(pt, "z", 0.0))
    except Exception:
        return None


def _raw_vis(pt: Any) -> float:
    try:
        if isinstance(pt, dict):
            return float(pt.get("visibility", 0.0) or 0.0)
        return float(getattr(pt, "visibility", 0.0) or 0.0)
    except Exception:
        return 0.0


def _vis_weight(raw_vis: float) -> float:
    if raw_vis < MIN_VIS_HARD:
        return 0.0
    if raw_vis >= MIN_VIS_FULL:
        return 1.0
    mid = 0.5
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-10.0 * (raw_vis - mid)))))


def _mag(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    mag = _mag(v)
    if mag <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _dump(video_path: Optional[str], idx: Optional[int], tag: str) -> None:
    if not DEBUG or not video_path or idx is None or idx < 0:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{tag}_{idx}.png"), frame)


def _upper_arm_angle(
    shoulder_xyz: Optional[Tuple[float, float, float]],
    elbow_xyz: Optional[Tuple[float, float, float]],
    forward: Tuple[float, float, float],
) -> Optional[float]:
    if not shoulder_xyz or not elbow_xyz:
        return None
    vector = (
        elbow_xyz[0] - shoulder_xyz[0],
        elbow_xyz[1] - shoulder_xyz[1],
        elbow_xyz[2] - shoulder_xyz[2],
    )
    mag = _mag(vector)
    if mag <= 1e-9:
        return None
    unit = (vector[0] / mag, vector[1] / mag, vector[2] / mag)
    dot = max(-1.0, min(1.0, _dot(unit, forward)))
    return math.degrees(math.acos(dot))


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


def _interp_short_gaps(series: np.ndarray, max_gap: int) -> np.ndarray:
    out = series.astype(float, copy=True)
    valid = np.isfinite(out)
    if valid.sum() < 2:
        return out

    idx = np.arange(len(out))
    valid_idx = idx[valid]
    out[~valid] = np.interp(idx[~valid], valid_idx, out[valid])

    gap_start: Optional[int] = None
    original_valid = valid.copy()
    for i, is_valid in enumerate(original_valid):
        if is_valid:
            if gap_start is not None:
                gap_len = i - gap_start
                if gap_len > max_gap:
                    out[gap_start:i] = np.nan
                gap_start = None
            continue
        if gap_start is None:
            gap_start = i

    if gap_start is not None:
        gap_len = len(out) - gap_start
        if gap_len > max_gap:
            out[gap_start:] = np.nan

    return out


def _smooth_nan_safe(series: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(series)
    if valid.sum() < 2:
        return np.full_like(series, np.nan, dtype=float)
    interp = series.copy()
    idx = np.arange(len(series))
    interp[~valid] = np.interp(idx[~valid], idx[valid], series[valid])
    smoothed = gaussian_filter1d(interp, sigma=sigma)
    smoothed[~valid] = np.nan
    return smoothed


def _gradient_nan_safe(series: np.ndarray, dt: float) -> np.ndarray:
    valid = np.isfinite(series)
    if valid.sum() < 2:
        return np.full_like(series, np.nan, dtype=float)
    interp = series.copy()
    idx = np.arange(len(series))
    interp[~valid] = np.interp(idx[~valid], idx[valid], series[valid])
    grad = np.gradient(interp, dt)
    grad[~valid] = np.nan
    return grad


def _clip_mean_weight(raw_vis: np.ndarray, weights: np.ndarray, start: int, end: int) -> float:
    mask = raw_vis[start:end] >= MIN_VIS_HARD
    if not np.any(mask):
        return 0.0
    return float(np.mean(weights[start:end][mask]))


def _distribution_from_signal(
    signal: np.ndarray,
    raw_vis: np.ndarray,
    weights: np.ndarray,
    start: int,
    end: int,
    sigma: float,
    *,
    mode: str,
) -> Optional[np.ndarray]:
    if end - start < 4:
        return None

    win = signal[start:end].copy()
    vis = raw_vis[start:end]
    wts = weights[start:end]

    valid = np.isfinite(win) & (vis >= MIN_VIS_HARD)
    min_valid = max(3, min(int(0.10 * (end - start)), 12))
    if int(valid.sum()) < min_valid:
        return None

    idx = np.arange(len(win))
    win[~valid] = np.interp(idx[~valid], idx[valid], win[valid])

    vmin = float(np.nanmin(win))
    vmax = float(np.nanmax(win))
    if vmax - vmin < 1e-9:
        return None

    if mode == "peak_min":
        norm = (vmax - win) / (vmax - vmin)
    else:
        norm = (win - vmin) / (vmax - vmin)

    norm *= np.where(vis >= MIN_VIS_HARD, wts, 0.0)
    dist = gaussian_filter1d(norm, sigma=sigma)
    dmax = float(np.max(dist))
    if dmax <= 1e-9:
        return None
    return dist / dmax


def _local_peak(signal: np.ndarray, start: int, end: int, *, mode: str) -> Optional[int]:
    win = signal[start:end]
    valid = np.isfinite(win)
    if not np.any(valid):
        return None
    work = win.copy()
    if mode == "peak_min":
        work[~valid] = np.inf
        return int(start + np.argmin(work))
    work[~valid] = -np.inf
    return int(start + np.argmax(work))


def _compute_forward_direction(pelvis_xyz: np.ndarray) -> Tuple[float, float, float]:
    valid = np.isfinite(pelvis_xyz[:, 0])
    if valid.sum() < 4:
        return (1.0, 0.0, 0.0)
    idx = np.where(valid)[0]
    steps = pelvis_xyz[idx[1:]] - pelvis_xyz[idx[:-1]]
    med = np.median(steps, axis=0)
    return _unit((float(med[0]), float(med[1]), 0.0))


def _extract_signals(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    tc: Dict[str, float],
) -> Dict[str, Any]:
    cache = build_signal_cache(
        pose_frames=pose_frames,
        hand=hand,
        fps=float(tc["fps"]),
        smooth_sigma=float(tc["smooth_sigma"]),
        max_interp_gap=int(tc["max_interp_gap"]),
    )
    return {
        "forward": cache["forward_vector"],
        "wrist_fwd_vel": cache["wrist_forward_velocity"],
        "pelvis_fwd_vel": cache["pelvis_forward_velocity"],
        "pelvis_jerk": cache["pelvis_jerk"],
        "shoulder_ang_vel": cache["shoulder_angular_velocity"],
        "wrist_height_rel": cache["wrist_height_relative"],
        "nb_elbow_y": cache["nb_elbow_y"],
        "bowl_elbow_y": cache["bowling_elbow_y"],
        "hip_ang_vel": cache["hip_line_angular_velocity"],
        "upper_arm_angle": cache["upper_arm_angle"],
        "wrist_vis_raw": cache["wrist_vis_raw"],
        "wrist_vis_weight": cache["wrist_vis_weight"],
        "shoulder_vis_raw": cache["shoulder_vis_raw"],
        "shoulder_vis_weight": cache["shoulder_vis_weight"],
        "pelvis_vis_raw": cache["pelvis_vis_raw"],
        "pelvis_vis_weight": cache["pelvis_vis_weight"],
        "nb_elbow_vis_raw": cache["nb_elbow_vis_raw"],
        "nb_elbow_vis_weight": cache["nb_elbow_vis_weight"],
        "bowl_elbow_vis_raw": cache["bowling_elbow_vis_raw"],
        "bowl_elbow_vis_weight": cache["bowling_elbow_vis_weight"],
        "hips_vis_raw": cache["hips_vis_raw"],
        "hips_vis_weight": cache["hips_vis_weight"],
        "wrist_peak": cache["wrist_peak"],
        "signal_cache": cache,
    }


def _release_search_window(
    signals: Dict[str, Any],
    n_frames: int,
    tc: Dict[str, float],
    delivery_window: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    start = int(tc["search_start"])
    end = max(start + 4, n_frames - int(tc["search_end_margin"]))

    if delivery_window and isinstance(delivery_window, dict):
        try:
            dw_start = int(delivery_window.get("analysis_start") or delivery_window.get("start") or 0)
            dw_end = int(delivery_window.get("analysis_end") or delivery_window.get("end") or (n_frames - 1))
            slack = int(tc["delivery_slack"])
            start = max(0, min(n_frames - 1, dw_start - slack))
            end = max(start + 4, min(n_frames - 1, dw_end + max(1, slack // 2)))
        except Exception:
            pass

    wrist_rel = signals["wrist_height_rel"]
    wrist_vis = signals["wrist_vis_raw"]
    above = np.where((wrist_rel < -0.02) & (wrist_vis >= MIN_VIS_SOFT))[0]
    if len(above) > 0:
        start = max(start, int(above[-1]) - int(tc["wrist_cross_slack"]))

    return max(0, start), min(n_frames - 1, end)


def _build_release_candidates(
    *,
    consensus_frame: int,
    confidence: float,
    start: int,
    end: int,
    signals: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Optional[Dict[str, Any]]] = [
        build_candidate(
            frame=consensus_frame,
            method="multi_signal_consensus",
            confidence=confidence,
            score=1.0,
        )
    ]

    for name, signal, mode in (
        ("nb_elbow_peak", signals["nb_elbow_y"], "peak_min"),
        ("shoulder_ang_vel_peak", signals["shoulder_ang_vel"], "peak_max"),
        ("wrist_fwd_peak", signals["wrist_fwd_vel"], "peak_max"),
        ("pelvis_jerk_quiet", signals["pelvis_jerk"], "peak_min"),
    ):
        frame = _local_peak(signal, start, end, mode=mode)
        if frame is None:
            continue
        score = 0.75 if frame == consensus_frame else 0.55
        candidates.append(
            build_candidate(
                frame=frame,
                method=name,
                confidence=max(0.30, confidence - 0.10),
                score=score,
            )
        )

    return compact_candidates(candidates)


def _release_consensus(
    signals: Dict[str, Any],
    start: int,
    end: int,
    tc: Dict[str, float],
) -> Tuple[Optional[int], float, List[Dict[str, Any]]]:
    vote_sigma = float(tc["vote_sigma"])
    consensus = np.zeros(end - start, dtype=float)
    total_weight = 0.0
    used_signals: List[Dict[str, Any]] = []

    locators = (
        ("nb_elbow_y", signals["nb_elbow_y"], signals["nb_elbow_vis_raw"], signals["nb_elbow_vis_weight"], 0.40, "peak_min"),
        ("shoulder_ang_vel", signals["shoulder_ang_vel"], signals["shoulder_vis_raw"], signals["shoulder_vis_weight"], 0.25, "peak_max"),
        ("wrist_fwd_vel", signals["wrist_fwd_vel"], signals["wrist_vis_raw"], signals["wrist_vis_weight"], 0.20, "peak_max"),
        ("pelvis_jerk", signals["pelvis_jerk"], signals["pelvis_vis_raw"], signals["pelvis_vis_weight"], 0.10, "peak_min"),
    )

    for name, signal, raw_vis, weights, base_weight, mode in locators:
        clip_weight = _clip_mean_weight(raw_vis, weights, start, end)
        eff_weight = base_weight * clip_weight
        if eff_weight < 0.02:
            used_signals.append(
                {
                    "name": name,
                    "clip_vis": round(clip_weight, 3),
                    "effective_weight": 0.0,
                    "used": False,
                }
            )
            continue

        dist = _distribution_from_signal(
            signal,
            raw_vis,
            weights,
            start,
            end,
            vote_sigma,
            mode=mode,
        )
        if dist is None:
            used_signals.append(
                {
                    "name": name,
                    "clip_vis": round(clip_weight, 3),
                    "effective_weight": 0.0,
                    "used": False,
                    "reason": "insufficient_data",
                }
            )
            continue

        consensus += dist * eff_weight
        total_weight += eff_weight
        used_signals.append(
            {
                "name": name,
                "clip_vis": round(clip_weight, 3),
                "effective_weight": round(eff_weight, 3),
                "used": True,
            }
        )

    if total_weight <= 1e-6:
        return None, 0.0, used_signals

    consensus /= total_weight

    pelvis_gate_vis = _clip_mean_weight(
        signals["pelvis_vis_raw"],
        signals["pelvis_vis_weight"],
        start,
        end,
    )
    if pelvis_gate_vis >= 0.20:
        pelvis = signals["pelvis_fwd_vel"][start:end].copy()
        valid = np.isfinite(pelvis)
        if np.any(valid):
            baseline = float(np.nanpercentile(np.abs(pelvis[valid]), 85)) + 1e-9
            gate = 1.0 - np.clip(np.abs(pelvis) / baseline, 0.0, 1.0)
            gate[~valid] = 0.9
            consensus *= (0.95 + 0.05 * gate)
            used_signals.append(
                {
                    "name": "pelvis_fwd_vel_gate",
                    "clip_vis": round(pelvis_gate_vis, 3),
                    "effective_weight": 0.05,
                    "used": True,
                }
            )

    wrist_gate_vis = _clip_mean_weight(
        signals["wrist_vis_raw"],
        signals["wrist_vis_weight"],
        start,
        end,
    )
    if wrist_gate_vis >= 0.25:
        wrist_rel = signals["wrist_height_rel"][start:end]
        wrist_vis = signals["wrist_vis_raw"][start:end]
        post_mask = (wrist_rel > float(tc["post_release_rel"])) & (wrist_vis >= MIN_VIS_SOFT)
        consensus[post_mask] *= 0.20

    peak_local = int(np.argmax(consensus))
    peak_value = float(consensus[peak_local])
    peak_frame = start + peak_local

    peak_window_start = max(0, peak_local - 2)
    peak_window_end = min(len(consensus), peak_local + 3)
    neighborhood = float(np.mean(consensus[peak_window_start:peak_window_end]))
    sharpness = max(0.0, peak_value - neighborhood)
    confidence = min(0.95, peak_value * (0.65 + sharpness * 2.5) * min(1.0, total_weight / 0.65))

    return peak_frame, round(confidence, 2), used_signals


def _uah_window(release_frame: int, tc: Dict[str, float], *, primary: bool = True) -> Tuple[int, int]:
    max_before = tc["uah_primary_max_before_release"] if primary else tc["uah_max_before_release"]
    start = max(0, release_frame - int(max_before))
    end = max(start, release_frame - int(tc["uah_min_before_release"]))
    return start, end


def _detect_uah(
    signals: Dict[str, Any],
    release_frame: int,
    tc: Dict[str, float],
) -> Tuple[int, float, str, List[Dict[str, Any]], Tuple[int, int]]:
    def collect_candidates(start: int, end: int, *, broad_window: bool) -> Tuple[List[Dict[str, Any]], List[Tuple[float, int, str]]]:
        candidates: List[Dict[str, Any]] = []
        ranked: List[Tuple[float, int, str]] = []
        if end <= start:
            return candidates, ranked

        window_len = max(1, end - start)

        def add_candidate(frame: Optional[int], method: str, confidence: float, score: float, reason: Optional[str] = None) -> None:
            if frame is None:
                return
            frame = min(int(frame), release_frame - 1)
            if frame < start or frame > end:
                return
            proximity = (frame - start) / float(window_len)
            selection_score = float(confidence) + 0.30 * proximity + score * 0.05
            if broad_window:
                selection_score -= 0.05
            candidate = build_candidate(
                frame=frame,
                method=method,
                confidence=confidence,
                score=round(selection_score, 4),
                reason=reason,
            )
            if candidate is not None:
                candidates.append(candidate)
                ranked.append((selection_score, frame, method))

        elbow_frame = _local_peak(signals["bowl_elbow_y"], start, end, mode="peak_min")
        elbow_vis = _clip_mean_weight(
            signals["bowl_elbow_vis_raw"],
            signals["bowl_elbow_vis_weight"],
            start,
            end,
        )
        if elbow_frame is not None and elbow_vis >= 0.20:
            elbow_conf = max(0.35, min(0.90, elbow_vis * 0.95))
            add_candidate(elbow_frame, "bowling_elbow_height_minimum", elbow_conf, 1.0)

        shoulder_ang_acc = np.abs(_gradient_nan_safe(signals["shoulder_ang_vel"], float(tc["dt"])))
        hip_ang_acc = _gradient_nan_safe(signals["hip_ang_vel"], float(tc["dt"]))
        kinetic = np.full_like(shoulder_ang_acc, np.nan, dtype=float)
        valid = np.isfinite(shoulder_ang_acc) & np.isfinite(hip_ang_acc)
        kinetic[valid] = np.maximum(0.0, shoulder_ang_acc[valid]) * np.maximum(0.0, -hip_ang_acc[valid])
        kinetic_frame = _local_peak(kinetic, start, end, mode="peak_max")
        kin_vis = min(
            _clip_mean_weight(signals["shoulder_vis_raw"], signals["shoulder_vis_weight"], start, end),
            _clip_mean_weight(signals["hips_vis_raw"], signals["hips_vis_weight"], start, end),
        )
        if kinetic_frame is not None and kin_vis >= 0.20:
            add_candidate(
                kinetic_frame,
                "kinetic_chain_crossover",
                max(0.30, min(0.75, kin_vis * 0.80)),
                0.7,
            )

        upper_arm = signals["upper_arm_angle"][start:end]
        valid_upper = np.isfinite(upper_arm)
        if np.any(valid_upper):
            global_idx = np.arange(start, end)[valid_upper]
            deviations = np.abs(upper_arm[valid_upper] - 90.0)
            best_idx = int(global_idx[int(np.argmin(deviations))])
            if float(np.min(deviations)) <= 25.0:
                deviation = float(np.min(deviations))
                angle_conf = max(0.30, 0.72 - deviation / 60.0)
                add_candidate(
                    best_idx,
                    "upper_arm_horizontal",
                    angle_conf,
                    max(0.0, 1.0 - deviation / 25.0),
                )

        return candidates, ranked

    for broad_window in (False, True):
        start, end = _uah_window(release_frame, tc, primary=not broad_window)
        candidates, ranked = collect_candidates(start, end, broad_window=broad_window)
        chosen = compact_candidates(candidates, limit=4)
        if chosen and ranked:
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_score, best_frame, best_method = ranked[0]
            best_candidate = next(
                (
                    item for item in chosen
                    if int(item.get("frame", -1)) == int(best_frame)
                    and str(item.get("method")) == best_method
                ),
                chosen[0],
            )
            return (
                int(best_frame),
                float(best_candidate["confidence"]),
                str(best_method),
                chosen,
                (start, end),
            )

    fallback = max(0, release_frame - 1)
    chosen = compact_candidates(
        [
            build_candidate(
                frame=fallback,
                method="release_minus_one_fallback",
                confidence=0.20,
                score=0.0,
            )
        ]
    )
    return fallback, 0.20, "release_minus_one_fallback", chosen, (start, end)


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
    signals = _extract_signals(pose_frames, hand, tc)
    search_start, search_end = _release_search_window(signals, n_frames, tc, delivery_window)
    wrist_peak = int(np.clip(signals["wrist_peak"], 0, n_frames - 1))

    release_frame, release_confidence, signals_used = _release_consensus(
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

    release_candidates = _build_release_candidates(
        consensus_frame=release_frame,
        confidence=release_confidence,
        start=search_start,
        end=search_end,
        signals=signals,
    )

    uah_frame, uah_confidence, uah_method, uah_candidates, uah_window = _detect_uah(
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

    return {
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
    }
