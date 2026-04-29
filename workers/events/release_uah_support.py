from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.workers.events.event_confidence import build_candidate, compact_candidates
from app.workers.events.signal_cache import build_signal_cache

MIN_VIS_HARD = 0.20
MIN_VIS_SOFT = 0.35


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


def _late_release_cap(
    signals: Dict[str, Any],
    start: int,
    end: int,
    tc: Dict[str, float],
) -> int:
    nb_vis = _clip_mean_weight(
        signals["nb_elbow_vis_raw"],
        signals["nb_elbow_vis_weight"],
        start,
        end,
    )
    nb_peak = _local_peak(signals["nb_elbow_y"], start, end, mode="peak_min")
    if nb_peak is None or nb_vis < 0.25:
        return end

    fps = float(tc["fps"])
    agree_gap = max(3, int(round(fps * 0.20)))
    tail_frames = max(5, int(round(fps * 0.28)))
    anchor_frames = [nb_peak]

    wrist_peak = _local_peak(signals["wrist_fwd_vel"], start, end, mode="peak_max")
    if wrist_peak is not None and abs(int(wrist_peak) - int(nb_peak)) <= agree_gap:
        anchor_frames.append(int(wrist_peak))

    shoulder_peak = _local_peak(signals["shoulder_ang_vel"], start, end, mode="peak_max")
    if shoulder_peak is not None and abs(int(shoulder_peak) - int(nb_peak)) <= agree_gap:
        anchor_frames.append(int(shoulder_peak))

    reference = int(round(float(np.median(anchor_frames))))
    return min(end, reference + tail_frames)


def refine_release_frame(
    signals: Dict[str, Any],
    consensus_frame: Optional[int],
    tc: Dict[str, float],
    n_frames: int,
) -> Optional[int]:
    if consensus_frame is None:
        return None

    half = int(tc["geometric_refine_half"])
    start = max(0, int(consensus_frame) - half)
    end = min(n_frames - 1, int(consensus_frame) + half)

    wrist_vis = signals["wrist_vis_raw"]
    wrist_rel = signals["wrist_height_rel"]
    wrist_vel = signals["wrist_fwd_vel"]

    visible = [
        idx for idx in range(start, end + 1)
        if idx < len(wrist_vis) and wrist_vis[idx] >= MIN_VIS_HARD and np.isfinite(wrist_rel[idx])
    ]
    if len(visible) < 3:
        return int(consensus_frame)

    best_frame = int(consensus_frame)
    best_score = float("inf")
    for idx in visible:
        height_dev = abs(float(wrist_rel[idx]))
        vel = float(np.nan_to_num(wrist_vel[idx], nan=0.0))
        vel_penalty = max(0.0, -vel) * 0.5
        score = height_dev + vel_penalty
        if score < best_score:
            best_score = score
            best_frame = idx

    if abs(int(best_frame) - int(consensus_frame)) > 2:
        return int(consensus_frame)
    return int(best_frame)


def extract_signals(
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


def release_search_window(
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
    end = _late_release_cap(signals, start, end, tc)
    end = max(start + 4, end)

    return max(0, start), min(n_frames - 1, end)


def build_release_candidates(
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
        ("pelvis_jerk_peak", signals["pelvis_jerk"], "peak_max"),
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


def release_consensus(
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
        ("pelvis_jerk", signals["pelvis_jerk"], signals["pelvis_vis_raw"], signals["pelvis_vis_weight"], 0.10, "peak_max"),
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


def detect_uah(
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

    start = 0
    end = 0
    for broad_window in (False, True):
        start, end = _uah_window(release_frame, tc, primary=not broad_window)
        candidates, ranked = collect_candidates(start, end, broad_window=broad_window)
        chosen = compact_candidates(candidates, limit=4)
        if chosen and ranked:
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            _, best_frame, best_method = ranked[0]
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
