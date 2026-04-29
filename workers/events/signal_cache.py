from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.workers.events.timing_constants import signal_cache_timing


# MediaPipe pose landmarks
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24
LA, RA = 27, 28
LFI, RFI = 31, 32

MIN_VIS_HARD = 0.20
MIN_VIS_FULL = 0.80


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
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-10.0 * (raw_vis - 0.5)))))


def _mag(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    mag = _mag(v)
    if mag <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


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


def _compute_forward_direction(pelvis_xy: np.ndarray) -> Tuple[float, float, float]:
    valid = np.isfinite(pelvis_xy[:, 0])
    if valid.sum() < 4:
        return (1.0, 0.0, 0.0)
    idx = np.where(valid)[0]
    steps = pelvis_xy[idx[1:]] - pelvis_xy[idx[:-1]]
    med = np.median(steps, axis=0)
    return _unit((float(med[0]), float(med[1]), 0.0))


def build_signal_cache(
    *,
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
    smooth_sigma: Optional[float] = None,
    max_interp_gap: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(pose_frames)
    timing = signal_cache_timing(fps)
    fps = float(timing["fps"])
    dt = float(timing["dt"])
    sigma = float(smooth_sigma if smooth_sigma is not None else timing["smooth_sigma"])
    max_gap = int(max_interp_gap if max_interp_gap is not None else timing["max_interp_gap"])
    hand = (hand or "R").upper()

    s_idx, e_idx, w_idx = (RS, RE, RW) if hand == "R" else (LS, LE, LW)
    nb_e_idx = LE if hand == "R" else RE
    nb_s_idx = LS if hand == "R" else RS

    wrist_xy = np.full((n, 2), np.nan)
    shoulder_xy = np.full((n, 2), np.nan)
    elbow_xy = np.full((n, 2), np.nan)
    pelvis_xy = np.full((n, 2), np.nan)
    nb_elbow_y = np.full(n, np.nan)
    bowling_elbow_y = np.full(n, np.nan)
    left_shoulder_xy = np.full((n, 2), np.nan)
    right_shoulder_xy = np.full((n, 2), np.nan)
    left_hip_xy = np.full((n, 2), np.nan)
    right_hip_xy = np.full((n, 2), np.nan)
    left_ankle_y = np.full(n, np.nan)
    right_ankle_y = np.full(n, np.nan)
    left_toe_y = np.full(n, np.nan)
    right_toe_y = np.full(n, np.nan)

    wrist_vis_raw = np.zeros(n)
    wrist_vis_weight = np.zeros(n)
    shoulder_vis_raw = np.zeros(n)
    shoulder_vis_weight = np.zeros(n)
    pelvis_vis_raw = np.zeros(n)
    pelvis_vis_weight = np.zeros(n)
    hips_vis_raw = np.zeros(n)
    hips_vis_weight = np.zeros(n)
    nb_elbow_vis_raw = np.zeros(n)
    nb_elbow_vis_weight = np.zeros(n)
    bowling_elbow_vis_raw = np.zeros(n)
    bowling_elbow_vis_weight = np.zeros(n)
    left_ankle_vis = np.zeros(n)
    right_ankle_vis = np.zeros(n)
    left_toe_vis = np.zeros(n)
    right_toe_vis = np.zeros(n)

    shoulder_xyz_list: List[Optional[Tuple[float, float, float]]] = [None] * n
    elbow_xyz_list: List[Optional[Tuple[float, float, float]]] = [None] * n

    for i, frame in enumerate(pose_frames):
        landmarks = frame.get("landmarks") or []

        def pt(idx: int) -> Optional[Any]:
            return landmarks[idx] if idx < len(landmarks) else None

        wrist = pt(w_idx)
        shoulder = pt(s_idx)
        elbow = pt(e_idx)
        nb_elbow = pt(nb_e_idx)
        nb_shoulder = pt(nb_s_idx)
        left_shoulder = pt(LS)
        right_shoulder = pt(RS)
        left_hip = pt(LH)
        right_hip = pt(RH)
        left_ankle = pt(LA)
        right_ankle = pt(RA)
        left_toe = pt(LFI)
        right_toe = pt(RFI)

        if wrist and shoulder:
            w_xyz = _xyz(wrist)
            s_xyz = _xyz(shoulder)
            if w_xyz and s_xyz:
                raw = min(_raw_vis(wrist), _raw_vis(shoulder))
                wrist_xy[i] = [w_xyz[0], w_xyz[1]]
                shoulder_xy[i] = [s_xyz[0], s_xyz[1]]
                wrist_vis_raw[i] = raw
                wrist_vis_weight[i] = _vis_weight(raw)
                shoulder_xyz_list[i] = s_xyz

        if elbow:
            e_xyz = _xyz(elbow)
            if e_xyz:
                raw = _raw_vis(elbow)
                elbow_xy[i] = [e_xyz[0], e_xyz[1]]
                bowling_elbow_y[i] = e_xyz[1]
                bowling_elbow_vis_raw[i] = raw
                bowling_elbow_vis_weight[i] = _vis_weight(raw)
                elbow_xyz_list[i] = e_xyz

        if nb_elbow and nb_shoulder:
            nb_xyz = _xyz(nb_elbow)
            nbs_xyz = _xyz(nb_shoulder)
            if nb_xyz and nbs_xyz:
                raw = min(_raw_vis(nb_elbow), _raw_vis(nb_shoulder))
                nb_elbow_y[i] = nb_xyz[1]
                nb_elbow_vis_raw[i] = raw
                nb_elbow_vis_weight[i] = _vis_weight(raw)

        if left_shoulder and right_shoulder:
            ls_xyz = _xyz(left_shoulder)
            rs_xyz = _xyz(right_shoulder)
            if ls_xyz and rs_xyz:
                raw = min(_raw_vis(left_shoulder), _raw_vis(right_shoulder))
                left_shoulder_xy[i] = [ls_xyz[0], ls_xyz[1]]
                right_shoulder_xy[i] = [rs_xyz[0], rs_xyz[1]]
                shoulder_vis_raw[i] = raw
                shoulder_vis_weight[i] = _vis_weight(raw)

        if left_hip and right_hip:
            lh_xyz = _xyz(left_hip)
            rh_xyz = _xyz(right_hip)
            if lh_xyz and rh_xyz:
                raw = min(_raw_vis(left_hip), _raw_vis(right_hip))
                pelvis_xy[i] = [(lh_xyz[0] + rh_xyz[0]) * 0.5, (lh_xyz[1] + rh_xyz[1]) * 0.5]
                left_hip_xy[i] = [lh_xyz[0], lh_xyz[1]]
                right_hip_xy[i] = [rh_xyz[0], rh_xyz[1]]
                pelvis_vis_raw[i] = raw
                pelvis_vis_weight[i] = _vis_weight(raw)
                hips_vis_raw[i] = raw
                hips_vis_weight[i] = _vis_weight(raw)

        if left_ankle:
            la_xyz = _xyz(left_ankle)
            if la_xyz:
                left_ankle_y[i] = la_xyz[1]
                left_ankle_vis[i] = _raw_vis(left_ankle)
        if right_ankle:
            ra_xyz = _xyz(right_ankle)
            if ra_xyz:
                right_ankle_y[i] = ra_xyz[1]
                right_ankle_vis[i] = _raw_vis(right_ankle)
        if left_toe:
            lt_xyz = _xyz(left_toe)
            if lt_xyz:
                left_toe_y[i] = lt_xyz[1]
                left_toe_vis[i] = _raw_vis(left_toe)
        if right_toe:
            rt_xyz = _xyz(right_toe)
            if rt_xyz:
                right_toe_y[i] = rt_xyz[1]
                right_toe_vis[i] = _raw_vis(right_toe)

    forward = _compute_forward_direction(pelvis_xy)

    wrist_x = _smooth_nan_safe(_interp_short_gaps(wrist_xy[:, 0], max_gap), sigma)
    wrist_y = _smooth_nan_safe(_interp_short_gaps(wrist_xy[:, 1], max_gap), sigma)
    shoulder_y = _smooth_nan_safe(_interp_short_gaps(shoulder_xy[:, 1], max_gap), sigma)
    pelvis_x = _smooth_nan_safe(_interp_short_gaps(pelvis_xy[:, 0], max_gap), sigma)
    pelvis_y = _smooth_nan_safe(_interp_short_gaps(pelvis_xy[:, 1], max_gap), sigma)
    nb_elbow_y_s = _smooth_nan_safe(_interp_short_gaps(nb_elbow_y, max_gap), sigma)
    bowling_elbow_y_s = _smooth_nan_safe(_interp_short_gaps(bowling_elbow_y, max_gap), sigma)
    left_shoulder_x = _smooth_nan_safe(_interp_short_gaps(left_shoulder_xy[:, 0], max_gap), sigma)
    left_shoulder_y = _smooth_nan_safe(_interp_short_gaps(left_shoulder_xy[:, 1], max_gap), sigma)
    right_shoulder_x = _smooth_nan_safe(_interp_short_gaps(right_shoulder_xy[:, 0], max_gap), sigma)
    right_shoulder_y = _smooth_nan_safe(_interp_short_gaps(right_shoulder_xy[:, 1], max_gap), sigma)
    left_hip_x = _smooth_nan_safe(_interp_short_gaps(left_hip_xy[:, 0], max_gap), sigma)
    left_hip_y = _smooth_nan_safe(_interp_short_gaps(left_hip_xy[:, 1], max_gap), sigma)
    right_hip_x = _smooth_nan_safe(_interp_short_gaps(right_hip_xy[:, 0], max_gap), sigma)
    right_hip_y = _smooth_nan_safe(_interp_short_gaps(right_hip_xy[:, 1], max_gap), sigma)
    left_ankle_y_s = _smooth_nan_safe(_interp_short_gaps(left_ankle_y, max_gap), sigma)
    right_ankle_y_s = _smooth_nan_safe(_interp_short_gaps(right_ankle_y, max_gap), sigma)
    left_toe_y_s = _smooth_nan_safe(_interp_short_gaps(left_toe_y, max_gap), sigma)
    right_toe_y_s = _smooth_nan_safe(_interp_short_gaps(right_toe_y, max_gap), sigma)

    wrist_dx = _gradient_nan_safe(wrist_x, dt)
    wrist_dy = _gradient_nan_safe(wrist_y, dt)
    pelvis_dx = _gradient_nan_safe(pelvis_x, dt)
    pelvis_dy = _gradient_nan_safe(pelvis_y, dt)

    wrist_forward_velocity = wrist_dx * forward[0] + wrist_dy * forward[1]
    pelvis_forward_velocity = pelvis_dx * forward[0] + pelvis_dy * forward[1]
    pelvis_linear_speed = np.sqrt(np.square(pelvis_dx) + np.square(pelvis_dy))
    pelvis_jerk = np.abs(_gradient_nan_safe(pelvis_forward_velocity, dt))

    shoulder_dx = right_shoulder_x - left_shoulder_x
    shoulder_dy = right_shoulder_y - left_shoulder_y
    shoulder_angle = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))
    shoulder_angle_unwrapped = np.degrees(np.unwrap(np.radians(np.nan_to_num(shoulder_angle, nan=0.0))))
    shoulder_angular_velocity = np.abs(_gradient_nan_safe(shoulder_angle_unwrapped, dt))

    hip_dx = right_hip_x - left_hip_x
    hip_dy = right_hip_y - left_hip_y
    hip_line_angle = np.degrees(np.arctan2(hip_dy, hip_dx))
    hip_line_angle_unwrapped = np.degrees(np.unwrap(np.radians(np.nan_to_num(hip_line_angle, nan=0.0))))
    hip_line_angular_velocity = np.abs(_gradient_nan_safe(hip_line_angle_unwrapped, dt))
    pelvis_angular_velocity = hip_line_angular_velocity.copy()

    wrist_height_relative = wrist_y - shoulder_y

    upper_arm_angle = np.full(n, np.nan, dtype=float)
    for i in range(n):
        ang = _upper_arm_angle(shoulder_xyz_list[i], elbow_xyz_list[i], forward)
        if ang is not None:
            upper_arm_angle[i] = float(ang)
    upper_arm_angle = _smooth_nan_safe(_interp_short_gaps(upper_arm_angle, max_gap), sigma)

    wrist_peak = int(np.argmax(np.nan_to_num(wrist_forward_velocity, nan=-np.inf))) if n else 0

    return {
        "n_frames": n,
        "fps": fps,
        "dt": dt,
        "smooth_sigma": sigma,
        "max_interp_gap": max_gap,
        "forward_vector": forward,
        "pelvis_centre_xy": np.column_stack((pelvis_x, pelvis_y)),
        "pelvis_forward_velocity": pelvis_forward_velocity,
        "pelvis_linear_speed": pelvis_linear_speed,
        "pelvis_angular_velocity": pelvis_angular_velocity,
        "pelvis_jerk": pelvis_jerk,
        "hip_line_angular_velocity": hip_line_angular_velocity,
        "shoulder_angular_velocity": shoulder_angular_velocity,
        "bowling_elbow_y": bowling_elbow_y_s,
        "nb_elbow_y": nb_elbow_y_s,
        "wrist_forward_velocity": wrist_forward_velocity,
        "wrist_height_relative": wrist_height_relative,
        "upper_arm_angle": upper_arm_angle,
        "wrist_vis_raw": wrist_vis_raw,
        "wrist_vis_weight": wrist_vis_weight,
        "shoulder_vis_raw": shoulder_vis_raw,
        "shoulder_vis_weight": shoulder_vis_weight,
        "pelvis_vis_raw": pelvis_vis_raw,
        "pelvis_vis_weight": pelvis_vis_weight,
        "hips_vis_raw": hips_vis_raw,
        "hips_vis_weight": hips_vis_weight,
        "bowling_elbow_vis_raw": bowling_elbow_vis_raw,
        "bowling_elbow_vis_weight": bowling_elbow_vis_weight,
        "nb_elbow_vis_raw": nb_elbow_vis_raw,
        "nb_elbow_vis_weight": nb_elbow_vis_weight,
        "left_ankle_y": left_ankle_y_s,
        "right_ankle_y": right_ankle_y_s,
        "left_toe_y": left_toe_y_s,
        "right_toe_y": right_toe_y_s,
        "left_ankle_vis": left_ankle_vis,
        "right_ankle_vis": right_ankle_vis,
        "left_toe_vis": left_toe_vis,
        "right_toe_vis": right_toe_vis,
        "wrist_peak": wrist_peak,
    }
