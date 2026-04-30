from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

BALL_WEIGHT_OZ = 5.25
MIN_VIS = 0.35
METHOD = "release_kinematics_research_v2"

LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28


def _get_landmark(
    landmarks: Optional[List[Dict[str, Any]]],
    idx: int,
    *,
    min_vis: float = MIN_VIS,
) -> Optional[Dict[str, float]]:
    if not isinstance(landmarks, list) or idx < 0 or idx >= len(landmarks):
        return None
    point = landmarks[idx]
    if not isinstance(point, dict):
        return None
    try:
        if float(point.get("visibility", 0.0)) < min_vis:
            return None
        return {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "z": float(point.get("z", 0.0)),
            "visibility": float(point.get("visibility", 0.0)),
        }
    except Exception:
        return None


def _pixel_xy(
    point: Optional[Dict[str, float]],
    *,
    width: float,
    height: float,
) -> Optional[Tuple[float, float]]:
    if not point or width <= 0.0 or height <= 0.0:
        return None
    return (float(point["x"]) * width, float(point["y"]) * height)


def _distance(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _interpolate_series(values: List[Optional[float]]) -> Optional[np.ndarray]:
    n = len(values)
    if n == 0:
        return None
    mask = np.array([value is not None for value in values], dtype=bool)
    if mask.sum() < 3:
        return None
    idx = np.arange(n, dtype=float)
    arr = np.zeros(n, dtype=float)
    arr[mask] = [float(value) for value in values if value is not None]
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def _smoothed_track(
    points: List[Optional[Tuple[float, float]]],
    *,
    fps: float,
    sigma_scale: float = 0.03,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    xs = _interpolate_series([point[0] if point else None for point in points])
    ys = _interpolate_series([point[1] if point else None for point in points])
    if xs is None or ys is None:
        return None
    sigma = max(1.0, sigma_scale * fps)
    return (
        gaussian_filter1d(xs, sigma=sigma),
        gaussian_filter1d(ys, sigma=sigma),
    )


def _smoothed_series(
    values: List[Optional[float]],
    *,
    fps: float,
    sigma_scale: float = 0.03,
) -> Optional[np.ndarray]:
    series = _interpolate_series(values)
    if series is None:
        return None
    sigma = max(1.0, sigma_scale * fps)
    return gaussian_filter1d(series, sigma=sigma)


def _body_height_px(
    pose_frames: List[Dict[str, Any]],
    frame_idx: int,
    *,
    height: float,
) -> Optional[float]:
    if frame_idx < 0 or frame_idx >= len(pose_frames) or height <= 0.0:
        return None
    landmarks = (pose_frames[frame_idx] or {}).get("landmarks")
    y_values: List[float] = []
    for idx in (LS, RS, LH, RH, LK, RK, LA, RA):
        point = _get_landmark(landmarks, idx, min_vis=0.25)
        if point:
            y_values.append(point["y"] * height)
    if len(y_values) < 4:
        return None
    span = max(y_values) - min(y_values)
    return span if span >= 60.0 else None


def _arm_length_px(
    landmarks: Optional[List[Dict[str, Any]]],
    *,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    width: float,
    height: float,
) -> Optional[float]:
    shoulder = _pixel_xy(_get_landmark(landmarks, shoulder_idx), width=width, height=height)
    elbow = _pixel_xy(_get_landmark(landmarks, elbow_idx), width=width, height=height)
    wrist = _pixel_xy(_get_landmark(landmarks, wrist_idx), width=width, height=height)
    upper = _distance(shoulder, elbow)
    lower = _distance(elbow, wrist)
    if upper is None or lower is None:
        return None
    total = upper + lower
    return total if total >= 20.0 else None


def _elbow_angle_deg(
    landmarks: Optional[List[Dict[str, Any]]],
    *,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
) -> Optional[float]:
    shoulder = _get_landmark(landmarks, shoulder_idx)
    elbow = _get_landmark(landmarks, elbow_idx)
    wrist = _get_landmark(landmarks, wrist_idx)
    if shoulder is None or elbow is None or wrist is None:
        return None

    a = np.array(
        [float(shoulder["x"] - elbow["x"]), float(shoulder["y"] - elbow["y"])],
        dtype=float,
    )
    b = np.array(
        [float(wrist["x"] - elbow["x"]), float(wrist["y"] - elbow["y"])],
        dtype=float,
    )
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return None
    dot = max(-1.0, min(1.0, float(np.dot(a, b) / (norm_a * norm_b))))
    return math.degrees(math.acos(dot))


def _window_indices(
    release_frame: int,
    total_frames: int,
    *,
    before: int = 3,
    after: int = 3,
) -> List[int]:
    start = max(0, int(release_frame) - before)
    end = min(int(total_frames), int(release_frame) + after + 1)
    return list(range(start, end))


def _median_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(median(valid))


def _percentile_or_none(values: List[Optional[float]], percentile: float) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(np.percentile(np.asarray(valid, dtype=float), percentile))


def _cv(values: List[Optional[float]]) -> float:
    valid = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if valid.size == 0:
        return float("inf")
    return float(np.std(valid) / max(float(np.mean(valid)), 1.0))


def _unavailable_result(ball_weight_oz: float) -> Dict[str, Any]:
    return {
        "available": False,
        "display_policy": "suppress",
        "method": METHOD,
        "confidence": 0.0,
        "ball_weight_oz": float(ball_weight_oz),
    }


def _estimate_release_speed_pass(
    *,
    pose_frames: List[Dict[str, Any]],
    video: Dict[str, Any],
    hand: str,
    release_frame: int,
    ball_weight_oz: float,
    window_before: int,
    window_after: int,
    scale_before: int,
    scale_after: int,
    sigma_scale: float,
    max_arm_cv: float,
    max_wrist_cv: float,
    salvage_mode: bool,
    release_confidence: float,
) -> Dict[str, Any]:
    fps = float(video.get("fps") or 0.0)
    width = float(video.get("width") or 0.0)
    height = float(video.get("height") or 0.0)
    unavailable = _unavailable_result(ball_weight_oz)

    if fps <= 0.0 or width <= 0.0 or height <= 0.0:
        return {**unavailable, "reason": "missing_video_geometry"}

    if release_frame < window_before:
        return {**unavailable, "reason": "release_too_close_to_clip_start"}
    if release_frame + window_after >= len(pose_frames):
        return {**unavailable, "reason": "release_too_close_to_clip_end"}

    h = (hand or "R").upper()
    shoulder_idx, elbow_idx, wrist_idx = (
        (RS, RE, RW) if h == "R" else (LS, LE, LW)
    )

    wrist_points: List[Optional[Tuple[float, float]]] = []
    shoulder_points: List[Optional[Tuple[float, float]]] = []
    pelvis_points: List[Optional[Tuple[float, float]]] = []
    wrist_depths: List[Optional[float]] = []
    arm_lengths: List[Optional[float]] = []
    body_heights: List[Optional[float]] = []
    elbow_angles: List[Optional[float]] = []

    for frame in pose_frames:
        landmarks = (frame or {}).get("landmarks")
        wrist_points.append(
            _pixel_xy(_get_landmark(landmarks, wrist_idx), width=width, height=height)
        )
        wrist_depths.append(
            float(_get_landmark(landmarks, wrist_idx)["z"])
            if _get_landmark(landmarks, wrist_idx) is not None
            else None
        )
        shoulder_points.append(
            _pixel_xy(_get_landmark(landmarks, shoulder_idx), width=width, height=height)
        )

        left_hip = _pixel_xy(_get_landmark(landmarks, LH), width=width, height=height)
        right_hip = _pixel_xy(_get_landmark(landmarks, RH), width=width, height=height)
        if left_hip and right_hip:
            pelvis_points.append(
                (
                    (left_hip[0] + right_hip[0]) * 0.5,
                    (left_hip[1] + right_hip[1]) * 0.5,
                )
            )
        else:
            pelvis_points.append(None)

        arm_lengths.append(
            _arm_length_px(
                landmarks,
                shoulder_idx=shoulder_idx,
                elbow_idx=elbow_idx,
                wrist_idx=wrist_idx,
                width=width,
                height=height,
            )
        )
        body_heights.append(
            _body_height_px(
                pose_frames,
                int((frame or {}).get("frame", len(body_heights))),
                height=height,
            )
        )
        elbow_angles.append(
            _elbow_angle_deg(
                landmarks,
                shoulder_idx=shoulder_idx,
                elbow_idx=elbow_idx,
                wrist_idx=wrist_idx,
            )
        )

    wrist_track = _smoothed_track(wrist_points, fps=fps, sigma_scale=sigma_scale)
    shoulder_track = _smoothed_track(shoulder_points, fps=fps, sigma_scale=sigma_scale)
    pelvis_track = _smoothed_track(pelvis_points, fps=fps, sigma_scale=sigma_scale)
    wrist_depth_series = _smoothed_series(wrist_depths, fps=fps, sigma_scale=sigma_scale)
    arm_series = _interpolate_series(arm_lengths)
    body_series = _interpolate_series(body_heights)
    elbow_series = _interpolate_series(elbow_angles)

    if (
        wrist_track is None
        or shoulder_track is None
        or pelvis_track is None
        or wrist_depth_series is None
        or arm_series is None
        or body_series is None
        or elbow_series is None
    ):
        return {**unavailable, "reason": "insufficient_release_landmarks"}

    sigma = max(1.0, sigma_scale * fps)
    arm_series = gaussian_filter1d(arm_series, sigma=sigma)
    body_series = gaussian_filter1d(body_series, sigma=sigma)
    elbow_series = gaussian_filter1d(elbow_series, sigma=sigma)

    dt = 1.0 / fps
    wrist_speed = np.hypot(np.gradient(wrist_track[0], dt), np.gradient(wrist_track[1], dt))
    shoulder_speed = np.hypot(
        np.gradient(shoulder_track[0], dt),
        np.gradient(shoulder_track[1], dt),
    )
    pelvis_speed = np.hypot(
        np.gradient(pelvis_track[0], dt),
        np.gradient(pelvis_track[1], dt),
    )
    wrist_depth_speed_norm = np.abs(np.gradient(wrist_depth_series, dt))
    elbow_extension_speed = np.abs(np.gradient(elbow_series, dt))

    metric_window = _window_indices(
        release_frame,
        len(pose_frames),
        before=window_before,
        after=window_after,
    )
    scale_window = _window_indices(
        release_frame,
        len(pose_frames),
        before=scale_before,
        after=scale_after,
    )

    arm_metric = [float(arm_series[idx]) for idx in metric_window]
    body_metric = [float(body_series[idx]) for idx in metric_window]
    wrist_metric = [float(wrist_speed[idx]) for idx in metric_window]
    shoulder_metric = [float(shoulder_speed[idx]) for idx in metric_window]
    pelvis_metric = [float(pelvis_speed[idx]) for idx in metric_window]
    wrist_depth_metric_norm = [float(wrist_depth_speed_norm[idx]) for idx in metric_window]
    elbow_metric = [float(elbow_extension_speed[idx]) for idx in metric_window]
    arm_scale_metric = [float(arm_series[idx]) for idx in scale_window]
    body_scale_metric = [float(body_series[idx]) for idx in scale_window]

    arm_length_px = _median_or_none(arm_scale_metric if salvage_mode else arm_metric)
    body_height_px = _median_or_none(body_scale_metric if salvage_mode else body_metric)
    if arm_length_px is None or body_height_px is None or body_height_px <= 1.0:
        return {**unavailable, "reason": "missing_body_scale"}

    soft_release_recovery_mode = (not salvage_mode) and release_confidence < 0.60

    if salvage_mode:
        wrist_reference = _percentile_or_none(wrist_metric, 75.0)
        elbow_velocity = _percentile_or_none(elbow_metric, 70.0)
        shoulder_reference = _median_or_none(shoulder_metric)
        pelvis_reference = _median_or_none(pelvis_metric)
    elif soft_release_recovery_mode:
        wrist_reference = max(
            _median_or_none(wrist_metric) or 0.0,
            _percentile_or_none(wrist_metric, 65.0) or 0.0,
        )
        elbow_velocity = max(
            _median_or_none(elbow_metric) or 0.0,
            _percentile_or_none(elbow_metric, 70.0) or 0.0,
        )
        shoulder_reference = _median_or_none(shoulder_metric)
        pelvis_reference = max(
            _median_or_none(pelvis_metric) or 0.0,
            _percentile_or_none(pelvis_metric, 60.0) or 0.0,
        )
    else:
        wrist_reference = _median_or_none(wrist_metric)
        elbow_velocity = _median_or_none(elbow_metric)
        shoulder_reference = _median_or_none(shoulder_metric)
        pelvis_reference = _median_or_none(pelvis_metric)

    if (
        wrist_reference is None
        or elbow_velocity is None
        or shoulder_reference is None
        or pelvis_reference is None
    ):
        return {**unavailable, "reason": "insufficient_release_landmarks"}

    wrist_arm_ratio = float(wrist_reference / max(arm_length_px, 1.0))
    shoulder_body_ratio = float(shoulder_reference / max(body_height_px, 1.0))
    pelvis_body_ratio = float(pelvis_reference / max(body_height_px, 1.0))
    elbow_velocity = float(elbow_velocity)

    arm_length_cv_local = _cv(arm_metric)
    arm_length_cv_broad = _cv(arm_scale_metric)
    arm_length_cv = min(arm_length_cv_local, arm_length_cv_broad) if salvage_mode else arm_length_cv_local
    wrist_window_cv = _cv(wrist_metric)
    overall_wrist_visibility = sum(1.0 for point in wrist_points if point is not None) / float(len(wrist_points))

    if arm_length_cv > max_arm_cv:
        return {
            **unavailable,
            "reason": "unstable_arm_scale",
            "debug": {
                "arm_length_cv": round(arm_length_cv, 3),
                "arm_length_cv_local": round(arm_length_cv_local, 3),
                "arm_length_cv_broad": round(arm_length_cv_broad, 3),
                "salvage_mode": salvage_mode,
            },
        }

    scoring_wrist_arm_ratio = min(wrist_arm_ratio, 10.0) if salvage_mode else wrist_arm_ratio
    scoring_elbow_velocity = min(elbow_velocity, 220.0) if salvage_mode else elbow_velocity
    scoring_shoulder_body_ratio = shoulder_body_ratio

    depth_scale_px = max(float(body_height_px) * 1.6, float(width) * 2.4)
    wrist_depth_reference = max(
        _median_or_none([value * depth_scale_px for value in wrist_depth_metric_norm]) or 0.0,
        _percentile_or_none([value * depth_scale_px for value in wrist_depth_metric_norm], 85.0) or 0.0,
    )
    depth_wrist_arm_ratio = (
        float(wrist_depth_reference / max(arm_length_px, 1.0))
        if wrist_depth_reference is not None
        else None
    )
    depth_dominant_mode = (
        not salvage_mode
        and release_confidence >= 0.60
        and overall_wrist_visibility >= 0.75
        and shoulder_body_ratio >= 0.50
        and pelvis_body_ratio <= 0.35
        and depth_wrist_arm_ratio is not None
        and depth_wrist_arm_ratio >= max(5.0, wrist_arm_ratio * 2.2)
    )
    if depth_dominant_mode:
        scoring_wrist_arm_ratio = max(scoring_wrist_arm_ratio, min(depth_wrist_arm_ratio, 24.0))
        scoring_elbow_velocity = max(
            scoring_elbow_velocity,
            _percentile_or_none(elbow_metric, 85.0) or scoring_elbow_velocity,
        )
        scoring_shoulder_body_ratio = min(scoring_shoulder_body_ratio, 0.28)

    release_score = (
        68.0
        + (0.22 * scoring_elbow_velocity)
        + (1.25 * scoring_wrist_arm_ratio)
        + (4.0 * pelvis_body_ratio)
        - (14.0 * scoring_shoulder_body_ratio)
    )

    weight_factor = math.pow(max(0.1, BALL_WEIGHT_OZ / max(ball_weight_oz, 0.1)), 0.05)
    value_kph = release_score * weight_factor

    saturated = False
    if value_kph < 75.0:
        value_kph = 75.0
        saturated = True
    if value_kph > 145.0:
        value_kph = 145.0
        saturated = True

    visibility_window = _window_indices(
        release_frame,
        len(pose_frames),
        before=min(1, window_before),
        after=min(1, window_after),
    )
    visibility_parts = [
        1.0 if wrist_points[idx] is not None else 0.0
        for idx in visibility_window
    ] + [
        1.0 if shoulder_points[idx] is not None else 0.0
        for idx in visibility_window
    ]
    visibility_score = sum(visibility_parts) / float(len(visibility_parts))
    scale_stability_score = max(0.0, 1.0 - min(1.0, arm_length_cv / 0.16))
    motion_stability_score = max(0.0, 1.0 - min(1.0, wrist_window_cv / (0.70 if salvage_mode else 0.50)))
    shoulder_penalty = max(0.0, min(1.0, (shoulder_body_ratio - 0.35) / 0.75))
    confidence = (
        (0.30 * visibility_score)
        + (0.25 * scale_stability_score)
        + (0.20 * motion_stability_score)
        + (0.15 * min(1.0, elbow_velocity / 180.0))
        + (0.10 * max(0.0, 1.0 - shoulder_penalty))
    )
    if saturated:
        confidence *= 0.75
    if salvage_mode:
        confidence *= 0.88
    confidence = max(0.20, min(0.90, confidence))

    if salvage_mode:
        low_visibility_penalty = max(0.0, min(0.14, (0.40 - overall_wrist_visibility) * 0.80))
        small_subject_penalty = max(0.0, min(0.05, (160.0 - body_height_px) / 1000.0))
        body_height_ratio = body_height_px / max(height, 1.0)
        close_camera_penalty = max(0.0, min(0.12, (body_height_ratio - 0.22) * 1.50))
        perspective_penalty = max(0.0, min(0.12, (shoulder_body_ratio - 0.60) * 0.90))
        salvage_penalty = (
            low_visibility_penalty
            + small_subject_penalty
            + close_camera_penalty
            + perspective_penalty
        )
        value_kph *= max(0.80, 1.0 - salvage_penalty)

    if wrist_window_cv > max_wrist_cv:
        return {
            **unavailable,
            "reason": "unstable_release_window",
            "debug": {
                "release_frame": int(release_frame),
                "wrist_window_cv": round(wrist_window_cv, 3),
                "arm_length_cv": round(arm_length_cv, 3),
                "salvage_mode": salvage_mode,
            },
        }

    if confidence < (0.30 if salvage_mode else 0.35):
        return {
            **unavailable,
            "reason": "low_confidence_estimate",
            "debug": {
                "release_frame": int(release_frame),
                "wrist_arm_ratio": round(wrist_arm_ratio, 3),
                "shoulder_body_ratio": round(shoulder_body_ratio, 3),
                "pelvis_body_ratio": round(pelvis_body_ratio, 3),
                "elbow_extension_velocity_deg_per_sec": round(elbow_velocity, 1),
                "arm_length_cv": round(arm_length_cv, 3),
                "wrist_window_cv": round(wrist_window_cv, 3),
                "salvage_mode": salvage_mode,
            },
        }

    rounded_kph = int(round(value_kph))
    return {
        "available": True,
        "display_policy": "show_low_confidence" if salvage_mode else "show",
        "value_kph": rounded_kph,
        "display": f"~{rounded_kph} km/h",
        "confidence": round(confidence, 2),
        "method": f"{METHOD}_salvage" if salvage_mode else METHOD,
        "ball_weight_oz": float(ball_weight_oz),
        "reason": None if not salvage_mode else "salvaged_recovery_pass",
        "debug": {
            "release_frame": int(release_frame),
            "wrist_arm_ratio": round(wrist_arm_ratio, 3),
            "scoring_wrist_arm_ratio": round(scoring_wrist_arm_ratio, 3),
            "shoulder_body_ratio": round(shoulder_body_ratio, 3),
            "scoring_shoulder_body_ratio": round(scoring_shoulder_body_ratio, 3),
            "pelvis_body_ratio": round(pelvis_body_ratio, 3),
            "elbow_extension_velocity_deg_per_sec": round(elbow_velocity, 1),
            "scoring_elbow_velocity_deg_per_sec": round(scoring_elbow_velocity, 1),
            "wrist_depth_reference": round(float(wrist_depth_reference), 1)
            if wrist_depth_reference is not None
            else None,
            "depth_wrist_arm_ratio": round(depth_wrist_arm_ratio, 3)
            if depth_wrist_arm_ratio is not None
            else None,
            "depth_scale_px": round(depth_scale_px, 1),
            "arm_length_px": round(float(arm_length_px), 1),
            "body_height_px": round(float(body_height_px), 1),
            "arm_length_cv": round(arm_length_cv, 3),
            "arm_length_cv_local": round(arm_length_cv_local, 3),
            "arm_length_cv_broad": round(arm_length_cv_broad, 3),
            "wrist_window_cv": round(wrist_window_cv, 3),
            "saturated": saturated,
            "salvage_mode": salvage_mode,
            "soft_release_recovery_mode": soft_release_recovery_mode,
            "depth_dominant_mode": depth_dominant_mode,
            "release_confidence": round(float(release_confidence), 3),
            "overall_wrist_visibility": round(overall_wrist_visibility, 3),
            "body_height_ratio": round(body_height_px / max(height, 1.0), 3),
            "perspective_penalty_trigger_ratio": round(shoulder_body_ratio, 3),
            "window_before": window_before,
            "window_after": window_after,
        },
    }


def _apply_low_confidence_neighbor_recovery(
    primary: Dict[str, Any],
    neighbor_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not primary.get("available") or primary.get("display_policy") != "show":
        return primary

    primary_value = int(primary.get("value_kph") or 0)
    primary_confidence = float(primary.get("confidence") or 0.0)
    stable_confidence_floor = max(0.55, primary_confidence - 0.03)

    stable_values: List[int] = []
    stable_frames: List[int] = []
    for result in neighbor_results:
        if not result.get("available") or result.get("display_policy") != "show":
            continue
        debug = result.get("debug") or {}
        if bool(debug.get("saturated")):
            continue
        confidence = float(result.get("confidence") or 0.0)
        value_kph = result.get("value_kph")
        if value_kph is None or confidence < stable_confidence_floor:
            continue
        stable_values.append(int(value_kph))
        if debug.get("release_frame") is not None:
            stable_frames.append(int(debug["release_frame"]))

    if len(stable_values) < 3:
        return primary

    recovered_value = int(round(float(np.percentile(np.asarray(stable_values, dtype=float), 75.0))))
    if recovered_value <= primary_value + 4:
        return primary

    recovered = dict(primary)
    recovered["value_kph"] = recovered_value
    recovered["display"] = f"~{recovered_value} km/h"
    recovered["debug"] = {
        **dict(primary.get("debug") or {}),
        "low_confidence_neighbor_recovery": True,
        "neighbor_recovery_values": stable_values,
        "neighbor_recovery_frames": stable_frames,
        "neighbor_recovery_confidence_floor": round(stable_confidence_floor, 2),
        "primary_value_kph": primary_value,
    }
    return recovered


def _apply_clean_salvage_promotion(
    result: Dict[str, Any],
    *,
    events: Dict[str, Any],
) -> Dict[str, Any]:
    if not result.get("available"):
        return result
    if not str(result.get("method") or "").endswith("_salvage"):
        return result
    if str(result.get("reason") or "") == "recovered_implausible_saturation":
        return result

    chain = (events or {}).get("event_chain") or {}
    ordered = bool(chain.get("ordered"))
    chain_quality = float(chain.get("quality") or 0.0)
    release_confidence = float(((events or {}).get("release") or {}).get("confidence") or 0.0)
    debug = dict(result.get("debug") or {})

    shoulder_body_ratio = float(debug.get("shoulder_body_ratio") or 0.0)
    pelvis_body_ratio = float(debug.get("pelvis_body_ratio") or 0.0)
    wrist_arm_ratio = float(debug.get("wrist_arm_ratio") or 0.0)
    elbow_velocity = float(debug.get("elbow_extension_velocity_deg_per_sec") or 0.0)
    overall_wrist_visibility = float(debug.get("overall_wrist_visibility") or 0.0)
    saturated = bool(debug.get("saturated"))

    if (
        not ordered
        or chain_quality < 0.25
        or release_confidence < 0.50
        or overall_wrist_visibility < 0.30
        or shoulder_body_ratio > 0.40
        or pelvis_body_ratio < 0.55
        or wrist_arm_ratio < 8.0
        or elbow_velocity < 120.0
        or elbow_velocity > 260.0
        or saturated
    ):
        return result

    shoulder_term = max(0.0, min(1.0, (0.40 - shoulder_body_ratio) / 0.15))
    pelvis_term = max(0.0, min(1.0, (pelvis_body_ratio - 0.55) / 0.20))
    wrist_vis_term = max(0.0, min(1.0, (overall_wrist_visibility - 0.30) / 0.20))
    wrist_ratio_term = max(0.0, min(1.0, (wrist_arm_ratio - 8.0) / 4.0))
    release_term = max(0.0, min(1.0, (release_confidence - 0.50) / 0.20))
    chain_term = max(0.0, min(1.0, (chain_quality - 0.25) / 0.20))

    uplift = 1.0 + (
        0.18 * shoulder_term
        + 0.14 * pelvis_term
        + 0.12 * wrist_vis_term
        + 0.10 * wrist_ratio_term
        + 0.08 * release_term
        + 0.08 * chain_term
    )
    value_kph = min(145, int(round(int(result.get("value_kph") or 0) * uplift)))

    debug["clean_salvage_promotion"] = round(uplift, 3)
    return {
        **result,
        "value_kph": value_kph,
        "display": f"~{value_kph} km/h",
        "display_policy": "show",
        "debug": debug,
    }


def _apply_small_subject_compensation(
    result: Dict[str, Any],
    *,
    events: Dict[str, Any],
) -> Dict[str, Any]:
    if not result.get("available"):
        return result

    debug = dict(result.get("debug") or {})
    reason = str(result.get("reason") or "")
    body_height_ratio = float(debug.get("body_height_ratio") or 0.0)
    body_height_px = float(debug.get("body_height_px") or 0.0)
    overall_wrist_visibility = float(debug.get("overall_wrist_visibility") or 0.0)
    shoulder_body_ratio = float(debug.get("shoulder_body_ratio") or 0.0)
    pelvis_body_ratio = float(debug.get("pelvis_body_ratio") or 0.0)
    saturated = bool(debug.get("saturated"))

    chain = (events or {}).get("event_chain") or {}
    ordered = bool(chain.get("ordered"))
    chain_quality = float(chain.get("quality") or 0.0)

    if (
        reason != "recovered_implausible_saturation"
        or not ordered
        or chain_quality < 0.35
        or body_height_ratio <= 0.0
        or body_height_ratio > 0.26
        or body_height_px > 240.0
        or overall_wrist_visibility > 0.16
        or shoulder_body_ratio > 0.65
        or pelvis_body_ratio < 0.35
        or saturated
    ):
        return result

    ratio_term = max(0.0, min(1.0, (0.26 - body_height_ratio) / 0.08))
    wrist_vis_term = max(0.0, min(1.0, (0.16 - overall_wrist_visibility) / 0.10))
    chain_term = max(0.0, min(1.0, (chain_quality - 0.35) / 0.15))
    uplift = 1.0 + (
        0.05 * ratio_term
        + 0.03 * wrist_vis_term
        + 0.02 * chain_term
    )
    value_kph = min(145, int(round(int(result.get("value_kph") or 0) * uplift)))

    debug["small_subject_compensation"] = round(uplift, 3)
    return {
        **result,
        "value_kph": value_kph,
        "display": f"~{value_kph} km/h",
        "debug": debug,
    }


def _is_implausible_saturated_estimate(result: Dict[str, Any]) -> bool:
    if not result.get("available"):
        return False
    debug = result.get("debug") or {}
    if not bool(debug.get("saturated")):
        return False
    if int(result.get("value_kph") or 0) < 145:
        return False

    confidence = float(result.get("confidence") or 0.0)
    elbow_velocity = float(debug.get("elbow_extension_velocity_deg_per_sec") or 0.0)
    shoulder_body_ratio = float(debug.get("shoulder_body_ratio") or 0.0)
    overall_wrist_visibility = float(debug.get("overall_wrist_visibility") or 0.0)
    release_confidence = float(debug.get("release_confidence") or 0.0)

    return (
        confidence <= 0.55
        and release_confidence <= 0.60
        and (
            elbow_velocity >= 320.0
            or shoulder_body_ratio >= 0.80
            or overall_wrist_visibility <= 0.25
        )
    )


def estimate_release_speed(
    *,
    pose_frames: List[Dict[str, Any]],
    events: Dict[str, Any],
    video: Dict[str, Any],
    hand: str,
    ball_weight_oz: float = BALL_WEIGHT_OZ,
) -> Dict[str, Any]:
    release_frame = int(((events.get("release") or {}).get("frame")) or -1)
    release_confidence = float(((events.get("release") or {}).get("confidence")) or 0.0)
    primary = _estimate_release_speed_pass(
        pose_frames=pose_frames,
        video=video,
        hand=hand,
        release_frame=release_frame,
        ball_weight_oz=ball_weight_oz,
        window_before=3,
        window_after=3,
        scale_before=3,
        scale_after=3,
        sigma_scale=0.03,
        max_arm_cv=0.22,
        max_wrist_cv=0.60,
        salvage_mode=False,
        release_confidence=release_confidence,
    )
    if primary.get("reason") in {
        "release_too_close_to_clip_start",
        "release_too_close_to_clip_end",
    }:
        debug = dict(primary.get("debug") or {})
        debug["primary_failure_reason"] = primary.get("reason")
        return {
            **primary,
            "reason": "missing_release_window",
            "debug": debug,
        }
    if primary.get("available"):
        if release_confidence < 0.60:
            neighbor_results = [primary]
            for candidate_frame in range(max(3, release_frame - 2), release_frame + 3):
                if candidate_frame == release_frame:
                    continue
                neighbor_results.append(
                    _estimate_release_speed_pass(
                        pose_frames=pose_frames,
                        video=video,
                        hand=hand,
                        release_frame=candidate_frame,
                        ball_weight_oz=ball_weight_oz,
                        window_before=3,
                        window_after=3,
                        scale_before=3,
                        scale_after=3,
                        sigma_scale=0.03,
                        max_arm_cv=0.22,
                        max_wrist_cv=0.60,
                        salvage_mode=False,
                        release_confidence=release_confidence,
                    )
                )
            primary = _apply_low_confidence_neighbor_recovery(primary, neighbor_results)
        if _is_implausible_saturated_estimate(primary):
            recovery = _estimate_release_speed_pass(
                pose_frames=pose_frames,
                video=video,
                hand=hand,
                release_frame=release_frame,
                ball_weight_oz=ball_weight_oz,
                window_before=5,
                window_after=5,
                scale_before=10,
                scale_after=10,
                sigma_scale=0.05,
                max_arm_cv=0.30,
                max_wrist_cv=1.10,
                salvage_mode=True,
                release_confidence=release_confidence,
            )
            if recovery.get("available") and not bool((recovery.get("debug") or {}).get("saturated")):
                recovery["reason"] = "recovered_implausible_saturation"
                recovery_debug = recovery.setdefault("debug", {})
                recovery_debug["primary_failure_reason"] = "implausible_saturated_estimate"
                return _apply_small_subject_compensation(recovery, events=events)
            debug = dict(primary.get("debug") or {})
            debug["primary_failure_reason"] = "implausible_saturated_estimate"
            return {
                **_unavailable_result(ball_weight_oz),
                "reason": "implausible_saturated_estimate",
                "debug": debug,
            }
        return primary

    if primary.get("reason") not in {"unstable_arm_scale", "unstable_release_window"}:
        return primary

    salvage = _estimate_release_speed_pass(
        pose_frames=pose_frames,
        video=video,
        hand=hand,
        release_frame=release_frame,
        ball_weight_oz=ball_weight_oz,
        window_before=5,
        window_after=5,
        scale_before=10,
        scale_after=10,
        sigma_scale=0.05,
        max_arm_cv=0.30,
        max_wrist_cv=1.10,
        salvage_mode=True,
        release_confidence=release_confidence,
    )
    if salvage.get("available"):
        salvage["reason"] = f"recovered_{primary.get('reason')}"
        salvage_debug = salvage.setdefault("debug", {})
        salvage_debug["primary_failure_reason"] = primary.get("reason")
        return _apply_clean_salvage_promotion(salvage, events=events)

    if salvage.get("reason") in {
        "release_too_close_to_clip_start",
        "release_too_close_to_clip_end",
    }:
        salvage = {
            **salvage,
            "reason": primary.get("reason"),
            "debug": {
                **dict(salvage.get("debug") or {}),
                "primary_failure_reason": primary.get("reason"),
            },
        }

    result = salvage if salvage.get("reason") != primary.get("reason") else primary
    return _apply_clean_salvage_promotion(result, events=events) if result.get("available") else result
