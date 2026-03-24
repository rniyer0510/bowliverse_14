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
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    xs = _interpolate_series([point[0] if point else None for point in points])
    ys = _interpolate_series([point[1] if point else None for point in points])
    if xs is None or ys is None:
        return None
    sigma = max(1.0, 0.03 * fps)
    return (
        gaussian_filter1d(xs, sigma=sigma),
        gaussian_filter1d(ys, sigma=sigma),
    )


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


def estimate_release_speed(
    *,
    pose_frames: List[Dict[str, Any]],
    events: Dict[str, Any],
    video: Dict[str, Any],
    hand: str,
    ball_weight_oz: float = BALL_WEIGHT_OZ,
) -> Dict[str, Any]:
    fps = float(video.get("fps") or 0.0)
    width = float(video.get("width") or 0.0)
    height = float(video.get("height") or 0.0)
    release_frame = int(((events.get("release") or {}).get("frame")) or -1)

    unavailable = {
        "available": False,
        "method": METHOD,
        "confidence": 0.0,
        "ball_weight_oz": float(ball_weight_oz),
    }

    if fps <= 0.0 or width <= 0.0 or height <= 0.0:
        return {**unavailable, "reason": "missing_video_geometry"}

    if release_frame < 3 or release_frame + 3 >= len(pose_frames):
        return {**unavailable, "reason": "missing_release_window"}

    h = (hand or "R").upper()
    shoulder_idx, elbow_idx, wrist_idx = (
        (RS, RE, RW) if h == "R" else (LS, LE, LW)
    )

    wrist_points: List[Optional[Tuple[float, float]]] = []
    shoulder_points: List[Optional[Tuple[float, float]]] = []
    pelvis_points: List[Optional[Tuple[float, float]]] = []
    arm_lengths: List[Optional[float]] = []
    body_heights: List[Optional[float]] = []
    elbow_angles: List[Optional[float]] = []

    for frame in pose_frames:
        landmarks = (frame or {}).get("landmarks")
        wrist_points.append(
            _pixel_xy(_get_landmark(landmarks, wrist_idx), width=width, height=height)
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

    wrist_track = _smoothed_track(wrist_points, fps=fps)
    shoulder_track = _smoothed_track(shoulder_points, fps=fps)
    pelvis_track = _smoothed_track(pelvis_points, fps=fps)
    arm_series = _interpolate_series(arm_lengths)
    body_series = _interpolate_series(body_heights)
    elbow_series = _interpolate_series(elbow_angles)

    if (
        wrist_track is None
        or shoulder_track is None
        or pelvis_track is None
        or arm_series is None
        or body_series is None
        or elbow_series is None
    ):
        return {**unavailable, "reason": "insufficient_release_landmarks"}

    sigma = max(1.0, 0.03 * fps)
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
    elbow_extension_speed = np.abs(np.gradient(elbow_series, dt))

    window = _window_indices(release_frame, len(pose_frames), before=3, after=3)
    arm_window = [float(arm_series[idx]) for idx in window]
    body_window = [float(body_series[idx]) for idx in window]
    wrist_window = [float(wrist_speed[idx]) for idx in window]
    shoulder_window = [float(shoulder_speed[idx]) for idx in window]
    pelvis_window = [float(pelvis_speed[idx]) for idx in window]
    elbow_window = [float(elbow_extension_speed[idx]) for idx in window]

    arm_length_px = _median_or_none(arm_window)
    body_height_px = _median_or_none(body_window)
    if arm_length_px is None or body_height_px is None or body_height_px <= 1.0:
        return {**unavailable, "reason": "missing_body_scale"}

    wrist_arm_ratio = float(median([value / max(arm_length_px, 1.0) for value in wrist_window]))
    shoulder_body_ratio = float(median([value / max(body_height_px, 1.0) for value in shoulder_window]))
    pelvis_body_ratio = float(median([value / max(body_height_px, 1.0) for value in pelvis_window]))
    elbow_velocity = float(median(elbow_window))

    arm_length_cv = float(np.std(arm_window) / max(np.mean(arm_window), 1.0))
    wrist_window_cv = float(np.std(wrist_window) / max(np.mean(wrist_window), 1.0))

    if arm_length_cv > 0.22:
        return {
            **unavailable,
            "reason": "unstable_arm_scale",
            "debug": {
                "arm_length_cv": round(arm_length_cv, 3),
            },
        }

    # Research mapping inspired by baseball pitch-velocity literature:
    # - distal segment speed near release
    # - elbow-extension velocity
    # - pelvis drive into release
    # - penalty for excessive full-body / camera-relative shoulder motion
    release_score = (
        68.0
        + (0.22 * elbow_velocity)
        + (1.25 * wrist_arm_ratio)
        + (4.0 * pelvis_body_ratio)
        - (14.0 * shoulder_body_ratio)
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

    visibility_parts = [
        1.0 if point is not None else 0.0
        for point in (
            wrist_points[release_frame - 1],
            wrist_points[release_frame],
            wrist_points[release_frame + 1],
            shoulder_points[release_frame - 1],
            shoulder_points[release_frame],
            shoulder_points[release_frame + 1],
        )
    ]
    visibility_score = sum(visibility_parts) / float(len(visibility_parts))
    scale_stability_score = max(0.0, 1.0 - min(1.0, arm_length_cv / 0.16))
    motion_stability_score = max(0.0, 1.0 - min(1.0, wrist_window_cv / 0.50))
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
    confidence = max(0.20, min(0.90, confidence))

    if wrist_window_cv > 0.60:
        return {
            **unavailable,
            "reason": "unstable_release_window",
            "debug": {
                "release_frame": int(release_frame),
                "wrist_window_cv": round(wrist_window_cv, 3),
                "arm_length_cv": round(arm_length_cv, 3),
            },
        }

    if confidence < 0.35:
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
            },
        }

    rounded_kph = int(round(value_kph))
    return {
        "available": True,
        "value_kph": rounded_kph,
        "display": f"~{rounded_kph} km/h",
        "confidence": round(confidence, 2),
        "method": METHOD,
        "ball_weight_oz": float(ball_weight_oz),
        "debug": {
            "release_frame": int(release_frame),
            "wrist_arm_ratio": round(wrist_arm_ratio, 3),
            "shoulder_body_ratio": round(shoulder_body_ratio, 3),
            "pelvis_body_ratio": round(pelvis_body_ratio, 3),
            "elbow_extension_velocity_deg_per_sec": round(elbow_velocity, 1),
            "arm_length_px": round(float(arm_length_px), 1),
            "body_height_px": round(float(body_height_px), 1),
            "arm_length_cv": round(arm_length_cv, 3),
            "wrist_window_cv": round(wrist_window_cv, 3),
            "saturated": saturated,
        },
    }
