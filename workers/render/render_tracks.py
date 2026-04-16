from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .render_constants import (
    JOINT_OUTER, MIN_VISIBILITY, RUNNER_MIN_CONSECUTIVE_FRAMES, RUNNER_MIN_MOTION_PX, RUNNER_MIN_VISIBILITY,
    SKELETON_COLOR, SKELETON_EDGES, SKELETON_SHADOW, TRACKED_JOINTS,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from .render_helpers import _safe_float

def _point_from_landmarks(
    landmarks: Optional[List[Dict[str, Any]]],
    idx: int,
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    if not isinstance(landmarks, list) or idx >= len(landmarks):
        return None
    point = landmarks[idx]
    if not isinstance(point, dict):
        return None
    vis = _safe_float(point.get("visibility"))
    x = _safe_float(point.get("x"))
    y = _safe_float(point.get("y"))
    if vis is None or x is None or y is None or vis < MIN_VISIBILITY:
        return None
    return (x * width, y * height)


def _joint_visible(
    landmarks: Optional[List[Dict[str, Any]]],
    idx: int,
    *,
    min_visibility: float = RUNNER_MIN_VISIBILITY,
) -> bool:
    if not isinstance(landmarks, list) or idx >= len(landmarks):
        return False
    point = landmarks[idx]
    if not isinstance(point, dict):
        return False
    x = _safe_float(point.get("x"))
    y = _safe_float(point.get("y"))
    vis = _safe_float(point.get("visibility"))
    if x is None or y is None or vis is None:
        return False
    return vis >= min_visibility


def _body_centroid(
    landmarks: Optional[List[Dict[str, Any]]],
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    points: List[Tuple[float, float]] = []
    for idx in indices:
        point = _point_from_landmarks(landmarks, idx, width=width, height=height)
        if point is not None:
            points.append(point)
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def _is_valid_human_pose(
    landmarks: Optional[List[Dict[str, Any]]],
    *,
    width: int,
    height: int,
) -> bool:
    if not isinstance(landmarks, list) or len(landmarks) <= RIGHT_ANKLE:
        return False

    torso_ok = (
        _joint_visible(landmarks, LEFT_SHOULDER)
        and _joint_visible(landmarks, RIGHT_SHOULDER)
        and _joint_visible(landmarks, LEFT_HIP)
        and _joint_visible(landmarks, RIGHT_HIP)
    )
    if not torso_ok:
        return False

    left_leg_ok = _joint_visible(landmarks, LEFT_KNEE) and _joint_visible(landmarks, LEFT_ANKLE)
    right_leg_ok = _joint_visible(landmarks, RIGHT_KNEE) and _joint_visible(landmarks, RIGHT_ANKLE)
    if not (left_leg_ok or right_leg_ok):
        return False

    ls = _point_from_landmarks(landmarks, LEFT_SHOULDER, width=width, height=height)
    rs = _point_from_landmarks(landmarks, RIGHT_SHOULDER, width=width, height=height)
    lh = _point_from_landmarks(landmarks, LEFT_HIP, width=width, height=height)
    rh = _point_from_landmarks(landmarks, RIGHT_HIP, width=width, height=height)
    if not (ls and rs and lh and rh):
        return False

    shoulder_width = abs(rs[0] - ls[0])
    hip_width = abs(rh[0] - lh[0])
    torso_height = abs(((lh[1] + rh[1]) * 0.5) - ((ls[1] + rs[1]) * 0.5))

    if shoulder_width < 6 or hip_width < 6 or torso_height < 10:
        return False

    if torso_height < shoulder_width * 0.25:
        return False
    if torso_height < hip_width * 0.25:
        return False

    centroid = _body_centroid(landmarks, width=width, height=height)
    return centroid is not None


def _compute_runner_mask(
    pose_frames: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
) -> List[bool]:
    frame_valid = [
        _is_valid_human_pose((frame or {}).get("landmarks"), width=width, height=height)
        for frame in pose_frames
    ]

    centroids: List[Optional[Tuple[float, float]]] = [
        _body_centroid((frame or {}).get("landmarks"), width=width, height=height)
        if frame_valid[idx] else None
        for idx, frame in enumerate(pose_frames)
    ]

    motion_ok = [False] * len(pose_frames)
    for idx in range(1, len(pose_frames)):
        prev_c = centroids[idx - 1]
        curr_c = centroids[idx]
        if prev_c is None or curr_c is None:
            continue
        dx = curr_c[0] - prev_c[0]
        dy = curr_c[1] - prev_c[1]
        motion = float((dx * dx + dy * dy) ** 0.5)
        if motion >= RUNNER_MIN_MOTION_PX:
            motion_ok[idx] = True

    runner_mask = [False] * len(pose_frames)
    streak = 0
    for idx in range(len(pose_frames)):
        if frame_valid[idx] and motion_ok[idx]:
            streak += 1
        else:
            streak = 0
        if streak >= RUNNER_MIN_CONSECUTIVE_FRAMES:
            start_idx = idx - RUNNER_MIN_CONSECUTIVE_FRAMES + 1
            for j in range(start_idx, idx + 1):
                runner_mask[j] = True

    return runner_mask


def _smooth_series(values: Iterable[Optional[float]], sigma: float) -> Optional[np.ndarray]:
    value_list = list(values)
    if not value_list:
        return None
    valid = np.array([value is not None for value in value_list], dtype=bool)
    if valid.sum() < 3:
        return None
    idx = np.arange(len(value_list), dtype=float)
    arr = np.zeros(len(value_list), dtype=float)
    arr[valid] = [float(value) for value in value_list if value is not None]
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return gaussian_filter1d(arr, sigma=max(1.0, sigma))


def _build_smoothed_tracks(
    pose_frames: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: float,
) -> Dict[int, Dict[str, Any]]:
    sigma = max(1.0, float(fps or 30.0) * 0.03)
    tracks: Dict[int, Dict[str, Any]] = {}
    for joint_idx in TRACKED_JOINTS:
        raw_points = [
            _point_from_landmarks((frame or {}).get("landmarks"), joint_idx, width=width, height=height)
            for frame in pose_frames
        ]
        xs = _smooth_series([point[0] if point else None for point in raw_points], sigma=sigma)
        ys = _smooth_series([point[1] if point else None for point in raw_points], sigma=sigma)
        tracks[joint_idx] = {
            "raw": raw_points,
            "xs": xs,
            "ys": ys,
        }
    return tracks


def _track_point(
    tracks: Dict[int, Dict[str, Any]],
    joint_idx: int,
    frame_idx: int,
) -> Optional[Tuple[int, int]]:
    track = tracks.get(joint_idx) or {}
    xs = track.get("xs")
    ys = track.get("ys")
    raw = track.get("raw") or []
    if xs is not None and ys is not None and frame_idx < len(xs) and frame_idx < len(ys):
        return (int(round(float(xs[frame_idx]))), int(round(float(ys[frame_idx]))))
    if frame_idx < len(raw) and raw[frame_idx] is not None:
        point = raw[frame_idx]
        return (int(round(point[0])), int(round(point[1])))
    return None


def _frame_point(
    pose_frames: List[Dict[str, Any]],
    *,
    frame_idx: int,
    joint_idx: int,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    if frame_idx < 0 or frame_idx >= len(pose_frames):
        return None
    point = _point_from_landmarks((pose_frames[frame_idx] or {}).get("landmarks"), joint_idx, width=width, height=height)
    if point is None:
        return None
    return (int(round(point[0])), int(round(point[1])))


def _draw_joint(frame: np.ndarray, point: Tuple[int, int], scale: int) -> None:
    outer = max(3, scale // 180)
    inner = max(2, outer - 1)
    cv2.circle(frame, point, outer + 2, SKELETON_SHADOW, -1, cv2.LINE_AA)
    cv2.circle(frame, point, outer, JOINT_OUTER, -1, cv2.LINE_AA)
    cv2.circle(frame, point, inner, SKELETON_COLOR, -1, cv2.LINE_AA)


def _draw_skeleton(frame: np.ndarray, tracks: Dict[int, Dict[str, Any]], frame_idx: int) -> None:
    scale = min(frame.shape[0], frame.shape[1])
    shadow_thickness = max(5, scale // 110)
    line_thickness = max(3, scale // 180)
    for start_idx, end_idx in SKELETON_EDGES:
        start = _track_point(tracks, start_idx, frame_idx)
        end = _track_point(tracks, end_idx, frame_idx)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, SKELETON_SHADOW, shadow_thickness, cv2.LINE_AA)
        cv2.line(frame, start, end, SKELETON_COLOR, line_thickness, cv2.LINE_AA)
    for joint_idx in TRACKED_JOINTS:
        point = _track_point(tracks, joint_idx, frame_idx)
        if point is None:
            continue
        _draw_joint(frame, point, scale)


