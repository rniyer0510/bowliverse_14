from __future__ import annotations

from .shared import *
from .analytics import _safe_float
from .occlusion_manager import (
    _visibility_weight,
    _smooth_series,
    _draw_mode_for_quality,
    _frame_quality_from_joint_qualities,
)


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
    x = _safe_float(point.get("x"))
    y = _safe_float(point.get("y"))
    vis = _safe_float(point.get("visibility"))
    if x is None or y is None or _visibility_weight(vis) <= 0.0:
        return None
    return (x * width, y * height)


def _landmark_visibility(
    landmarks: Optional[List[Dict[str, Any]]],
    idx: int,
) -> float:
    if not isinstance(landmarks, list) or idx >= len(landmarks):
        return 0.0
    point = landmarks[idx]
    if not isinstance(point, dict):
        return 0.0
    return float(_safe_float(point.get("visibility")) or 0.0)


def _build_smoothed_tracks(
    pose_frames: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: float,
) -> Dict[int, Dict[str, Any]]:
    sigma = min(max(1.0, float(fps or 30.0) * 0.03), 2.5)
    tracks: Dict[int, Dict[str, Any]] = {}
    for joint_idx in TRACKED_JOINTS:
        raw_points = []
        raw_visibility = []
        raw_weights = []
        for frame in pose_frames:
            landmarks = (frame or {}).get("landmarks")
            raw_points.append(
                _point_from_landmarks(landmarks, joint_idx, width=width, height=height)
            )
            vis = _landmark_visibility(landmarks, joint_idx)
            raw_visibility.append(vis)
            raw_weights.append(_visibility_weight(vis))

        xs = _smooth_series(
            [point[0] if point else None for point in raw_points],
            sigma=sigma,
            weights=raw_weights,
        )
        ys = _smooth_series(
            [point[1] if point else None for point in raw_points],
            sigma=sigma,
            weights=raw_weights,
        )
        quality = _smooth_series(raw_weights, sigma=max(1.0, sigma * 0.6))
        if quality is None:
            quality = np.asarray(raw_weights, dtype=float)
        quality = np.clip(np.nan_to_num(quality, nan=0.0), 0.0, 1.0)

        draw_modes = [
            _draw_mode_for_quality(
                float(quality[idx]) if idx < len(quality) else 0.0,
                float(raw_weights[idx]) if idx < len(raw_weights) else 0.0,
                raw_points[idx] is not None,
            )
            for idx in range(len(raw_points))
        ]

        tracks[joint_idx] = {
            "raw": raw_points,
            "raw_visibility": np.asarray(raw_visibility, dtype=float),
            "raw_weight": np.asarray(raw_weights, dtype=float),
            "quality": quality,
            "draw_modes": draw_modes,
            "xs": xs,
            "ys": ys,
        }
    return tracks


def _track_state(
    tracks: Dict[int, Dict[str, Any]],
    joint_idx: int,
    frame_idx: int,
) -> Optional[Dict[str, Any]]:
    track = tracks.get(joint_idx) or {}
    xs = track.get("xs")
    ys = track.get("ys")
    raw = track.get("raw") or []
    quality = track.get("quality")
    draw_modes = track.get("draw_modes") or []

    point: Optional[Tuple[int, int]] = None
    if xs is not None and ys is not None and frame_idx < len(xs) and frame_idx < len(ys):
        x = float(xs[frame_idx])
        y = float(ys[frame_idx])
        if np.isfinite(x) and np.isfinite(y):
            point = (int(round(x)), int(round(y)))
    if point is None and frame_idx < len(raw) and raw[frame_idx] is not None:
        raw_point = raw[frame_idx]
        point = (int(round(raw_point[0])), int(round(raw_point[1])))

    if point is None:
        return None

    joint_quality = 0.0
    if quality is not None and frame_idx < len(quality):
        joint_quality = float(np.clip(quality[frame_idx], 0.0, 1.0))
    mode = draw_modes[frame_idx] if frame_idx < len(draw_modes) else None
    if mode is None and joint_quality > 0.0:
        mode = "placeholder"

    return {
        "point": point,
        "quality": joint_quality,
        "draw_mode": mode,
    }


def _track_quality(
    tracks: Dict[int, Dict[str, Any]],
    joint_idx: int,
    frame_idx: int,
) -> float:
    state = _track_state(tracks, joint_idx, frame_idx)
    if not state:
        return 0.0
    return float(state.get("quality") or 0.0)


def _track_draw_mode(
    tracks: Dict[int, Dict[str, Any]],
    joint_idx: int,
    frame_idx: int,
) -> Optional[str]:
    state = _track_state(tracks, joint_idx, frame_idx)
    if not state:
        return None
    return state.get("draw_mode")


def _track_point(
    tracks: Dict[int, Dict[str, Any]],
    joint_idx: int,
    frame_idx: int,
) -> Optional[Tuple[int, int]]:
    state = _track_state(tracks, joint_idx, frame_idx)
    if not state:
        return None
    return state.get("point")


def _track_frame_quality(
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
) -> float:
    return _frame_quality_from_joint_qualities(
        [
        _track_quality(tracks, joint_idx, frame_idx)
        for joint_idx in TRACKED_JOINTS
        ],
        len(TRACKED_JOINTS),
    )


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
    point = _point_from_landmarks(
        (pose_frames[frame_idx] or {}).get("landmarks"),
        joint_idx,
        width=width,
        height=height,
    )
    if point is None:
        return None
    return (int(round(point[0])), int(round(point[1])))


__all__ = [name for name in globals() if name != "__builtins__"]
