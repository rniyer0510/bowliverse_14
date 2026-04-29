from __future__ import annotations

from .shared import *
from .analytics import _safe_float


def _visibility_weight(vis: Optional[float]) -> float:
    value = _safe_float(vis)
    if value is None or value <= MIN_VISIBILITY_HARD:
        return 0.0
    if value >= FULL_VISIBILITY:
        return 1.0
    span = max(1e-6, FULL_VISIBILITY - MIN_VISIBILITY_HARD)
    return max(0.0, min(1.0, (float(value) - MIN_VISIBILITY_HARD) / span))


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


def _smooth_series(
    values: Iterable[Optional[float]],
    sigma: float,
    *,
    weights: Optional[Iterable[float]] = None,
) -> Optional[np.ndarray]:
    value_list = list(values)
    if not value_list:
        return None
    valid = np.array([value is not None for value in value_list], dtype=bool)
    if valid.sum() < 3:
        return None
    idx = np.arange(len(value_list), dtype=float)
    raw = np.full(len(value_list), np.nan, dtype=float)
    raw[valid] = [float(value) for value in value_list if value is not None]
    interp = raw.copy()
    interp[~valid] = np.interp(idx[~valid], idx[valid], raw[valid])

    if weights is not None:
        weight_arr = np.asarray(list(weights), dtype=float)
        if weight_arr.shape[0] != len(value_list):
            raise ValueError("weights length must match values length")
        weight_arr = np.clip(weight_arr, 0.0, 1.0)
        source = interp.copy()
        finite_raw = np.isfinite(raw)
        source[finite_raw] = (
            interp[finite_raw] * (1.0 - weight_arr[finite_raw])
            + raw[finite_raw] * weight_arr[finite_raw]
        )
    else:
        source = interp

    smoothed = gaussian_filter1d(source, sigma=max(1.0, sigma))
    return smoothed


def _track_mode_for_quality(
    quality: float,
    raw_weight: float,
    has_raw_point: bool,
) -> Optional[str]:
    if raw_weight >= 0.72 and has_raw_point:
        return "solid"
    if raw_weight > 0.0 and has_raw_point:
        return "dashed"
    if quality > 0.0:
        return "placeholder"
    return None


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
            _track_mode_for_quality(
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
    qualities = [
        _track_quality(tracks, joint_idx, frame_idx)
        for joint_idx in TRACKED_JOINTS
    ]
    finite = [quality for quality in qualities if quality > 0.0]
    if not finite:
        return 0.0
    return float(sum(finite) / max(1, len(TRACKED_JOINTS)))


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
