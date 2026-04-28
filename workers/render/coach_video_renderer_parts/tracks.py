from __future__ import annotations
from .shared import *
from .analytics import _safe_float

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


__all__ = [name for name in globals() if name != "__builtins__"]
