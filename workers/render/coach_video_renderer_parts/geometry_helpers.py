from __future__ import annotations
from .shared import *

def _midpoint(
    point_a: Optional[Tuple[int, int]],
    point_b: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if point_a is None or point_b is None:
        return None
    return (
        int(round((point_a[0] + point_b[0]) / 2.0)),
        int(round((point_a[1] + point_b[1]) / 2.0)),
    )
def _point_between(
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    progress: float,
) -> Tuple[int, int]:
    t = max(0.0, min(1.0, float(progress)))
    return (
        int(round(point_a[0] + (point_b[0] - point_a[0]) * t)),
        int(round(point_a[1] + (point_b[1] - point_a[1]) * t)),
    )
def _partial_polyline(
    points: List[Tuple[int, int]],
    *,
    progress: float,
) -> List[Tuple[int, int]]:
    if len(points) < 2:
        return list(points)
    clamped = max(0.0, min(1.0, float(progress)))
    if clamped <= 0.0:
        return [points[0]]
    if clamped >= 1.0:
        return list(points)

    segment_lengths = [
        float(np.hypot(float(end[0] - start[0]), float(end[1] - start[1])))
        for start, end in zip(points, points[1:])
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-6:
        return list(points[:2])

    target_length = total_length * clamped
    consumed = 0.0
    partial: List[Tuple[int, int]] = [points[0]]
    for idx, segment_length in enumerate(segment_lengths):
        start = points[idx]
        end = points[idx + 1]
        if consumed + segment_length < target_length - 1e-6:
            partial.append(end)
            consumed += segment_length
            continue
        remainder = max(0.0, target_length - consumed)
        segment_progress = 0.0 if segment_length <= 1e-6 else remainder / segment_length
        partial.append(_point_between(start, end, segment_progress))
        break
    return partial
def _draw_partial_polyline(
    frame: np.ndarray,
    *,
    points: List[Tuple[int, int]],
    progress: float,
    color: Tuple[int, int, int],
    thickness: int,
    shadow_thickness: int,
) -> None:
    partial_points = _partial_polyline(points, progress=progress)
    if len(partial_points) < 2:
        return
    for start, end in zip(partial_points, partial_points[1:]):
        cv2.line(frame, start, end, SKELETON_SHADOW, shadow_thickness, cv2.LINE_AA)
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
