import math
from statistics import median
from typing import Any, Optional

MIN_VIS = 0.45
ANGLE_OUTLIER_GAP_DEG = 35.0


def point(landmarks: Any, index: int) -> Optional[tuple[float, float, float]]:
    if not isinstance(landmarks, list) or index < 0 or index >= len(landmarks):
        return None
    item = landmarks[index]
    if not isinstance(item, dict):
        return None
    visibility = item.get("visibility")
    x = item.get("x")
    y = item.get("y")
    if not isinstance(visibility, (int, float)) or float(visibility) < MIN_VIS:
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return (float(x), float(y), float(visibility))


def angle_between_deg(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    ax, ay = a
    bx, by = b
    amag = math.hypot(ax, ay)
    bmag = math.hypot(bx, by)
    if amag < 1e-6 or bmag < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (amag * bmag)))
    return math.degrees(math.acos(cosine))


def stable_angles(angles: list[float]) -> list[float]:
    if len(angles) < 3:
        return angles
    center = float(median(angles))
    stable = [angle for angle in angles if abs(angle - center) <= ANGLE_OUTLIER_GAP_DEG]
    return stable if len(stable) >= 2 else angles
