"""
Elbow signal extraction — ActionLab V14

LOCKED RULES:
- Bowling arm identity is fixed by input `hand` (R/L)
- NEVER switch arms
- NEVER substitute the non-bowling arm
- Do not discard frames: emit angle_deg=None when invalid
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

# MediaPipe landmark indices (LOCKED)
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16

# Signal-quality thresholds
VISIBILITY_MIN = 0.5
ANGLE_JUMP_MAX_DEG = 25.0   # per-frame jump guard (camera jitter protection)


def _to_xyz(pt: Any) -> Optional[Tuple[float, float, float]]:
    """Convert a landmark point to (x,y,z)."""
    if pt is None:
        return None
    if isinstance(pt, dict):
        if "x" in pt and "y" in pt:
            return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
        return None
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        x = float(pt[0])
        y = float(pt[1])
        z = float(pt[2]) if len(pt) >= 3 else 0.0
        return (x, y, z)
    try:
        return (float(pt.x), float(pt.y), float(getattr(pt, "z", 0.0)))
    except Exception:
        return None


def _angle(a: Tuple[float, float, float],
           b: Tuple[float, float, float],
           c: Tuple[float, float, float]) -> Optional[float]:
    """
    Angle ABC at B, degrees.
    Shoulder–Elbow–Wrist (inside elbow angle).
    """
    bax, bay, baz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    bcx, bcy, bcz = c[0] - b[0], c[1] - b[1], c[2] - b[2]

    mag_ba = math.sqrt(bax*bax + bay*bay + baz*baz)
    mag_bc = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None

    dot = bax*bcx + bay*bcy + baz*bcz
    cosv = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return float(math.degrees(math.acos(cosv)))


def _normalize_pose_frames(pose_frames: Any) -> List[Dict[str, Any]]:
    """
    Normalize pose_frames into:
      [{ "frame": int, "landmarks": list|None }, ...]
    """
    if isinstance(pose_frames, list):
        return pose_frames

    if isinstance(pose_frames, dict):
        if "frames" in pose_frames:
            return pose_frames["frames"]
        if "pose_frames" in pose_frames:
            return pose_frames["pose_frames"]

        out: List[Dict[str, Any]] = []
        for k, v in pose_frames.items():
            try:
                f = int(k)
            except Exception:
                continue
            out.append({"frame": f, "landmarks": v})
        out.sort(key=lambda d: d.get("frame", 0))
        return out

    raise TypeError(f"Unsupported pose_frames type/shape: {type(pose_frames)}")


def compute_elbow_signal(pose_frames: Any, hand: str) -> List[Dict[str, Any]]:
    """
    Returns list of:
      {
        "frame": int,
        "angle_deg": float | None,
        "valid": bool
      }
    """
    frames = _normalize_pose_frames(pose_frames)

    # LOCK bowling arm once
    if hand == "R":
        s_idx, e_idx, w_idx = RS, RE, RW
    else:
        s_idx, e_idx, w_idx = LS, LE, LW

    signal: List[Dict[str, Union[int, float, bool, None]]] = []
    prev_angle: Optional[float] = None

    for item in frames:
        frame = int(item.get("frame", 0))
        lm = item.get("landmarks")

        # Missing landmarks
        if lm is None or len(lm) <= max(s_idx, e_idx, w_idx):
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        s = lm[s_idx]
        e = lm[e_idx]
        w = lm[w_idx]

        # Visibility guard (bowling arm only)
        try:
            if (
                s.get("visibility", 1.0) < VISIBILITY_MIN or
                e.get("visibility", 1.0) < VISIBILITY_MIN or
                w.get("visibility", 1.0) < VISIBILITY_MIN
            ):
                signal.append({"frame": frame, "angle_deg": None, "valid": False})
                prev_angle = None
                continue
        except Exception:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            prev_angle = None
            continue

        a = _to_xyz(s)
        b = _to_xyz(e)
        c = _to_xyz(w)

        if a is None or b is None or c is None:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            prev_angle = None
            continue

        ang = _angle(a, b, c)
        if ang is None:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            prev_angle = None
            continue

        # Jitter / spike suppression
        if prev_angle is not None and abs(ang - prev_angle) > ANGLE_JUMP_MAX_DEG:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            prev_angle = None
            continue

        signal.append({"frame": frame, "angle_deg": ang, "valid": True})
        prev_angle = ang

    return signal

