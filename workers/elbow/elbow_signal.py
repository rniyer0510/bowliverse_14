"""
Elbow signal extraction — ActionLab V14 (WRIST-ROBUST)

LOCKED RULES:
- Bowling arm identity is fixed by input `hand` (R/L)
- NEVER switch arms
- NEVER substitute the non-bowling arm
- Do not discard frames: emit angle_deg=None when invalid

CRITICAL FIX:
- Wrist landmark can drift (especially amateurs + spinners) and contaminate elbow angle.
- We stabilize the distal forearm reference using a "virtual distal point"
  formed by a weighted centroid of wrist + hand landmarks (index/pinky/thumb).
- We still compute the INTERNAL elbow angle at elbow:
    angle = ∠(shoulder - elbow - distal)
  but distal is now robust to wrist flick/roll/occlusion artifacts.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

# MediaPipe landmark indices (Pose, 33 landmarks)
# Shoulder/Elbow/Wrist (LOCKED)
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16

# Hand landmarks available in Pose (important for distal stabilization)
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

# Signal-quality thresholds
VISIBILITY_MIN = 0.5

# Per-frame jump guard (camera jitter protection)
ANGLE_JUMP_MAX_DEG = 25.0

# Distal blending weights (sum doesn't have to be 1; we normalize)
W_WRIST = 0.55
W_INDEX = 0.20
W_PINKY = 0.15
W_THUMB = 0.10


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


def _get_vis(pt: Any) -> float:
    """Extract visibility if present, else assume visible."""
    try:
        if isinstance(pt, dict):
            return float(pt.get("visibility", 1.0))
        return float(getattr(pt, "visibility", 1.0))
    except Exception:
        return 1.0


def _angle(a: Tuple[float, float, float],
           b: Tuple[float, float, float],
           c: Tuple[float, float, float]) -> Optional[float]:
    """
    Angle ABC at B, degrees.
    Shoulder–Elbow–Distal (inside elbow angle).
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


def _weighted_centroid(points: List[Tuple[Tuple[float, float, float], float]]) -> Optional[Tuple[float, float, float]]:
    """
    points: [((x,y,z), weight), ...]
    returns weighted centroid or None.
    """
    if not points:
        return None
    sw = sum(w for _, w in points)
    if sw < 1e-9:
        return None
    x = sum(p[0] * w for p, w in points) / sw
    y = sum(p[1] * w for p, w in points) / sw
    z = sum(p[2] * w for p, w in points) / sw
    return (float(x), float(y), float(z))


def _virtual_distal_point(lm: Any, w_idx: int, i_idx: int, p_idx: int, t_idx: int) -> Optional[Tuple[float, float, float]]:
    """
    Build a wrist-robust distal point using wrist + hand landmarks.
    We only include a landmark if its visibility >= VISIBILITY_MIN.
    """
    pts: List[Tuple[Tuple[float, float, float], float]] = []

    # Wrist
    w = lm[w_idx]
    if _get_vis(w) >= VISIBILITY_MIN:
        w_xyz = _to_xyz(w)
        if w_xyz is not None:
            pts.append((w_xyz, W_WRIST))

    # Index
    idx = lm[i_idx]
    if _get_vis(idx) >= VISIBILITY_MIN:
        idx_xyz = _to_xyz(idx)
        if idx_xyz is not None:
            pts.append((idx_xyz, W_INDEX))

    # Pinky
    pk = lm[p_idx]
    if _get_vis(pk) >= VISIBILITY_MIN:
        pk_xyz = _to_xyz(pk)
        if pk_xyz is not None:
            pts.append((pk_xyz, W_PINKY))

    # Thumb
    th = lm[t_idx]
    if _get_vis(th) >= VISIBILITY_MIN:
        th_xyz = _to_xyz(th)
        if th_xyz is not None:
            pts.append((th_xyz, W_THUMB))

    # If only wrist exists, centroid == wrist (fine).
    return _weighted_centroid(pts)


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
        i_idx, p_idx, t_idx = R_INDEX, R_PINKY, R_THUMB
    else:
        s_idx, e_idx, w_idx = LS, LE, LW
        i_idx, p_idx, t_idx = L_INDEX, L_PINKY, L_THUMB

    signal: List[Dict[str, Union[int, float, bool, None]]] = []
    prev_angle: Optional[float] = None

    for item in frames:
        frame = int(item.get("frame", 0))
        lm = item.get("landmarks")

        if lm is None or len(lm) <= max(s_idx, e_idx, w_idx, i_idx, p_idx, t_idx):
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        s = lm[s_idx]
        e = lm[e_idx]

        # Need shoulder & elbow visibility; distal is handled by virtual point logic
        try:
            if (
                _get_vis(s) < VISIBILITY_MIN or
                _get_vis(e) < VISIBILITY_MIN
            ):
                signal.append({"frame": frame, "angle_deg": None, "valid": False})
                continue
        except Exception:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        a = _to_xyz(s)  # shoulder
        b = _to_xyz(e)  # elbow
        if a is None or b is None:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        # Wrist-robust distal point (KEY FIX)
        c = _virtual_distal_point(lm, w_idx=w_idx, i_idx=i_idx, p_idx=p_idx, t_idx=t_idx)
        if c is None:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        ang = _angle(a, b, c)
        if ang is None:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        # Jitter / spike suppression:
        if prev_angle is not None and abs(ang - prev_angle) > ANGLE_JUMP_MAX_DEG:
            signal.append({"frame": frame, "angle_deg": None, "valid": False})
            continue

        signal.append({"frame": frame, "angle_deg": ang, "valid": True})
        prev_angle = ang

    return signal

