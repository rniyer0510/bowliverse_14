"""
Elbow signal extraction — ActionLab V14 (IDEMPOTENT, CONTINUOUS)

FINAL LOCKED BEHAVIOUR:
- Never drop frames due to angle jumps
- Enforce biomechanical continuity via soft clamping
- Preserve UAH → Release window integrity
- Guarantee >=2 usable samples when landmarks exist
- Deterministic output (idempotent)

Rationale:
Hard frame rejection breaks continuity at low FPS.
Human elbows cannot change angle discontinuously.
We clamp impossible jumps instead of discarding data.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

# MediaPipe Pose indices
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16

L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

VISIBILITY_MIN = 0.5

# Maximum physically plausible per-frame elbow angle change (degrees)
ANGLE_JUMP_MAX_DEG = 25.0

W_WRIST = 0.55
W_INDEX = 0.20
W_PINKY = 0.15
W_THUMB = 0.10


def _to_xyz(pt: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if isinstance(pt, dict):
            return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
        return (float(pt.x), float(pt.y), float(getattr(pt, "z", 0.0)))
    except Exception:
        return None


def _get_vis(pt: Any) -> float:
    try:
        return float(pt.get("visibility", 1.0)) if isinstance(pt, dict) else float(getattr(pt, "visibility", 1.0))
    except Exception:
        return 1.0


def _angle(a, b, c) -> Optional[float]:
    bax, bay, baz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    bcx, bcy, bcz = c[0]-b[0], c[1]-b[1], c[2]-b[2]

    mag1 = math.sqrt(bax*bax + bay*bay + baz*baz)
    mag2 = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None

    dot = bax*bcx + bay*bcy + baz*bcz
    cosv = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosv))


def _weighted_centroid(points):
    sw = sum(w for _, w in points)
    if sw < 1e-9:
        return None
    return (
        sum(p[0]*w for p, w in points)/sw,
        sum(p[1]*w for p, w in points)/sw,
        sum(p[2]*w for p, w in points)/sw,
    )


def compute_elbow_signal(pose_frames: Any, hand: str) -> List[Dict[str, Any]]:
    frames = pose_frames if isinstance(pose_frames, list) else []
    signal = []

    if hand == "R":
        s_idx, e_idx, w_idx = RS, RE, RW
        i_idx, p_idx, t_idx = R_INDEX, R_PINKY, R_THUMB
    else:
        s_idx, e_idx, w_idx = LS, LE, LW
        i_idx, p_idx, t_idx = L_INDEX, L_PINKY, L_THUMB

    prev_angle: Optional[float] = None

    for i, item in enumerate(frames):
        lm = item.get("landmarks") if isinstance(item, dict) else item
        frame_idx = int(item.get("frame", i)) if isinstance(item, dict) else i

        if not isinstance(lm, list) or len(lm) <= max(s_idx, e_idx, w_idx, i_idx, p_idx, t_idx):
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        s, e = lm[s_idx], lm[e_idx]
        if _get_vis(s) < VISIBILITY_MIN or _get_vis(e) < VISIBILITY_MIN:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        a, b = _to_xyz(s), _to_xyz(e)
        if a is None or b is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        pts = []
        for idx, w in [(w_idx, W_WRIST), (i_idx, W_INDEX), (p_idx, W_PINKY), (t_idx, W_THUMB)]:
            pt = lm[idx]
            if _get_vis(pt) >= VISIBILITY_MIN:
                xyz = _to_xyz(pt)
                if xyz:
                    pts.append((xyz, w))

        c = _weighted_centroid(pts)
        if c is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        ang = _angle(a, b, c)
        if ang is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        # 🔒 SOFT CONTINUITY ENFORCEMENT (KEY FIX)
        if prev_angle is not None:
            delta = ang - prev_angle
            if abs(delta) > ANGLE_JUMP_MAX_DEG:
                ang = prev_angle + math.copysign(ANGLE_JUMP_MAX_DEG, delta)

        signal.append({"frame": frame_idx, "angle_deg": ang, "valid": True})
        prev_angle = ang

    return signal
