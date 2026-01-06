import math
from typing import Any, Dict, List, Optional, Tuple

LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

VISIBILITY_MIN = 0.5
CARRY_MAX = 5  # frames

def _to_xyz(pt: Any) -> Optional[Tuple[float, float, float]]:
    try:
        return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
    except Exception:
        return None

def _get_vis(pt: Any) -> float:
    try:
        return float(pt.get("visibility", 1.0))
    except Exception:
        return 1.0

def _angle(a, b, c) -> Optional[float]:
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]
    mag_ba = math.sqrt(sum(x*x for x in ba))
    mag_bc = math.sqrt(sum(x*x for x in bc))
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cosv = max(-1.0, min(1.0, sum(ba[j]*bc[j] for j in range(3)) / (mag_ba*mag_bc)))
    return math.degrees(math.acos(cosv))

def compute_elbow_signal(pose_frames: List[Any], hand: str) -> List[Dict[str, Any]]:
    s_idx, e_idx, w_idx = (RS, RE, RW) if hand == "R" else (LS, LE, LW)
    i_idx, p_idx, t_idx = (R_INDEX, R_PINKY, R_THUMB) if hand == "R" else (L_INDEX, L_PINKY, L_THUMB)

    # -------------------------------------------------
    # PRE-PASS: decide distal mode ONCE
    # -------------------------------------------------
    wrist_vis = 0
    finger_vis = 0

    for frame in pose_frames:
        lm = frame.get("landmarks")
        if not lm:
            continue
        if _get_vis(lm[w_idx]) >= VISIBILITY_MIN:
            wrist_vis += 1
        if any(_get_vis(lm[x]) >= VISIBILITY_MIN for x in (i_idx, p_idx, t_idx)):
            finger_vis += 1

    use_wrist = wrist_vis >= finger_vis

    signal = []
    last_c = None
    carry = 0

    for f, frame in enumerate(pose_frames):
        lm = frame.get("landmarks")
        frame_idx = frame.get("frame", f)

        if not lm or len(lm) <= max(s_idx, e_idx, w_idx, i_idx, p_idx, t_idx):
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        s, e = lm[s_idx], lm[e_idx]
        if _get_vis(s) < VISIBILITY_MIN or _get_vis(e) < VISIBILITY_MIN:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        a = _to_xyz(s)
        b = _to_xyz(e)
        c = None

        if use_wrist and _get_vis(lm[w_idx]) >= VISIBILITY_MIN:
            c = _to_xyz(lm[w_idx])
        else:
            pts = []
            for idx in (i_idx, p_idx, t_idx):
                if _get_vis(lm[idx]) >= VISIBILITY_MIN:
                    xyz = _to_xyz(lm[idx])
                    if xyz:
                        pts.append(xyz)
            if pts:
                c = (
                    sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts),
                    sum(p[2] for p in pts) / len(pts),
                )

        if c is None:
            if last_c is not None and carry < CARRY_MAX:
                c = last_c
                carry += 1
            else:
                signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
                continue
        else:
            last_c = c
            carry = 0

        ang = _angle(a, b, c)
        if ang is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        signal.append({"frame": frame_idx, "angle_deg": ang, "valid": True})

    return signal
