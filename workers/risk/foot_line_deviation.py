from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

def _xy(lms: Any, idx: int) -> Optional[tuple[float, float]]:
    try:
        if not isinstance(lms, list) or idx >= len(lms):
            return None
        pt = lms[idx]
        if pt is None:
            return None
        x = float(pt.get("x"))
        y = float(pt.get("y"))
        if math.isnan(x) or math.isnan(y):
            return None
        return (x, y)
    except Exception:
        return None

def _mid_hip(lms: Any) -> Optional[tuple[float, float]]:
    lh = _xy(lms, LEFT_HIP)
    rh = _xy(lms, RIGHT_HIP)
    if lh is None or rh is None:
        return None
    return ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)

def _norm(v: tuple[float, float]) -> tuple[float, float]:
    mag = math.hypot(v[0], v[1])
    if mag < 1e-9:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)

def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_foot_line_deviation(
    pose_frames: List[Dict[str, Any]],
    bfc_frame: Optional[int],
    ffc_frame: Optional[int],
    fps: float,
    cfg: Dict[str, Any],
    *,
    action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Foot-line deviation risk (directional).

    ACTIONLAB INVARIANT:
    - If frames exist, we ALWAYS return a risk object.
    - Weak geometry => confidence=0.0, metrics fallback to 0.0.
    - risk_worker._emit enforces the floor signal_strength.
    """

    out: Dict[str, Any] = {
        "signal_strength": 0.0,
        "confidence": 0.0,
        "mode": "OUTWARD_STEP",  # safe default for clinician mapping
        "metrics": {"offset_norm": 0.0},
    }

    if bfc_frame is None or ffc_frame is None:
        return out

    n = len(pose_frames)
    if n <= 0:
        return out

    bfc = max(0, min(int(bfc_frame), n - 1))
    ffc = max(0, min(int(ffc_frame), n - 1))

    lms_b = pose_frames[bfc].get("landmarks")
    lms_f = pose_frames[ffc].get("landmarks")

    # Hand from action (best-effort). If unknown => confidence forced to 0.
    hand = None
    if isinstance(action, dict):
        h = action.get("hand") or action.get("bowling_hand") or action.get("input_hand")
        if isinstance(h, str) and h.strip():
            hand = h.strip().upper()

    if hand == "R":
        front_idx, back_idx, hand_conf = LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, 1.0
    elif hand == "L":
        front_idx, back_idx, hand_conf = RIGHT_FOOT_INDEX, LEFT_FOOT_INDEX, 1.0
    else:
        front_idx, back_idx, hand_conf = LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, 0.0

    B = _xy(lms_b, back_idx) if lms_b is not None else None
    F = _xy(lms_f, front_idx) if lms_f is not None else None
    LH = _xy(lms_f, LEFT_HIP) if lms_f is not None else None
    RH = _xy(lms_f, RIGHT_HIP) if lms_f is not None else None

    mh_b = _mid_hip(lms_b) if lms_b is not None else None
    mh_f = _mid_hip(lms_f) if lms_f is not None else None

    D = (0.0, 0.0)
    if mh_b is not None and mh_f is not None:
        D = _norm((mh_f[0] - mh_b[0], mh_f[1] - mh_b[1]))
    if abs(D[0]) + abs(D[1]) < 1e-9 and B is not None and F is not None:
        D = _norm((F[0] - B[0], F[1] - B[1]))

    if (abs(D[0]) + abs(D[1]) < 1e-9) or (B is None) or (F is None) or (LH is None) or (RH is None):
        return out

    hip_w = _dist(LH, RH)
    if hip_w < 1e-9:
        return out

    V = (F[0] - B[0], F[1] - B[1])
    signed_offset = _cross(D, V)
    mode = "INWARD_CROSS" if signed_offset > 0 else "OUTWARD_STEP"
    offset_norm = abs(signed_offset) / hip_w

    t_low = float(cfg.get("low", 0.08))
    t_med = float(cfg.get("med", 0.15))

    if offset_norm <= t_low:
        signal = 0.15
    elif offset_norm <= t_med:
        signal = 0.35
    else:
        signal = 0.65

    conf = max(0.0, min(1.0, 1.0 * hand_conf))

    out["signal_strength"] = float(signal)
    out["confidence"] = float(conf)
    out["mode"] = mode
    out["metrics"] = {"offset_norm": round(float(offset_norm), 3)}
    return out
