# app/workers/risk/foot_line_deviation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

# MediaPipe Pose indices
LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

EPS = 1e-9


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
    if mag < EPS:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _signed_offset(D: tuple[float, float], origin: tuple[float, float], pt: tuple[float, float]) -> float:
    return _cross(D, (pt[0] - origin[0], pt[1] - origin[1]))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


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
    Front-foot line alignment risk.

    Principle:
    - Penalize medial crossing / load-line collapse.
    - Do not mistake external rotation or an outward elite step for a moderate pathology.
    - Judge the plant from heel/ankle support line first, toe only second.
    """

    out: Dict[str, Any] = {
        "signal_strength": 0.0,
        "confidence": 0.0,
        "mode": "OUTWARD_STEP",
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

    hand = None
    if isinstance(action, dict):
        h = action.get("hand") or action.get("bowling_hand") or action.get("input_hand")
        if isinstance(h, str) and h.strip():
            hand = h.strip().upper()

    if hand == "R":
        front_toe_idx, back_toe_idx, hand_conf = LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, 1.0
        front_heel_idx, front_ankle_idx, front_knee_idx = LEFT_HEEL, LEFT_ANKLE, LEFT_KNEE
    elif hand == "L":
        front_toe_idx, back_toe_idx, hand_conf = RIGHT_FOOT_INDEX, LEFT_FOOT_INDEX, 1.0
        front_heel_idx, front_ankle_idx, front_knee_idx = RIGHT_HEEL, RIGHT_ANKLE, RIGHT_KNEE
    else:
        front_toe_idx, back_toe_idx, hand_conf = LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, 0.35
        front_heel_idx, front_ankle_idx, front_knee_idx = LEFT_HEEL, LEFT_ANKLE, LEFT_KNEE

    B_toe = _xy(lms_b, back_toe_idx) if lms_b is not None else None
    F_toe = _xy(lms_f, front_toe_idx) if lms_f is not None else None
    F_heel = _xy(lms_f, front_heel_idx) if lms_f is not None else None
    F_ank = _xy(lms_f, front_ankle_idx) if lms_f is not None else None
    F_knee = _xy(lms_f, front_knee_idx) if lms_f is not None else None

    LH = _xy(lms_f, LEFT_HIP) if lms_f is not None else None
    RH = _xy(lms_f, RIGHT_HIP) if lms_f is not None else None

    mh_b = _mid_hip(lms_b) if lms_b is not None else None
    mh_f = _mid_hip(lms_f) if lms_f is not None else None

    D = (0.0, 0.0)
    if mh_b is not None and mh_f is not None:
        D = _norm((mh_f[0] - mh_b[0], mh_f[1] - mh_b[1]))
    if (abs(D[0]) + abs(D[1]) < EPS) and (B_toe is not None) and (F_toe is not None):
        D = _norm((F_toe[0] - B_toe[0], F_toe[1] - B_toe[1]))

    if (abs(D[0]) + abs(D[1]) < EPS) or (B_toe is None) or (LH is None) or (RH is None) or (F_toe is None):
        return out

    hip_w = _dist(LH, RH)
    if hip_w < EPS:
        return out

    toe_off = _signed_offset(D, B_toe, F_toe)
    heel_off = _signed_offset(D, B_toe, F_heel) if F_heel is not None else None
    ank_off = _signed_offset(D, B_toe, F_ank) if F_ank is not None else None
    knee_off = _signed_offset(D, B_toe, F_knee) if F_knee is not None else None

    toe_inward = toe_off > 0.0

    plant_votes = 0
    plant_checks = 0
    if heel_off is not None:
        plant_checks += 1
        plant_votes += 1 if ((heel_off > 0.0) == toe_inward) else 0
    if ank_off is not None:
        plant_checks += 1
        plant_votes += 1 if ((ank_off > 0.0) == toe_inward) else 0

    toe_only_cross = False
    if toe_inward:
        if heel_off is not None:
            toe_only_cross = heel_off <= 0.0
        elif ank_off is not None:
            toe_only_cross = ank_off <= 0.0

    collapse_norm = 0.0
    has_collapse = False
    if (knee_off is not None) and (ank_off is not None):
        collapse_norm = max(0.0, (knee_off - ank_off) / hip_w)
        collapse_thr = float(cfg.get("collapse", 0.035))
        has_collapse = collapse_norm >= collapse_thr

    plant_confirmed = (plant_checks > 0) and (plant_votes >= max(1, plant_checks))
    mode = "INWARD_CROSS" if toe_inward and (plant_confirmed or has_collapse) else "OUTWARD_STEP"

    support_offsets = [abs(v) for v in (heel_off, ank_off) if v is not None]
    support_line_norm = (_median(support_offsets) / hip_w) if support_offsets else (abs(toe_off) / hip_w)
    plant_line_norm = (_median([abs(toe_off)] + support_offsets) / hip_w)
    toe_norm = abs(toe_off) / hip_w

    if toe_only_cross and mode == "OUTWARD_STEP":
        support_line_norm *= 0.45
        plant_line_norm *= 0.45

    if mode == "INWARD_CROSS":
        offset_norm = plant_line_norm * (1.0 + min(0.25, collapse_norm * 2.0))
    else:
        offset_norm = support_line_norm

    t_low = float(cfg.get("low", 0.08))
    t_med = float(cfg.get("med", 0.15))
    outward_high = float(cfg.get("outward_high", 0.22))

    if mode == "OUTWARD_STEP":
        if offset_norm <= t_low:
            signal = 0.10
        elif offset_norm <= t_med:
            signal = 0.16
        elif offset_norm <= outward_high:
            signal = 0.18
        else:
            signal = 0.22
    else:
        if offset_norm <= t_low:
            signal = 0.20
        elif offset_norm <= t_med:
            signal = 0.45
        else:
            signal = 0.70

    geom_conf = 1.0
    if (heel_off is None) and (ank_off is None):
        geom_conf *= 0.55
    if (knee_off is None) and mode == "INWARD_CROSS":
        geom_conf *= 0.75
    if toe_only_cross and mode == "OUTWARD_STEP":
        geom_conf *= 0.70

    conf = max(0.0, min(1.0, hand_conf * geom_conf))

    out["signal_strength"] = float(max(0.0, min(1.0, signal)))
    out["confidence"] = float(conf)
    out["mode"] = mode
    out["metrics"] = {
        "offset_norm": round(float(offset_norm), 3),
        "toe_norm": round(float(toe_norm), 3),
        "support_line_norm": round(float(support_line_norm), 3),
        "plant_line_norm": round(float(plant_line_norm), 3),
        "toe_only": bool(toe_only_cross),
        "plant_checks": int(plant_checks),
        "plant_votes": int(plant_votes),
        "collapse_norm": round(float(collapse_norm), 3),
    }
    return out
