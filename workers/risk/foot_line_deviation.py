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
    """Signed lateral offset of pt relative to progression direction D, using cross(D, pt-origin)."""
    return _cross(D, (pt[0] - origin[0], pt[1] - origin[1]))


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
    Foot-line deviation risk (stress-based, NOT style-based).

    FIXED BIOMECH PRINCIPLE:
    - Toe crossing alone is NOT "bad". Many elites (incl. McGrath) show toe-out / external rotation.
    - "Bad" deviation is when the *whole plant* crosses/loads medially:
        heel + ankle (and often knee) also cross the line of progression,
      OR knee collapses medially relative to ankle under load (valgus-like pattern).

    We therefore require:
      (A) Plant crossing evidence: heel+ankle agree with toe (not toe-only), AND/OR
      (B) Knee-over-foot collapse: knee offset noticeably more medial than ankle offset.

    Output modes:
      - INWARD_CROSS  : medial-crossing / collapse pattern (potential groin load)
      - OUTWARD_STEP  : neutral / toe-out / externally rotated (often protective)

    Returns always a risk object; weak geometry => confidence low.
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

    # Hand from action (best-effort). If unknown => confidence forced low.
    hand = None
    if isinstance(action, dict):
        h = action.get("hand") or action.get("bowling_hand") or action.get("input_hand")
        if isinstance(h, str) and h.strip():
            hand = h.strip().upper()

    # Decide which side is "front foot" by handedness (existing convention in your pipeline).
    if hand == "R":
        front_toe_idx, back_toe_idx, hand_conf = LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, 1.0
        front_heel_idx, front_ankle_idx, front_knee_idx = LEFT_HEEL, LEFT_ANKLE, LEFT_KNEE
    elif hand == "L":
        front_toe_idx, back_toe_idx, hand_conf = RIGHT_FOOT_INDEX, LEFT_FOOT_INDEX, 1.0
        front_heel_idx, front_ankle_idx, front_knee_idx = RIGHT_HEEL, RIGHT_ANKLE, RIGHT_KNEE
    else:
        # Unknown: still compute but lower confidence; keep a consistent guess.
        front_toe_idx, back_toe_idx, hand_conf = LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, 0.35
        front_heel_idx, front_ankle_idx, front_knee_idx = LEFT_HEEL, LEFT_ANKLE, LEFT_KNEE

    # Points
    B_toe = _xy(lms_b, back_toe_idx) if lms_b is not None else None  # back-foot toe at BFC
    F_toe = _xy(lms_f, front_toe_idx) if lms_f is not None else None  # front-foot toe at FFC
    F_heel = _xy(lms_f, front_heel_idx) if lms_f is not None else None
    F_ank = _xy(lms_f, front_ankle_idx) if lms_f is not None else None
    F_knee = _xy(lms_f, front_knee_idx) if lms_f is not None else None

    LH = _xy(lms_f, LEFT_HIP) if lms_f is not None else None
    RH = _xy(lms_f, RIGHT_HIP) if lms_f is not None else None

    mh_b = _mid_hip(lms_b) if lms_b is not None else None
    mh_f = _mid_hip(lms_f) if lms_f is not None else None

    # Progression direction D: pelvis travel BFC->FFC (preferred), else B_toe->F_toe
    D = (0.0, 0.0)
    if mh_b is not None and mh_f is not None:
        D = _norm((mh_f[0] - mh_b[0], mh_f[1] - mh_b[1]))
    if (abs(D[0]) + abs(D[1]) < EPS) and (B_toe is not None) and (F_toe is not None):
        D = _norm((F_toe[0] - B_toe[0], F_toe[1] - B_toe[1]))

    # Minimal geometry requirement: need D + back origin + hips for scale + at least toe&heel (or toe&ankle)
    if (abs(D[0]) + abs(D[1]) < EPS) or (B_toe is None) or (LH is None) or (RH is None) or (F_toe is None):
        return out

    hip_w = _dist(LH, RH)
    if hip_w < EPS:
        return out

    # Signed offsets (medial/lateral relative to D). Positive/negative is just a side label in image plane.
    toe_off = _signed_offset(D, B_toe, F_toe)
    heel_off = _signed_offset(D, B_toe, F_heel) if F_heel is not None else None
    ank_off = _signed_offset(D, B_toe, F_ank) if F_ank is not None else None
    knee_off = _signed_offset(D, B_toe, F_knee) if F_knee is not None else None

    # Determine "inward" using toe sign (existing convention); we then require heel/ankle agreement to confirm.
    toe_inward = toe_off > 0.0

    # Confirmation: whole-plant crossing should move heel and ankle the same way as toe.
    plant_votes = 0
    plant_checks = 0
    if heel_off is not None:
        plant_checks += 1
        plant_votes += 1 if ((heel_off > 0.0) == toe_inward) else 0
    if ank_off is not None:
        plant_checks += 1
        plant_votes += 1 if ((ank_off > 0.0) == toe_inward) else 0

    # External rotation (toe-out) false positive pattern:
    # toe indicates "cross", but heel does NOT follow (heel opposite sign or near-zero) -> treat as OUTWARD_STEP / safe.
    toe_only_cross = False
    if toe_inward:
        if heel_off is not None:
            toe_only_cross = (heel_off <= 0.0)
        elif ank_off is not None:
            toe_only_cross = (ank_off <= 0.0)

    # Knee-over-ankle collapse cue (valgus-like):
    # if knee is "more inward" than ankle by a meaningful margin, that's a stress pattern.
    # Normalize by hip width so it scales across zoom.
    collapse_norm = 0.0
    has_collapse = False
    if (knee_off is not None) and (ank_off is not None):
        collapse_norm = abs(knee_off - ank_off) / hip_w
        # Threshold is deliberately modest; cfg can tune.
        # Default 0.035 ~ "knee drifting inward noticeably" at typical MediaPipe scale.
        collapse_thr = float(cfg.get("collapse", 0.035))
        has_collapse = (collapse_norm >= collapse_thr) and (((knee_off > 0.0) == toe_inward) and ((ank_off > 0.0) == toe_inward))

    # Plant crossing evidence (needs at least one confirming segment; two is stronger)
    plant_confirmed = (plant_checks > 0) and (plant_votes >= max(1, plant_checks))  # strict agreement
    plant_partial = (plant_checks > 0) and (plant_votes >= 1)

    # Decide mode:
    # - True inward cross if toe indicates inward AND (plant_confirmed OR collapse)
    # - Otherwise outward/neutral (including external rotation)
    if toe_inward and (plant_confirmed or has_collapse):
        mode = "INWARD_CROSS"
    else:
        mode = "OUTWARD_STEP"

    # Score magnitude: use the best available "plant" point, NOT just toe.
    # This reduces toe-out false positives dramatically.
    offsets = [abs(toe_off)]
    if heel_off is not None:
        offsets.append(abs(heel_off))
    if ank_off is not None:
        offsets.append(abs(ank_off))
    # Use median to avoid one landmark glitch dominating.
    offsets.sort()
    off_med = offsets[len(offsets) // 2]
    offset_norm = off_med / hip_w

    # If this is toe-only cross (classic McGrath-style toe-out), squash the signal aggressively.
    if toe_only_cross and mode == "OUTWARD_STEP":
        offset_norm *= 0.35

    # If we see collapse, boost relevance a bit (still bounded).
    if has_collapse and mode == "INWARD_CROSS":
        offset_norm *= 1.15

    # Thresholds (kept configurable)
    t_low = float(cfg.get("low", 0.08))
    t_med = float(cfg.get("med", 0.15))

    # Convert to signal strength with biomechanical gating:
    # OUTWARD_STEP should never be "high load" from geometry alone.
    if mode == "OUTWARD_STEP":
        # keep it low unless it's extreme and confirmed (rare)
        if offset_norm <= t_low:
            signal = 0.10
        elif offset_norm <= t_med:
            signal = 0.18
        else:
            signal = 0.25
    else:
        # INWARD_CROSS (confirmed/collapse)
        if offset_norm <= t_low:
            signal = 0.20
        elif offset_norm <= t_med:
            signal = 0.45
        else:
            signal = 0.70

    # Confidence: depends on handedness + geometry completeness
    geom_conf = 1.0
    # Need at least one of heel/ankle to confirm plant
    if (heel_off is None) and (ank_off is None):
        geom_conf *= 0.55
    # Knee helps collapse logic
    if (knee_off is None) and mode == "INWARD_CROSS":
        geom_conf *= 0.75
    # If it’s toe-only cross, confidence that it’s risky should be lower
    if toe_only_cross and mode == "OUTWARD_STEP":
        geom_conf *= 0.70

    conf = max(0.0, min(1.0, hand_conf * geom_conf))

    out["signal_strength"] = float(max(0.0, min(1.0, signal)))
    out["confidence"] = float(conf)
    out["mode"] = mode
    out["metrics"] = {
        "offset_norm": round(float(offset_norm), 3),
        "toe_only": bool(toe_only_cross),
        "plant_checks": int(plant_checks),
        "plant_votes": int(plant_votes),
        "collapse_norm": round(float(collapse_norm), 3),
    }
    return out

