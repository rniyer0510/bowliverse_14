import math
from app.common.logger import get_logger
from app.workers.pose.landmarks import get_lm

logger = get_logger(__name__)

# ============================================================
# Tunables (centralized – NO hardcoding inside logic)
# ============================================================

BFC_WINDOW = 3
FFC_WINDOW = 5
GAUSS_SIGMA = 1.0

VIS_THR_FOOT = 0.35
VIS_THR_HIP = 0.50
MIN_SAMPLES = 2

# Toe intent bands
TOE_SIDE_ON_MIN = 60.0
TOE_SEMI_OPEN_MIN = 30.0

# Expected hip corridors (deg)
HIP_CORRIDOR_SIDE_ON = (45.0, 90.0)
HIP_CORRIDOR_SEMI_OPEN = (25.0, 70.0)
HIP_CORRIDOR_FRONT_ON = (0.0, 40.0)


# ============================================================
# Geometry helpers
# ============================================================

def _angle_xy(a, b):
    """
    Angle of vector a -> b in XY plane, folded to [0, 90]
    0   = horizontal (-- like)
    90  = vertical (|)
    """
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    return ang if ang <= 90 else 180 - ang


def _gaussian_weights(n, sigma):
    c = n // 2
    return [math.exp(-0.5 * ((i - c) / sigma) ** 2) for i in range(n)]


def _smooth(values, sigma):
    if not values:
        return None
    if len(values) < 3:
        return sum(values) / len(values)
    w = _gaussian_weights(len(values), sigma)
    return sum(v * wi for v, wi in zip(values, w)) / sum(w)


def _window(center, radius, total):
    lo = max(0, center - radius)
    hi = min(total - 1, center + radius)
    return range(lo, hi + 1)


def _vis_ok(lm, thr):
    return lm and lm.get("visibility", 0.0) >= thr


def _collect_angles(frames, idxs, a_name, b_name, thr):
    out = []
    for i in idxs:
        a = get_lm(frames[i], a_name)
        b = get_lm(frames[i], b_name)
        if _vis_ok(a, thr) and _vis_ok(b, thr):
            out.append(_angle_xy(a, b))
    return out


# ============================================================
# Intent & corridor logic
# ============================================================

def corridor_from_toe_angle(toe_deg):
    """
    Declare action intent from toe direction @ BFC.
    """
    if toe_deg >= TOE_SIDE_ON_MIN:
        return "side_on", HIP_CORRIDOR_SIDE_ON
    if toe_deg >= TOE_SEMI_OPEN_MIN:
        return "semi_open", HIP_CORRIDOR_SEMI_OPEN
    return "front_on", HIP_CORRIDOR_FRONT_ON


def within_corridor(angle, corridor):
    lo, hi = corridor
    return lo <= angle <= hi


# ============================================================
# Main classifier (FINAL, ROBUST)
# ============================================================

def classify_action(pose_frames, hand, bfc_frame, ffc_frame=None):
    """
    FINAL v14 Action Classification Logic

    - Toe direction @ BFC declares intent (range-based)
    - Bowling-side hip orientation @ FFC validates intent
    - MIXED only if contradiction is sustained (distribution-based)
    - Robust to occlusion via windowing + smoothing
    - Conservative confidence handling when validation is partial
    """

    if not pose_frames or bfc_frame is None:
        return {"intent": "unknown", "type": "unknown", "compliance": 0.0}

    total_frames = len(pose_frames)
    if bfc_frame < 0 or bfc_frame >= total_frames:
        return {"intent": "unknown", "type": "unknown", "compliance": 0.0}

    is_r = (hand or "R").upper() == "R"

    # --------------------------------------------------------
    # 1. TOE @ BFC (intent)
    # --------------------------------------------------------

    heel = "RIGHT_HEEL" if is_r else "LEFT_HEEL"
    toe = "RIGHT_FOOT_INDEX" if is_r else "LEFT_FOOT_INDEX"

    bfc_idxs = _window(bfc_frame, BFC_WINDOW, total_frames)
    toe_vals = _collect_angles(
        pose_frames, bfc_idxs, heel, toe, VIS_THR_FOOT
    )

    if len(toe_vals) < MIN_SAMPLES:
        logger.info("[ACTION] Toe intent not recoverable at BFC")
        return {"intent": "unknown", "type": "unknown", "compliance": 0.0}

    toe_deg = _smooth(toe_vals, GAUSS_SIGMA)
    intent, corridor = corridor_from_toe_angle(toe_deg)

    # --------------------------------------------------------
    # 2. HIP @ FFC (validation)
    #    Use hip → opposite shoulder vector
    # --------------------------------------------------------

    if ffc_frame is None or ffc_frame < 0 or ffc_frame >= total_frames:
        return {
            "intent": intent,
            "type": intent,
            "toe_bfc_xy": round(toe_deg, 2),
            "bfc_frame": bfc_frame,
            "ffc_frame": ffc_frame,
            "compliance": 0.5,
        }

    hip = "RIGHT_HIP" if is_r else "LEFT_HIP"
    opp_shoulder = "LEFT_SHOULDER" if is_r else "RIGHT_SHOULDER"

    ffc_idxs = _window(ffc_frame, FFC_WINDOW, total_frames)
    hip_vals = _collect_angles(
        pose_frames, ffc_idxs, hip, opp_shoulder, VIS_THR_HIP
    )

    # --------------------------------------------------------
    # 2a. Partial hip evidence → confidence refinement
    # --------------------------------------------------------

    if len(hip_vals) < MIN_SAMPLES:
        if hip_vals:
            violations = [v for v in hip_vals if not within_corridor(v, corridor)]
            if not violations:
                return {
                    "intent": intent,
                    "type": intent,
                    "toe_bfc_xy": round(toe_deg, 2),
                    "bfc_frame": bfc_frame,
                    "ffc_frame": ffc_frame,
                    "compliance": 0.7,
                }

        return {
            "intent": intent,
            "type": intent,
            "toe_bfc_xy": round(toe_deg, 2),
            "bfc_frame": bfc_frame,
            "ffc_frame": ffc_frame,
            "compliance": 0.5,
        }

    hip_deg = _smooth(hip_vals, GAUSS_SIGMA)

    # --------------------------------------------------------
    # 3. Robust MIXED detection (distribution-based)
    # --------------------------------------------------------

    violations = [v for v in hip_vals if not within_corridor(v, corridor)]
    violation_ratio = len(violations) / max(1, len(hip_vals))

    if violation_ratio < 0.5:
        final_type = intent
        comp = 1.0
    else:
        final_type = "mixed"
        comp = 0.0

    return {
        "intent": intent,
        "type": final_type,
        "toe_bfc_xy": round(toe_deg, 2),
        "hip_ffc_xy": round(hip_deg, 2),
        "bfc_frame": bfc_frame,
        "ffc_frame": ffc_frame,
        "toe_samples": len(toe_vals),
        "hip_samples": len(hip_vals),
        "violation_ratio": round(violation_ratio, 2),
        "compliance": comp,
    }

