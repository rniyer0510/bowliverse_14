import statistics
from app.common.signal_quality import landmarks_visible
from app.workers.action.geometry import (
    compute_batsman_axis,
    vec,
    norm,
    angle_deg,
)
from app.workers.action.foot_orientation import compute_foot_intent

# MediaPipe landmark indices
LS, RS = 11, 12
LH, RH = 23, 24

# -----------------------------
# Pelvic tolerance bands
# -----------------------------
HIP_OPEN_LIMIT = 55.0      # deg
HIP_DRIFT_LIMIT = 12.0     # deg (BFC → FFC)


def classify_action(pose_frames, hand, bfc_frame, ffc_frame):
    """
    Action classification (V14 LOCKED)

    Decision hierarchy:
    1. FOOT intent @ BFC (PRIMARY)
    2. Shoulder intent (SECONDARY)
    3. MIXED only on STRUCTURAL pelvic violation
    """

    if bfc_frame is None:
        return {"intent": None, "action": "UNKNOWN", "confidence": 0.0}

    axis = compute_batsman_axis(pose_frames, bfc_frame, ffc_frame)
    if axis is None:
        return {"intent": None, "action": "UNKNOWN", "confidence": 0.0}

    # --------------------------------
    # PRIMARY: Foot-based intent
    # --------------------------------
    foot = compute_foot_intent(
        pose_frames=pose_frames,
        hand=hand,
        bfc_frame=bfc_frame,
        axis=axis,
    )

    foot_intent = foot["intent"] if foot else None
    foot_conf = foot["confidence"] if foot else 0.0

    # --------------------------------
    # SECONDARY: Shoulder intent
    # --------------------------------
    shoulder_angles = []
    used = 0

    for f in range(bfc_frame - 3, bfc_frame + 4):
        if f < 0 or f >= len(pose_frames):
            continue

        frame = pose_frames[f]
        if not landmarks_visible(frame, [LS, RS, LH, RH]):
            continue

        lm = frame["landmarks"]
        sh_vec = norm(vec(lm[RS], lm[LS]))
        if not sh_vec:
            continue

        ang = angle_deg(sh_vec, axis)
        shoulder_angles.append(ang)
        used += 1

    shoulder_intent = None
    shoulder_conf = 0.0

    if len(shoulder_angles) >= 3:
        shoulder_med = statistics.median(shoulder_angles)
        shoulder_var = statistics.pstdev(shoulder_angles) if len(shoulder_angles) > 1 else 0.0

        if shoulder_med < 35:
            shoulder_intent = "SIDE_ON"
        elif shoulder_med > 65:
            shoulder_intent = "FRONT_ON"
        else:
            shoulder_intent = "SEMI_OPEN"

        vis_conf = min(1.0, used / 7.0)
        var_conf = max(0.0, 1.0 - (shoulder_var / 20.0))
        shoulder_conf = vis_conf * var_conf

    # --------------------------------
    # Final intent
    # --------------------------------
    if foot_intent:
        intent = foot_intent
        confidence = foot_conf
    elif shoulder_intent:
        intent = shoulder_intent
        confidence = shoulder_conf
    else:
        return {"intent": None, "action": "UNKNOWN", "confidence": 0.0}

    # --------------------------------
    # STRUCTURAL MIXED detection
    # --------------------------------
    mixed = False

    lm_bfc = pose_frames[bfc_frame]["landmarks"]
    hip_vec_bfc = norm(vec(lm_bfc[RH], lm_bfc[LH]))
    hip_ang_bfc = angle_deg(hip_vec_bfc, axis) if hip_vec_bfc else None

    hip_ang_ffc = None
    if ffc_frame is not None:
        lm_ffc = pose_frames[ffc_frame]["landmarks"]
        hip_vec_ffc = norm(vec(lm_ffc[RH], lm_ffc[LH]))
        hip_ang_ffc = angle_deg(hip_vec_ffc, axis) if hip_vec_ffc else None

    # Case 1: pelvis already too open at BFC
    if intent == "SIDE_ON" and hip_ang_bfc is not None:
        if hip_ang_bfc > HIP_OPEN_LIMIT:
            mixed = True

    # Case 2: pelvis keeps opening after BFC
    if hip_ang_bfc is not None and hip_ang_ffc is not None:
        if (hip_ang_ffc - hip_ang_bfc) > HIP_DRIFT_LIMIT:
            mixed = True

    action = "MIXED" if mixed else intent

    return {
        "intent": intent.lower(),
        "action": action,
        "confidence": round(confidence, 2),
    }
