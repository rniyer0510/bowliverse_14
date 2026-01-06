import numpy as np
from app.workers.action.geometry import (
    signed_angle_deg,
    compute_batsman_axis,
)

# -----------------------------
# STRUCTURAL thresholds
# -----------------------------
HIP_OPEN_LEAD_THRESHOLD = 15.0   # hips opening ahead of shoulders (deg)
SEP_HARD_LIMIT = 35.0            # absolute contradiction cap (deg)

# -----------------------------
# DYNAMIC thresholds
# -----------------------------
SCR_THRESHOLD = 30.0
SCR_MAX = 90.0


def compute_compliance(ctx):
    """
    Compliance model (biomechanics-correct):

    - Global plane: batsman axis
    - Intent: absolute shoulder angle vs batsman (handled upstream)
    - Structure: relative hip–shoulder delta at BFC
    - Dynamics: shoulder counter-rotation (SCR)
    """

    pose_frames = ctx.pose.frames
    bfc_f = ctx.events.bfc.frame
    ffc_f = ctx.events.ffc.frame

    batsman_axis = compute_batsman_axis(
        pose_frames, bfc_f, ffc_f
    )

    pose_bfc = pose_frames[bfc_f]
    pose_ffc = pose_frames[ffc_f]

    # -----------------------------
    # BFC geometry (STRUCTURE)
    # -----------------------------
    ls_b = np.array(pose_bfc["left_shoulder"])
    rs_b = np.array(pose_bfc["right_shoulder"])
    lh_b = np.array(pose_bfc["left_hip"])
    rh_b = np.array(pose_bfc["right_hip"])

    shoulder_vec_b = ls_b - rs_b
    hip_vec_b = lh_b - rh_b

    shoulder_angle_b = signed_angle_deg(
        shoulder_vec_b, batsman_axis
    )
    hip_angle_b = signed_angle_deg(
        hip_vec_b, batsman_axis
    )

    # Relative structure (THIS is MIXED logic)
    delta = hip_angle_b - shoulder_angle_b

    structural_ok = not (
        delta > HIP_OPEN_LEAD_THRESHOLD or
        abs(delta) > SEP_HARD_LIMIT
    )

    # -----------------------------
    # FFC geometry (DYNAMICS)
    # -----------------------------
    ls_f = np.array(pose_ffc["left_shoulder"])
    rs_f = np.array(pose_ffc["right_shoulder"])
    shoulder_vec_f = ls_f - rs_f

    shoulder_angle_f = signed_angle_deg(
        shoulder_vec_f, batsman_axis
    )

    scr = abs(shoulder_angle_f - shoulder_angle_b)

    dynamic_score = 1.0
    if scr > SCR_THRESHOLD:
        dynamic_score = max(
            0.0,
            1.0 - (scr - SCR_THRESHOLD) / (SCR_MAX - SCR_THRESHOLD)
        )

    structural_score = 1.0 if structural_ok else 0.0
    overall = 0.7 * structural_score + 0.3 * dynamic_score

    return {
        # absolute angles (for intent & debug)
        "shoulder_angle_bfc_deg": shoulder_angle_b,
        "hip_angle_bfc_deg": hip_angle_b,

        # relative structure (key biomech signal)
        "delta_deg": delta,

        # dynamics
        "scr_deg": scr,

        # decisions
        "structural_ok": structural_ok,
        "structural_score": structural_score,
        "dynamic_score": dynamic_score,
        "overall_score": overall,
    }
