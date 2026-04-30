"""
Foot orientation intent — ActionLab V14 (FIXED)

PRIMARY INTENT SIGNAL
- BFC-anchored
- Foot PLANE orientation (ankle → foot centre)
- STRUCTURE ONLY
"""

from app.workers.action.geometry import vec, norm, angle_deg
from app.common.signal_quality import landmarks_visible

# MediaPipe landmarks
L_ANKLE, R_ANKLE = 27, 28
L_HEEL,  R_HEEL  = 29, 30
L_TOE,   R_TOE   = 31, 32

SIDE_ON_MAX = 35.0
FRONT_ON_MIN = 80.0


def _midpoint(a, b):
    return {
        "x": 0.5 * (a["x"] + b["x"]),
        "y": 0.5 * (a["y"] + b["y"]),
    }


def compute_foot_intent(pose_frames, hand, bfc_frame, axis):
    if bfc_frame is None or axis is None:
        return None

    frame = pose_frames[bfc_frame]

    lm = frame["landmarks"]

    # Select FRONT foot based on handedness
    if hand == "R":
        ankle = lm[L_ANKLE]
        heel  = lm[L_HEEL]
        toe   = lm[L_TOE]
        required = [L_ANKLE, L_HEEL, L_TOE]
    else:
        ankle = lm[R_ANKLE]
        heel  = lm[R_HEEL]
        toe   = lm[R_TOE]
        required = [R_ANKLE, R_HEEL, R_TOE]

    # Only require the selected front foot landmarks. Requiring both feet
    # drops the primary BFC intent on non-standard clips even when the
    # bowling-side front foot is clearly visible.
    if not landmarks_visible(frame, required):
        return None

    foot_center = _midpoint(heel, toe)

    foot_vec = norm(vec(ankle, foot_center))
    if not foot_vec:
        return None

    ang = angle_deg(foot_vec, axis)

    # Intent bands (STRUCTURAL)
    if ang < SIDE_ON_MAX:
        intent = "SIDE_ON"
    elif ang >= FRONT_ON_MIN:
        intent = "FRONT_ON"
    else:
        intent = "SEMI_OPEN"

    return {
        "angle": ang,
        "intent": intent,
        "confidence": 1.0,  # structural authority
    }
