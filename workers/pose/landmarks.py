# Minimal MediaPipe Pose landmark indices needed for ActionLab action classification

IDX = {
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

def get_lm(pose_frame, name):
    lms = (pose_frame or {}).get("landmarks")
    if not lms:
        return None
    idx = IDX.get(name)
    if idx is None or idx >= len(lms):
        return None
    return lms[idx]

def vis_ok(lm, thr=0.6):
    if not lm:
        return False
    v = lm.get("visibility", 0.0)
    return isinstance(v, (int, float)) and v >= thr
