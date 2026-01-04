from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.trunk_rotation_snap import compute_trunk_rotation_snap
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean

RISK_CONFIG = {
    "ffbs": {"pre_window": 6, "post_window": 4, "jerk_reference": 15.0, "floor": 0.15},
    "knee": {"post_window": 8, "collapse_ref": 20.0, "floor": 0.15},
    "trunk": {"snap_ref": 120.0, "floor": 0.15},
    "hsm": {"mismatch_ref": 2.0, "floor": 0.15},
    "lean": {"lean_ref": 0.08, "floor": 0.15},
}


def _emit(obj, floor):
    if obj is None:
        return {"signal_strength": floor, "confidence": 0.4}
    obj["signal_strength"] = max(obj["signal_strength"], floor)
    return obj


def run_risk_worker(pose_frames, video, events, action):
    fps = float(video.get("fps") or 0.0)
    risks = []

    ffc = (events.get("ffc") or {}).get("frame")
    bfc = (events.get("bfc") or {}).get("frame")
    uah = (events.get("uah") or {}).get("frame")
    rel = (events.get("release") or {}).get("frame")

    risks.append(_emit(
        compute_front_foot_braking_shock(pose_frames, ffc, fps, RISK_CONFIG["ffbs"]),
        RISK_CONFIG["ffbs"]["floor"]
    ))

    risks.append(_emit(
        compute_knee_brace_failure(pose_frames, ffc, fps, RISK_CONFIG["knee"]),
        RISK_CONFIG["knee"]["floor"]
    ))

    risks.append(_emit(
        compute_trunk_rotation_snap(pose_frames, ffc, uah, fps, RISK_CONFIG["trunk"]),
        RISK_CONFIG["trunk"]["floor"]
    ))

    risks.append(_emit(
        compute_hip_shoulder_mismatch(pose_frames, ffc, rel, fps, RISK_CONFIG["hsm"]),
        RISK_CONFIG["hsm"]["floor"]
    ))

    risks.append(_emit(
        compute_lateral_trunk_lean(pose_frames, ffc, rel, fps, RISK_CONFIG["lean"]),
        RISK_CONFIG["lean"]["floor"]
    ))

    return risks
