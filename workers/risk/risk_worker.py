from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.trunk_rotation_snap import compute_trunk_rotation_snap
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean

from app.workers.risk.visual_utils import draw_and_save_visual


# ---------------------------------------------------------------------
# Risk configuration (LOCKED)
# ---------------------------------------------------------------------
RISK_CONFIG = {
    "ffbs": {"floor": 0.15},
    "knee": {"floor": 0.15},
    "trunk": {"floor": 0.15},
    "hsm": {"floor": 0.15},
    "lean": {"floor": 0.15},
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _emit(obj, floor, risk_id: str):
    if obj is None:
        return {
            "risk_id": risk_id,
            "signal_strength": floor,
            "confidence": 0.0,
            "note": "Insufficient samples",
        }

    obj.setdefault("risk_id", risk_id)
    obj["signal_strength"] = max(float(obj.get("signal_strength", 0.0)), floor)
    obj.setdefault("confidence", 0.0)
    return obj


def _select_visual_window(risk_id, events):
    ffc = (events.get("ffc") or {}).get("frame")
    bfc = (events.get("bfc") or {}).get("frame")
    uah = (events.get("uah") or {}).get("frame")
    rel = (events.get("release") or {}).get("frame")

    if risk_id in ("front_foot_braking_shock", "knee_brace_failure"):
        if ffc is not None:
            return max(0, ffc - 5), ffc + 10

    if risk_id == "trunk_rotation_snap":
        if uah is not None:
            return max(0, uah - 4), uah + 4
        if ffc is not None:
            return max(0, ffc - 3), ffc + 3

    if risk_id == "hip_shoulder_mismatch":
        if uah is not None and rel is not None:
            return min(uah, rel) - 4, max(uah, rel) + 4

    if risk_id == "lateral_trunk_lean":
        if bfc is not None and rel is not None:
            return max(0, bfc - 4), rel + 4

    if ffc is not None:
        return max(0, ffc - 5), ffc + 5

    return None, None


def _attach_visual(risk, pose_frames, video, events):
    if not isinstance(risk, dict):
        return risk

    video_path = video.get("path") or video.get("file_path")
    if not video_path:
        return risk

    start, end = _select_visual_window(risk["risk_id"], events)
    if start is None or end is None:
        return risk

    total = len(pose_frames)
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))

    frame_idx = (start + end) // 2

    visual = draw_and_save_visual(
        risk_id=risk["risk_id"],
        video_path=video_path,
        frame_idx=frame_idx,
        pose_frames=pose_frames,
    )

    if visual:
        conf = float(risk.get("confidence", 0.0))
        visual["visual_confidence"] = (
            "HIGH" if conf >= 0.6 else
            "MEDIUM" if conf >= 0.3 else
            "LOW"
        )
        risk["visual"] = visual
        risk["visual_window"] = {"start": start, "end": end}

    return risk


# ---------------------------------------------------------------------
# Main worker (FINAL)
# ---------------------------------------------------------------------
def run_risk_worker(pose_frames, video, events, action):
    fps = float(video.get("fps") or 0.0)

    ffc = (events.get("ffc") or {}).get("frame")
    bfc = (events.get("bfc") or {}).get("frame")
    uah = (events.get("uah") or {}).get("frame")
    rel = (events.get("release") or {}).get("frame")

    risks = [
        _emit(
            compute_front_foot_braking_shock(
                pose_frames, ffc, fps, {}, action=action or {}
            ),
            RISK_CONFIG["ffbs"]["floor"],
            "front_foot_braking_shock",
        ),
        _emit(
            compute_knee_brace_failure(
                pose_frames, ffc, fps, {}
            ),
            RISK_CONFIG["knee"]["floor"],
            "knee_brace_failure",
        ),
        _emit(
            compute_trunk_rotation_snap(
                pose_frames, ffc, uah, fps, {}
            ),
            RISK_CONFIG["trunk"]["floor"],
            "trunk_rotation_snap",
        ),
        _emit(
            compute_hip_shoulder_mismatch(
                pose_frames, ffc, rel, fps, {}
            ),
            RISK_CONFIG["hsm"]["floor"],
            "hip_shoulder_mismatch",
        ),
        _emit(
            compute_lateral_trunk_lean(
                pose_frames, bfc, ffc, rel, fps, {}
            ),
            RISK_CONFIG["lean"]["floor"],
            "lateral_trunk_lean",
        ),
    ]

    out = []
    for r in risks:
        out.append(_attach_visual(r, pose_frames, video, events))

    return out

