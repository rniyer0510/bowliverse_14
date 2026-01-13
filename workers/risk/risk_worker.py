from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.trunk_rotation_snap import compute_trunk_rotation_snap
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean

from app.workers.risk.visual_utils import draw_and_save_visual

RISK_CONFIG = {
    "ffbs": {
        "pre_window": 6,
        "post_window": 4,
        "jerk_reference": 15.0,
        "travel_min": 0.012,
        "floor": 0.15
    },
    "knee": {"post_window": 8, "collapse_ref": 20.0, "floor": 0.15},
    "trunk": {"snap_ref": 120.0, "floor": 0.15},
    "hsm": {"mismatch_ref": 2.0, "floor": 0.15},
    "lean": {"lean_ref": 0.08, "floor": 0.15},
}


def _emit(obj, floor, risk_id: str):
    if obj is None:
        return {"risk_id": risk_id, "signal_strength": floor, "confidence": 0.4}

    if "risk_id" not in obj or not obj.get("risk_id"):
        obj["risk_id"] = risk_id

    obj["signal_strength"] = max(float(obj.get("signal_strength", 0.0)), float(floor))
    return obj


def _attach_visual_best_effort(risk, pose_frames, video, events):
    try:
        if not isinstance(risk, dict):
            return risk

        vid_path = video.get("path") or video.get("file_path") or video.get("video_path")
        if not vid_path:
            return risk

        rid = risk.get("risk_id")

        ffc = (events.get("ffc") or {}).get("frame")
        bfc = (events.get("bfc") or {}).get("frame")
        uah = (events.get("uah") or {}).get("frame")
        rel = (events.get("release") or {}).get("frame")

        anchor = None
        if rid == "front_foot_braking_shock":
            anchor = ffc
        elif rid == "knee_brace_failure":
            anchor = ffc
        elif rid == "trunk_rotation_snap":
            anchor = uah or ffc
        elif rid == "hip_shoulder_mismatch":
            anchor = rel or uah
        elif rid == "lateral_trunk_lean":
            anchor = bfc or ffc

        if anchor is None:
            return risk

        vis = draw_and_save_visual(
            risk=risk,
            video_path=vid_path,
            frame_idx=int(anchor),
            pose_frames=pose_frames,
        )
        if vis:
            risk["visual"] = vis

        return risk
    except Exception:
        return risk


def run_risk_worker(pose_frames, video, events, action):
    fps = float(video.get("fps") or 0.0)
    risks = []

    ffc = (events.get("ffc") or {}).get("frame")
    bfc = (events.get("bfc") or {}).get("frame")
    uah = (events.get("uah") or {}).get("frame")
    rel = (events.get("release") or {}).get("frame")

    risks.append(
        _emit(
            compute_front_foot_braking_shock(
                pose_frames, ffc, fps, RISK_CONFIG["ffbs"], action=action or {}
            ),
            RISK_CONFIG["ffbs"]["floor"],
            "front_foot_braking_shock",
        )
    )

    risks.append(
        _emit(
            compute_knee_brace_failure(pose_frames, ffc, fps, RISK_CONFIG["knee"]),
            RISK_CONFIG["knee"]["floor"],
            "knee_brace_failure",
        )
    )

    risks.append(
        _emit(
            compute_trunk_rotation_snap(pose_frames, ffc, uah, fps, RISK_CONFIG["trunk"]),
            RISK_CONFIG["trunk"]["floor"],
            "trunk_rotation_snap",
        )
    )

    risks.append(
        _emit(
            compute_hip_shoulder_mismatch(pose_frames, ffc, rel, fps, RISK_CONFIG["hsm"]),
            RISK_CONFIG["hsm"]["floor"],
            "hip_shoulder_mismatch",
        )
    )

    risks.append(
        _emit(
            compute_lateral_trunk_lean(
                pose_frames,
                bfc,
                ffc,
                rel,
                fps,
                RISK_CONFIG["lean"],
            ),
            RISK_CONFIG["lean"]["floor"],
            "lateral_trunk_lean",
        )
    )

    moderate_or_higher = [
        r for r in risks
        if r.get("risk_id") != "lateral_trunk_lean"
        and float(r.get("signal_strength", 0.0)) >= 0.4
    ]

    for r in risks:
        if (
            r.get("risk_id") == "lateral_trunk_lean"
            and float(r.get("signal_strength", 0.0)) >= 0.6
            and len(moderate_or_higher) == 0
        ):
            r["signal_strength"] = 0.45
            r.setdefault(
                "note",
                "Lateral posture is stable and isolated. "
                "Not a short-term risk; consider long-term load monitoring."
            )

    out = []
    for r in risks:
        out.append(_attach_visual_best_effort(r, pose_frames, video, events))

    return out
