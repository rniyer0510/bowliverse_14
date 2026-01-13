# app/workers/risk/trunk_rotation_snap.py

import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)


def compute_trunk_rotation_snap(pose_frames, ffc_frame, uah_frame, fps, config):
    floor = float(config.get("floor", 0.15))

    if ffc_frame is None or ffc_frame < 0:
        return {"risk_id": "trunk_rotation_snap", "signal_strength": floor, "confidence": 0.0}

    angles = []
    vis = []

    for i in range(max(0, ffc_frame - 6), min(len(pose_frames), ffc_frame + 6)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue

        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue

        if "LEFT_SHOULDER" not in lm or "RIGHT_SHOULDER" not in lm:
            continue

        dx = lm["RIGHT_SHOULDER"]["x"] - lm["LEFT_SHOULDER"]["x"]
        dy = lm["RIGHT_SHOULDER"]["y"] - lm["LEFT_SHOULDER"]["y"]

        angles.append(np.arctan2(dy, dx))
        vis.append(min(lm["LEFT_SHOULDER"].get("v", 0), lm["RIGHT_SHOULDER"].get("v", 0)))

    if len(angles) < 5:
        return {
            "risk_id": "trunk_rotation_snap",
            "signal_strength": floor,
            "confidence": 0.0,
            "note": "Torso rotation proxy insufficient",
        }

    jerk = np.max(np.abs(np.diff(np.diff(angles))))
    signal = min(1.0, jerk / 1.2)
    confidence = float(np.mean(vis)) * 0.7

    return {
        "risk_id": "trunk_rotation_snap",
        "signal_strength": round(max(signal, floor), 3),
        "confidence": round(confidence, 3),
        "debug": {"rot_jerk": round(jerk, 3), "mode": "proxy"},
    }

# -----------------------------
# Visualisation (PNG export)
# -----------------------------
from app.workers.risk.visual_utils import extract_frame, save_visual

def _attach_visual(
    risk_obj,
    video,
    frame_idx,
    region="torso_core",
    direction="rotational_torque"
):
    try:
        video_path = video.get("path") or video.get("file_path")
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            return risk_obj

        sev = "low"
        if risk_obj.get("signal_strength", 0) >= 0.6:
            sev = "high"
        elif risk_obj.get("signal_strength", 0) >= 0.3:
            sev = "moderate"

        img_path = save_visual(
            risk_id=risk_obj["risk_id"],
            frame_idx=frame_idx,
            frame=frame,
            region=region,
            direction=direction,
            severity=sev
        )

        risk_obj["visual"] = {
            "image_path": img_path,
            "region": region,
            "direction": direction
        }
    except Exception:
        pass

    return risk_obj

# ---- VISUAL ATTACH (FINAL STEP) ----
def _finalize_with_visual(risk, video, frame_idx):
    try:
        return _attach_visual(risk, video=video, frame_idx=frame_idx)
    except Exception:
        return risk

# --------------------------------------------------
# FINAL RETURN GATE (MANDATORY FOR VISUALS)
# --------------------------------------------------
def finalize_risk(risk, video, frame_idx=None):
    try:
        if frame_idx is not None:
            from app.workers.risk.visual_utils import extract_frame, save_visual

            frame = extract_frame(
                video.get("path") or video.get("file_path"),
                frame_idx
            )

            if frame is not None:
                sev = "low"
                if risk.get("signal_strength", 0) >= 0.6:
                    sev = "high"
                elif risk.get("signal_strength", 0) >= 0.3:
                    sev = "moderate"

                img = save_visual(
                    risk_id=risk["risk_id"],
                    frame_idx=frame_idx,
                    frame=frame,
                    region=risk.get("risk_region", "generic"),
                    direction=risk.get("risk_direction", "load"),
                    severity=sev
                )

                risk["visual"] = {
                    "image_path": img,
                    "region": risk.get("risk_region"),
                    "direction": risk.get("risk_direction"),
                }
    except Exception:
        pass

    return risk

# --------------------------------------------------
# FINAL VISUAL ATTACHMENT (CALLED EXPLICITLY)
# --------------------------------------------------
def finalize_risk(risk, video, frame_idx=None):
    try:
        if frame_idx is None:
            return risk

        from app.workers.risk.visual_utils import extract_frame, save_visual

        video_path = video.get("path") or video.get("file_path")
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            return risk

        sev = "low"
        if risk.get("signal_strength", 0) >= 0.6:
            sev = "high"
        elif risk.get("signal_strength", 0) >= 0.3:
            sev = "moderate"

        img = save_visual(
            risk_id=risk["risk_id"],
            frame_idx=frame_idx,
            frame=frame,
            region=risk.get("risk_region", "generic"),
            direction=risk.get("risk_direction", "load"),
            severity=sev
        )

        risk["visual"] = {
            "image_path": img,
            "region": risk.get("risk_region"),
            "direction": risk.get("risk_direction"),
        }
    except Exception:
        pass

    return risk
