# app/workers/risk/hip_shoulder_mismatch.py

import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)


def compute_hip_shoulder_mismatch(pose_frames, ffc_frame, rel_frame, fps, config):
    floor = float(config.get("floor", 0.15))

    hips = []
    shoulders = []

    for i in range(max(0, ffc_frame - 6), min(len(pose_frames), ffc_frame + 6)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue

        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue

        if not all(k in lm for k in ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]):
            continue

        hips.append(lm["RIGHT_HIP"]["x"] - lm["LEFT_HIP"]["x"])
        shoulders.append(lm["RIGHT_SHOULDER"]["x"] - lm["LEFT_SHOULDER"]["x"])

    if len(hips) < 4:
        return {
            "risk_id": "hip_shoulder_mismatch",
            "signal_strength": floor,
            "confidence": 0.0,
            "note": "Phase proxy insufficient",
        }

    phase = np.mean(np.abs(np.array(hips) - np.array(shoulders)))
    signal = min(1.0, phase / 0.25)

    return {
        "risk_id": "hip_shoulder_mismatch",
        "signal_strength": round(max(signal, floor), 3),
        "confidence": 0.6,
        "debug": {"phase_lag": round(phase, 3), "mode": "proxy"},
    }

# -----------------------------
# Visualisation (PNG export)
# -----------------------------
from app.workers.risk.visual_utils import extract_frame, save_visual

def _attach_visual(
    risk_obj,
    video,
    frame_idx,
    region="lumbar_spine",
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
