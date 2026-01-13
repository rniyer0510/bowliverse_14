# app/workers/risk/knee_brace_failure.py

import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)


def compute_knee_brace_failure(pose_frames, ffc_frame, fps, config):
    floor = float(config.get("floor", 0.15))

    if ffc_frame is None or ffc_frame < 0:
        return {"risk_id": "knee_brace_failure", "signal_strength": floor, "confidence": 0.0}

    pelvis_y = []
    vis = []

    for i in range(max(0, ffc_frame - 5), min(len(pose_frames), ffc_frame + 8)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue

        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue

        if "LEFT_HIP" not in lm or "RIGHT_HIP" not in lm:
            continue

        y = (lm["LEFT_HIP"]["y"] + lm["RIGHT_HIP"]["y"]) / 2.0
        v = min(lm["LEFT_HIP"].get("v", 0), lm["RIGHT_HIP"].get("v", 0))

        pelvis_y.append(y)
        vis.append(v)

    if len(pelvis_y) < 4:
        return {
            "risk_id": "knee_brace_failure",
            "signal_strength": floor,
            "confidence": 0.0,
            "note": "Pelvis motion insufficient",
        }

    drop = max(pelvis_y) - min(pelvis_y)
    signal = min(1.0, drop / 0.04)  # conservative
    confidence = float(np.mean(vis)) * 0.7

    return {
        "risk_id": "knee_brace_failure",
        "signal_strength": round(max(signal, floor), 3),
        "confidence": round(confidence, 3),
        "debug": {"pelvis_drop": round(drop, 4), "mode": "proxy"},
    }

# -----------------------------
# Visualisation (PNG export)
# -----------------------------
from app.workers.risk.visual_utils import extract_frame, save_visual

def _attach_visual(
    risk_obj,
    video,
    frame_idx,
    region="inner_thigh",
    direction="medial_load"
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
