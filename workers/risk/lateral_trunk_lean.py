# app/workers/risk/lateral_trunk_lean.py

import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)


def compute_lateral_trunk_lean(pose_frames, bfc_frame, ffc_frame, rel_frame, fps, config):
    floor = float(config.get("floor", 0.15))

    xs = []
    vis = []

    for i in range(max(0, bfc_frame - 6), min(len(pose_frames), bfc_frame + 6)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue

        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue

        if not all(k in lm for k in ["LEFT_HIP", "RIGHT_HIP"]):
            continue

        x = (lm["LEFT_HIP"]["x"] + lm["RIGHT_HIP"]["x"]) / 2.0
        xs.append(x)
        vis.append(min(lm["LEFT_HIP"].get("v", 0), lm["RIGHT_HIP"].get("v", 0)))

    if len(xs) < 4:
        return {
            "risk_id": "lateral_trunk_lean",
            "signal_strength": floor,
            "confidence": 0.0,
            "note": "COM drift insufficient",
        }

    drift = max(xs) - min(xs)
    signal = min(1.0, drift / 0.06)
    confidence = float(np.mean(vis)) * 0.7

    return {
        "risk_id": "lateral_trunk_lean",
        "signal_strength": round(max(signal, floor), 3),
        "confidence": round(confidence, 3),
        "debug": {"lateral_drift": round(drift, 4), "mode": "proxy"},
    }

# -----------------------------
# Visualisation (PNG export)
# -----------------------------
from app.workers.risk.visual_utils import extract_frame, save_visual

def _attach_visual(
    risk_obj,
    video,
    frame_idx,
    region="non_bowling_shoulder",
    direction="lateral_shear"
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
