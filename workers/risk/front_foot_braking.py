# app/workers/risk/front_foot_braking.py

import numpy as np
from app.common.logger import get_logger

logger = get_logger(__name__)


def compute_front_foot_braking_shock(
    pose_frames,
    ffc_frame,
    fps,
    config,
    action,
):
    if ffc_frame is None or ffc_frame < 0:
        return {
            "risk_id": "front_foot_braking_shock",
            "signal_strength": 0.0,
            "confidence": 0.0,
            "note": "FFC not available",
        }

    pre = int(config.get("pre_window", 6))
    post = int(config.get("post_window", 4))

    ys = []

    for i in range(max(0, ffc_frame - pre), min(len(pose_frames), ffc_frame + post)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue

        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue

        if "LEFT_ANKLE" not in lm or "RIGHT_ANKLE" not in lm:
            continue

        ys.append(min(lm["LEFT_ANKLE"]["y"], lm["RIGHT_ANKLE"]["y"]))

    if len(ys) < 5:
        return {
            "risk_id": "front_foot_braking_shock",
            "signal_strength": 0.0,
            "confidence": 0.0,
            "note": "Insufficient ankle samples",
        }

    travel = np.percentile(ys, 95) - np.percentile(ys, 5)
    signal = min(1.0, travel / float(config.get("travel_min", 0.012)))

    return {
        "risk_id": "front_foot_braking_shock",
        "signal_strength": round(signal, 3),
        "confidence": 1.0,
        "debug": {"travel": round(float(travel), 4)},
    }

# -----------------------------
# Visualisation (PNG export)
# -----------------------------
from app.workers.risk.visual_utils import extract_frame, save_visual

def _attach_visual(
    risk_obj,
    video,
    frame_idx,
    region="ankle_chain",
    direction="vertical_braking"
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
