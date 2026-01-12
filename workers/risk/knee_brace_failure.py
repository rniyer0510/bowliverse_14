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
