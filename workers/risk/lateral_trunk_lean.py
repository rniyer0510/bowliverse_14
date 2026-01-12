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
