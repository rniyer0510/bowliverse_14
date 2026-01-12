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
