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
