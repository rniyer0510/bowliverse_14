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
