from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

def compute_front_foot_braking_shock(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    if ffc_frame is None or not isinstance(ffc_frame, int) or ffc_frame < 0:
        return {"risk_id": "front_foot_braking_shock", "signal_strength": floor, "confidence": 0.0}

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
        try:
            ys.append(min(float(lm["LEFT_ANKLE"]["y"]), float(lm["RIGHT_ANKLE"]["y"])))
        except Exception:
            continue

    if len(ys) < 5:
        return {"risk_id": "front_foot_braking_shock", "signal_strength": floor, "confidence": 0.0}

    travel = float(np.percentile(ys, 95) - np.percentile(ys, 5))
    signal = min(1.0, travel / float(config.get("travel_min", 0.012)))
    signal = round(max(signal, floor), 3)

    # Confidence: proxy for sample support
    confidence = round(min(1.0, len(ys) / 10.0), 3)

    return {
        "risk_id": "front_foot_braking_shock",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"travel": round(travel, 4)},
    }
