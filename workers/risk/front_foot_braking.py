from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _landmark(lms: Any, key: str, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(lms, dict):
        val = lms.get(key)
        return val if isinstance(val, dict) else None
    if isinstance(lms, list) and 0 <= idx < len(lms):
        val = lms[idx]
        return val if isinstance(val, dict) else None
    return None


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
    vis = []
    for i in range(max(0, ffc_frame - pre), min(len(pose_frames), ffc_frame + post)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, (dict, list)):
            continue
        left_ankle = _landmark(lm, "LEFT_ANKLE", LEFT_ANKLE)
        right_ankle = _landmark(lm, "RIGHT_ANKLE", RIGHT_ANKLE)
        if left_ankle is None or right_ankle is None:
            continue
        try:
            ys.append(min(float(left_ankle["y"]), float(right_ankle["y"])))
            vis.append(min(
                float(left_ankle.get("visibility", left_ankle.get("v", 0.0))),
                float(right_ankle.get("visibility", right_ankle.get("v", 0.0))),
            ))
        except Exception:
            continue

    if len(ys) < 5:
        return {"risk_id": "front_foot_braking_shock", "signal_strength": floor, "confidence": 0.0}

    travel = float(np.percentile(ys, 95) - np.percentile(ys, 5))
    signal = min(1.0, travel / float(config.get("travel_min", 0.010)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else round(min(1.0, len(ys) / 10.0), 3)

    return {
        "risk_id": "front_foot_braking_shock",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"travel": round(travel, 4)},
    }
