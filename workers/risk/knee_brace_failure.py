from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

LEFT_HIP = 23
RIGHT_HIP = 24


def _landmark(lms: Any, key: str, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(lms, dict):
        val = lms.get(key)
        return val if isinstance(val, dict) else None
    if isinstance(lms, list) and 0 <= idx < len(lms):
        val = lms[idx]
        return val if isinstance(val, dict) else None
    return None


def compute_knee_brace_failure(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    if ffc_frame is None or not isinstance(ffc_frame, int) or ffc_frame < 0:
        return {"risk_id": "knee_brace_failure", "signal_strength": floor, "confidence": 0.0}

    pelvis_y = []
    vis = []

    for i in range(max(0, ffc_frame - 5), min(len(pose_frames), ffc_frame + 8)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, (dict, list)):
            continue
        left_hip = _landmark(lm, "LEFT_HIP", LEFT_HIP)
        right_hip = _landmark(lm, "RIGHT_HIP", RIGHT_HIP)
        if left_hip is None or right_hip is None:
            continue

        try:
            y = (float(left_hip["y"]) + float(right_hip["y"])) / 2.0
            v1 = float(left_hip.get("visibility", left_hip.get("v", 0.0)))
            v2 = float(right_hip.get("visibility", right_hip.get("v", 0.0)))
        except Exception:
            continue

        pelvis_y.append(y)
        vis.append(min(v1, v2))

    if len(pelvis_y) < 4:
        return {"risk_id": "knee_brace_failure", "signal_strength": floor, "confidence": 0.0}

    drop = float(max(pelvis_y) - min(pelvis_y))
    signal = min(1.0, drop / float(config.get("drop_threshold", 0.025)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else 0.0

    return {
        "risk_id": "knee_brace_failure",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"pelvis_drop": round(drop, 4)},
    }
