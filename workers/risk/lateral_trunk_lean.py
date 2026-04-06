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


def compute_lateral_trunk_lean(
    pose_frames: List[Dict[str, Any]],
    bfc_frame: Optional[int],
    ffc_frame: Optional[int],
    rel_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    anchor = rel_frame if isinstance(rel_frame, int) else (
        ffc_frame if isinstance(ffc_frame, int) else (
            bfc_frame if isinstance(bfc_frame, int) else None
        )
    )
    if anchor is None or anchor < 0:
        return {"risk_id": "lateral_trunk_lean", "signal_strength": floor, "confidence": 0.0}

    xs = []
    vis = []

    for i in range(max(0, anchor - 6), min(len(pose_frames), anchor + 6)):
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
            x = (float(left_hip["x"]) + float(right_hip["x"])) / 2.0
            v1 = float(left_hip.get("visibility", left_hip.get("v", 0.0)))
            v2 = float(right_hip.get("visibility", right_hip.get("v", 0.0)))
        except Exception:
            continue

        xs.append(x)
        vis.append(min(v1, v2))

    if len(xs) < 4:
        return {"risk_id": "lateral_trunk_lean", "signal_strength": floor, "confidence": 0.0}

    drift = float(max(xs) - min(xs))
    signal = min(1.0, drift / float(config.get("drift_norm", 0.03)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else 0.0

    return {
        "risk_id": "lateral_trunk_lean",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"lateral_drift": round(drift, 4)},
    }
