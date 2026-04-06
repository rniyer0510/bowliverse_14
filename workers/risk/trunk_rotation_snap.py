from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _landmark(lms: Any, key: str, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(lms, dict):
        val = lms.get(key)
        return val if isinstance(val, dict) else None
    if isinstance(lms, list) and 0 <= idx < len(lms):
        val = lms[idx]
        return val if isinstance(val, dict) else None
    return None


def compute_trunk_rotation_snap(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    uah_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    anchor = uah_frame if isinstance(uah_frame, int) else (ffc_frame if isinstance(ffc_frame, int) else None)
    if anchor is None or anchor < 0:
        return {"risk_id": "trunk_rotation_snap", "signal_strength": floor, "confidence": 0.0}

    angles = []
    vis = []

    for i in range(max(0, anchor - 7), min(len(pose_frames), anchor + 5)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, (dict, list)):
            continue
        left_shoulder = _landmark(lm, "LEFT_SHOULDER", LEFT_SHOULDER)
        right_shoulder = _landmark(lm, "RIGHT_SHOULDER", RIGHT_SHOULDER)
        if left_shoulder is None or right_shoulder is None:
            continue

        try:
            dx = float(right_shoulder["x"]) - float(left_shoulder["x"])
            dy = float(right_shoulder["y"]) - float(left_shoulder["y"])
            v1 = float(left_shoulder.get("visibility", left_shoulder.get("v", 0.0)))
            v2 = float(right_shoulder.get("visibility", right_shoulder.get("v", 0.0)))
        except Exception:
            continue

        angles.append(float(np.arctan2(dy, dx)))
        vis.append(min(v1, v2))

    if len(angles) < 5:
        return {"risk_id": "trunk_rotation_snap", "signal_strength": floor, "confidence": 0.0}

    jerk = float(np.max(np.abs(np.diff(np.diff(np.array(angles, dtype=float))))))
    signal = min(1.0, jerk / float(config.get("jerk_norm", 0.55)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else 0.0

    return {
        "risk_id": "trunk_rotation_snap",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"rot_jerk": round(jerk, 3)},
    }
