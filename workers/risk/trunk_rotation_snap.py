from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

def compute_trunk_rotation_snap(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    uah_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    # Prefer FFC anchor if present; else fall back to UAH
    anchor = ffc_frame if isinstance(ffc_frame, int) else (uah_frame if isinstance(uah_frame, int) else None)
    if anchor is None or anchor < 0:
        return {"risk_id": "trunk_rotation_snap", "signal_strength": floor, "confidence": 0.0}

    angles = []
    vis = []

    for i in range(max(0, anchor - 6), min(len(pose_frames), anchor + 6)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue
        if "LEFT_SHOULDER" not in lm or "RIGHT_SHOULDER" not in lm:
            continue

        try:
            dx = float(lm["RIGHT_SHOULDER"]["x"]) - float(lm["LEFT_SHOULDER"]["x"])
            dy = float(lm["RIGHT_SHOULDER"]["y"]) - float(lm["LEFT_SHOULDER"]["y"])
            v1 = float(lm["LEFT_SHOULDER"].get("visibility", lm["LEFT_SHOULDER"].get("v", 0.0)))
            v2 = float(lm["RIGHT_SHOULDER"].get("visibility", lm["RIGHT_SHOULDER"].get("v", 0.0)))
        except Exception:
            continue

        angles.append(float(np.arctan2(dy, dx)))
        vis.append(min(v1, v2))

    if len(angles) < 5:
        return {"risk_id": "trunk_rotation_snap", "signal_strength": floor, "confidence": 0.0}

    jerk = float(np.max(np.abs(np.diff(np.diff(np.array(angles, dtype=float))))))
    signal = min(1.0, jerk / float(config.get("jerk_norm", 1.2)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.7, 3) if vis else 0.0

    return {
        "risk_id": "trunk_rotation_snap",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"rot_jerk": round(jerk, 3)},
    }
