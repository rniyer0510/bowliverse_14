from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

def compute_hip_shoulder_mismatch(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    rel_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    anchor = ffc_frame if isinstance(ffc_frame, int) else (rel_frame if isinstance(rel_frame, int) else None)
    if anchor is None or anchor < 0:
        return {"risk_id": "hip_shoulder_mismatch", "signal_strength": floor, "confidence": 0.0}

    hips = []
    shoulders = []
    vis = []

    for i in range(max(0, anchor - 6), min(len(pose_frames), anchor + 6)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue

        needed = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        if not all(k in lm for k in needed):
            continue

        try:
            hx = float(lm["RIGHT_HIP"]["x"]) - float(lm["LEFT_HIP"]["x"])
            sx = float(lm["RIGHT_SHOULDER"]["x"]) - float(lm["LEFT_SHOULDER"]["x"])
            v = min(
                float(lm["LEFT_HIP"].get("visibility", lm["LEFT_HIP"].get("v", 0.0))),
                float(lm["RIGHT_HIP"].get("visibility", lm["RIGHT_HIP"].get("v", 0.0))),
                float(lm["LEFT_SHOULDER"].get("visibility", lm["LEFT_SHOULDER"].get("v", 0.0))),
                float(lm["RIGHT_SHOULDER"].get("visibility", lm["RIGHT_SHOULDER"].get("v", 0.0))),
            )
        except Exception:
            continue

        hips.append(hx)
        shoulders.append(sx)
        vis.append(v)

    if len(hips) < 4:
        return {"risk_id": "hip_shoulder_mismatch", "signal_strength": floor, "confidence": 0.0}

    phase = float(np.mean(np.abs(np.array(hips, dtype=float) - np.array(shoulders, dtype=float))))
    signal = min(1.0, phase / float(config.get("phase_norm", 0.25)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.7, 3) if vis else 0.0

    return {
        "risk_id": "hip_shoulder_mismatch",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"phase_lag": round(phase, 3)},
    }
