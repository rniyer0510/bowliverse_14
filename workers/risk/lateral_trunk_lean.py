from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

def compute_lateral_trunk_lean(
    pose_frames: List[Dict[str, Any]],
    bfc_frame: Optional[int],
    ffc_frame: Optional[int],
    rel_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    # Lean is lower-body to trunk stability: prefer BFC; else FFC; else Release
    anchor = bfc_frame if isinstance(bfc_frame, int) else (ffc_frame if isinstance(ffc_frame, int) else (rel_frame if isinstance(rel_frame, int) else None))
    if anchor is None or anchor < 0:
        return {"risk_id": "lateral_trunk_lean", "signal_strength": floor, "confidence": 0.0}

    xs = []
    vis = []

    for i in range(max(0, anchor - 6), min(len(pose_frames), anchor + 6)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, dict):
            continue
        if "LEFT_HIP" not in lm or "RIGHT_HIP" not in lm:
            continue

        try:
            x = (float(lm["LEFT_HIP"]["x"]) + float(lm["RIGHT_HIP"]["x"])) / 2.0
            v1 = float(lm["LEFT_HIP"].get("visibility", lm["LEFT_HIP"].get("v", 0.0)))
            v2 = float(lm["RIGHT_HIP"].get("visibility", lm["RIGHT_HIP"].get("v", 0.0)))
        except Exception:
            continue

        xs.append(x)
        vis.append(min(v1, v2))

    if len(xs) < 4:
        return {"risk_id": "lateral_trunk_lean", "signal_strength": floor, "confidence": 0.0}

    drift = float(max(xs) - min(xs))
    signal = min(1.0, drift / float(config.get("drift_norm", 0.06)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.7, 3) if vis else 0.0

    return {
        "risk_id": "lateral_trunk_lean",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {"lateral_drift": round(drift, 4)},
    }
