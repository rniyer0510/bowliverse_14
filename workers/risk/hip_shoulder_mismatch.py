from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger
logger = get_logger(__name__)

LEFT_HIP = 23
RIGHT_HIP = 24
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


def compute_hip_shoulder_mismatch(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    rel_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    anchor = rel_frame if isinstance(rel_frame, int) else (ffc_frame if isinstance(ffc_frame, int) else None)
    if anchor is None or anchor < 0:
        return {"risk_id": "hip_shoulder_mismatch", "signal_strength": floor, "confidence": 0.0}

    hips = []
    shoulders = []
    vis = []

    for i in range(max(0, anchor - 8), min(len(pose_frames), anchor + 4)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get("landmarks")
        if not isinstance(lm, (dict, list)):
            continue

        left_hip = _landmark(lm, "LEFT_HIP", LEFT_HIP)
        right_hip = _landmark(lm, "RIGHT_HIP", RIGHT_HIP)
        left_shoulder = _landmark(lm, "LEFT_SHOULDER", LEFT_SHOULDER)
        right_shoulder = _landmark(lm, "RIGHT_SHOULDER", RIGHT_SHOULDER)
        if None in (left_hip, right_hip, left_shoulder, right_shoulder):
            continue

        try:
            hx = float(right_hip["x"]) - float(left_hip["x"])
            sx = float(right_shoulder["x"]) - float(left_shoulder["x"])
            v = min(
                float(left_hip.get("visibility", left_hip.get("v", 0.0))),
                float(right_hip.get("visibility", right_hip.get("v", 0.0))),
                float(left_shoulder.get("visibility", left_shoulder.get("v", 0.0))),
                float(right_shoulder.get("visibility", right_shoulder.get("v", 0.0))),
            )
        except Exception:
            continue

        hips.append(hx)
        shoulders.append(sx)
        vis.append(v)

    if len(hips) < 4:
        return {"risk_id": "hip_shoulder_mismatch", "signal_strength": floor, "confidence": 0.0}

    hips_arr = np.array(hips, dtype=float)
    shoulders_arr = np.array(shoulders, dtype=float)

    phase = float(np.mean(np.abs(hips_arr - shoulders_arr)))
    signal = min(1.0, phase / float(config.get("phase_norm", 0.12)))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else 0.0

    hip_velocity = np.abs(np.diff(hips_arr))
    shoulder_velocity = np.abs(np.diff(shoulders_arr))
    hip_peak_idx = int(np.argmax(hip_velocity)) if len(hip_velocity) else 0
    shoulder_peak_idx = int(np.argmax(shoulder_velocity)) if len(shoulder_velocity) else 0
    hip_peak_frame = max(0, anchor - 8) + hip_peak_idx + 1
    shoulder_peak_frame = max(0, anchor - 8) + shoulder_peak_idx + 1
    sequence_delta_frames = int(shoulder_peak_frame - hip_peak_frame)

    if sequence_delta_frames <= -2:
        sequence_pattern = "shoulders_lead"
    elif sequence_delta_frames >= 2:
        sequence_pattern = "hips_lead"
    else:
        sequence_pattern = "in_sync"

    return {
        "risk_id": "hip_shoulder_mismatch",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {
            "phase_lag": round(phase, 3),
            "hip_peak_frame": int(hip_peak_frame),
            "shoulder_peak_frame": int(shoulder_peak_frame),
            "sequence_delta_frames": sequence_delta_frames,
            "sequence_pattern": sequence_pattern,
        },
    }
