from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _landmark(lms: Any, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(lms, list) and 0 <= idx < len(lms):
        value = lms[idx]
        return value if isinstance(value, dict) else None
    if isinstance(lms, dict):
        values = list(lms.values())
        if 0 <= idx < len(values):
            value = values[idx]
            return value if isinstance(value, dict) else None
    return None


def compute_neck_tilt_left_bfc(
    pose_frames: List[Dict[str, Any]],
    bfc_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    del fps
    floor = float(config.get("floor", 0.15))
    if not isinstance(bfc_frame, int) or bfc_frame < 0 or bfc_frame >= len(pose_frames):
        return {"risk_id": "neck_tilt_left_bfc", "signal_strength": floor, "confidence": 0.0}

    offsets: List[float] = []
    visibilities: List[float] = []
    for frame_idx in range(max(0, bfc_frame - 1), min(len(pose_frames), bfc_frame + 2)):
        frame = pose_frames[frame_idx]
        if not isinstance(frame, dict):
            continue
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, (list, dict)):
            continue

        nose = _landmark(landmarks, NOSE)
        left_shoulder = _landmark(landmarks, LEFT_SHOULDER)
        right_shoulder = _landmark(landmarks, RIGHT_SHOULDER)
        if None in (nose, left_shoulder, right_shoulder):
            continue

        try:
            shoulder_mid_x = (float(left_shoulder["x"]) + float(right_shoulder["x"])) / 2.0
            shoulder_width = abs(float(right_shoulder["x"]) - float(left_shoulder["x"]))
            nose_x = float(nose["x"])
            visibility = min(
                float(nose.get("visibility", nose.get("v", 0.0))),
                float(left_shoulder.get("visibility", left_shoulder.get("v", 0.0))),
                float(right_shoulder.get("visibility", right_shoulder.get("v", 0.0))),
            )
        except Exception:
            continue

        if shoulder_width <= 1e-6:
            continue

        left_offset = (shoulder_mid_x - nose_x) / shoulder_width
        offsets.append(left_offset)
        visibilities.append(visibility)

    if not offsets:
        return {"risk_id": "neck_tilt_left_bfc", "signal_strength": floor, "confidence": 0.0}

    median_offset = float(np.median(offsets))
    normalized = max(0.0, (median_offset - 0.03) / float(config.get("offset_norm", 0.22)))
    signal = round(max(floor, min(1.0, normalized)), 3)
    confidence = round(float(np.mean(visibilities)) * 0.82, 3) if visibilities else 0.0
    return {
        "risk_id": "neck_tilt_left_bfc",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {
            "median_left_offset": round(median_offset, 4),
            "sample_count": len(offsets),
        },
    }
