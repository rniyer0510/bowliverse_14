from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

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


def _point(lms: Any, key: str, idx: int) -> Optional[Tuple[float, float, float]]:
    pt = _landmark(lms, key, idx)
    if pt is None:
        return None
    try:
        return (
            float(pt['x']),
            float(pt['y']),
            float(pt.get('visibility', pt.get('v', 0.0))),
        )
    except Exception:
        return None


def _side_from_hand(hand: Optional[str]) -> Optional[str]:
    if not isinstance(hand, str):
        return None
    hand = hand.strip().upper()
    if hand == 'R':
        return 'RIGHT'
    if hand == 'L':
        return 'LEFT'
    return None


def _trunk_angle_deg(shoulder: Tuple[float, float, float], hip: Tuple[float, float, float]) -> float:
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    return math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))


def compute_lateral_trunk_lean(
    pose_frames: List[Dict[str, Any]],
    bfc_frame: Optional[int],
    ffc_frame: Optional[int],
    rel_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    del fps
    floor = float(config.get('floor', 0.15))

    anchor = rel_frame if isinstance(rel_frame, int) else (
        ffc_frame if isinstance(ffc_frame, int) else (
            bfc_frame if isinstance(bfc_frame, int) else None
        )
    )
    if anchor is None or anchor < 0:
        return {'risk_id': 'lateral_trunk_lean', 'signal_strength': floor, 'confidence': 0.0}

    side = _side_from_hand(config.get('hand'))
    if side == 'LEFT':
        shoulder_idx = ('LEFT_SHOULDER', LEFT_SHOULDER)
        hip_idx = ('LEFT_HIP', LEFT_HIP)
    else:
        shoulder_idx = ('RIGHT_SHOULDER', RIGHT_SHOULDER)
        hip_idx = ('RIGHT_HIP', RIGHT_HIP)

    angles = []
    vis = []
    for i in range(max(0, anchor - 6), min(len(pose_frames), anchor + 1)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lms = frame.get('landmarks')
        if not isinstance(lms, (dict, list)):
            continue
        shoulder = _point(lms, shoulder_idx[0], shoulder_idx[1])
        hip = _point(lms, hip_idx[0], hip_idx[1])
        if shoulder is None or hip is None:
            continue
        angles.append(_trunk_angle_deg(shoulder, hip))
        vis.append(min(shoulder[2], hip[2]))

    if len(angles) < 4:
        return {'risk_id': 'lateral_trunk_lean', 'signal_strength': floor, 'confidence': 0.0}

    late_window = np.asarray(angles[-4:], dtype=float)
    late_angle = float(np.percentile(late_window, 75))
    low = float(config.get('lean_low_deg', 24.0))
    high = float(config.get('lean_high_deg', 48.0))
    signal_raw = (late_angle - low) / max(high - low, 1e-6)
    signal = round(max(floor, min(1.0, signal_raw)), 3)
    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else 0.0

    return {
        'risk_id': 'lateral_trunk_lean',
        'signal_strength': signal,
        'confidence': confidence,
        'debug': {
            'side': (side or 'RIGHT').lower(),
            'late_angle_deg': round(late_angle, 2),
            'sample_count': len(angles),
        },
    }
