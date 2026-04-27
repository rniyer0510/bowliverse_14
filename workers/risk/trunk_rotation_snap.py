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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_trunk_rotation_snap(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    uah_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get('floor', 0.15))

    anchor = uah_frame if isinstance(uah_frame, int) else (ffc_frame if isinstance(ffc_frame, int) else None)
    if anchor is None or anchor < 0:
        return {'risk_id': 'trunk_rotation_snap', 'signal_strength': floor, 'confidence': 0.0}

    angles = []
    vis = []
    for i in range(max(0, anchor - 7), min(len(pose_frames), anchor + 2)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lm = frame.get('landmarks')
        if not isinstance(lm, (dict, list)):
            continue
        left_shoulder = _landmark(lm, 'LEFT_SHOULDER', LEFT_SHOULDER)
        right_shoulder = _landmark(lm, 'RIGHT_SHOULDER', RIGHT_SHOULDER)
        if left_shoulder is None or right_shoulder is None:
            continue
        try:
            dx = float(right_shoulder['x']) - float(left_shoulder['x'])
            dy = float(right_shoulder['y']) - float(left_shoulder['y'])
            v1 = float(left_shoulder.get('visibility', left_shoulder.get('v', 0.0)))
            v2 = float(right_shoulder.get('visibility', right_shoulder.get('v', 0.0)))
        except Exception:
            continue
        angles.append(float(np.arctan2(dy, dx)))
        vis.append(min(v1, v2))

    if len(angles) < 5:
        return {'risk_id': 'trunk_rotation_snap', 'signal_strength': floor, 'confidence': 0.0}

    arr = np.unwrap(np.asarray(angles, dtype=float))
    kernel = np.asarray(config.get('smoothing_kernel', [0.25, 0.5, 0.25]), dtype=float)
    kernel = kernel / np.sum(kernel)
    if len(arr) >= len(kernel):
        pad = len(kernel) // 2
        smooth = np.convolve(np.pad(arr, (pad, pad), mode='edge'), kernel, mode='valid')
    else:
        smooth = arr

    velocity = np.diff(smooth)
    accel = np.diff(velocity)
    if len(accel) == 0 or len(velocity) == 0:
        return {'risk_id': 'trunk_rotation_snap', 'signal_strength': floor, 'confidence': 0.0}

    core_accel = accel[1:-1] if len(accel) > 2 else accel
    peak_accel = float(np.max(np.abs(core_accel)))
    peak_velocity = float(np.percentile(np.abs(velocity), 75))
    snap_index = peak_accel / max(peak_velocity, float(config.get('min_velocity_window', 0.05)))
    rotation_range_deg = float(np.degrees(np.max(smooth) - np.min(smooth)))

    peak_acc_idx = int(np.argmax(np.abs(core_accel))) if len(core_accel) else 0
    acc_offset = 1 if len(accel) > 2 else 0
    timing_index = peak_acc_idx + acc_offset
    timing_ratio = (timing_index / max(1, len(accel) - 1)) if len(accel) > 1 else 0.0

    abruptness_norm = _clip01((snap_index - float(config.get('snap_low', 0.55))) / max(1e-6, float(config.get('snap_high', 1.35)) - float(config.get('snap_low', 0.55))))
    range_gate = _clip01(rotation_range_deg / float(config.get('rotation_range_gate_deg', 35.0)))
    late_gate = _clip01((timing_ratio - float(config.get('late_timing_start', 0.55))) / max(1e-6, 1.0 - float(config.get('late_timing_start', 0.55))))

    raw_signal = abruptness_norm * max(0.35, range_gate) * max(0.25, late_gate)
    signal = round(max(floor, min(1.0, floor + ((1.0 - floor) * raw_signal))), 3)
    confidence = round(float(np.mean(vis)) * 0.85, 3) if vis else 0.0

    return {
        'risk_id': 'trunk_rotation_snap',
        'signal_strength': signal,
        'confidence': confidence,
        'debug': {
            'peak_accel': round(peak_accel, 3),
            'peak_velocity': round(peak_velocity, 3),
            'snap_index': round(snap_index, 3),
            'rotation_range_deg': round(rotation_range_deg, 3),
            'timing_ratio': round(float(timing_ratio), 3),
        },
    }
