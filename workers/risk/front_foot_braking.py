from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from app.common.logger import get_logger
logger = get_logger(__name__)

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


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


def _front_side_from_action(action: Dict[str, Any]) -> Optional[str]:
    if not isinstance(action, dict):
        return None
    hand = action.get('hand') or action.get('bowling_hand') or action.get('input_hand')
    if not isinstance(hand, str):
        return None
    hand = hand.strip().upper()
    if hand == 'R':
        return 'LEFT'
    if hand == 'L':
        return 'RIGHT'
    return None


def _front_side_from_frame(lms: Any) -> str:
    left_points = [
        _point(lms, 'LEFT_ANKLE', LEFT_ANKLE),
        _point(lms, 'LEFT_HEEL', LEFT_HEEL),
        _point(lms, 'LEFT_FOOT_INDEX', LEFT_FOOT_INDEX),
    ]
    right_points = [
        _point(lms, 'RIGHT_ANKLE', RIGHT_ANKLE),
        _point(lms, 'RIGHT_HEEL', RIGHT_HEEL),
        _point(lms, 'RIGHT_FOOT_INDEX', RIGHT_FOOT_INDEX),
    ]
    left_y = [pt[1] for pt in left_points if pt is not None]
    right_y = [pt[1] for pt in right_points if pt is not None]
    if left_y and right_y:
        return 'LEFT' if float(np.mean(left_y)) >= float(np.mean(right_y)) else 'RIGHT'
    return 'LEFT'


def _foot_indices(side: str) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
    if side == 'RIGHT':
        return (
            ('RIGHT_ANKLE', RIGHT_ANKLE),
            ('RIGHT_HEEL', RIGHT_HEEL),
            ('RIGHT_FOOT_INDEX', RIGHT_FOOT_INDEX),
        )
    return (
        ('LEFT_ANKLE', LEFT_ANKLE),
        ('LEFT_HEEL', LEFT_HEEL),
        ('LEFT_FOOT_INDEX', LEFT_FOOT_INDEX),
    )


def compute_front_foot_braking_shock(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get('floor', 0.15))

    if ffc_frame is None or not isinstance(ffc_frame, int) or ffc_frame < 0 or ffc_frame >= len(pose_frames):
        return {'risk_id': 'front_foot_braking_shock', 'signal_strength': floor, 'confidence': 0.0}

    ref_frame = pose_frames[ffc_frame]
    ref_lms = ref_frame.get('landmarks') if isinstance(ref_frame, dict) else None
    if not isinstance(ref_lms, (dict, list)):
        return {'risk_id': 'front_foot_braking_shock', 'signal_strength': floor, 'confidence': 0.0}

    front_side = _front_side_from_action(action) or _front_side_from_frame(ref_lms)
    ankle_idx, heel_idx, toe_idx = _foot_indices(front_side)

    left_hip = _point(ref_lms, 'LEFT_HIP', LEFT_HIP)
    right_hip = _point(ref_lms, 'RIGHT_HIP', RIGHT_HIP)
    heel_ref = _point(ref_lms, heel_idx[0], heel_idx[1])
    toe_ref = _point(ref_lms, toe_idx[0], toe_idx[1])
    if left_hip is None or right_hip is None or heel_ref is None or toe_ref is None:
        return {'risk_id': 'front_foot_braking_shock', 'signal_strength': floor, 'confidence': 0.0}

    hip_width = abs(right_hip[0] - left_hip[0])
    foot_vector_ref = (toe_ref[0] - heel_ref[0], toe_ref[1] - heel_ref[1])
    foot_length = math.hypot(*foot_vector_ref)
    if hip_width <= 1e-6 or foot_length <= 1e-6:
        return {'risk_id': 'front_foot_braking_shock', 'signal_strength': floor, 'confidence': 0.0}

    foot_mid_ref = ((heel_ref[0] + toe_ref[0]) * 0.5, (heel_ref[1] + toe_ref[1]) * 0.5)

    post = int(config.get('post_window', max(4, int(round((fps or 30.0) * 0.08)))))
    movement_threshold = float(config.get('movement_threshold', 0.18))
    shape_tolerance = float(config.get('shape_tolerance', 1.0))

    coherent_movements = []
    coherences = []
    visibilities = []
    shape_errors = []

    for i in range(ffc_frame + 1, min(len(pose_frames), ffc_frame + post + 1)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lms = frame.get('landmarks')
        if not isinstance(lms, (dict, list)):
            continue

        heel = _point(lms, heel_idx[0], heel_idx[1])
        toe = _point(lms, toe_idx[0], toe_idx[1])
        if heel is None or toe is None:
            continue

        foot_mid = ((heel[0] + toe[0]) * 0.5, (heel[1] + toe[1]) * 0.5)
        movement = math.hypot(foot_mid[0] - foot_mid_ref[0], foot_mid[1] - foot_mid_ref[1]) / hip_width

        foot_vector = (toe[0] - heel[0], toe[1] - heel[1])
        shape_error = math.hypot(foot_vector[0] - foot_vector_ref[0], foot_vector[1] - foot_vector_ref[1]) / foot_length
        coherence = max(0.0, 1.0 - (shape_error / max(shape_tolerance, 1e-6))) ** 2
        visibility = min(heel[2], toe[2])

        coherent_movements.append(movement * coherence)
        coherences.append(coherence)
        visibilities.append(visibility)
        shape_errors.append(shape_error)

    if len(coherent_movements) < 2:
        return {'risk_id': 'front_foot_braking_shock', 'signal_strength': floor, 'confidence': 0.0}

    coherent_motion = float(np.percentile(coherent_movements, 75))
    mean_coherence = float(np.mean(coherences)) if coherences else 0.0
    mean_visibility = float(np.mean(visibilities)) if visibilities else 0.0
    representative_shape = float(np.percentile(shape_errors, 50)) if shape_errors else 0.0

    signal_raw = (coherent_motion / max(movement_threshold, 1e-6)) * mean_visibility * max(0.25, mean_coherence)
    signal = round(max(floor, min(1.0, signal_raw)), 3)
    confidence = round(min(1.0, mean_visibility * (0.45 + 0.55 * mean_coherence)), 3)

    return {
        'risk_id': 'front_foot_braking_shock',
        'signal_strength': signal,
        'confidence': confidence,
        'debug': {
            'front_side': front_side.lower(),
            'coherent_motion': round(coherent_motion, 4),
            'mean_coherence': round(mean_coherence, 4),
            'mean_visibility': round(mean_visibility, 4),
            'median_shape_error': round(representative_shape, 4),
            'sample_count': len(coherent_movements),
        },
    }
