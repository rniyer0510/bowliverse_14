from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from app.common.logger import get_logger
logger = get_logger(__name__)

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
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


def _front_side_from_hand(hand: Optional[str]) -> Optional[str]:
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


def _leg_indices(side: str) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
    if side == 'RIGHT':
        return (
            ('RIGHT_HIP', RIGHT_HIP),
            ('RIGHT_KNEE', RIGHT_KNEE),
            ('RIGHT_ANKLE', RIGHT_ANKLE),
        )
    return (
        ('LEFT_HIP', LEFT_HIP),
        ('LEFT_KNEE', LEFT_KNEE),
        ('LEFT_ANKLE', LEFT_ANKLE),
    )


def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    nba = math.hypot(*ba)
    nbc = math.hypot(*bc)
    if nba <= 1e-6 or nbc <= 1e-6:
        return None
    cosang = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (nba * nbc)))
    return math.degrees(math.acos(cosang))


def compute_front_knee_brace_profile(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    fps: float,
    hand: Optional[str] = None,
    post_window: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if ffc_frame is None or not isinstance(ffc_frame, int) or ffc_frame < 0 or ffc_frame >= len(pose_frames):
        return None

    ref_frame = pose_frames[ffc_frame]
    ref_lms = ref_frame.get('landmarks') if isinstance(ref_frame, dict) else None
    if not isinstance(ref_lms, (dict, list)):
        return None

    front_side = _front_side_from_hand(hand) or _front_side_from_frame(ref_lms)
    hip_idx, knee_idx, ankle_idx = _leg_indices(front_side)
    window = int(post_window or max(6, int(round((fps or 30.0) * 0.16))))

    samples = []
    for i in range(max(0, ffc_frame - 1), min(len(pose_frames), ffc_frame + window + 1)):
        frame = pose_frames[i]
        if not isinstance(frame, dict):
            continue
        lms = frame.get('landmarks')
        if not isinstance(lms, (dict, list)):
            continue
        hip = _point(lms, hip_idx[0], hip_idx[1])
        knee = _point(lms, knee_idx[0], knee_idx[1])
        ankle = _point(lms, ankle_idx[0], ankle_idx[1])
        if hip is None or knee is None or ankle is None:
            continue
        angle = _angle((hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1]))
        if angle is None:
            continue
        samples.append({
            'frame': i,
            'angle': angle,
            'visibility': min(hip[2], knee[2], ankle[2]),
        })

    if len(samples) < 4:
        return None

    contact_samples = [sample for sample in samples if sample['frame'] <= ffc_frame + 1]
    post_samples = [sample for sample in samples if sample['frame'] >= ffc_frame]
    if not contact_samples or len(post_samples) < 3:
        return None

    angle_ffc = float(np.median([sample['angle'] for sample in contact_samples]))
    min_post_angle = min(sample['angle'] for sample in post_samples)
    release_angle = float(np.median([sample['angle'] for sample in post_samples[-2:]]))
    collapse_deg = max(0.0, angle_ffc - min_post_angle)
    support_deficit_deg = max(0.0, 150.0 - release_angle)
    mean_visibility = float(np.mean([sample['visibility'] for sample in post_samples]))

    return {
        'front_side': front_side.lower(),
        'angle_ffc': angle_ffc,
        'min_post_angle': float(min_post_angle),
        'release_angle': release_angle,
        'collapse_deg': collapse_deg,
        'support_deficit_deg': support_deficit_deg,
        'mean_visibility': mean_visibility,
        'sample_count': len(post_samples),
    }


def compute_knee_brace_failure(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get('floor', 0.15))
    profile = compute_front_knee_brace_profile(
        pose_frames=pose_frames,
        ffc_frame=ffc_frame,
        fps=fps,
        hand=config.get('hand'),
        post_window=config.get('post_window'),
    )
    if profile is None:
        return {'risk_id': 'knee_brace_failure', 'signal_strength': floor, 'confidence': 0.0}

    collapse_signal = min(1.0, profile['collapse_deg'] / float(config.get('collapse_threshold_deg', 18.0)))
    support_signal = min(1.0, profile['support_deficit_deg'] / float(config.get('support_threshold_deg', 30.0)))
    signal_raw = (collapse_signal * 0.65) + (support_signal * 0.35)
    signal = round(max(floor, min(1.0, signal_raw)), 3)
    confidence = round(min(1.0, profile['mean_visibility'] * min(1.0, profile['sample_count'] / 5.0)), 3)

    return {
        'risk_id': 'knee_brace_failure',
        'signal_strength': signal,
        'confidence': confidence,
        'debug': {
            'front_side': profile['front_side'],
            'angle_ffc': round(profile['angle_ffc'], 2),
            'min_post_angle': round(profile['min_post_angle'], 2),
            'release_angle': round(profile['release_angle'], 2),
            'collapse_deg': round(profile['collapse_deg'], 2),
            'support_deficit_deg': round(profile['support_deficit_deg'], 2),
            'sample_count': profile['sample_count'],
        },
    }
