import math
from statistics import median
from typing import Any, Dict, Iterable, Optional

from .geometry import angle_between_deg, point, stable_angles

LS, LE, LW, LH, LA = 11, 13, 15, 23, 27
RS, RE, RW, RH, RA = 12, 14, 16, 24, 28
ROUND_ARM_THRESHOLD_DEG = 60.0
ANGLE_MARGIN_TRUST_DEG, CHAIN_QUALITY_FOR_ARC = 8.0, 0.4
RELEASE_CONFIDENCE_FOR_ARC = 0.6


def _normalized_hand(hand: Optional[str]) -> Optional[str]:
    text = str(hand or "").strip().lower()
    return text or None


def _event_chain_quality(events: Optional[Dict[str, Any]]) -> float:
    value = ((events or {}).get("event_chain") or {}).get("quality")
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return 0.0


def _geometry_trust(
    *,
    angle_deg: float,
    coverage: float,
    visibility: float,
    release_confidence: float,
    event_chain_quality: float,
    stable_angle_count: int,
    wrist_height_count: int,
) -> Dict[str, bool]:
    clear_category_margin = abs(angle_deg - ROUND_ARM_THRESHOLD_DEG) >= ANGLE_MARGIN_TRUST_DEG
    return {
        "category": coverage >= 0.6 and visibility >= 0.7 and clear_category_margin,
        "angle_deg": coverage >= 0.6 and visibility >= 0.65,
        "arc_deg": stable_angle_count >= 3 and release_confidence >= RELEASE_CONFIDENCE_FOR_ARC and event_chain_quality >= CHAIN_QUALITY_FOR_ARC,
        "height_ratio": wrist_height_count >= 2 and visibility >= 0.7 and release_confidence >= 0.55,
    }


def compute_release_shape_metrics(
    *,
    pose_frames: Optional[Iterable[Dict[str, Any]]],
    events: Optional[Dict[str, Any]],
    hand: Optional[str],
) -> Dict[str, Any]:
    frames = list(pose_frames) if pose_frames is not None else []
    release = (events or {}).get("release") or {}
    release_frame = release.get("frame")
    release_confidence = float(release.get("confidence") or 0.0)
    event_chain_quality = _event_chain_quality(events)
    if not isinstance(release_frame, (int, float)):
        return {"reason": "release_frame_missing"}
    release_index = int(release_frame)
    if release_index < 0 or release_index >= len(frames):
        return {"reason": "release_frame_out_of_range"}
    is_left = _normalized_hand(hand) == "left"
    shoulder_idx, elbow_idx, wrist_idx, hip_idx, ankle_idx = (LS, LE, LW, LH, LA) if is_left else (RS, RE, RW, RH, RA)
    start = max(0, release_index - 2)
    end = min(len(frames) - 1, release_index + 2)
    arm_angles = []
    wrist_heights = []
    visibilities = []
    for frame_idx in range(start, end + 1):
        landmarks = (frames[frame_idx] or {}).get("landmarks")
        shoulder = point(landmarks, shoulder_idx)
        elbow = point(landmarks, elbow_idx)
        wrist = point(landmarks, wrist_idx)
        hip = point(landmarks, hip_idx)
        ankle = point(landmarks, ankle_idx)
        if shoulder and elbow and wrist and hip:
            torso = (shoulder[0] - hip[0], shoulder[1] - hip[1])
            upper_arm = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
            forearm = (wrist[0] - elbow[0], wrist[1] - elbow[1])
            upper_arm_angle = angle_between_deg(upper_arm, torso)
            forearm_angle = angle_between_deg(forearm, torso)
            arm_angle = (
                (upper_arm_angle * 0.7) + (forearm_angle * 0.3)
                if upper_arm_angle is not None and forearm_angle is not None
                else upper_arm_angle if upper_arm_angle is not None else forearm_angle
            )
            if arm_angle is not None:
                arm_angles.append(arm_angle)
            visibilities.append(min(shoulder[2], elbow[2], wrist[2], hip[2]))
        if shoulder and wrist and ankle and ankle[1] > shoulder[1]:
            wrist_heights.append((ankle[1] - wrist[1]) / (ankle[1] - shoulder[1]))
    if not arm_angles:
        return {
            "release_frame": release_index,
            "release_confidence": round(release_confidence, 2),
            "event_chain_quality": event_chain_quality,
            "reason": "insufficient_release_side_landmarks",
        }
    stable = stable_angles(arm_angles)
    angle_deg = round(float(median(arm_angles)), 1)
    coverage = len(arm_angles) / max(1, end - start + 1)
    visibility = float(median(visibilities)) if visibilities else 0.0
    trusted_fields = _geometry_trust(
        angle_deg=angle_deg,
        coverage=coverage,
        visibility=visibility,
        release_confidence=release_confidence,
        event_chain_quality=event_chain_quality,
        stable_angle_count=len(stable),
        wrist_height_count=len(wrist_heights),
    )
    return {
        "release_frame": release_index,
        "release_confidence": round(release_confidence, 2),
        "event_chain_quality": event_chain_quality,
        "category_key": "round_arm" if angle_deg >= ROUND_ARM_THRESHOLD_DEG else "standard",
        "angle_deg": angle_deg,
        "arc_deg": round(float(max(stable) - min(stable)), 1),
        "height_ratio": round(float(median(wrist_heights)), 3) if wrist_heights else None,
        "trusted_fields": trusted_fields,
        "confidence": round((release_confidence + visibility + coverage + event_chain_quality) / 4, 2),
        "reason": "computed_from_torso_relative_release_geometry",
    }
