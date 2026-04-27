from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

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


def _segment_angle_deg(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    dx = float(right["x"]) - float(left["x"])
    dy = float(right["y"]) - float(left["y"])
    return math.degrees(math.atan2(dy, dx))


def _unwrap_deg(values: List[float]) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.array(values, dtype=float))))


def _wrapped_delta_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_hip_shoulder_mismatch(
    pose_frames: List[Dict[str, Any]],
    ffc_frame: Optional[int],
    rel_frame: Optional[int],
    fps: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    floor = float(config.get("floor", 0.15))

    if isinstance(ffc_frame, int):
        start = max(0, int(ffc_frame) - 4)
    elif isinstance(rel_frame, int):
        start = max(0, int(rel_frame) - 8)
    else:
        start = 0

    if isinstance(rel_frame, int):
        end = min(len(pose_frames), int(rel_frame) + 2)
    elif isinstance(ffc_frame, int):
        end = min(len(pose_frames), int(ffc_frame) + 8)
    else:
        return {"risk_id": "hip_shoulder_mismatch", "signal_strength": floor, "confidence": 0.0}

    shoulder_angles: List[float] = []
    hip_angles: List[float] = []
    vis: List[float] = []
    frame_ids: List[int] = []

    for i in range(start, end):
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
            shoulder_angle = _segment_angle_deg(left_shoulder, right_shoulder)
            hip_angle = _segment_angle_deg(left_hip, right_hip)
            v = min(
                float(left_hip.get("visibility", left_hip.get("v", 0.0))),
                float(right_hip.get("visibility", right_hip.get("v", 0.0))),
                float(left_shoulder.get("visibility", left_shoulder.get("v", 0.0))),
                float(right_shoulder.get("visibility", right_shoulder.get("v", 0.0))),
            )
        except Exception:
            continue

        shoulder_angles.append(shoulder_angle)
        hip_angles.append(hip_angle)
        vis.append(v)
        frame_ids.append(i)

    if len(shoulder_angles) < 4:
        return {"risk_id": "hip_shoulder_mismatch", "signal_strength": floor, "confidence": 0.0}

    shoulder_arr = _unwrap_deg(shoulder_angles)
    hip_arr = _unwrap_deg(hip_angles)
    relative_phase = np.abs(np.array([_wrapped_delta_deg(s, h) for s, h in zip(shoulder_angles, hip_angles)], dtype=float))

    shoulder_vel = np.diff(shoulder_arr)
    hip_vel = np.diff(hip_arr)
    abs_shoulder_vel = np.abs(shoulder_vel)
    abs_hip_vel = np.abs(hip_vel)

    shoulder_peak_idx = int(np.argmax(abs_shoulder_vel)) if len(abs_shoulder_vel) else 0
    hip_peak_idx = int(np.argmax(abs_hip_vel)) if len(abs_hip_vel) else 0
    shoulder_peak_frame = frame_ids[min(len(frame_ids) - 1, shoulder_peak_idx + 1)]
    hip_peak_frame = frame_ids[min(len(frame_ids) - 1, hip_peak_idx + 1)]
    sequence_delta_frames = int(shoulder_peak_frame - hip_peak_frame)

    if sequence_delta_frames <= -1:
        sequence_pattern = "shoulders_lead"
    elif sequence_delta_frames >= 1:
        sequence_pattern = "hips_lead"
    else:
        sequence_pattern = "in_sync"

    phase_lag_deg = float(np.percentile(relative_phase, 75))
    velocity_gap_deg = float(np.mean(np.abs(abs_shoulder_vel - abs_hip_vel))) if len(abs_shoulder_vel) else 0.0
    shoulder_range_deg = float(np.max(shoulder_arr) - np.min(shoulder_arr))
    hip_range_deg = float(np.max(hip_arr) - np.min(hip_arr))
    movement_energy = max(shoulder_range_deg, hip_range_deg)

    phase_low = float(config.get("phase_low_deg", 10.0))
    phase_high = float(config.get("phase_high_deg", 36.0))
    phase_norm = _clip01((phase_lag_deg - phase_low) / max(1.0, phase_high - phase_low))
    velocity_norm = _clip01(velocity_gap_deg / float(config.get("velocity_gap_norm_deg", 12.0)))
    sequence_penalty = {
        "hips_lead": 0.10,
        "in_sync": 0.35,
        "shoulders_lead": 1.0,
    }[sequence_pattern]
    motion_conf = _clip01(movement_energy / float(config.get("movement_norm_deg", 18.0)))

    raw_signal = (phase_norm * 0.50) + (velocity_norm * 0.25) + (sequence_penalty * 0.25)
    signal = floor + ((1.0 - floor) * raw_signal * max(0.35, motion_conf))
    signal = round(max(signal, floor), 3)

    confidence = round(float(np.mean(vis)) * min(1.0, len(frame_ids) / 8.0) * max(0.5, motion_conf), 3) if vis else 0.0

    return {
        "risk_id": "hip_shoulder_mismatch",
        "signal_strength": signal,
        "confidence": confidence,
        "debug": {
            "phase_lag_deg": round(phase_lag_deg, 3),
            "relative_velocity_gap_deg": round(velocity_gap_deg, 3),
            "shoulder_range_deg": round(shoulder_range_deg, 3),
            "hip_range_deg": round(hip_range_deg, 3),
            "hip_peak_frame": int(hip_peak_frame),
            "shoulder_peak_frame": int(shoulder_peak_frame),
            "sequence_delta_frames": sequence_delta_frames,
            "sequence_pattern": sequence_pattern,
        },
    }
