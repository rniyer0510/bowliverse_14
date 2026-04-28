from __future__ import annotations
from .shared import *
from .tracks import *
from .joints import _front_leg_joints
from .geometry_helpers import _midpoint

def _draw_transfer_leak_particles(
    frame: np.ndarray,
    *,
    anchor: Tuple[int, int],
    direction: Tuple[float, float],
    accent: Tuple[int, int, int],
    scale: int,
    intensity: float,
) -> None:
    intensity = max(0.0, min(1.0, float(intensity)))
    if intensity <= 0.0:
        return
    dx, dy = float(direction[0]), float(direction[1])
    norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / norm, dy / norm
    count = 2 + int(round(intensity * 3.0))
    for idx in range(1, count + 1):
        travel = scale * (0.026 + idx * (0.022 + intensity * 0.012))
        center = (
            int(round(anchor[0] + ux * travel)),
            int(round(anchor[1] + uy * travel)),
        )
        radius = max(3, int(round(scale * (0.013 + intensity * 0.004 - idx * 0.0020))))
        alpha = max(0.20, 0.24 + intensity * 0.32 - idx * 0.05)
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, accent, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
def _transfer_leak_geometry(
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk_id: str,
) -> Optional[Dict[str, Any]]:
    left_hip = _track_point(tracks, LEFT_HIP, frame_idx)
    right_hip = _track_point(tracks, RIGHT_HIP, frame_idx)
    left_shoulder = _track_point(tracks, LEFT_SHOULDER, frame_idx)
    right_shoulder = _track_point(tracks, RIGHT_SHOULDER, frame_idx)
    hip_mid = _midpoint(left_hip, right_hip)
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    front_hip_idx, front_knee_idx, front_ankle_idx = _front_leg_joints(hand)
    front_hip = _track_point(tracks, front_hip_idx, frame_idx)
    front_knee = _track_point(tracks, front_knee_idx, frame_idx)
    front_ankle = _track_point(tracks, front_ankle_idx, frame_idx)

    if risk_id in {"knee_brace_failure", "front_foot_braking_shock", "foot_line_deviation"}:
        anchor = front_knee if risk_id != "foot_line_deviation" else (front_ankle or front_knee)
        if hip_mid is None or anchor is None:
            return None
        path_points = [
            point
            for point in [
                hip_mid,
                front_hip,
                front_knee if risk_id != "foot_line_deviation" else anchor,
            ]
            if point is not None
        ]
        if len(path_points) < 2:
            return None
        direction_target = shoulder_mid or front_hip or hip_mid
        direction = (
            float(anchor[0] - direction_target[0]),
            float(anchor[1] - direction_target[1]),
        )
        if risk_id == "foot_line_deviation":
            direction = (
                float(anchor[0] - hip_mid[0]),
                float(max(-24, anchor[1] - hip_mid[1])),
            )
        elif risk_id == "front_foot_braking_shock":
            direction = (
                float(anchor[0] - hip_mid[0]),
                float(max(18, anchor[1] - hip_mid[1])),
            )
        return {
            "anchor": anchor,
            "path_points": path_points,
            "direction": direction,
        }

    if hip_mid is None or shoulder_mid is None:
        return None
    if risk_id == "hip_shoulder_mismatch":
        anchor = _midpoint(hip_mid, shoulder_mid)
        if anchor is None:
            return None
        lower_anchor = front_knee or hip_mid
        return {
            "anchor": anchor,
            "path_points": [point for point in [lower_anchor, hip_mid, anchor] if point is not None],
            "direction": (
                float(shoulder_mid[0] - hip_mid[0]),
                float(shoulder_mid[1] - hip_mid[1]),
            ),
        }

    shoulder_dx = shoulder_mid[0] - hip_mid[0]
    return {
        "anchor": shoulder_mid,
        "path_points": [point for point in [front_knee or hip_mid, hip_mid, shoulder_mid] if point is not None],
        "direction": (
            float(shoulder_dx if shoulder_dx != 0 else 1.0),
            float(-max(24, abs(shoulder_dx) * 0.40)),
        ),
    }
def _draw_transfer_break_phase(
    frame: np.ndarray,
    *,
    points: List[Tuple[int, int]],
    anchor: Tuple[int, int],
    direction: Tuple[float, float],
    intensity: float,
) -> None:
    intensity = max(0.0, min(1.0, float(intensity)))
    if len(points) < 2:
        return
    scale = min(frame.shape[0], frame.shape[1])
    thickness = max(3, scale // 165)
    shadow = max(5, thickness + 2)
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, SKELETON_SHADOW, shadow, cv2.LINE_AA)
        cv2.line(frame, start, end, FLOW_CARRY, thickness, cv2.LINE_AA)
    ring_r = max(13, scale // 20)
    cv2.circle(frame, anchor, ring_r + 3, SKELETON_SHADOW, max(2, thickness), cv2.LINE_AA)
    cv2.circle(frame, anchor, ring_r, FLOW_BREAK, max(2, thickness - 1), cv2.LINE_AA)
    cv2.circle(frame, anchor, max(4, ring_r // 4), FLOW_BREAK, -1, cv2.LINE_AA)
    slash_dx = max(8, int(round(scale * (0.014 + intensity * 0.008))))
    slash_dy = max(8, int(round(scale * (0.012 + intensity * 0.008))))
    cv2.line(
        frame,
        (anchor[0] - slash_dx, anchor[1] - slash_dy),
        (anchor[0] + slash_dx, anchor[1] + slash_dy),
        FLOW_BREAK,
        max(2, thickness - 1),
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (anchor[0] - slash_dx, anchor[1] + slash_dy),
        (anchor[0] + slash_dx, anchor[1] - slash_dy),
        FLOW_BREAK,
        max(2, thickness - 1),
        cv2.LINE_AA,
    )
    dx, dy = float(direction[0]), float(direction[1])
    norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / norm, dy / norm
    dash_gap = max(10, int(round(scale * 0.030)))
    dash_len = max(12, int(round(scale * 0.040)))
    ghost_start = (
        int(round(anchor[0] + ux * dash_gap)),
        int(round(anchor[1] + uy * dash_gap)),
    )
    for idx in range(3):
        seg_start = (
            int(round(ghost_start[0] + ux * idx * (dash_len + dash_gap))),
            int(round(ghost_start[1] + uy * idx * (dash_len + dash_gap))),
        )
        seg_end = (
            int(round(seg_start[0] + ux * dash_len)),
            int(round(seg_start[1] + uy * dash_len)),
        )
        cv2.line(frame, seg_start, seg_end, FLOW_GHOST, max(1, thickness - 1), cv2.LINE_AA)
    _draw_transfer_leak_particles(
        frame,
        anchor=anchor,
        direction=direction,
        accent=FLOW_BREAK,
        scale=scale,
        intensity=intensity,
    )
