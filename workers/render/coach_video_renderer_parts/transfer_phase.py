from __future__ import annotations
from .shared import *
from .geometry_helpers import _draw_partial_polyline
from .transfer_core import _transfer_leak_geometry, _draw_transfer_break_phase
from .bubble_base import _draw_pointer_bubble

def _draw_transfer_leak_phase(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    payload: Dict[str, Any],
    progress: float = 1.0,
) -> None:
    risk_id = str((payload or {}).get("risk_id") or "").strip()
    if not risk_id:
        return
    geometry = _transfer_leak_geometry(
        tracks=tracks,
        frame_idx=frame_idx,
        hand=hand,
        risk_id=risk_id,
    )
    if not geometry:
        return
    points = list(geometry.get("path_points") or [])
    anchor = geometry.get("anchor")
    direction = geometry.get("direction")
    if len(points) < 2 or not isinstance(anchor, tuple) or not isinstance(direction, tuple):
        return
    scale = min(frame.shape[0], frame.shape[1])
    thickness = max(3, scale // 165)
    shadow = max(5, thickness + 2)
    carry_progress = min(1.0, max(0.0, float(progress)) / 0.44)
    _draw_partial_polyline(
        frame,
        points=points,
        progress=carry_progress,
        color=FLOW_CARRY,
        thickness=thickness,
        shadow_thickness=shadow,
    )
    if carry_progress >= 0.72:
        anchor_overlay = frame.copy()
        cv2.circle(anchor_overlay, anchor, max(9, scale // 26), FLOW_CARRY, max(2, thickness - 1), cv2.LINE_AA)
        cv2.addWeighted(anchor_overlay, 0.34, frame, 0.66, 0.0, frame)
    if progress <= 0.44:
        return
    break_progress = min(1.0, (max(0.0, float(progress)) - 0.44) / 0.56)
    _draw_transfer_break_phase(
        frame,
        points=points,
        anchor=anchor,
        direction=direction,
        intensity=break_progress,
    )
    if break_progress >= 0.32:
        _draw_pointer_bubble(
            frame,
            anchor=anchor,
            text=str((payload or {}).get("bubble") or "Energy leaks here."),
            accent=FLOW_BREAK,
        )
