from __future__ import annotations
from .shared import *
from .geometry_helpers import _draw_partial_polyline
from .hotspot_draw import _draw_hotspot_compact_label
from .transfer_core import _transfer_leak_geometry

def _body_pay_region_priority(risk_id: str) -> List[str]:
    if risk_id == "knee_brace_failure":
        return ["groin", "shin", "knee"]
    if risk_id == "foot_line_deviation":
        return ["groin", "knee", "shin"]
    if risk_id == "front_foot_braking_shock":
        return ["shin", "knee", "groin"]
    if risk_id == "hip_shoulder_mismatch":
        return ["side_trunk", "lumbar", "upper_trunk"]
    if risk_id == "lateral_trunk_lean":
        return ["lumbar", "side_trunk", "upper_trunk"]
    if risk_id == "trunk_rotation_snap":
        return ["lumbar", "side_trunk", "upper_trunk"]
    return []
def _select_body_pay_region(
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk_id: str,
    risk_by_id: Dict[str, Dict[str, Any]],
    region_priority: Optional[List[str]],
    anchor: Tuple[int, int],
    scale: int,
) -> Optional[Dict[str, Any]]:
    regions = _load_hotspot_regions(
        tracks=tracks,
        frame_idx=frame_idx,
        hand=hand,
        risk_id=risk_id,
        risk_by_id=risk_by_id,
    )
    if not regions:
        return None
    by_key = {
        str(region.get("region_key") or ""): region
        for region in regions
        if isinstance(region, dict)
    }
    ordered_keys: List[str] = []
    for key in list(region_priority or []) + _body_pay_region_priority(risk_id):
        if key and key not in ordered_keys:
            ordered_keys.append(key)
    min_distance = max(14.0, scale * 0.045)
    for key in ordered_keys:
        region = by_key.get(key)
        if not region:
            continue
        center = region.get("center")
        if not isinstance(center, tuple) or len(center) != 2:
            continue
        weight = float(region.get("weight") or 0.0)
        distance = float(np.hypot(float(center[0] - anchor[0]), float(center[1] - anchor[1])))
        if weight >= 0.30 and distance >= min_distance:
            return region
    ranked = sorted(
        (
            region
            for region in regions
            if isinstance(region, dict) and isinstance(region.get("center"), tuple)
        ),
        key=lambda region: float(region.get("weight") or 0.0),
        reverse=True,
    )
    return ranked[0] if ranked else None
def _draw_body_pay_phase(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk_id: str,
    risk_by_id: Dict[str, Dict[str, Any]],
    region_priority: Optional[List[str]],
    progress: float,
) -> None:
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
    anchor = geometry.get("anchor")
    anchor_xy = _safe_point(anchor)
    if anchor_xy is None:
        return
    scale = min(frame.shape[0], frame.shape[1])
    region = _select_body_pay_region(
        tracks=tracks,
        frame_idx=frame_idx,
        hand=hand,
        risk_id=risk_id,
        risk_by_id=risk_by_id,
        region_priority=region_priority,
        anchor=anchor_xy,
        scale=scale,
    )
    if not region:
        return
    pay_center = _safe_point(region.get("center"))
    if pay_center is None:
        return
    thickness = max(3, scale // 180)
    shadow = max(5, thickness + 2)
    path_points = [anchor_xy, pay_center]
    _draw_partial_polyline(
        frame,
        points=path_points,
        progress=progress,
        color=FLOW_PAY,
        thickness=thickness,
        shadow_thickness=shadow,
    )
    ring_r = max(10, scale // 24)
    overlay = frame.copy()
    cv2.circle(overlay, pay_center, ring_r, FLOW_PAY_CORE, max(2, thickness - 1), cv2.LINE_AA)
    cv2.circle(overlay, pay_center, max(4, ring_r // 3), FLOW_PAY_CORE, -1, cv2.LINE_AA)
    alpha = 0.25 + max(0.0, min(1.0, float(progress))) * 0.30
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    if progress >= 0.42:
        direction = (
            float(pay_center[0] - anchor_xy[0]),
            float(pay_center[1] - anchor_xy[1]),
        )
        _draw_hotspot_compact_label(
            frame,
            center=(int(pay_center[0]), int(pay_center[1])),
            direction=direction,
            label=str(region.get("label") or "Body pay"),
            scale=scale,
            occupied_rects=[],
        )
