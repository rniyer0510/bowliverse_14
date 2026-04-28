from __future__ import annotations
from .shared import *
from .hotspot_draw import _draw_hotspot_marker, _draw_hotspot_pointer_line, _draw_hotspot_compact_label
from .hotspot_support import _stacked_hotspot_region_keys, _draw_load_watch_card

def _draw_load_watch_phase(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk_id: Optional[str],
    risk_by_id: Dict[str, Dict[str, Any]],
    load_watch_text: str,
    pulse_phase: float,
    stage: str = "rings",
    region_priority: Optional[List[str]] = None,
) -> None:
    if not risk_id:
        return
    regions = _load_hotspot_regions(
        tracks=tracks,
        frame_idx=frame_idx,
        hand=hand,
        risk_id=risk_id,
        risk_by_id=risk_by_id,
    )
    if not regions:
        return
    ranked_regions = sorted(
        (
            region
            for region in regions
            if float(region.get("weight") or 0.0) >= 0.34
        ),
        key=lambda region: float(region.get("weight") or 0.0),
        reverse=True,
    )
    if not ranked_regions:
        ranked_regions = sorted(
            regions,
            key=lambda region: float(region.get("weight") or 0.0),
            reverse=True,
        )[:1]
    elif risk_id in FFC_DEPENDENT_RISKS:
        ranked_regions = ranked_regions[:3]
    elif len(ranked_regions) > 1:
        lead = float(ranked_regions[0].get("weight") or 0.0)
        ranked_regions = [
            region
            for idx, region in enumerate(ranked_regions)
            if idx == 0
            or (
                idx == 1
                and float(region.get("weight") or 0.0) >= max(0.50, lead * 0.72)
            )
        ][:2]

    scale = min(frame.shape[0], frame.shape[1])
    _draw_load_watch_card(frame, load_watch_text=load_watch_text)
    stacked_keys = [
        str(item).strip()
        for item in list(region_priority or _stacked_hotspot_region_keys(risk_id))
        if str(item).strip()
    ]
    stacked_regions: List[Dict[str, Any]] = []
    for region_key in stacked_keys:
        match = next(
            (region for region in ranked_regions if str(region.get("region_key") or "") == region_key),
            None,
        )
        if isinstance(match, dict):
            stacked_regions.append(match)
    if not stacked_regions and ranked_regions:
        stacked_regions = [ranked_regions[0]]
    primary_region = stacked_regions[0] if stacked_regions else None
    if not isinstance(primary_region, dict):
        return
    center = primary_region.get("center")
    if not isinstance(center, tuple) or len(center) != 2:
        return
    direction = tuple(primary_region.get("direction") or (1.0, -0.25))
    label = str(primary_region.get("label") or "")
    center_xy = (int(center[0]), int(center[1]))
    if stage == "line":
        _draw_hotspot_pointer_line(
            frame,
            center=center_xy,
            direction=direction,
            scale=scale,
        )
        return
    if stage in {"rings", "label"}:
        for region in stacked_regions:
            stacked_center = region.get("center")
            if not isinstance(stacked_center, tuple) or len(stacked_center) != 2:
                continue
            _draw_hotspot_marker(
                frame,
                center=(int(stacked_center[0]), int(stacked_center[1])),
                scale=scale,
                weight=float(region.get("weight") or 0.8),
                pulse_phase=pulse_phase,
            )
    if stage == "label":
        rect = _draw_hotspot_compact_label(
            frame,
            center=center_xy,
            direction=direction,
            label=label,
            scale=scale,
            occupied_rects=[],
        )
