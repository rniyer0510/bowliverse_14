from __future__ import annotations
from .shared import *
from .analytics import _risk_weight, _safe_int
from .leg_callouts import _front_leg_support_caption
from .trunk_callout import _trunk_lean_caption
from .hip_callout import _hip_shoulder_caption
from .bubble_base import _bubble_copy
from .hotspot_support import _root_cause_phase_target, _root_cause_proof_step
from .timeline_events import _render_timeline_events

def _hotspot_stage_plan(hotspot_hold: int) -> List[Tuple[str, int]]:
    hotspot_hold = max(0, int(hotspot_hold))
    if hotspot_hold <= 0:
        return []
    if hotspot_hold == 1:
        return [("label", 1)]
    if hotspot_hold == 2:
        return [("rings", 1), ("label", 1)]
    if hotspot_hold == 3:
        return [("line", 1), ("rings", 1), ("label", 1)]
    line_count = max(1, int(round(hotspot_hold * 0.14)))
    label_count = max(1, int(round(hotspot_hold * 0.48)))
    rings_count = max(1, hotspot_hold - line_count - label_count)
    while line_count + rings_count + label_count > hotspot_hold:
        if rings_count > 1:
            rings_count -= 1
        elif label_count > 1:
            label_count -= 1
        else:
            line_count -= 1
    while line_count + rings_count + label_count < hotspot_hold:
        label_count += 1
    return [
        ("line", line_count),
        ("rings", rings_count),
        ("label", label_count),
    ]
def _hotspot_search_window(
    *,
    phase_key: str,
    anchor_frame: int,
    start: int,
    stop: int,
) -> range:
    if phase_key == "ffc":
        lo = max(start, anchor_frame - 4)
        hi = min(stop - 1, anchor_frame + 4)
    elif phase_key == "release":
        lo = max(start, anchor_frame - 3)
        hi = min(stop - 1, anchor_frame + 2)
    else:
        lo = max(start, anchor_frame - 2)
        hi = min(stop - 1, anchor_frame + 2)
    return range(lo, hi + 1)
def _select_hotspot_frame_idx(
    *,
    tracks: Dict[int, Dict[str, Any]],
    hand: Optional[str],
    risk_id: Optional[str],
    risk_by_id: Dict[str, Dict[str, Any]],
    phase_key: str,
    anchor_frame: int,
    start: int,
    stop: int,
) -> int:
    if not risk_id:
        return int(anchor_frame)
    best_frame = int(anchor_frame)
    best_score = float("-inf")
    for candidate in _hotspot_search_window(
        phase_key=phase_key,
        anchor_frame=int(anchor_frame),
        start=start,
        stop=stop,
    ):
        regions = _load_hotspot_regions(
            tracks=tracks,
            frame_idx=int(candidate),
            hand=hand,
            risk_id=risk_id,
            risk_by_id=risk_by_id,
        )
        if not regions:
            continue
        weights = [float(region.get("weight") or 0.0) for region in regions]
        strong_regions = sum(1 for weight in weights if weight >= 0.34)
        score = (
            sum(weights)
            + strong_regions * 0.18
            + len(regions) * 0.05
            - abs(int(candidate) - int(anchor_frame)) * 0.04
        )
        if score > best_score:
            best_score = score
            best_frame = int(candidate)
    return best_frame
def _pause_anchor_frames(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> Dict[int, str]:
    render_events = _render_timeline_events(start=start, stop=stop, events=events)
    anchors: Dict[int, str] = {}
    for key in ("bfc", "ffc", "release"):
        frame_value = _safe_int(((render_events or {}).get(key) or {}).get("frame"))
        if frame_value is None:
            continue
        if start <= frame_value < stop:
            anchors[int(frame_value)] = key
    return dict(sorted(anchors.items()))
def _pause_hold_plan(*, pause_frames: int, has_hotspot: bool) -> Tuple[int, int]:
    base_pause = max(0, int(pause_frames))
    if base_pause <= 0:
        return 0, 0
    if not has_hotspot:
        return base_pause, 0
    cue_hold = max(1, int(round(base_pause * 0.45)))
    hotspot_hold = max(1, base_pause - cue_hold)
    hotspot_bonus = max(1, int(round(base_pause * 0.35)))
    return cue_hold, hotspot_hold + hotspot_bonus
def _pause_sequence_plan(
    *,
    pause_frames: int,
    has_hotspot: bool,
    has_leakage: bool,
) -> Dict[str, int]:
    cue_hold, hotspot_hold = _pause_hold_plan(
        pause_frames=pause_frames,
        has_hotspot=has_hotspot,
    )
    if not has_leakage:
        return {
            "proof": cue_hold,
            "leak": 0,
            "pay": 0,
            "hotspot": hotspot_hold,
        }

    if cue_hold <= 1:
        proof_hold = 0
        leak_hold = cue_hold
    else:
        leak_hold = max(1, int(round(cue_hold * 0.48)))
        proof_hold = max(1, cue_hold - leak_hold)

    pay_hold = 0
    hotspot_core_hold = hotspot_hold
    if has_hotspot and hotspot_hold > 1:
        pay_hold = max(1, int(round(hotspot_hold * 0.24)))
        hotspot_core_hold = max(1, hotspot_hold - pay_hold)
    elif has_hotspot:
        hotspot_core_hold = hotspot_hold

    return {
        "proof": proof_hold,
        "leak": leak_hold,
        "pay": pay_hold,
        "hotspot": hotspot_core_hold,
    }
def _proof_bubble_text_for_phase(
    *,
    phase_key: str,
    risk_id: Optional[str],
    proof_step: Optional[Dict[str, Any]],
    risk_by_id: Dict[str, Dict[str, Any]],
) -> str:
    normalized_phase = str(phase_key or "").strip().lower()
    normalized_risk = str(risk_id or "").strip()
    if normalized_phase == "ffc":
        if normalized_risk == "knee_brace_failure":
            caption = _front_leg_support_caption(risk_by_id.get("knee_brace_failure")) or {}
            return _bubble_copy(
                title=str((proof_step or {}).get("title") or RISK_TITLE_BY_ID["knee_brace_failure"]),
                headline=str((proof_step or {}).get("headline") or caption.get("title") or ""),
                body=str((proof_step or {}).get("body") or caption.get("body") or ""),
            )
        if normalized_risk == "foot_line_deviation":
            return _bubble_copy(
                title=str((proof_step or {}).get("title") or "Where It Starts"),
                headline=str((proof_step or {}).get("headline") or "Front foot lands across here."),
                body=str((proof_step or {}).get("body") or "The landing line is leaking force sideways."),
            )
    if normalized_phase == "release":
        if normalized_risk == "lateral_trunk_lean":
            caption = _trunk_lean_caption(risk_by_id.get("lateral_trunk_lean")) or {}
            return _bubble_copy(
                title=str((proof_step or {}).get("title") or RISK_TITLE_BY_ID["lateral_trunk_lean"]),
                headline=str((proof_step or {}).get("headline") or caption.get("title") or ""),
                body=str((proof_step or {}).get("body") or caption.get("body") or ""),
            )
        if normalized_risk == "hip_shoulder_mismatch":
            caption = _hip_shoulder_caption(risk_by_id.get("hip_shoulder_mismatch")) or {}
            return _bubble_copy(
                title=str((proof_step or {}).get("title") or RISK_TITLE_BY_ID["hip_shoulder_mismatch"]),
                headline=str((proof_step or {}).get("headline") or caption.get("title") or ""),
                body=str((proof_step or {}).get("body") or caption.get("body") or ""),
            )
    return ""
