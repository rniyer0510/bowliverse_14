from __future__ import annotations
from .shared import *
from .analytics import _supports_ffc_story
from .story_logic import _story_risk_for_phase
from .kinetic_story import _phase_leakage_payload
from .bubble_base import _draw_top_risk_panel
from .anchor_panels import _draw_phase_anchor_panel
from .leg_callouts import _draw_front_leg_support_callout, _draw_foot_line_overlay
from .release_callout import _draw_release_callout
from .hotspot_support import _should_render_warning_hotspots, _root_cause_phase_target, _root_cause_proof_step


def _hotspot_payload(risk_id: str, phase_target: Optional[Dict[str, Any]], risk_by_id: Dict[str, Dict[str, Any]], render_events: Optional[Dict[str, Any]], report_story: Optional[Dict[str, Any]], root_cause: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "risk_id": risk_id,
        "region_priority": list((phase_target or {}).get("region_priority") or []),
        "symptom_text": _summary_symptom_text(risk_by_id, events=render_events, report_story=report_story, root_cause=root_cause),
        "load_watch_text": str((phase_target or {}).get("load_watch_label") or "").strip() or _load_watch_label(risk_id) or _summary_load_watch_text(risk_by_id, events=render_events, report_story=report_story, root_cause=root_cause),
    }


def _prepare_pause_context(*, frame: np.ndarray, pose_frames: List[Dict[str, Any]], tracks: Dict[int, Dict[str, Any]], frame_idx: int, pause_key: str, hand: Optional[str], risk_by_id: Dict[str, Dict[str, Any]], render_events: Optional[Dict[str, Any]], report_story: Optional[Dict[str, Any]], root_cause: Optional[Dict[str, Any]], kinetic_chain: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    paused_frame = frame.copy()
    hotspot_payload: Optional[Dict[str, Any]] = None
    leakage_payload: Optional[Dict[str, Any]] = None
    proof_step = _root_cause_proof_step(root_cause, phase_key=pause_key)
    phase_target = _root_cause_phase_target(root_cause, phase_key=pause_key)
    allow_warning_hotspots = _should_render_warning_hotspots(report_story=report_story, root_cause=root_cause)
    if pause_key == "bfc":
        _draw_phase_anchor_panel(paused_frame, phase_key="bfc")
        return {"paused_frame": paused_frame, "hotspot_payload": None, "leakage_payload": None, "proof_step": proof_step}
    if pause_key == "ffc" and allow_warning_hotspots and _supports_ffc_story(render_events):
        preferred_ffc_risk = _preferred_ffc_cue_risk_id(risk_by_id, report_story=report_story, events=render_events, root_cause=root_cause) or _story_risk_for_phase(report_story, phase_key="ffc", events=render_events, root_cause=root_cause)
        if preferred_ffc_risk == "knee_brace_failure":
            _draw_front_leg_support_callout(paused_frame, tracks=tracks, frame_idx=frame_idx, hand=hand, risk=risk_by_id.get("knee_brace_failure"), proof_step=proof_step)
        elif preferred_ffc_risk == "foot_line_deviation":
            _draw_foot_line_overlay(paused_frame, pose_frames=pose_frames, frame_idx=frame_idx, events=render_events, hand=hand, risk=risk_by_id.get("foot_line_deviation"), proof_step=proof_step)
        elif preferred_ffc_risk and proof_step:
            _draw_top_risk_panel(paused_frame, title=str((proof_step or {}).get("title") or "Where It Starts"), headline=str((proof_step or {}).get("headline") or ""), body=str((proof_step or {}).get("body") or ""), accent=(92, 220, 255))
        else:
            _draw_phase_anchor_panel(paused_frame, phase_key="ffc")
        if preferred_ffc_risk:
            hotspot_payload = _hotspot_payload(preferred_ffc_risk, phase_target, risk_by_id, render_events, report_story, root_cause)
            leakage_payload = _phase_leakage_payload(kinetic_chain=kinetic_chain, phase_key="ffc", risk_id=preferred_ffc_risk, events=render_events)
        return {"paused_frame": paused_frame, "hotspot_payload": hotspot_payload, "leakage_payload": leakage_payload, "proof_step": proof_step}
    if pause_key == "release" and allow_warning_hotspots:
        release_hotspot_risk = _release_hotspot_risk_id(risk_by_id, events=render_events, report_story=report_story, root_cause=root_cause)
        _draw_release_callout(paused_frame, tracks=tracks, frame_idx=frame_idx, risk_by_id=risk_by_id, report_story=report_story, events=render_events, root_cause=root_cause, proof_step=proof_step)
        if not release_hotspot_risk and proof_step:
            _draw_top_risk_panel(paused_frame, title=str((proof_step or {}).get("title") or "What Happens Next"), headline=str((proof_step or {}).get("headline") or ""), body=str((proof_step or {}).get("body") or ""), accent=(0, 132, 255))
        elif not release_hotspot_risk:
            _draw_phase_anchor_panel(paused_frame, phase_key="release")
        if release_hotspot_risk:
            hotspot_payload = _hotspot_payload(release_hotspot_risk, phase_target, risk_by_id, render_events, report_story, root_cause)
            leakage_payload = _phase_leakage_payload(kinetic_chain=kinetic_chain, phase_key="release", risk_id=release_hotspot_risk, events=render_events)
        return {"paused_frame": paused_frame, "hotspot_payload": hotspot_payload, "leakage_payload": leakage_payload, "proof_step": proof_step}
    _draw_phase_anchor_panel(paused_frame, phase_key=pause_key)
    return {"paused_frame": paused_frame, "hotspot_payload": None, "leakage_payload": None, "proof_step": proof_step}
