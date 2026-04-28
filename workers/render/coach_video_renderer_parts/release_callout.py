from __future__ import annotations
from .shared import *
from .analytics import _risk_weight
from .story_logic import _story_risk_for_phase
from .trunk_callout import _draw_trunk_lean_callout
from .hip_callout import _draw_hip_shoulder_callout

def _draw_release_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    risk_by_id: Dict[str, Dict[str, Any]],
    report_story: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, Any]] = None,
    root_cause: Optional[Dict[str, Any]] = None,
    proof_step: Optional[Dict[str, Any]] = None,
) -> None:
    preferred_risk_id = _story_risk_for_phase(
        report_story,
        phase_key="release",
        events=events,
        root_cause=root_cause,
    )
    if preferred_risk_id == "hip_shoulder_mismatch":
        _draw_hip_shoulder_callout(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            risk=risk_by_id.get("hip_shoulder_mismatch"),
            proof_step=proof_step,
        )
        return
    if preferred_risk_id == "lateral_trunk_lean":
        _draw_trunk_lean_callout(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            risk=risk_by_id.get("lateral_trunk_lean"),
            proof_step=proof_step,
        )
        return
    if isinstance(report_story, dict) and str(report_story.get("theme") or "") in {
        "working_pattern",
        "good_base",
    }:
        return

    hip_shoulder = risk_by_id.get("hip_shoulder_mismatch")
    trunk_lean = risk_by_id.get("lateral_trunk_lean")
    if _risk_weight(hip_shoulder) >= max(0.20, _risk_weight(trunk_lean) + 0.03):
        _draw_hip_shoulder_callout(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            risk=hip_shoulder,
            proof_step=proof_step,
        )
        return
    _draw_trunk_lean_callout(
        frame,
        tracks=tracks,
        frame_idx=frame_idx,
        risk=trunk_lean,
        proof_step=proof_step,
    )
