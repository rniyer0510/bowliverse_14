from __future__ import annotations
from .shared import *

def _preferred_hotspot_region_key(risk_id: Optional[str]) -> Optional[str]:
    mapping = {
        "knee_brace_failure": "knee",
        "foot_line_deviation": "shin",
        "front_foot_braking_shock": "knee",
        "lateral_trunk_lean": "side_trunk",
        "hip_shoulder_mismatch": "side_trunk",
        "trunk_rotation_snap": "lumbar",
    }
    return mapping.get(str(risk_id or "").strip())
def _stacked_hotspot_region_keys(risk_id: Optional[str]) -> List[str]:
    risk_id = str(risk_id or "").strip()
    if risk_id == "knee_brace_failure":
        return ["knee", "shin", "groin"]
    if risk_id == "foot_line_deviation":
        return ["shin", "knee", "groin"]
    if risk_id == "front_foot_braking_shock":
        return ["knee", "shin", "groin"]
    if risk_id in FFC_DEPENDENT_RISKS:
        return ["groin", "knee", "shin"]
    if risk_id in {
        "lateral_trunk_lean",
        "hip_shoulder_mismatch",
        "trunk_rotation_snap",
    }:
        return ["upper_trunk", "side_trunk", "lumbar"]
    preferred = _preferred_hotspot_region_key(risk_id)
    return [preferred] if preferred else []
def _load_watch_support_text(load_watch_text: str) -> str:
    lower = str(load_watch_text or "").lower()
    if "lower back" in lower or "side trunk" in lower:
        return "This is where extra body load may build if the pattern repeats."
    return "This is where extra body load may build if the pattern repeats."
def _should_render_warning_hotspots(
    *,
    report_story: Optional[Dict[str, Any]],
    root_cause: Optional[Dict[str, Any]],
) -> bool:
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    if root_cause_status == "no_clear_problem":
        return False
    story_theme = str((report_story or {}).get("theme") or "").strip().lower()
    if story_theme in {"working_pattern", "good_base"}:
        return False
    renderer_guidance = ((root_cause or {}).get("renderer_guidance") or {})
    if "warning_hotspots_allowed" in renderer_guidance:
        return bool(renderer_guidance.get("warning_hotspots_allowed"))
    return True
def _root_cause_phase_target(
    root_cause: Optional[Dict[str, Any]],
    *,
    phase_key: str,
) -> Optional[Dict[str, Any]]:
    renderer_guidance = ((root_cause or {}).get("renderer_guidance") or {})
    phase_targets = renderer_guidance.get("phase_targets") or {}
    if not isinstance(phase_targets, dict):
        return None
    target = phase_targets.get(phase_key)
    return target if isinstance(target, dict) else None
def _root_cause_proof_step(
    root_cause: Optional[Dict[str, Any]],
    *,
    phase_key: str,
) -> Optional[Dict[str, Any]]:
    phase_target = _root_cause_phase_target(root_cause, phase_key=phase_key) or {}
    proof_step = phase_target.get("proof_step")
    return proof_step if isinstance(proof_step, dict) else None
def _draw_load_watch_card(
    frame: np.ndarray,
    *,
    load_watch_text: str,
) -> None:
    return
