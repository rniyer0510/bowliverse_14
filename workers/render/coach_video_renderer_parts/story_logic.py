from __future__ import annotations
from .shared import *
from .analytics import _risk_supported_for_phase

def _story_feature_labels(report_story: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report_story, dict):
        return []
    labels: List[str] = []
    watch_focus = report_story.get("watch_focus") or {}
    watch_label = str(watch_focus.get("label") or "").strip()
    if watch_label:
        labels.append(watch_label)
    for item in report_story.get("key_metrics") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        label = str(item.get("label") or "").strip()
        resolved = FEATURE_TO_RENDER_LABEL.get(key) or label
        if resolved and resolved not in labels:
            labels.append(resolved)
    return labels
def _positive_recap_lines(report_story: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report_story, dict):
        return []
    theme = str(report_story.get("theme") or "").strip()
    if theme not in {"working_pattern", "good_base"}:
        return []

    lines: List[str] = []
    watch_focus = report_story.get("watch_focus") or {}
    watch_label = str(watch_focus.get("label") or "").strip()
    if watch_label:
        lines.append(f"Keep watching {watch_label}")

    positive_by_key = {
        "upper_body_alignment": "Upper body stays aligned",
        "lower_body_alignment": "Lower body stays aligned",
        "whole_body_alignment": "Action shape stays connected",
        "momentum_forward": "Carries forward well",
        "front_leg_support": "Front leg support looks steady",
        "trunk_lean": "Body stays fairly tall",
        "upper_body_opening": "Top half stays in order",
        "action_flow": "Action carries through well",
        "front_foot_line": "Feet stay balanced and in line",
    }

    for item in report_story.get("key_metrics") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        label = positive_by_key.get(key)
        if not label or label in lines:
            continue
        lines.append(label)
        if len(lines) >= 4:
            break

    if not lines:
        headline = str(report_story.get("headline") or "").strip()
        if "strong working pattern" in headline.lower():
            lines = ["Strong working pattern", "Keep repeating this shape"]
        else:
            lines = ["Action has a good base", "Keep repeating this shape"]
    return lines[:4]
def _story_risk_for_phase(
    report_story: Optional[Dict[str, Any]],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
    root_cause: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    root_cause_guidance = ((root_cause or {}).get("renderer_guidance") or {})
    phase_targets = root_cause_guidance.get("phase_targets") or {}
    if isinstance(phase_targets, dict):
        phase_target = phase_targets.get(phase_key) or {}
        phase_target_risk_id = str((phase_target or {}).get("risk_id") or "").strip()
        if _risk_supported_for_phase(phase_target_risk_id, phase_key=phase_key, events=events):
            return phase_target_risk_id
    anchor_risk_ids = root_cause_guidance.get("anchor_risk_ids") or {}
    if isinstance(anchor_risk_ids, dict):
        root_cause_risk_id = str(anchor_risk_ids.get(phase_key) or "").strip()
        if _risk_supported_for_phase(root_cause_risk_id, phase_key=phase_key, events=events):
            return root_cause_risk_id
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    if root_cause_status in {"clear", "holdback", "no_clear_problem", "not_interpretable"}:
        return None
    if not isinstance(report_story, dict):
        return None
    hero_risk_id = str(report_story.get("hero_risk_id") or "").strip()
    if _risk_supported_for_phase(hero_risk_id, phase_key=phase_key, events=events):
        return hero_risk_id

    watch_focus = report_story.get("watch_focus") or {}
    watch_key = str(watch_focus.get("key") or "").strip()
    mapped = {
        "front_leg_support": "knee_brace_failure",
        "front_foot_line": "foot_line_deviation",
        "trunk_lean": "lateral_trunk_lean",
        "upper_body_opening": "hip_shoulder_mismatch",
        "action_flow": "front_foot_braking_shock",
        "trunk_rotation_load": "trunk_rotation_snap",
    }.get(watch_key)
    if _risk_supported_for_phase(mapped, phase_key=phase_key, events=events):
        return mapped
    return None
def _format_action_label(action: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(action, dict):
        return None
    for key in ("intent", "action"):
        raw = str(action.get(key) or "").strip()
        if raw and raw.upper() != "UNKNOWN":
            return raw.replace("_", " ").title()
    return None
