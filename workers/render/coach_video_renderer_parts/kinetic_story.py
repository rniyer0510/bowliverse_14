from __future__ import annotations
from .shared import *
from .analytics import _supports_ffc_story

def _kinetic_pace_translation(
    kinetic_chain: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(kinetic_chain, dict):
        return {}
    payload = kinetic_chain.get("pace_translation") or {}
    if isinstance(payload, dict):
        return payload
    return {}
def _pace_leakage_stage(
    kinetic_chain: Optional[Dict[str, Any]],
    stage_key: str,
) -> Optional[Dict[str, Any]]:
    pace_translation = _kinetic_pace_translation(kinetic_chain)
    for item in list(pace_translation.get("pace_leakage") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("stage") or "").strip() == str(stage_key or "").strip():
            return item
    return None
def _phase_leakage_payload(
    *,
    kinetic_chain: Optional[Dict[str, Any]],
    phase_key: str,
    risk_id: Optional[str],
    events: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    pace_translation = _kinetic_pace_translation(kinetic_chain)
    if not pace_translation or not risk_id:
        return None

    transfer_efficiency = float(pace_translation.get("transfer_efficiency") or 0.0)
    leakage_before_block = float(pace_translation.get("leakage_before_block") or 0.0)
    leakage_at_block = float(pace_translation.get("leakage_at_block") or 0.0)
    late_arm_chase = float(pace_translation.get("late_arm_chase") or 0.0)
    dissipation_burden = float(pace_translation.get("dissipation_burden") or 0.0)
    normalized_phase = str(phase_key or "").strip().lower()
    normalized_risk = str(risk_id or "").strip()

    if normalized_phase == "ffc":
        if not _supports_ffc_story(events):
            return None
        transfer_shortfall = max(0.0, 1.0 - transfer_efficiency)
        stage_entry = _pace_leakage_stage(kinetic_chain, "front_foot_block") or _pace_leakage_stage(
            kinetic_chain, "transfer_and_block"
        )
        severity = max(
            leakage_at_block,
            transfer_shortfall,
            float((stage_entry or {}).get("severity") or 0.0),
        )
        if severity < 0.46:
            return None
        if normalized_risk == "knee_brace_failure":
            return {
                "title": "Transfer leak",
                "headline": "Carry leaks through the landing leg here.",
                "body": "The chain reaches front foot contact, but the leg does not turn it into a firm transfer point.",
                "bubble": "Energy leaks here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
        if normalized_risk == "foot_line_deviation":
            return {
                "title": "Transfer leak",
                "headline": "Carry leaks across the landing line here.",
                "body": "The landing foot is arriving off line, so some force spills sideways before it can stack upward.",
                "bubble": "Energy leaks off line here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
        if normalized_risk == "front_foot_braking_shock":
            return {
                "title": "Transfer leak",
                "headline": "Landing absorbs force instead of passing it on.",
                "body": "The front foot is taking the hit sharply here, so the chain loses calm carry into release.",
                "bubble": "Energy gets stuck here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
        if leakage_before_block >= 0.52:
            return {
                "title": "Transfer leak",
                "headline": "Some carry is already missing before the block.",
                "body": "The body arrives disorganized enough that the landing base has less clean momentum to work with.",
                "bubble": "Energy is already leaking here.",
                "severity": max(severity, leakage_before_block),
                "risk_id": normalized_risk,
            }
        return None

    if normalized_phase == "release":
        transfer_shortfall = max(0.0, 1.0 - transfer_efficiency)
        severity = max(
            leakage_at_block,
            transfer_shortfall,
            late_arm_chase,
            dissipation_burden * 0.85,
        )
        if severity < 0.48:
            return None
        if normalized_risk == "hip_shoulder_mismatch":
            return {
                "title": "Transfer leak",
                "headline": "Carry leaks between hips and shoulders here.",
                "body": "The lower and upper halves stop passing force cleanly into the release window, so the chain has to rescue timing late.",
                "bubble": "Energy breaks here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
        if normalized_risk == "lateral_trunk_lean":
            return {
                "title": "Transfer leak",
                "headline": "Carry leaks out through the trunk here.",
                "body": "The lower body gets the chain this far, but the trunk keeps travelling past the stack instead of releasing from a calmer base.",
                "bubble": "Energy spills here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
        if normalized_risk == "trunk_rotation_snap":
            return {
                "title": "Transfer leak",
                "headline": "The chain dumps late through the trunk here.",
                "body": "Timing arrives late enough that the top half has to turn sharply instead of carrying force through in sequence.",
                "bubble": "Energy dumps here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
        if normalized_risk == "front_foot_braking_shock" and leakage_at_block >= 0.46:
            return {
                "title": "Transfer leak",
                "headline": "The chain is still paying for the landing hit here.",
                "body": "Too much force gets stuck at landing, so the release is working from a harsher transfer instead of a clean carry.",
                "bubble": "Energy is still stuck here.",
                "severity": severity,
                "risk_id": normalized_risk,
            }
    return None
