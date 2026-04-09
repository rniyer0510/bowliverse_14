"""
ActionLab persistence READ APIs (Phase-I – Auth Locked)

Supports:
- Player listing
- Player creation
- Player profile update
- Deterministic analysis history (run-based)
- Run-id based report fetch

SECURITY:
- All endpoints require authentication
- All reads are scoped to current_account via AccountPlayerLink
"""

from collections import Counter
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.persistence.session import get_db
from app.persistence.models import (
    Player,
    AccountPlayerLink,
    AnalysisRun,
    AnalysisResultRaw,
)
from app.common.auth import get_current_account

router = APIRouter()

ACTION_CHANGE_BASELINE_WINDOW = 4
ACTION_CHANGE_MIN_BASELINE_RUNS = 2
BASELINE_REFRESH_WINDOW = 4
BASELINE_REFRESH_MIN_RUNS = 3
BASELINE_REVIEW_GAP_DAYS = 180
ACTION_CHANGE_NUMERIC_RULES = {
    "overall": {"min_range": 4.0, "spread_cushion": 2.0, "watch_buffer": 4.0},
    "upper_body_alignment": {"min_range": 4.0, "spread_cushion": 2.0, "watch_buffer": 4.0},
    "lower_body_alignment": {"min_range": 4.0, "spread_cushion": 2.0, "watch_buffer": 4.0},
    "whole_body_alignment": {"min_range": 4.0, "spread_cushion": 2.0, "watch_buffer": 4.0},
    "momentum_forward": {"min_range": 4.0, "spread_cushion": 2.0, "watch_buffer": 4.0},
    "front_foot_toe_angle_deg": {"min_range": 5.0, "spread_cushion": 3.0, "watch_buffer": 5.0},
    "risk_front_foot_braking_shock": {"min_range": 0.06, "spread_cushion": 0.04, "watch_buffer": 0.08},
    "risk_knee_brace_failure": {"min_range": 0.06, "spread_cushion": 0.04, "watch_buffer": 0.08},
    "risk_trunk_rotation_snap": {"min_range": 0.06, "spread_cushion": 0.04, "watch_buffer": 0.08},
    "risk_hip_shoulder_mismatch": {"min_range": 0.06, "spread_cushion": 0.04, "watch_buffer": 0.08},
    "risk_lateral_trunk_lean": {"min_range": 0.06, "spread_cushion": 0.04, "watch_buffer": 0.08},
    "risk_foot_line_deviation": {"min_range": 0.06, "spread_cushion": 0.04, "watch_buffer": 0.08},
    "kinetic_chain_delta_frames": {"min_range": 1.0, "spread_cushion": 1.0, "watch_buffer": 1.0},
}
ACTION_CHANGE_LABELS = {
    "overall": "Overall",
    "upper_body_alignment": "Upper Body Alignment",
    "lower_body_alignment": "Lower Body Alignment",
    "whole_body_alignment": "Whole Body Alignment",
    "momentum_forward": "Momentum Forward",
    "front_foot_toe_angle_deg": "Front-foot Toe Angle",
    "action_type": "Action Type",
    "action_intent": "Action Intent",
    "front_foot_toe_alignment": "Front-foot Toe Alignment",
    "risk_front_foot_braking_shock": "Front-foot Braking Shock",
    "risk_knee_brace_failure": "Knee Brace Failure",
    "risk_trunk_rotation_snap": "Trunk Rotation Snap",
    "risk_hip_shoulder_mismatch": "Hip-shoulder Separation Timing",
    "risk_lateral_trunk_lean": "Lateral Trunk Lean",
    "risk_foot_line_deviation": "Foot Line Deviation",
    "kinetic_chain_delta_frames": "Hip-shoulder Timing Delta",
    "kinetic_chain_sequence": "Kinetic Chain Sequence",
}
ACTION_CHANGE_SUMMARY_PRIORITY = (
    "kinetic_chain_sequence",
    "risk_hip_shoulder_mismatch",
    "risk_trunk_rotation_snap",
    "risk_knee_brace_failure",
    "risk_foot_line_deviation",
    "risk_lateral_trunk_lean",
    "risk_front_foot_braking_shock",
    "lower_body_alignment",
    "upper_body_alignment",
    "whole_body_alignment",
    "momentum_forward",
    "action_type",
    "action_intent",
    "front_foot_toe_alignment",
    "front_foot_toe_angle_deg",
    "overall",
    "kinetic_chain_delta_frames",
)
ACTION_CHANGE_RISK_IDS = (
    "front_foot_braking_shock",
    "knee_brace_failure",
    "trunk_rotation_snap",
    "hip_shoulder_mismatch",
    "lateral_trunk_lean",
    "foot_line_deviation",
)


def _extract_scorecard(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    clinician = result_json.get("clinician")
    if not isinstance(clinician, dict):
        return None
    scorecard = clinician.get("scorecard")
    return scorecard if isinstance(scorecard, dict) else None


def _extract_rating_system_v2(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    clinician = result_json.get("clinician")
    if not isinstance(clinician, dict):
        return None
    rating_system = clinician.get("rating_system_v2")
    return rating_system if isinstance(rating_system, dict) else None


def _score_band(score: Optional[int]) -> str:
    if score is None:
        return "unknown"
    if score >= 90:
        return "elite"
    if score >= 75:
        return "strong"
    if score >= 60:
        return "watch"
    if score >= 45:
        return "concern"
    return "review"


def _extract_score_summary(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scorecard = _extract_scorecard(result_json)
    if not scorecard:
        return None

    overall = scorecard.get("overall") or {}
    pillars = scorecard.get("pillars") or {}
    confidence = scorecard.get("confidence") or {}

    def pillar_score(key: str) -> Optional[int]:
        pillar = pillars.get(key) or {}
        value = pillar.get("score")
        return int(value) if isinstance(value, (int, float)) else None

    overall_score = overall.get("score")
    if not isinstance(overall_score, (int, float)):
        return None

    confidence_score = confidence.get("score")
    return {
        "overall": {
            "score": int(overall_score),
            "band": _score_band(int(overall_score)),
            "label": overall.get("label"),
        },
        "balance": {
            "score": pillar_score("balance"),
            "band": _score_band(pillar_score("balance")),
        },
        "carry": {
            "score": pillar_score("carry"),
            "band": _score_band(pillar_score("carry")),
        },
        "body_load": {
            "score": pillar_score("body_load"),
            "band": _score_band(pillar_score("body_load")),
        },
        "confidence": {
            "score": int(confidence_score) if isinstance(confidence_score, (int, float)) else None,
            "band": _score_band(int(confidence_score)) if isinstance(confidence_score, (int, float)) else "unknown",
        },
    }


def _extract_rating_summary_v2(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rating_system = _extract_rating_system_v2(result_json)
    if not rating_system:
        return None

    overall = rating_system.get("overall") or {}
    player_view = rating_system.get("player_view") or {}
    coach_view = rating_system.get("coach_view") or {}
    confidence = rating_system.get("confidence") or {}
    metrics = player_view.get("metrics") or {}
    coach_metrics = coach_view.get("metrics") or {}

    def metric(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = raw or {}
        value = raw.get("score")
        score = int(value) if isinstance(value, (int, float)) else None
        return {
            "score": score,
            "band": _score_band(score),
            "label": raw.get("label"),
        }

    overall_score = overall.get("score")
    if not isinstance(overall_score, (int, float)):
        return None

    return {
        "overall": metric(overall),
        "upper_body_alignment": metric(metrics.get("upper_body_alignment")),
        "lower_body_alignment": metric(metrics.get("lower_body_alignment")),
        "whole_body_alignment": metric(metrics.get("whole_body_alignment")),
        "momentum_forward": metric(metrics.get("momentum_forward")),
        "safety": metric(coach_metrics.get("safety")),
        "confidence": metric(confidence),
    }


def _extract_action_change_traits(result_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rating_summary = _extract_rating_summary_v2(result_json) or {}
    action = (result_json or {}).get("action") or {}
    basics = (result_json or {}).get("basics") or {}
    risks = (result_json or {}).get("risks") or []
    toe_alignment = basics.get("front_foot_toe_alignment") or {}
    toe_debug = toe_alignment.get("debug") or {}
    risks_by_id = {
        str(risk.get("risk_id") or ""): risk
        for risk in risks
        if isinstance(risk, dict) and risk.get("risk_id")
    }
    hip_shoulder_risk = risks_by_id.get("hip_shoulder_mismatch") or {}
    hip_shoulder_debug = hip_shoulder_risk.get("debug") or {}

    def _numeric_metric(key: str) -> Optional[int]:
        metric = rating_summary.get(key) or {}
        value = metric.get("score")
        return int(value) if isinstance(value, (int, float)) else None

    toe_angle = toe_debug.get("toe_angle_deg")
    numeric: Dict[str, Optional[float]] = {
        "overall": _numeric_metric("overall"),
        "upper_body_alignment": _numeric_metric("upper_body_alignment"),
        "lower_body_alignment": _numeric_metric("lower_body_alignment"),
        "whole_body_alignment": _numeric_metric("whole_body_alignment"),
        "momentum_forward": _numeric_metric("momentum_forward"),
        "front_foot_toe_angle_deg": round(float(toe_angle), 2)
        if isinstance(toe_angle, (int, float))
        else None,
        "kinetic_chain_delta_frames": float(hip_shoulder_debug.get("sequence_delta_frames"))
        if isinstance(hip_shoulder_debug.get("sequence_delta_frames"), (int, float))
        else None,
    }
    for risk_id in ACTION_CHANGE_RISK_IDS:
        risk = risks_by_id.get(risk_id) or {}
        signal = risk.get("signal_strength")
        numeric[f"risk_{risk_id}"] = (
            round(float(signal), 3) if isinstance(signal, (int, float)) else None
        )

    return {
        "numeric": numeric,
        "categorical": {
            "action_type": str((action.get("action") or "unknown")).lower(),
            "action_intent": str((action.get("intent") or "unknown")).lower(),
            "front_foot_toe_alignment": str((toe_alignment.get("status") or "unknown")).lower(),
            "kinetic_chain_sequence": str((hip_shoulder_debug.get("sequence_pattern") or "unknown")).lower(),
        },
        "confidence": {
            "action_confidence": round(float(action.get("confidence") or 0.0), 2),
            "front_foot_toe_alignment_confidence": round(float(toe_alignment.get("confidence") or 0.0), 2),
        },
    }


def _extract_visual_walkthrough(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    walkthrough = result_json.get("visual_walkthrough")
    return walkthrough if isinstance(walkthrough, dict) else None


def _analysis_is_baseline_eligible(result_json: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(result_json, dict):
        return False
    action_confidence = float(((result_json.get("action") or {}).get("confidence") or 0.0))
    chain_quality = float((((result_json.get("events") or {}).get("event_chain") or {}).get("quality") or 0.0))
    if action_confidence < 0.35:
        return False
    if chain_quality < 0.20:
        return False
    return True


def _build_player_baseline_state(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_entries = [entry for entry in run_entries if isinstance(entry.get("result_json"), dict)]
    trusted_entries = [
        entry for entry in valid_entries
        if _analysis_is_baseline_eligible(entry.get("result_json"))
    ]
    if not trusted_entries:
        return {
            "version": "player_baseline_v1",
            "status": "collecting",
            "should_refresh_baseline": False,
            "headline": "ActionLab is still collecting trusted clips for your personal baseline.",
            "summary": "Keep recording a few more clear clips before ActionLab updates what it treats as your normal action pattern.",
            "recent_trusted_run_ids": [],
            "reference_run_ids": [],
            "trigger_reason": "no_trusted_clips",
        }

    recent_entries = trusted_entries[:BASELINE_REFRESH_WINDOW]
    older_entries = trusted_entries[BASELINE_REFRESH_WINDOW:BASELINE_REFRESH_WINDOW * 2]
    recent_traits = [
        _extract_action_change_traits(entry.get("result_json"))
        for entry in recent_entries
    ]
    recent_action_types = [
        str((traits.get("categorical") or {}).get("action_type") or "")
        for traits in recent_traits
        if str((traits.get("categorical") or {}).get("action_type") or "") not in {"", "unknown"}
    ]
    recent_mode = None
    recent_mode_ratio = 0.0
    if recent_action_types:
        counts = Counter(recent_action_types)
        recent_mode, recent_mode_count = counts.most_common(1)[0]
        recent_mode_ratio = recent_mode_count / len(recent_action_types)

    older_action_types = [
        str((_extract_action_change_traits(entry.get("result_json")).get("categorical") or {}).get("action_type") or "")
        for entry in older_entries
        if str((_extract_action_change_traits(entry.get("result_json")).get("categorical") or {}).get("action_type") or "") not in {"", "unknown"}
    ]
    older_mode = None
    if older_action_types:
        older_mode = Counter(older_action_types).most_common(1)[0][0]

    latest_created_at = recent_entries[0].get("created_at")
    oldest_recent_created_at = recent_entries[-1].get("created_at")
    long_gap = False
    if latest_created_at and oldest_recent_created_at:
        try:
            long_gap = abs((latest_created_at - oldest_recent_created_at).days) >= BASELINE_REVIEW_GAP_DAYS
        except Exception:
            long_gap = False

    if len(recent_entries) < BASELINE_REFRESH_MIN_RUNS:
        status = "collecting"
        headline = "ActionLab is still collecting trusted clips for your personal baseline."
        summary = "Need a few more strong clips before ActionLab updates your normal action pattern."
        trigger_reason = "not_enough_recent_trusted_runs"
        should_refresh = False
    elif older_mode and recent_mode and recent_mode != older_mode and recent_mode_ratio >= 0.75:
        status = "refresh_candidate"
        headline = "Your recent action pattern may be becoming the new normal."
        summary = (
            f"The last {len(recent_entries)} trusted clips are clustering around a {recent_mode.replace('_', ' ')} action, "
            f"which is different from the older baseline pattern."
        )
        trigger_reason = "sustained_action_type_shift"
        should_refresh = True
    elif long_gap:
        status = "review_due"
        headline = "Your personal baseline may need a fresh review."
        summary = "These trusted clips are spread across a long time gap, so ActionLab should review whether your current normal has changed."
        trigger_reason = "long_time_gap"
        should_refresh = False
    else:
        status = "stable"
        headline = "Your current personal baseline still looks usable."
        summary = "Recent trusted clips are still close enough to the current pattern, so ActionLab can keep using the same normal for now."
        trigger_reason = "recent_pattern_stable"
        should_refresh = False

    return {
        "version": "player_baseline_v1",
        "status": status,
        "should_refresh_baseline": should_refresh,
        "headline": headline,
        "summary": summary,
        "recent_trusted_run_ids": [entry.get("run_id") for entry in recent_entries],
        "reference_run_ids": [entry.get("run_id") for entry in older_entries],
        "recent_action_type_mode": recent_mode,
        "recent_action_type_mode_ratio": round(float(recent_mode_ratio), 2) if recent_mode_ratio else 0.0,
        "trigger_reason": trigger_reason,
    }


def _compare_numeric_action_trait(metric_key: str, current_value: float, baseline_values: List[float]) -> Dict[str, Any]:
    sorted_values = sorted(float(v) for v in baseline_values)
    median = sorted_values[len(sorted_values) // 2]
    spread = max(abs(v - median) for v in sorted_values) if sorted_values else 0.0
    rules = ACTION_CHANGE_NUMERIC_RULES[metric_key]
    allowed_delta = max(float(rules["min_range"]), spread + float(rules["spread_cushion"]))
    watch_limit = allowed_delta + float(rules["watch_buffer"])
    delta = float(current_value) - float(median)
    abs_delta = abs(delta)

    if abs_delta <= allowed_delta:
        status = "within_range"
    elif abs_delta <= watch_limit:
        status = "watch_change"
    else:
        status = "clear_change"

    return {
        "metric": metric_key,
        "label": ACTION_CHANGE_LABELS[metric_key],
        "type": "numeric",
        "status": status,
        "current": round(float(current_value), 2),
        "baseline_median": round(float(median), 2),
        "baseline_min": round(float(sorted_values[0]), 2),
        "baseline_max": round(float(sorted_values[-1]), 2),
        "allowed_delta": round(float(allowed_delta), 2),
        "watch_limit": round(float(watch_limit), 2),
        "delta_from_baseline": round(float(delta), 2),
        "direction": "higher" if delta > 0 else "lower" if delta < 0 else "steady",
    }


def _compare_categorical_action_trait(metric_key: str, current_value: str, baseline_values: List[str]) -> Dict[str, Any]:
    counts = Counter(baseline_values)
    mode_value, mode_count = counts.most_common(1)[0]
    baseline_size = len(baseline_values)
    mode_ratio = mode_count / baseline_size if baseline_size else 0.0
    current_matches_mode = current_value == mode_value
    seen_before = current_value in counts

    if current_matches_mode:
        status = "within_range"
    elif len(counts) == 1 or not seen_before or mode_ratio >= 0.75:
        status = "clear_change"
    else:
        status = "watch_change"

    return {
        "metric": metric_key,
        "label": ACTION_CHANGE_LABELS[metric_key],
        "type": "categorical",
        "status": status,
        "current": current_value,
        "baseline_mode": mode_value,
        "baseline_values": baseline_values,
        "mode_ratio": round(float(mode_ratio), 2),
    }


def _human_join(items: List[str]) -> str:
    filtered = [item for item in items if item]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return f"{', '.join(filtered[:-1])}, and {filtered[-1]}"


def _action_change_priority(metric_key: str) -> int:
    try:
        return ACTION_CHANGE_SUMMARY_PRIORITY.index(metric_key)
    except ValueError:
        return len(ACTION_CHANGE_SUMMARY_PRIORITY)


def _plain_change_message(item: Dict[str, Any]) -> str:
    metric = str(item.get("metric") or "")
    current = str(item.get("current") or "")
    baseline_mode = str(item.get("baseline_mode") or "")
    direction = str(item.get("direction") or "")

    if metric == "kinetic_chain_sequence":
        if current == "shoulders_lead":
            return "Shoulders are opening earlier than usual."
        if current == "hips_lead":
            return "Hips are opening earlier than usual."
        if current == "in_sync":
            return "Hips and shoulders are moving together again."
    if metric == "risk_hip_shoulder_mismatch":
        return "Hips and shoulders are less in sync than usual."
    if metric == "risk_trunk_rotation_snap":
        return "Upper body is turning more sharply than usual."
    if metric == "risk_knee_brace_failure":
        return "Front leg is softer at landing than usual."
    if metric == "risk_front_foot_braking_shock":
        return "Landing is taking a harder hit than usual."
    if metric == "risk_lateral_trunk_lean":
        return "Body is leaning more to the side than usual."
    if metric == "risk_foot_line_deviation":
        return "Front foot is landing wider than usual."
    if metric == "lower_body_alignment":
        return "Lower body is less lined up than usual."
    if metric == "upper_body_alignment":
        return "Upper body is less steady than usual."
    if metric == "whole_body_alignment":
        return "The whole action looks less connected than usual."
    if metric == "momentum_forward":
        return "The action is carrying through the ball less well than usual."
    if metric == "front_foot_toe_alignment":
        if current == "open":
            return "Front foot is opening out more than usual."
        if current == "semi_open":
            return "Front foot is opening out a bit more than usual."
    if metric == "action_type":
        if current == "front_on":
            return "The action shape looks more front-on than usual."
        if current == "mixed":
            return "The action shape looks more mixed than usual."
        if current == "side_on":
            return "The action shape looks more side-on than usual."
        if current == "semi_open":
            return "The action shape looks more open than usual."
    if metric == "action_intent":
        if current == "front_on":
            return "The body setup is opening earlier than usual."
        if current == "side_on":
            return "The body setup is staying more side-on than usual."
        if current == "semi_open":
            return "The body setup is a bit more open than usual."
    if metric == "front_foot_toe_angle_deg":
        return "Front foot direction has changed from the usual landing pattern."
    if metric == "overall":
        if direction == "lower":
            return "Overall action quality is down from the usual level."
        if direction == "higher":
            return "Overall action quality is up from the usual level."
    if metric == "kinetic_chain_delta_frames":
        if direction == "lower":
            return "Upper body is getting ahead of the hips more than usual."
        if direction == "higher":
            return "Hips are getting ahead of the upper body more than usual."

    if baseline_mode:
        return f"{ACTION_CHANGE_LABELS.get(metric, metric)} has changed from the usual pattern."
    return f"{ACTION_CHANGE_LABELS.get(metric, metric)} has shifted from the usual pattern."


def _plain_stable_message(item: Dict[str, Any]) -> str:
    metric = str(item.get("metric") or "")
    if metric == "kinetic_chain_sequence":
        return "Hip and shoulder timing still looks like the usual pattern."
    if metric == "risk_hip_shoulder_mismatch":
        return "Hips and shoulders are working together like they usually do."
    if metric == "risk_knee_brace_failure":
        return "Front leg support looks like the usual pattern."
    if metric == "risk_trunk_rotation_snap":
        return "Upper-body turn looks like the usual pattern."
    if metric == "risk_lateral_trunk_lean":
        return "Body height and balance look like the usual pattern."
    if metric == "risk_foot_line_deviation":
        return "Front-foot landing line looks like the usual pattern."
    if metric == "whole_body_alignment":
        return "The whole action still looks connected."
    if metric == "momentum_forward":
        return "The action is still carrying through the ball well."
    return f"{ACTION_CHANGE_LABELS.get(metric, metric)} is still within the usual range."


def _fix_tip_for_action_change(item: Dict[str, Any]) -> str:
    metric = str(item.get("metric") or "")
    current = str(item.get("current") or "")

    if metric == "kinetic_chain_sequence":
        if current == "shoulders_lead":
            return "Try to let the hips lead the move into the ball instead of pulling the shoulders around early."
        if current == "hips_lead":
            return "Try to let the upper body follow the hips smoothly instead of getting left behind."
        return "Keep the hips and shoulders moving together like they do in your better clips."
    if metric == "risk_hip_shoulder_mismatch":
        return "Try to keep the hips and shoulders working together through the action."
    if metric == "risk_trunk_rotation_snap":
        return "Try to let the upper body turn more smoothly instead of whipping around sharply."
    if metric == "risk_knee_brace_failure":
        return "Try to keep the front leg firmer when it lands."
    if metric == "risk_front_foot_braking_shock":
        return "Try to land and keep moving through the ball instead of taking a hard stop."
    if metric == "risk_lateral_trunk_lean":
        return "Try to stay taller instead of falling away to the side."
    if metric == "risk_foot_line_deviation":
        return "Try to line up the back foot and front foot so they stay balanced and in line."
    if metric == "lower_body_alignment":
        return "Try to keep the lower body more lined up through the action."
    if metric == "upper_body_alignment":
        return "Try to keep the upper body steadier through release."
    if metric == "whole_body_alignment":
        return "Try to keep the whole action connected from landing into release."
    if metric == "momentum_forward":
        return "Try to keep the action moving through the ball with the same flow each time."
    if metric == "front_foot_toe_alignment":
        return "Try to line up the back foot and front foot so they stay balanced and in line."
    if metric == "front_foot_toe_angle_deg":
        return "Try to line up the back foot and front foot so they stay balanced and in line."
    if metric == "action_type":
        return "Try to repeat the same action shape on each ball instead of letting it drift."
    if metric == "action_intent":
        return "Try to set up the body the same way each time before the ball comes out."
    if metric == "overall":
        return "Try to repeat the same simple shape that shows up in your better recent clips."
    if metric == "kinetic_chain_delta_frames":
        return "Try to let the whole body work in sequence instead of one part rushing ahead."

    return "Try to repeat the same shape from your better recent clips."


def _coach_prompt_for_action_change(item: Dict[str, Any]) -> str:
    metric = str(item.get("metric") or "")

    if metric == "kinetic_chain_sequence":
        return "If you can, show these clips to a coach and ask whether the shoulders are opening too early."
    if metric == "risk_hip_shoulder_mismatch":
        return "If you can, show these clips to a coach and ask whether the hips and shoulders are staying connected."
    if metric == "risk_trunk_rotation_snap":
        return "If you can, show these clips to a coach and ask whether the upper-body turn looks too sharp."
    if metric == "risk_knee_brace_failure":
        return "If you can, show these clips to a coach and ask whether the front leg is holding its shape at landing."
    if metric == "risk_front_foot_braking_shock":
        return "If you can, show these clips to a coach and ask whether the bowler is stopping too hard at landing."
    if metric == "risk_lateral_trunk_lean":
        return "If you can, show these clips to a coach and ask whether the body is falling away to the side."
    if metric == "risk_foot_line_deviation":
        return "If you can, show these clips to a coach and ask whether the front foot is landing too wide."
    if metric in {"lower_body_alignment", "upper_body_alignment", "whole_body_alignment"}:
        return "If you can, show these clips to a coach and ask which part of the action is drifting out of shape."
    if metric == "momentum_forward":
        return "If you can, show these clips to a coach and ask whether the action is losing flow into the ball."
    if metric in {"action_type", "action_intent"}:
        return "If you can, show these clips to a coach and ask whether the action shape is drifting."

    return "If you can, show these clips to a coach and check whether this change keeps repeating."


def _headline_for_action_change(status: str, top_items: List[Dict[str, Any]]) -> str:
    if status == "within_range":
        return "Your action looks like your usual pattern."
    if status == "watch_change":
        return "A few parts of the action are starting to change."
    if status == "clear_change":
        if top_items:
            return top_items[0].get("plain_message") or "Your action has changed from its usual pattern."
        return "Your action has changed from its usual pattern."
    return "Action change needs a few more clips before we can judge it well."


def _summary_for_action_change(status: str, top_items: List[Dict[str, Any]]) -> str:
    if status == "within_range":
        stable_bits = [item.get("plain_message") for item in top_items[:2]]
        detail = _human_join([bit.rstrip(".") for bit in stable_bits if isinstance(bit, str)])
        if detail:
            return f"{detail}."
        return "Latest action is still within the bowler's normal recent range."
    if status in {"watch_change", "clear_change"}:
        changed = [item.get("plain_message") for item in top_items[:3]]
        detail = _human_join([bit.rstrip(".") for bit in changed if isinstance(bit, str)])
        if detail:
            return detail + "."
        return "Latest action has moved away from the recent pattern."
    return "Need at least 2 recent comparison clips before ActionLab can track action change reliably."


def _top_level_guidance_for_action_change(status: str, top_items: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if status == "within_range":
        return {
            "what_to_try": "Keep repeating the same simple action shape from your better recent clips.",
            "coach_prompt": "If you can, keep a coach updated with a few clips over time so small changes do not get missed.",
        }
    if status in {"watch_change", "clear_change"} and top_items:
        lead = top_items[0]
        return {
            "what_to_try": _fix_tip_for_action_change(lead),
            "coach_prompt": _coach_prompt_for_action_change(lead),
        }
    return {
        "what_to_try": "Keep recording a few more clips before changing anything major.",
        "coach_prompt": "If you can, show a coach a few recent clips together instead of judging from one ball.",
    }


def _build_action_change_summary(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest_entries = [entry for entry in run_entries if isinstance(entry.get("result_json"), dict)]
    if not latest_entries:
        return {
            "version": "action_change_v1",
            "status": "insufficient_history",
            "headline": _headline_for_action_change("insufficient_history", []),
            "summary": "No completed analysis runs are available yet for change tracking.",
            "what_to_try": "Keep recording a few more clips before changing anything major.",
            "coach_prompt": "If you can, show a coach a few recent clips together instead of judging from one ball.",
            "baseline_window": ACTION_CHANGE_BASELINE_WINDOW,
            "baseline_sample_size": 0,
            "latest_run_id": None,
            "baseline_run_ids": [],
            "comparisons": [],
            "highlights": [],
        }

    latest = latest_entries[0]
    baseline_entries = latest_entries[1:1 + ACTION_CHANGE_BASELINE_WINDOW]
    latest_traits = _extract_action_change_traits(latest.get("result_json"))
    baseline_traits = [
        _extract_action_change_traits(entry.get("result_json"))
        for entry in baseline_entries
    ]

    comparisons: List[Dict[str, Any]] = []
    for metric_key, current_value in (latest_traits.get("numeric") or {}).items():
        if current_value is None:
            continue
        baseline_values = [
            traits_value
            for traits_value in ((traits.get("numeric") or {}).get(metric_key) for traits in baseline_traits)
            if isinstance(traits_value, (int, float))
        ]
        if len(baseline_values) < ACTION_CHANGE_MIN_BASELINE_RUNS:
            continue
        comparisons.append(
            _compare_numeric_action_trait(metric_key, float(current_value), [float(v) for v in baseline_values])
        )

    for metric_key, current_value in (latest_traits.get("categorical") or {}).items():
        if not current_value or current_value == "unknown":
            continue
        baseline_values = [
            traits_value
            for traits_value in ((traits.get("categorical") or {}).get(metric_key) for traits in baseline_traits)
            if traits_value and traits_value != "unknown"
        ]
        if len(baseline_values) < ACTION_CHANGE_MIN_BASELINE_RUNS:
            continue
        comparisons.append(
            _compare_categorical_action_trait(metric_key, str(current_value), [str(v) for v in baseline_values])
        )

    highlights = [
        item
        for item in comparisons
        if item.get("status") in {"watch_change", "clear_change"}
    ]
    highlights.sort(
        key=lambda item: (
            0 if item.get("status") == "clear_change" else 1,
            -abs(float(item.get("delta_from_baseline") or 0.0))
            if item.get("type") == "numeric"
            else -float(item.get("mode_ratio") or 0.0),
        )
    )
    for item in comparisons:
        item["plain_message"] = _plain_change_message(item)
        item["what_to_try"] = _fix_tip_for_action_change(item)
        item["coach_prompt"] = _coach_prompt_for_action_change(item)
    for item in highlights:
        item["plain_message"] = _plain_change_message(item)
        item["what_to_try"] = _fix_tip_for_action_change(item)
        item["coach_prompt"] = _coach_prompt_for_action_change(item)

    if len(baseline_entries) < ACTION_CHANGE_MIN_BASELINE_RUNS or not comparisons:
        status = "insufficient_history"
        headline = _headline_for_action_change("insufficient_history", [])
        summary = _summary_for_action_change("insufficient_history", [])
        guidance = _top_level_guidance_for_action_change("insufficient_history", [])
    elif not highlights:
        status = "within_range"
        stable_items = sorted(
            comparisons,
            key=lambda item: _action_change_priority(str(item.get("metric") or "")),
        )
        for item in stable_items:
            item["plain_message"] = _plain_stable_message(item)
            item["what_to_try"] = _fix_tip_for_action_change(item)
            item["coach_prompt"] = _coach_prompt_for_action_change(item)
        headline = _headline_for_action_change(status, stable_items[:2])
        summary = _summary_for_action_change(status, stable_items[:2])
        guidance = _top_level_guidance_for_action_change(status, stable_items[:2])
    else:
        prioritized_highlights = sorted(
            highlights,
            key=lambda item: (
                0 if item.get("status") == "clear_change" else 1,
                _action_change_priority(str(item.get("metric") or "")),
            ),
        )
        if any(item.get("status") == "clear_change" for item in prioritized_highlights):
            status = "clear_change"
        else:
            status = "watch_change"
        headline = _headline_for_action_change(status, prioritized_highlights)
        summary = _summary_for_action_change(status, prioritized_highlights)
        highlights = prioritized_highlights
        guidance = _top_level_guidance_for_action_change(status, prioritized_highlights)

    return {
        "version": "action_change_v1",
        "status": status,
        "headline": headline,
        "summary": summary,
        "what_to_try": guidance.get("what_to_try"),
        "coach_prompt": guidance.get("coach_prompt"),
        "baseline_window": ACTION_CHANGE_BASELINE_WINDOW,
        "baseline_sample_size": len(baseline_entries),
        "latest_run_id": latest.get("run_id"),
        "baseline_run_ids": [entry.get("run_id") for entry in baseline_entries],
        "comparisons": comparisons,
        "highlights": highlights,
    }


def _build_score_heatmap(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        ("overall", "Overall"),
        ("balance", "Balance"),
        ("carry", "Carry"),
        ("body_load", "Body Load"),
        ("confidence", "Confidence"),
    ]

    rows = []
    for metric_key, label in metrics:
        cells = []
        for entry in run_entries:
            summary = entry.get("score_summary") or {}
            metric = summary.get(metric_key) or {}
            cells.append(
                {
                    "run_id": entry["run_id"],
                    "created_at": entry["created_at"],
                    "score": metric.get("score"),
                    "band": metric.get("band") or "unknown",
                }
            )
        rows.append(
            {
                "metric": metric_key,
                "label": label,
                "cells": cells,
            }
        )

    return {
        "version": "score_heatmap_v1",
        "headline": "Score trend over recent runs",
        "latest_first": True,
        "rows": rows,
    }


def _build_rating_heatmap_v2(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        ("overall", "Overall"),
        ("upper_body_alignment", "Upper Body"),
        ("lower_body_alignment", "Lower Body"),
        ("whole_body_alignment", "Whole Body"),
        ("momentum_forward", "Momentum"),
        ("safety", "Safety"),
        ("confidence", "Confidence"),
    ]

    rows = []
    for metric_key, label in metrics:
        cells = []
        for entry in run_entries:
            summary = entry.get("rating_summary_v2") or {}
            metric = summary.get(metric_key) or {}
            cells.append(
                {
                    "run_id": entry["run_id"],
                    "created_at": entry["created_at"],
                    "score": metric.get("score"),
                    "band": metric.get("band") or "unknown",
                }
            )
        rows.append({"metric": metric_key, "label": label, "cells": cells})

    return {
        "version": "rating_heatmap_v2",
        "headline": "How the action is moving over recent runs",
        "latest_first": True,
        "rows": rows,
    }


# ---------------------------------------------------------------------
# Models (Pydantic v2 compatible)
# ---------------------------------------------------------------------

class PlayerCreateRequest(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    handedness: str = Field(..., pattern="^(R|L)$")
    age_group: str = Field(..., pattern="^(U10|U14|U16|U19|SENIOR)$")
    season: int = Field(..., ge=2000, le=2100)


class PlayerProfileUpdate(BaseModel):
    age_group: Optional[str] = Field(None, pattern="^(U10|U14|U16|U19|SENIOR)$")
    season: Optional[int] = Field(None, ge=2000, le=2100)


# ---------------------------------------------------------------------
# Players (Auth Scoped)
# ---------------------------------------------------------------------

@router.get("/players")
def list_players(
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    links = (
        db.query(AccountPlayerLink)
        .filter_by(account_id=current_account.account_id)
        .all()
    )

    player_ids = [link.player_id for link in links]
    players_by_id = {}
    if player_ids:
        player_rows = db.query(Player).filter(Player.player_id.in_(player_ids)).all()
        players_by_id = {p.player_id: p for p in player_rows}

    players = []
    for link in links:
        player = players_by_id.get(link.player_id)
        players.append({
            "player_id": str(link.player_id),
            "player_name": link.player_name,
            "link_type": link.link_type,
            "age_group": player.age_group if player else None,
            "season": player.season if player else None,
        })

    return {"players": players}


@router.post("/players", status_code=201)
def create_player(
    payload: PlayerCreateRequest,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    existing = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_name=payload.player_name.strip(),
        )
        .first()
    )

    if existing:
        player = db.query(Player).filter_by(player_id=existing.player_id).first()
        return {
            "player_id": str(existing.player_id),
            "player_name": existing.player_name,
            "age_group": player.age_group,
            "season": player.season,
            "already_exists": True,
        }

    player = Player(
        primary_owner_account_id=current_account.account_id,
        created_by_account_id=current_account.account_id,
        handedness=payload.handedness,
        age_group=payload.age_group,
        season=payload.season,
    )

    db.add(player)
    db.flush()

    db.add(AccountPlayerLink(
        account_id=current_account.account_id,
        player_id=player.player_id,
        link_type="owner",
        player_name=payload.player_name.strip(),
    ))

    db.commit()

    return {
        "player_id": str(player.player_id),
        "player_name": payload.player_name,
        "age_group": player.age_group,
        "season": player.season,
    }


@router.patch("/players/{player_id}/profile")
def update_player_profile(
    player_id: str,
    payload: PlayerProfileUpdate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    player = db.query(Player).filter_by(player_id=player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    if payload.age_group is not None:
        player.age_group = payload.age_group

    if payload.season is not None:
        player.season = payload.season

    db.commit()

    return {
        "player_id": str(player.player_id),
        "age_group": player.age_group,
        "season": player.season,
    }


# ---------------------------------------------------------------------
# Analysis (READ – Auth Scoped)
# ---------------------------------------------------------------------

@router.get("/players/{player_id}/latest")
def latest_analysis(
    player_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """
    Identity-only helper for Home screen.
    NEVER returns report JSON.
    """

    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    run = (
        db.query(AnalysisRun)
        .filter_by(player_id=player_id)
        .order_by(AnalysisRun.created_at.desc())
        .first()
    )

    if not run:
        raise HTTPException(status_code=404, detail="No analysis found")

    return {
        "run_id": str(run.run_id),
        "created_at": run.created_at,
        "season": run.season,
        "age_group": run.age_group,
    }


@router.get("/players/{player_id}/analysis-runs")
def list_analysis_runs(
    player_id: str,
    season: Optional[int] = None,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    q = db.query(AnalysisRun).filter_by(player_id=player_id)

    if season is not None:
        q = q.filter_by(season=season)

    runs = q.order_by(AnalysisRun.created_at.desc()).all()
    run_ids = [r.run_id for r in runs]
    raw_by_run_id: Dict[Any, AnalysisResultRaw] = {}
    if run_ids:
        raw_rows = db.query(AnalysisResultRaw).filter(AnalysisResultRaw.run_id.in_(run_ids)).all()
        raw_by_run_id = {row.run_id: row for row in raw_rows}

    items = []
    heatmap_entries = []
    for r in runs:
        raw = raw_by_run_id.get(r.run_id)
        score_summary = _extract_score_summary(raw.result_json if raw else None)
        rating_summary_v2 = _extract_rating_summary_v2(raw.result_json if raw else None)
        item = {
            "run_id": str(r.run_id),
            "created_at": r.created_at,
            "season": r.season,
            "age_group": r.age_group,
            "schema_version": r.schema_version,
            "fps": r.fps,
            "total_frames": r.total_frames,
            "score_summary": score_summary,
            "rating_summary_v2": rating_summary_v2,
            "visual_walkthrough": _extract_visual_walkthrough(raw.result_json if raw else None),
        }
        items.append(item)
        heatmap_entries.append(
            {
                "run_id": str(r.run_id),
                "created_at": r.created_at,
                "score_summary": score_summary,
                "rating_summary_v2": rating_summary_v2,
            }
        )

    return {
        "items": items,
        "heatmap": _build_score_heatmap(heatmap_entries),
        "heatmap_v2": _build_rating_heatmap_v2(heatmap_entries),
        "action_change": _build_action_change_summary(
            [
                {
                    "run_id": str(r.run_id),
                    "created_at": r.created_at,
                    "result_json": (raw_by_run_id.get(r.run_id).result_json if raw_by_run_id.get(r.run_id) else None),
                }
                for r in runs
            ]
        ),
        "baseline_state": _build_player_baseline_state(
            [
                {
                    "run_id": str(r.run_id),
                    "created_at": r.created_at,
                    "result_json": (raw_by_run_id.get(r.run_id).result_json if raw_by_run_id.get(r.run_id) else None),
                }
                for r in runs
            ]
        ),
    }


@router.get("/analysis-runs/{run_id}")
def get_analysis_run(
    run_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    run = db.query(AnalysisRun).filter_by(run_id=run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # 🔐 Ownership check
    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=run.player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    raw = db.query(AnalysisResultRaw).filter_by(run_id=run_id).first()

    return {
        "run_id": str(run.run_id),
        "player_id": str(run.player_id),
        "season": run.season,
        "age_group": run.age_group,
        "created_at": run.created_at,
        "schema_version": run.schema_version,
        "coach_notes": run.coach_notes,
        "visual_walkthrough": _extract_visual_walkthrough(raw.result_json if raw else None),
        "result": raw.result_json if raw else None,
    }
