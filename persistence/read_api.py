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
import os
import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.clinician.knowledge_pack import load_knowledge_pack
from app.persistence.session import get_db
from app.persistence.models import (
    Player,
    AccountPlayerLink,
    AnalysisRun,
    AnalysisResultRaw,
    AnalysisExplanationTrace,
    KnowledgePackMonitoringSnapshot,
    KnowledgePackRegressionCaseResult,
    KnowledgePackRegressionRun,
    KnowledgePackReleaseCandidate,
    KnowledgePackReleaseEvent,
    KnowledgePackRollbackAlert,
    LearningCase,
    LearningCaseCluster,
    CoachFlag,
    PrescriptionFollowup,
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
    if not isinstance(walkthrough, dict):
        return None

    normalized = dict(walkthrough)
    raw_url = str(normalized.get("url") or "").strip()
    relative_url = str(normalized.get("relative_url") or raw_url or "").strip()
    if not relative_url:
        return normalized
    if relative_url.startswith("http://") or relative_url.startswith("https://"):
        return normalized
    if not relative_url.startswith("/"):
        relative_url = f"/{relative_url}"

    base_url = (os.getenv("ACTIONLAB_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    normalized["relative_url"] = relative_url
    normalized["url"] = f"{base_url}{relative_url}" if base_url else relative_url
    return normalized


def _extract_kinetic_chain_summary(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    kinetic_chain = result_json.get("kinetic_chain_v1")
    explanation = result_json.get("mechanism_explanation_v1")
    prescription_plan = result_json.get("prescription_plan_v1")
    render_reasoning = result_json.get("render_reasoning_v1")
    root_cause_summary = _extract_root_cause_summary(result_json)
    if not isinstance(kinetic_chain, dict):
        return None

    prescription_title = None
    if isinstance(prescription_plan, dict):
        prescriptions = prescription_plan.get("prescriptions") or []
        if prescriptions and isinstance(prescriptions[0], dict):
            prescription_title = prescriptions[0].get("title")

    def phase_summary(key: str) -> Optional[Dict[str, Any]]:
        raw = kinetic_chain.get(key) or {}
        score = raw.get("score")
        return {
            "score": round(float(score), 3),
            "label": raw.get("label"),
        } if isinstance(score, (int, float)) else None

    return {
        "diagnosis_status": kinetic_chain.get("diagnosis_status"),
        "confidence": kinetic_chain.get("confidence"),
        "archetype": (kinetic_chain.get("archetype") or {}).get("short_label")
        if isinstance(kinetic_chain.get("archetype"), dict)
        else None,
        "primary_mechanism": (explanation or {}).get("primary_mechanism") if isinstance(explanation, dict) else None,
        "first_intervention": (explanation or {}).get("first_intervention") if isinstance(explanation, dict) else None,
        "primary_prescription_title": prescription_title,
        "root_cause": root_cause_summary,
        "renderer_mode": (
            render_reasoning.get("renderer_mode")
            if isinstance(render_reasoning, dict)
            else None
        ),
        "approach_build": phase_summary("approach_build"),
        "transfer": phase_summary("transfer"),
        "block": phase_summary("block"),
        "dissipation": phase_summary("dissipation"),
        "pace_translation": {
            "approach_momentum": ((kinetic_chain.get("pace_translation") or {}).get("approach_momentum")),
            "transfer_efficiency": ((kinetic_chain.get("pace_translation") or {}).get("transfer_efficiency")),
            "terminal_impulse": ((kinetic_chain.get("pace_translation") or {}).get("terminal_impulse")),
            "leakage_before_block": ((kinetic_chain.get("pace_translation") or {}).get("leakage_before_block")),
            "leakage_at_block": ((kinetic_chain.get("pace_translation") or {}).get("leakage_at_block")),
            "late_arm_chase": ((kinetic_chain.get("pace_translation") or {}).get("late_arm_chase")),
            "dissipation_burden": ((kinetic_chain.get("pace_translation") or {}).get("dissipation_burden")),
        },
    }


def _extract_root_cause_summary(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    coach_diagnosis = result_json.get("coach_diagnosis_v1") or {}
    root_cause = coach_diagnosis.get("root_cause")
    if not isinstance(root_cause, dict):
        presentation_payload = result_json.get("presentation_payload_v1") or {}
        for key in ("match", "ambiguous", "no_match"):
            payload = presentation_payload.get(key) or {}
            if isinstance(payload, dict) and isinstance(payload.get("root_cause"), dict):
                root_cause = payload.get("root_cause")
                break
    if not isinstance(root_cause, dict):
        frontend_surface = result_json.get("frontend_surface_v1") or {}
        hero = frontend_surface.get("hero") or {}
        if isinstance(hero, dict):
            root_cause = hero.get("root_cause")
    if not isinstance(root_cause, dict):
        return None

    primary_driver = root_cause.get("primary_driver") or {}
    compensation = root_cause.get("compensation") or {}
    renderer_guidance = root_cause.get("renderer_guidance") or {}
    where_it_starts = root_cause.get("where_it_starts") or {}
    return {
        "status": root_cause.get("status"),
        "mechanism_id": root_cause.get("mechanism_id"),
        "title": root_cause.get("title"),
        "summary": root_cause.get("summary"),
        "why_it_is_happening": root_cause.get("why_it_is_happening"),
        "chain_story": root_cause.get("chain_story"),
        "phase_id": where_it_starts.get("phase_id") if isinstance(where_it_starts, dict) else None,
        "primary_driver_id": primary_driver.get("id") if isinstance(primary_driver, dict) else None,
        "primary_driver_title": primary_driver.get("title") if isinstance(primary_driver, dict) else None,
        "compensation_id": compensation.get("id") if isinstance(compensation, dict) else None,
        "compensation_title": compensation.get("title") if isinstance(compensation, dict) else None,
        "render_story_id": renderer_guidance.get("story_id") if isinstance(renderer_guidance, dict) else None,
        "cue_points": list(renderer_guidance.get("cue_points") or [])[:3]
        if isinstance(renderer_guidance, dict)
        else [],
        "symptom_text": renderer_guidance.get("symptom_text") if isinstance(renderer_guidance, dict) else None,
        "load_watch_text": renderer_guidance.get("load_watch_text") if isinstance(renderer_guidance, dict) else None,
    }


def _extract_history_plan_summary(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    history_plan = result_json.get("history_plan_v1")
    if not isinstance(history_plan, dict):
        return None
    root_cause_summary = _extract_root_cause_summary(result_json)
    bindings = history_plan.get("history_bindings") or []
    binding_trends = history_plan.get("binding_trends") or []
    followup_checks = history_plan.get("followup_checks") or []
    render_stories = history_plan.get("render_stories") or []
    render_story_ids = [
        story.get("id")
        for story in render_stories
        if isinstance(story, dict) and story.get("id")
    ]
    root_cause_story_id = (
        root_cause_summary.get("render_story_id")
        if isinstance(root_cause_summary, dict)
        else None
    )
    if root_cause_story_id and root_cause_story_id not in render_story_ids:
        render_story_ids = [root_cause_story_id, *render_story_ids]
    return {
        "history_story": history_plan.get("history_story"),
        "coaching_priority": history_plan.get("coaching_priority"),
        "root_cause": root_cause_summary,
        "history_binding_ids": [
            binding.get("id")
            for binding in bindings
            if isinstance(binding, dict) and binding.get("id")
        ],
        "binding_trend_statuses": {
            str(binding.get("id")): binding.get("status")
            for binding in binding_trends
            if isinstance(binding, dict) and binding.get("id")
        },
        "followup_check_ids": [
            check.get("id")
            for check in followup_checks
            if isinstance(check, dict) and check.get("id")
        ],
        "render_story_ids": render_story_ids,
    }


def _explanation_trace_summary(trace_row: Optional[AnalysisExplanationTrace]) -> Optional[Dict[str, Any]]:
    if trace_row is None:
        return None
    return {
        "diagnosis_status": trace_row.diagnosis_status,
        "primary_mechanism_id": trace_row.primary_mechanism_id,
        "matched_symptom_ids": list(trace_row.matched_symptom_ids or []),
        "selected_render_story_ids": list(trace_row.selected_render_story_ids or []),
        "selected_history_binding_ids": list(trace_row.selected_history_binding_ids or []),
    }


def _explanation_trace_payload(trace_row: Optional[AnalysisExplanationTrace]) -> Optional[Dict[str, Any]]:
    if trace_row is None:
        return None
    return {
        "knowledge_pack_id": trace_row.knowledge_pack_id,
        "knowledge_pack_version": trace_row.knowledge_pack_version,
        "diagnosis_status": trace_row.diagnosis_status,
        "primary_mechanism_id": trace_row.primary_mechanism_id,
        "matched_symptom_ids": list(trace_row.matched_symptom_ids or []),
        "candidate_mechanisms": list(trace_row.candidate_mechanisms or []),
        "supporting_evidence": dict(trace_row.supporting_evidence or {}),
        "contradictions_triggered": list(trace_row.contradictions_triggered or []),
        "selected_trajectory_ids": list(trace_row.selected_trajectory_ids or []),
        "selected_prescription_ids": list(trace_row.selected_prescription_ids or []),
        "selected_render_story_ids": list(trace_row.selected_render_story_ids or []),
        "selected_history_binding_ids": list(trace_row.selected_history_binding_ids or []),
        "trace": dict(trace_row.explanation_trace_json or {}),
    }


def _history_uncertainty_thresholds() -> Dict[str, Any]:
    try:
        globals_cfg = load_knowledge_pack()["globals"]
    except Exception:
        return {
            "unresolved_min_runs": 3,
            "unresolved_window_runs": 8,
            "unresolved_rate_min": 0.35,
        }
    cfg = globals_cfg.get("history_uncertainty") or {}
    return {
        "unresolved_min_runs": int(cfg.get("unresolved_min_runs") or 3),
        "unresolved_window_runs": int(cfg.get("unresolved_window_runs") or 8),
        "unresolved_rate_min": float(cfg.get("unresolved_rate_min") or 0.35),
    }


def _build_history_uncertainty_summary(
    runs: List[AnalysisRun],
    runtime_learning_cases_by_run: Dict[Any, List[str]],
) -> Dict[str, Any]:
    thresholds = _history_uncertainty_thresholds()
    window_runs = max(1, int(thresholds["unresolved_window_runs"]))
    recent = list(runs[:window_runs])
    if not recent:
        return {
            "pattern_still_being_understood": False,
            "unresolved_runs": 0,
            "window_runs": 0,
            "unresolved_rate": 0.0,
        }

    unresolved_count = 0
    for run in recent:
        diagnosis_status = str(run.deterministic_diagnosis_status or "").strip().lower()
        runtime_case_statuses = runtime_learning_cases_by_run.get(run.run_id) or []
        if diagnosis_status in {"no_match", "ambiguous_match", "weak_match"}:
            unresolved_count += 1
            continue
        if any(status in {"OPEN", "CLUSTERED", "QUEUED", "UNDER_REVIEW"} for status in runtime_case_statuses):
            unresolved_count += 1

    rate = unresolved_count / len(recent)
    return {
        "pattern_still_being_understood": (
            unresolved_count >= int(thresholds["unresolved_min_runs"])
            and rate >= float(thresholds["unresolved_rate_min"])
        ),
        "unresolved_runs": unresolved_count,
        "window_runs": len(recent),
        "unresolved_rate": round(rate, 3),
    }


def _build_learning_cluster_item(
    cluster_row: LearningCaseCluster,
    *,
    case_rows: List[LearningCase],
    coach_flag_rows: List[CoachFlag],
) -> Dict[str, Any]:
    run_ids = [str(row.run_id) for row in case_rows]
    player_ids = sorted({str(row.player_id) for row in case_rows})
    account_ids = sorted(
        {
            str(row.account_id)
            for row in case_rows
            if getattr(row, "account_id", None) is not None
        }
    )
    followup_outcomes = Counter(
        str(row.followup_outcome)
        for row in case_rows
        if getattr(row, "followup_outcome", None)
    )
    return {
        "learning_case_cluster_id": str(cluster_row.learning_case_cluster_id),
        "knowledge_pack_id": cluster_row.knowledge_pack_id,
        "knowledge_pack_version": cluster_row.knowledge_pack_version,
        "source_type": cluster_row.source_type,
        "case_type": cluster_row.case_type,
        "priority": cluster_row.priority,
        "status": cluster_row.status,
        "suggested_gap_type": cluster_row.suggested_gap_type,
        "trigger_reason": cluster_row.trigger_reason,
        "symptom_bundle_hash": cluster_row.symptom_bundle_hash,
        "renderer_mode": cluster_row.renderer_mode,
        "chosen_mechanism_id": cluster_row.chosen_mechanism_id,
        "prescription_id": cluster_row.prescription_id,
        "candidate_mechanism_ids": list(cluster_row.candidate_mechanism_ids or []),
        "case_count": int(cluster_row.case_count or 0),
        "coach_flag_count": int(cluster_row.coach_flag_count or 0),
        "first_run_id": str(cluster_row.first_run_id) if cluster_row.first_run_id else None,
        "latest_run_id": str(cluster_row.latest_run_id) if cluster_row.latest_run_id else None,
        "representative_learning_case_id": (
            str(cluster_row.representative_learning_case_id)
            if cluster_row.representative_learning_case_id
            else None
        ),
        "created_at": cluster_row.created_at,
        "updated_at": cluster_row.updated_at,
        "run_ids": run_ids,
        "player_ids": player_ids,
        "account_ids": account_ids,
        "case_statuses": dict(Counter(str(row.status) for row in case_rows)),
        "followup_outcomes": dict(followup_outcomes),
        "coach_flag_types": dict(Counter(str(row.flag_type) for row in coach_flag_rows)),
        "latest_review": dict((cluster_row.cluster_payload or {}).get("latest_review") or {}),
        "cluster_payload": dict(cluster_row.cluster_payload or {}),
    }


def _render_reasoning_mode(result_json: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(result_json, dict):
        return None
    render_reasoning = result_json.get("render_reasoning_v1")
    if isinstance(render_reasoning, dict):
        mode = render_reasoning.get("renderer_mode")
        return str(mode) if isinstance(mode, str) and mode else None
    return None


def _coverage_metrics_payload(
    *,
    runs: List[AnalysisRun],
    raw_by_run_id: Dict[Any, AnalysisResultRaw],
    followups: List[PrescriptionFollowup],
    coach_flags: List[CoachFlag],
    include_breakdown: bool = True,
) -> Dict[str, Any]:
    total_runs = len(runs)
    diagnosis_counts = Counter(
        str(run.deterministic_diagnosis_status or "").strip().lower()
        for run in runs
        if getattr(run, "deterministic_diagnosis_status", None)
    )
    renderer_counts = Counter(
        mode
        for mode in (
            _render_reasoning_mode(
                raw_by_run_id.get(run.run_id).result_json if raw_by_run_id.get(run.run_id) else None
            )
            for run in runs
        )
        if mode
    )
    followup_status_counts = Counter(
        str(row.response_status or "")
        for row in followups
    )
    total_followups = len(followups)

    def rate(count: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return round(float(count) / float(denominator), 3)

    overall = {
        "total_runs": total_runs,
        "total_followups": total_followups,
        "total_coach_flags": len(coach_flags),
        "high_confidence_resolution_rate": rate(diagnosis_counts.get("confident_match", 0), total_runs),
        "partial_resolution_rate": rate(diagnosis_counts.get("partial_match", 0), total_runs),
        "no_match_rate": rate(diagnosis_counts.get("no_match", 0), total_runs),
        "ambiguity_rate": rate(diagnosis_counts.get("ambiguous_match", 0), total_runs),
        "weak_match_rate": rate(diagnosis_counts.get("weak_match", 0), total_runs),
        "prescription_non_response_rate": rate(
            followup_status_counts.get("NO_CLEAR_CHANGE", 0) + followup_status_counts.get("WORSENING", 0),
            total_followups,
        ),
        "renderer_event_only_rate": rate(renderer_counts.get("event_only", 0), total_runs),
        "coach_flag_rate": rate(len(coach_flags), total_runs),
    }

    by_pack_version: Dict[str, Dict[str, Any]] = {}
    if not include_breakdown:
        return {
            "overall": overall,
            "by_pack_version": by_pack_version,
        }
    runs_by_pack: Dict[str, List[AnalysisRun]] = {}
    for run in runs:
        pack_version = str(run.knowledge_pack_version or "unknown")
        runs_by_pack.setdefault(pack_version, []).append(run)
    followups_by_pack: Dict[str, List[PrescriptionFollowup]] = {}
    for row in followups:
        pack_version = str(row.knowledge_pack_version or "unknown")
        followups_by_pack.setdefault(pack_version, []).append(row)
    coach_flags_by_pack: Dict[str, List[CoachFlag]] = {}
    for row in coach_flags:
        pack_version = str(row.knowledge_pack_version or "unknown")
        coach_flags_by_pack.setdefault(pack_version, []).append(row)

    for pack_version, pack_runs in runs_by_pack.items():
        pack_followups = followups_by_pack.get(pack_version, [])
        pack_flags = coach_flags_by_pack.get(pack_version, [])
        pack_raw_by_run_id = {
            run.run_id: raw_by_run_id.get(run.run_id)
            for run in pack_runs
        }
        by_pack_version[pack_version] = _coverage_metrics_payload(
            runs=pack_runs,
            raw_by_run_id=pack_raw_by_run_id,
            followups=pack_followups,
            coach_flags=pack_flags,
            include_breakdown=False,
        )["overall"]

    return {
        "overall": overall,
        "by_pack_version": by_pack_version,
    }


def _build_release_candidate_item(row: KnowledgePackReleaseCandidate) -> Dict[str, Any]:
    return {
        "knowledge_pack_release_candidate_id": str(row.knowledge_pack_release_candidate_id),
        "knowledge_pack_id": row.knowledge_pack_id,
        "base_pack_version": row.base_pack_version,
        "candidate_pack_version": row.candidate_pack_version,
        "supersedes_pack_version": row.supersedes_pack_version,
        "status": row.status,
        "current_environment": row.current_environment,
        "summary": row.summary,
        "change_summary": dict(row.change_summary_json or {}),
        "motivating_cluster_ids": list(row.motivating_cluster_ids or []),
        "motivating_case_ids": list(row.motivating_case_ids or []),
        "tests_added": list(row.tests_added or []),
        "reinterpret_run_ids": list(row.reinterpret_run_ids or []),
        "schema_validated": bool(row.schema_validated),
        "referential_integrity_validated": bool(row.referential_integrity_validated),
        "regression_suite_passed": bool(row.regression_suite_passed),
        "staging_evaluation_passed": bool(row.staging_evaluation_passed),
        "approval_granted": bool(row.approval_granted),
        "created_by_account_id": str(row.created_by_account_id) if row.created_by_account_id else None,
        "updated_by_account_id": str(row.updated_by_account_id) if row.updated_by_account_id else None,
        "promoted_at": row.promoted_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _build_release_event_item(row: KnowledgePackReleaseEvent) -> Dict[str, Any]:
    return {
        "knowledge_pack_release_event_id": str(row.knowledge_pack_release_event_id),
        "knowledge_pack_release_candidate_id": str(row.knowledge_pack_release_candidate_id),
        "account_id": str(row.account_id) if row.account_id else None,
        "action": row.action,
        "from_status": row.from_status,
        "to_status": row.to_status,
        "from_environment": row.from_environment,
        "to_environment": row.to_environment,
        "notes": row.notes or "",
        "metadata": dict(row.metadata_json or {}),
        "created_at": row.created_at,
    }


def _build_regression_run_item(row: KnowledgePackRegressionRun) -> Dict[str, Any]:
    return {
        "knowledge_pack_regression_run_id": str(row.knowledge_pack_regression_run_id),
        "knowledge_pack_release_candidate_id": str(row.knowledge_pack_release_candidate_id),
        "baseline_pack_version": row.baseline_pack_version,
        "candidate_pack_version": row.candidate_pack_version,
        "status": row.status,
        "total_cases": int(row.total_cases or 0),
        "expected_change_cases": int(row.expected_change_cases or 0),
        "stable_cases": int(row.stable_cases or 0),
        "passed_cases": int(row.passed_cases or 0),
        "failed_cases": int(row.failed_cases or 0),
        "validated_regression_count": int(row.validated_regression_count or 0),
        "validated_regression_rate": float(row.validated_regression_rate or 0.0),
        "expected_change_success_count": int(row.expected_change_success_count or 0),
        "expected_change_success_rate": float(row.expected_change_success_rate or 0.0),
        "summary": dict(row.summary_json or {}),
        "created_by_account_id": str(row.created_by_account_id) if row.created_by_account_id else None,
        "created_at": row.created_at,
    }


def _build_regression_case_result_item(row: KnowledgePackRegressionCaseResult) -> Dict[str, Any]:
    return {
        "knowledge_pack_regression_case_result_id": str(row.knowledge_pack_regression_case_result_id),
        "knowledge_pack_regression_run_id": str(row.knowledge_pack_regression_run_id),
        "run_id": str(row.run_id),
        "learning_case_cluster_id": str(row.learning_case_cluster_id) if row.learning_case_cluster_id else None,
        "learning_case_id": str(row.learning_case_id) if row.learning_case_id else None,
        "expected_behavior": row.expected_behavior,
        "outcome": row.outcome,
        "baseline_pack_version": row.baseline_pack_version,
        "candidate_pack_version": row.candidate_pack_version,
        "baseline_diagnosis_status": row.baseline_diagnosis_status,
        "candidate_diagnosis_status": row.candidate_diagnosis_status,
        "baseline_primary_mechanism_id": row.baseline_primary_mechanism_id,
        "candidate_primary_mechanism_id": row.candidate_primary_mechanism_id,
        "baseline_renderer_mode": row.baseline_renderer_mode,
        "candidate_renderer_mode": row.candidate_renderer_mode,
        "reason": row.reason,
        "result": dict(row.result_json or {}),
        "created_at": row.created_at,
    }


def _build_monitoring_snapshot_item(row: KnowledgePackMonitoringSnapshot) -> Dict[str, Any]:
    return {
        "knowledge_pack_monitoring_snapshot_id": str(row.knowledge_pack_monitoring_snapshot_id),
        "knowledge_pack_release_candidate_id": str(row.knowledge_pack_release_candidate_id),
        "baseline_pack_version": row.baseline_pack_version,
        "candidate_pack_version": row.candidate_pack_version,
        "baseline_window_start": row.baseline_window_start,
        "baseline_window_end": row.baseline_window_end,
        "candidate_window_start": row.candidate_window_start,
        "candidate_window_end": row.candidate_window_end,
        "sufficient_data": bool(row.sufficient_data),
        "alert_triggered": bool(row.alert_triggered),
        "rollback_recommended": bool(row.rollback_recommended),
        "baseline_metrics": dict(row.baseline_metrics_json or {}),
        "candidate_metrics": dict(row.candidate_metrics_json or {}),
        "regression_metrics": dict(row.regression_metrics_json or {}),
        "alert_rules": dict(row.alert_rules_json or {}),
        "created_by_account_id": str(row.created_by_account_id) if row.created_by_account_id else None,
        "created_at": row.created_at,
    }


def _build_rollback_alert_item(row: KnowledgePackRollbackAlert) -> Dict[str, Any]:
    return {
        "knowledge_pack_rollback_alert_id": str(row.knowledge_pack_rollback_alert_id),
        "knowledge_pack_release_candidate_id": str(row.knowledge_pack_release_candidate_id),
        "knowledge_pack_monitoring_snapshot_id": str(row.knowledge_pack_monitoring_snapshot_id),
        "status": row.status,
        "summary": row.summary,
        "triggered_rules": dict(row.triggered_rules_json or {}),
        "resolved_at": row.resolved_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


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
    trace_by_run_id: Dict[Any, AnalysisExplanationTrace] = {}
    if run_ids:
        trace_rows = (
            db.query(AnalysisExplanationTrace)
            .filter(AnalysisExplanationTrace.run_id.in_(run_ids))
            .all()
        )
        trace_by_run_id = {row.run_id: row for row in trace_rows}
    runtime_learning_cases_by_run: Dict[Any, List[str]] = {}
    if run_ids:
        runtime_learning_cases = (
            db.query(LearningCase)
            .filter(
                LearningCase.run_id.in_(run_ids),
                LearningCase.source_type == "runtime_gap",
            )
            .all()
        )
        for row in runtime_learning_cases:
            runtime_learning_cases_by_run.setdefault(row.run_id, []).append(str(row.status))

    items = []
    heatmap_entries = []
    history_uncertainty = _build_history_uncertainty_summary(
        runs=runs,
        runtime_learning_cases_by_run=runtime_learning_cases_by_run,
    )
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
            "knowledge_pack_id": r.knowledge_pack_id,
            "knowledge_pack_version": r.knowledge_pack_version,
            "deterministic_diagnosis_status": r.deterministic_diagnosis_status,
            "deterministic_primary_mechanism_id": r.deterministic_primary_mechanism_id,
            "deterministic_archetype_id": r.deterministic_archetype_id,
            "fps": r.fps,
            "total_frames": r.total_frames,
            "score_summary": score_summary,
            "rating_summary_v2": rating_summary_v2,
            "visual_walkthrough": _extract_visual_walkthrough(raw.result_json if raw else None),
            "root_cause_summary_v1": _extract_root_cause_summary(raw.result_json if raw else None),
            "kinetic_chain_summary_v1": _extract_kinetic_chain_summary(raw.result_json if raw else None),
            "history_plan_summary_v1": _extract_history_plan_summary(raw.result_json if raw else None),
            "explanation_trace_summary_v1": _explanation_trace_summary(trace_by_run_id.get(r.run_id)),
            "history_uncertainty_flag": history_uncertainty["pattern_still_being_understood"],
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
        "history_uncertainty": history_uncertainty,
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
    trace = db.query(AnalysisExplanationTrace).filter_by(run_id=run_id).first()
    recent_runs = (
        db.query(AnalysisRun)
        .filter_by(player_id=run.player_id)
        .order_by(AnalysisRun.created_at.desc())
        .limit(_history_uncertainty_thresholds()["unresolved_window_runs"])
        .all()
    )
    recent_run_ids = [row.run_id for row in recent_runs]
    runtime_learning_cases_by_run: Dict[Any, List[str]] = {}
    if recent_run_ids:
        for case_row in (
            db.query(LearningCase)
            .filter(
                LearningCase.run_id.in_(recent_run_ids),
                LearningCase.source_type == "runtime_gap",
            )
            .all()
        ):
            runtime_learning_cases_by_run.setdefault(case_row.run_id, []).append(str(case_row.status))
    history_uncertainty = _build_history_uncertainty_summary(
        runs=recent_runs,
        runtime_learning_cases_by_run=runtime_learning_cases_by_run,
    )

    return {
        "run_id": str(run.run_id),
        "player_id": str(run.player_id),
        "season": run.season,
        "age_group": run.age_group,
        "created_at": run.created_at,
        "schema_version": run.schema_version,
        "knowledge_pack_id": run.knowledge_pack_id,
        "knowledge_pack_version": run.knowledge_pack_version,
        "deterministic_diagnosis_status": run.deterministic_diagnosis_status,
        "deterministic_primary_mechanism_id": run.deterministic_primary_mechanism_id,
        "deterministic_archetype_id": run.deterministic_archetype_id,
        "coach_notes": run.coach_notes,
        "history_uncertainty": history_uncertainty,
        "explanation_trace_v1": _explanation_trace_payload(trace),
        "visual_walkthrough": _extract_visual_walkthrough(raw.result_json if raw else None),
        "result": raw.result_json if raw else None,
    }


@router.get("/players/{player_id}/learning-case-clusters")
def list_learning_case_clusters(
    player_id: str,
    status: Optional[str] = None,
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

    cases_query = db.query(LearningCase).filter(LearningCase.player_id == player_id)
    if status:
        cases_query = cases_query.filter(LearningCase.status == status.upper())
    case_rows = cases_query.order_by(LearningCase.created_at.desc()).all()
    cluster_ids = [
        row.learning_case_cluster_id
        for row in case_rows
        if getattr(row, "learning_case_cluster_id", None) is not None
    ]
    if not cluster_ids:
        return {"items": []}

    cluster_rows = (
        db.query(LearningCaseCluster)
        .filter(LearningCaseCluster.learning_case_cluster_id.in_(cluster_ids))
        .order_by(LearningCaseCluster.updated_at.desc())
        .all()
    )
    coach_flag_rows = (
        db.query(CoachFlag)
        .filter(CoachFlag.learning_case_cluster_id.in_(cluster_ids))
        .all()
    )
    cases_by_cluster: Dict[Any, List[LearningCase]] = {}
    for row in case_rows:
        if row.learning_case_cluster_id is not None:
            cases_by_cluster.setdefault(row.learning_case_cluster_id, []).append(row)
    coach_flags_by_cluster: Dict[Any, List[CoachFlag]] = {}
    for row in coach_flag_rows:
        if row.learning_case_cluster_id is not None:
            coach_flags_by_cluster.setdefault(row.learning_case_cluster_id, []).append(row)

    return {
        "items": [
            _build_learning_cluster_item(
                cluster_row,
                case_rows=cases_by_cluster.get(cluster_row.learning_case_cluster_id, []),
                coach_flag_rows=coach_flags_by_cluster.get(cluster_row.learning_case_cluster_id, []),
            )
            for cluster_row in cluster_rows
        ]
    }


@router.get("/players/{player_id}/learning-coverage")
def get_learning_coverage(
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

    runs_query = db.query(AnalysisRun).filter(AnalysisRun.player_id == player_id)
    if season is not None:
        runs_query = runs_query.filter(AnalysisRun.season == season)
    runs = runs_query.order_by(AnalysisRun.created_at.desc()).all()
    run_ids = [row.run_id for row in runs]
    raw_by_run_id: Dict[Any, AnalysisResultRaw] = {}
    if run_ids:
        raw_rows = db.query(AnalysisResultRaw).filter(AnalysisResultRaw.run_id.in_(run_ids)).all()
        raw_by_run_id = {row.run_id: row for row in raw_rows}

    if run_ids:
        followups = (
            db.query(PrescriptionFollowup)
            .filter(
                PrescriptionFollowup.player_id == player_id,
                PrescriptionFollowup.prescription_assigned_at_run_id.in_(run_ids),
            )
            .all()
        )
        coach_flags = (
            db.query(CoachFlag)
            .filter(
                CoachFlag.player_id == player_id,
                CoachFlag.run_id.in_(run_ids),
            )
            .all()
        )
    else:
        followups = []
        coach_flags = []

    return _coverage_metrics_payload(
        runs=runs,
        raw_by_run_id=raw_by_run_id,
        followups=followups,
        coach_flags=coach_flags,
    )


@router.get("/knowledge-pack-release-candidates")
def list_knowledge_pack_release_candidates(
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_release_reviewer(current_account)

    rows = (
        db.query(KnowledgePackReleaseCandidate)
        .order_by(KnowledgePackReleaseCandidate.updated_at.desc())
        .all()
    )
    return {"items": [_build_release_candidate_item(row) for row in rows]}


@router.get("/knowledge-pack-release-candidates/{release_candidate_id}")
def get_knowledge_pack_release_candidate(
    release_candidate_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_release_reviewer(current_account)

    try:
        release_candidate_uuid = uuid.UUID(release_candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid release_candidate_id format")

    row = (
        db.query(KnowledgePackReleaseCandidate)
        .filter(KnowledgePackReleaseCandidate.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Knowledge-pack release candidate not found")

    events = (
        db.query(KnowledgePackReleaseEvent)
        .filter(KnowledgePackReleaseEvent.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .order_by(KnowledgePackReleaseEvent.created_at.desc())
        .all()
    )
    regression_runs = (
        db.query(KnowledgePackRegressionRun)
        .filter(KnowledgePackRegressionRun.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .order_by(KnowledgePackRegressionRun.created_at.desc())
        .all()
    )
    monitoring_snapshots = (
        db.query(KnowledgePackMonitoringSnapshot)
        .filter(KnowledgePackMonitoringSnapshot.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .order_by(KnowledgePackMonitoringSnapshot.created_at.desc())
        .all()
    )
    rollback_alerts = (
        db.query(KnowledgePackRollbackAlert)
        .filter(KnowledgePackRollbackAlert.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .order_by(KnowledgePackRollbackAlert.updated_at.desc())
        .all()
    )
    return {
        "candidate": _build_release_candidate_item(row),
        "events": [_build_release_event_item(event_row) for event_row in events],
        "regression_runs": [_build_regression_run_item(run_row) for run_row in regression_runs],
        "monitoring_snapshots": [_build_monitoring_snapshot_item(row) for row in monitoring_snapshots],
        "rollback_alerts": [_build_rollback_alert_item(row) for row in rollback_alerts],
    }


@router.get("/knowledge-pack-release-candidates/{release_candidate_id}/regression-runs/{regression_run_id}")
def get_knowledge_pack_release_regression_run(
    release_candidate_id: str,
    regression_run_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_release_reviewer(current_account)

    try:
        release_candidate_uuid = uuid.UUID(release_candidate_id)
        regression_run_uuid = uuid.UUID(regression_run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid release_candidate_id or regression_run_id format")

    run_row = (
        db.query(KnowledgePackRegressionRun)
        .filter(
            KnowledgePackRegressionRun.knowledge_pack_release_candidate_id == release_candidate_uuid,
            KnowledgePackRegressionRun.knowledge_pack_regression_run_id == regression_run_uuid,
        )
        .first()
    )
    if not run_row:
        raise HTTPException(status_code=404, detail="Knowledge-pack regression run not found")

    case_rows = (
        db.query(KnowledgePackRegressionCaseResult)
        .filter(KnowledgePackRegressionCaseResult.knowledge_pack_regression_run_id == regression_run_uuid)
        .order_by(KnowledgePackRegressionCaseResult.created_at.asc())
        .all()
    )
    return {
        "regression_run": _build_regression_run_item(run_row),
        "cases": [_build_regression_case_result_item(row) for row in case_rows],
    }


@router.get("/knowledge-pack-release-candidates/{release_candidate_id}/monitoring-snapshots")
def list_knowledge_pack_monitoring_snapshots(
    release_candidate_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_release_reviewer(current_account)
    try:
        release_candidate_uuid = uuid.UUID(release_candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid release_candidate_id format")

    rows = (
        db.query(KnowledgePackMonitoringSnapshot)
        .filter(KnowledgePackMonitoringSnapshot.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .order_by(KnowledgePackMonitoringSnapshot.created_at.desc())
        .all()
    )
    return {"items": [_build_monitoring_snapshot_item(row) for row in rows]}


@router.get("/knowledge-pack-release-candidates/{release_candidate_id}/rollback-alerts")
def list_knowledge_pack_rollback_alerts(
    release_candidate_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_release_reviewer(current_account)
    try:
        release_candidate_uuid = uuid.UUID(release_candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid release_candidate_id format")

    rows = (
        db.query(KnowledgePackRollbackAlert)
        .filter(KnowledgePackRollbackAlert.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .order_by(KnowledgePackRollbackAlert.updated_at.desc())
        .all()
    )
    return {"items": [_build_rollback_alert_item(row) for row in rows]}


def _require_release_reviewer(current_account) -> None:
    if str(getattr(current_account, "role", "")).lower() not in {"coach", "reviewer", "clinician"}:
        raise HTTPException(
            status_code=403,
            detail="Knowledge-pack release workflow is only available to coach, reviewer, or clinician accounts",
        )
