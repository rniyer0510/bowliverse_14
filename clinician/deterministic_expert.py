from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.clinician.knowledge_pack import load_knowledge_pack


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round3(value: float) -> float:
    return round(_clip01(value), 3)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _weighted_average(parts: Iterable[Tuple[float, float]]) -> float:
    total_weight = 0.0
    total = 0.0
    for value, weight in parts:
        if weight <= 0.0:
            continue
        total += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0.0:
        return 0.0
    return total / total_weight


def _average(values: Iterable[float], default: float = 0.0) -> float:
    items = [float(v) for v in values]
    if not items:
        return float(default)
    return sum(items) / float(len(items))


def _metric(value: float, confidence: float, *, label: Optional[str] = None) -> Dict[str, Any]:
    return {
        "value": _round3(value),
        "confidence": _round3(confidence),
        "available": bool(confidence > 0.0),
        "label": label,
    }


def _combine_metrics(
    parts: Iterable[Tuple[Dict[str, Any], float]],
    *,
    invert: bool = False,
) -> Dict[str, Any]:
    weighted_values: List[Tuple[float, float]] = []
    confidences: List[Tuple[float, float]] = []
    for metric, weight in parts:
        if not metric or weight <= 0.0:
            continue
        value = _safe_float(metric.get("value"), 0.0)
        confidence = _safe_float(metric.get("confidence"), 0.0)
        if invert:
            value = 1.0 - value
        weighted_values.append((value, confidence * weight))
        confidences.append((confidence, weight))
    if not weighted_values:
        return _metric(0.5, 0.0)
    value = _weighted_average(weighted_values)
    confidence = _weighted_average(confidences)
    return _metric(value, confidence)


def _metric_band(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.55:
        return "moderate"
    if score >= 0.35:
        return "watch"
    return "weak"


def _symptom_severity(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "moderate"
    if score >= 0.35:
        return "low"
    return "minimal"


def _risk_lookup(risks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for risk in risks or []:
        if not isinstance(risk, dict):
            continue
        risk_id = str(risk.get("risk_id") or "").strip()
        if risk_id:
            out[risk_id] = risk
    return out


def _basic_lookup(basics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(basics, dict):
        return {}
    return {
        key: value
        for key, value in basics.items()
        if isinstance(value, dict)
    }


_PRESENTATION_STATUS_RANK = {
    "no_match": 0,
    "weak_match": 1,
    "partial_match": 2,
    "ambiguous_match": 2,
    "confident_match": 3,
}

_RISK_CONTRIBUTOR_CATALOG: Dict[str, Dict[str, str]] = {
    "lateral_trunk_lean": {
        "title": "Trunk lean",
        "body_group": "upper_body",
        "phase": "RELEASE",
        "summary": "The trunk is leaning away instead of staying more stacked through release.",
    },
    "hip_shoulder_mismatch": {
        "title": "Hip-shoulder mismatch",
        "body_group": "upper_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "Upper and lower segments are not rotating together cleanly into release.",
    },
    "trunk_rotation_snap": {
        "title": "Shoulder rotation timing",
        "body_group": "upper_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "Shoulder-trunk rotation is arriving abruptly and too late in the chain.",
    },
    "foot_line_deviation": {
        "title": "Foot line deviation",
        "body_group": "lower_body",
        "phase": "FFC",
        "summary": "The landing line is not guiding force cleanly toward target.",
    },
    "front_foot_braking_shock": {
        "title": "Front-foot landing quality",
        "body_group": "lower_body",
        "phase": "FFC",
        "summary": "Landing is not becoming a calm, stable transfer point.",
    },
    "knee_brace_failure": {
        "title": "Knee brace loss",
        "body_group": "lower_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "The front knee is not supporting the action strongly enough through release.",
    },
}

_METRIC_CONTRIBUTOR_CATALOG: Dict[str, Dict[str, str]] = {
    "shoulder_rotation_timing": {
        "title": "Shoulder rotation timing",
        "body_group": "upper_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "Shoulder rotation timing is arriving later than we want for a calm release window.",
    },
    "pelvis_trunk_alignment": {
        "title": "Pelvic movement organization",
        "body_group": "lower_body",
        "phase": "BFC_TO_FFC",
        "summary": "Pelvis-to-trunk organization is not staying clean enough into landing.",
    },
    "chest_stack_over_landing": {
        "title": "Chest stack over landing",
        "body_group": "lower_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "The chest is not getting stacked over the landing leg early enough.",
    },
    "front_leg_support_score": {
        "title": "Front-leg support",
        "body_group": "lower_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "The landing leg is not turning landing into a stable transfer base.",
    },
    "trunk_drift_after_ffc": {
        "title": "Late trunk drift",
        "body_group": "upper_body",
        "phase": "FFC_TO_RELEASE",
        "summary": "The trunk keeps travelling after landing instead of settling into release.",
    },
}

_PHASE_LABELS: Dict[str, str] = {
    "approach_build": "Approach build",
    "gather_and_organize": "Gather and organize",
    "transfer_and_block": "Transfer and block",
    "whip_and_release": "Whip and release",
    "dissipation_and_recovery": "Dissipation and recovery",
    "BFC": "Back-foot contact",
    "FFC": "Front-foot contact",
    "UAH": "Upper-arm horizontal",
    "RELEASE": "Release",
    "FFC_TO_RELEASE": "Front-foot contact to release",
    "BFC_TO_FFC": "Back-foot contact to front-foot contact",
}


def _status_score(status: str) -> float:
    normalized = str(status or "").strip().lower()
    if normalized in {"ok", "aligned"}:
        return 0.85
    if normalized in {"semi_open", "warn"}:
        return 0.5
    if normalized in {"bad", "open"}:
        return 0.2
    return 0.5


def _risk_metric(
    risk_lookup: Dict[str, Dict[str, Any]],
    risk_id: str,
    *,
    chain_quality: float,
) -> Dict[str, Any]:
    risk = risk_lookup.get(risk_id) or {}
    signal = _clip01(_safe_float(risk.get("signal_strength"), 0.0))
    confidence = _clip01(max(_safe_float(risk.get("confidence"), 0.0), chain_quality * 0.45))
    return _metric(signal, confidence)


def _basic_metric(
    basic_lookup: Dict[str, Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    basic = basic_lookup.get(key) or {}
    score = _status_score(str(basic.get("status") or "unknown"))
    confidence = _clip01(_safe_float(basic.get("confidence"), 0.0))
    return _metric(score, confidence)


class DeterministicExpertSystem:
    def __init__(self, *, pack_version: Optional[str] = None):
        self._pack = load_knowledge_pack(pack_version)

    @property
    def history_window_runs(self) -> int:
        return int(self._pack["globals"]["history_window_defaults"]["trend_window_runs"])

    def build(
        self,
        *,
        events: Dict[str, Any],
        action: Dict[str, Any],
        risks: List[Dict[str, Any]],
        basics: Dict[str, Any],
        interpretation: Dict[str, Any],
        estimated_release_speed: Dict[str, Any],
        prior_results: Optional[List[Dict[str, Any]]] = None,
        account_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics = self._build_metrics(
            events=events,
            action=action,
            risks=risks,
            basics=basics,
            interpretation=interpretation,
            estimated_release_speed=estimated_release_speed,
        )
        history_context = self._summarize_prior_results(prior_results or [])
        capture_quality = self._build_capture_quality(
            events=events,
            action=action,
            metrics=metrics,
        )
        symptoms = self._build_symptoms(metrics)
        mechanics_evidence = self._build_mechanics_evidence_payload(
            events=events,
            symptoms=symptoms,
            metrics=metrics,
            capture_quality=capture_quality,
        )
        if capture_quality["status"] == "UNUSABLE":
            hypotheses = []
            selection = self._capture_quality_short_circuit_selection(capture_quality)
            archetype = None
        else:
            hypotheses, selection = self._score_mechanisms(symptoms, metrics)
            archetype = self._select_archetype(
                metrics=metrics,
                selection=selection,
                history_context=history_context,
            )
        render_reasoning = self._build_render_reasoning(
            selection=selection,
            hypotheses=hypotheses,
            capture_quality=capture_quality,
        )
        kinetic_chain = self._build_kinetic_chain_payload(
            metrics,
            hypotheses,
            selection,
            capture_quality=capture_quality,
            archetype=archetype,
            history_context=history_context,
        )
        mechanism_explanation = self._build_mechanism_explanation(
            symptoms,
            hypotheses,
            selection,
            archetype=archetype,
            account_role=account_role,
            history_context=history_context,
        )
        prescription_plan = self._build_prescription_plan(
            selection,
            prescription_allowed=render_reasoning["prescription_allowed"],
        )
        history_plan = self._build_history_plan(
            selection,
            metrics=metrics,
            archetype=archetype,
            history_context=history_context,
        )
        coach_diagnosis = self._build_coach_diagnosis(
            events=events,
            risks=risks,
            metrics=metrics,
            symptoms=symptoms,
            hypotheses=hypotheses,
            selection=selection,
            capture_quality=capture_quality,
            render_reasoning=render_reasoning,
            mechanism_explanation=mechanism_explanation,
            prescription_plan=prescription_plan,
            history_plan=history_plan,
            archetype=archetype,
            history_context=history_context,
        )
        presentation_payload = self._build_presentation_payload(
            selection=selection,
            hypotheses=hypotheses,
            capture_quality=capture_quality,
            render_reasoning=render_reasoning,
            mechanism_explanation=mechanism_explanation,
            prescription_plan=prescription_plan,
            archetype=archetype,
        )

        return {
            "version": "deterministic_expert_v1",
            "runtime_mode": "static",
            "knowledge_pack_id": self._pack["pack_id"],
            "knowledge_pack_version": self._pack["pack_version"],
            "unknown_path_enforced": True,
            "capture_quality_v1": capture_quality,
            "mechanics_evidence_v1": mechanics_evidence,
            "metrics": metrics,
            "symptoms": symptoms,
            "mechanism_hypotheses": hypotheses,
            "selection": selection,
            "history_context_v1": history_context,
            "archetype_v1": archetype,
            "kinetic_chain_v1": kinetic_chain,
            "render_reasoning_v1": render_reasoning,
            "mechanism_explanation_v1": mechanism_explanation,
            "prescription_plan_v1": prescription_plan,
            "history_plan_v1": history_plan,
            "coach_diagnosis_v1": coach_diagnosis,
            "presentation_payload_v1": presentation_payload,
        }

    def _build_capture_quality(
        self,
        *,
        events: Dict[str, Any],
        action: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        event_chain = (events or {}).get("event_chain") or {}
        ordered = bool(event_chain.get("ordered", True))
        chain_quality = _clip01(_safe_float(event_chain.get("quality"), 0.0))
        action_conf = _clip01(_safe_float((action or {}).get("confidence"), 0.0))
        notes: List[str] = []
        for name in ("bfc", "ffc", "release"):
            event = (events or {}).get(name) or {}
            if event.get("frame") is None:
                notes.append(f"{name}_missing")
        if not ordered:
            notes.append("event_chain_unordered")
        if chain_quality < 0.20:
            notes.append("event_chain_low_quality")
        elif chain_quality < 0.40:
            notes.append("event_chain_weak_quality")
        if action_conf < 0.25:
            notes.append("action_confidence_too_low")
        elif action_conf < 0.40:
            notes.append("action_confidence_weak")

        if not ordered or chain_quality < 0.20 or action_conf < 0.25:
            status = "UNUSABLE"
        elif chain_quality < 0.40 or action_conf < 0.40:
            status = "WEAK"
        else:
            status = "USABLE"

        return {
            "version": "capture_quality_v1",
            "status": status,
            "notes": notes,
            "event_chain_quality": metrics["event_chain_quality"]["value"],
            "action_confidence": metrics["action_confidence"]["value"],
        }

    def _build_mechanics_evidence_payload(
        self,
        *,
        events: Dict[str, Any],
        symptoms: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        capture_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        derived_metrics = {
            "runup_build_score": metrics["approach_momentum_score"]["value"],
            "transfer_efficiency": metrics["transfer_efficiency_score"]["value"],
            "late_thrust_dependence": metrics["terminal_impulse_score"]["value"],
            "dissipation_burden": metrics["dissipation_burden_score"]["value"],
        }
        confidences = [
            _safe_float((metric or {}).get("confidence"), 0.0)
            for metric in (
                metrics.get("approach_momentum_score"),
                metrics.get("transfer_efficiency_score"),
                metrics.get("terminal_impulse_score"),
                metrics.get("dissipation_burden_score"),
            )
            if isinstance(metric, dict)
        ]
        evidence_completeness = _round3(_average(confidences, default=0.0))
        serialized_events = {}
        for event_name in ("bfc", "ffc", "uah", "release"):
            event = (events or {}).get(event_name) or {}
            serialized_events[event_name] = {
                "frame": event.get("frame"),
                "confidence": _safe_float(event.get("confidence"), 0.0),
            }
        return {
            "version": "mechanics_evidence_v1",
            "knowledge_pack_version": self._pack["pack_version"],
            "capture_quality": capture_quality,
            "events": serialized_events,
            "symptoms": [
                {
                    "id": symptom["id"],
                    "severity_band": symptom["severity"],
                    "score": symptom["score"],
                }
                for symptom in symptoms
                if symptom.get("present")
            ],
            "derived_metrics": derived_metrics,
            "evidence_completeness": evidence_completeness,
        }

    def _capture_quality_short_circuit_selection(
        self,
        capture_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "diagnosis_status": "no_match",
            "ambiguous": False,
            "primary_mechanism_id": None,
            "primary_mechanism_title": None,
            "overall_confidence": 0.0,
            "primary": None,
            "secondary": [],
            "selected_mechanism_ids": [],
            "selected_trajectory_ids": [],
            "selected_prescription_ids": [],
            "selected_render_story_ids": [],
            "no_match_reason": (
                "Capture quality is too weak for deterministic mechanism scoring."
            ),
            "capture_quality_status": capture_quality.get("status"),
        }

    def _build_render_reasoning(
        self,
        *,
        selection: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        capture_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        rules = self._pack["globals"]["presentation_downgrade_rules"]
        diagnosis_status = str(selection.get("diagnosis_status") or "no_match")
        status_rank = _PRESENTATION_STATUS_RANK.get(diagnosis_status, 0)
        full_rank = _PRESENTATION_STATUS_RANK.get(
            str(rules.get("full_causal_story_requires") or "confident_match"),
            3,
        )
        partial_rank = _PRESENTATION_STATUS_RANK.get(
            str(rules.get("partial_evidence_requires") or "partial_match"),
            2,
        )
        event_only_below_rank = _PRESENTATION_STATUS_RANK.get(
            str(rules.get("event_only_below") or "partial_match"),
            2,
        )
        prescription_floor_rank = _PRESENTATION_STATUS_RANK.get(
            str(rules.get("prescription_suppressed_below") or "partial_match"),
            2,
        )
        full_min_evidence = _safe_float(
            rules.get("full_causal_story_min_evidence_completeness"),
            0.55,
        )

        primary = selection.get("primary") or {}
        if not isinstance(primary, dict):
            primary = {}
        evidence_completeness = _safe_float(primary.get("evidence_completeness"), 0.0)
        selected_story_ids = list(selection.get("selected_render_story_ids") or [])
        history_binding_ids = self._history_binding_ids(selection)
        matched_symptom_ids = list(primary.get("matched_symptom_ids") or [])
        contradictions = list(primary.get("contradiction_notes") or [])
        capture_quality_status = str(capture_quality.get("status") or "").upper()

        if capture_quality_status == "UNUSABLE":
            renderer_mode = "event_only"
            downgrade_reason = "capture_quality_unusable"
        elif status_rank < event_only_below_rank:
            renderer_mode = "event_only"
            downgrade_reason = "below_event_only_threshold"
        elif not selected_story_ids:
            renderer_mode = "event_only"
            downgrade_reason = "no_render_story_selected"
        elif status_rank >= full_rank and evidence_completeness >= full_min_evidence:
            renderer_mode = "full_causal_story"
            downgrade_reason = "full_causal_rules_satisfied"
        elif status_rank >= partial_rank:
            renderer_mode = "partial_evidence"
            downgrade_reason = "downgraded_from_full_due_to_confidence_or_evidence"
        else:
            renderer_mode = "event_only"
            downgrade_reason = "fell_through_to_event_only"

        prescription_allowed = (
            capture_quality_status != "UNUSABLE"
            and status_rank >= prescription_floor_rank
        )

        return {
            "version": "render_reasoning_v1",
            "knowledge_pack_version": self._pack["pack_version"],
            "diagnosis_status": diagnosis_status,
            "renderer_mode": renderer_mode,
            "selected_story_id": (
                selected_story_ids[0]
                if renderer_mode != "event_only" and selected_story_ids
                else None
            ),
            "selected_story_ids": (
                selected_story_ids
                if renderer_mode != "event_only"
                else []
            ),
            "suppressed_story_ids": (
                selected_story_ids
                if renderer_mode == "event_only"
                else []
            ),
            "downgrade_reason": downgrade_reason,
            "prescription_allowed": prescription_allowed,
            "overall_confidence": _safe_float(selection.get("overall_confidence"), 0.0),
            "evidence_completeness": _round3(evidence_completeness),
            "capture_quality_status": capture_quality_status,
            "primary_mechanism_id": selection.get("primary_mechanism_id"),
            "matched_symptom_ids": matched_symptom_ids,
            "contradictions_triggered": contradictions,
            "candidate_mechanisms": [
                {
                    "id": item.get("id"),
                    "confidence": item.get("overall_confidence"),
                }
                for item in hypotheses[:3]
            ],
            "causal_chain": {
                "mechanism_id": selection.get("primary_mechanism_id"),
                "trajectory_ids": list(selection.get("selected_trajectory_ids") or []),
                "prescription_ids": list(selection.get("selected_prescription_ids") or []),
                "render_story_ids": selected_story_ids,
                "history_binding_ids": history_binding_ids,
            },
        }

    def _build_metrics(
        self,
        *,
        events: Dict[str, Any],
        action: Dict[str, Any],
        risks: List[Dict[str, Any]],
        basics: Dict[str, Any],
        interpretation: Dict[str, Any],
        estimated_release_speed: Dict[str, Any],
    ) -> Dict[str, Any]:
        risk_lookup = _risk_lookup(risks)
        basic_lookup = _basic_lookup(basics)
        chain_quality = _clip01(_safe_float(((events or {}).get("event_chain") or {}).get("quality"), 0.0))
        action_conf = _clip01(_safe_float((action or {}).get("confidence"), 0.0))
        flow = (interpretation or {}).get("linear_flow") or {}
        flow_conf = _clip01(_safe_float(flow.get("confidence"), 0.0))
        flow_state = str(flow.get("flow_state") or "SMOOTH").upper()
        flow_state_score = {
            "SMOOTH": 0.85,
            "INTERRUPTED": 0.5,
            "FRAGMENTED": 0.2,
        }.get(flow_state, 0.45)

        back_foot_stability = _basic_metric(basic_lookup, "back_foot_stability")
        knee_brace_proxy = _basic_metric(basic_lookup, "knee_brace_proxy")
        toe_alignment = _basic_metric(basic_lookup, "front_foot_toe_alignment")

        front_foot_braking = _risk_metric(risk_lookup, "front_foot_braking_shock", chain_quality=chain_quality)
        knee_brace_failure = _risk_metric(risk_lookup, "knee_brace_failure", chain_quality=chain_quality)
        trunk_rotation_snap = _risk_metric(risk_lookup, "trunk_rotation_snap", chain_quality=chain_quality)
        hip_shoulder_mismatch = _risk_metric(risk_lookup, "hip_shoulder_mismatch", chain_quality=chain_quality)
        lateral_trunk_lean = _risk_metric(risk_lookup, "lateral_trunk_lean", chain_quality=chain_quality)
        foot_line_deviation = _risk_metric(risk_lookup, "foot_line_deviation", chain_quality=chain_quality)

        speed_conf = _clip01(_safe_float((estimated_release_speed or {}).get("confidence"), 0.0))
        speed_available = bool((estimated_release_speed or {}).get("available"))
        speed_debug = (estimated_release_speed or {}).get("debug") or {}
        elbow_velocity = _clip01(_safe_float(speed_debug.get("elbow_extension_velocity_deg_per_sec"), 0.0) / 220.0)
        wrist_arm_ratio = _clip01(_safe_float(speed_debug.get("wrist_arm_ratio"), 0.0) / 1.6)
        shoulder_body_ratio = _clip01(_safe_float(speed_debug.get("shoulder_body_ratio"), 0.0))
        pelvis_body_ratio = _clip01(_safe_float(speed_debug.get("pelvis_body_ratio"), 0.0))
        speed_metric_conf = speed_conf if speed_available else max(0.0, speed_conf * 0.6)

        runup_rhythm_stability = _combine_metrics(
            [
                (back_foot_stability, 0.45),
                (_metric(flow_state_score, flow_conf), 0.35),
                (_metric(action_conf, action_conf), 0.20),
            ]
        )
        approach_momentum = _combine_metrics(
            [
                (runup_rhythm_stability, 0.40),
                (_metric(action_conf, action_conf), 0.25),
                (_metric(speed_conf if speed_available else 0.45, speed_metric_conf), 0.20),
                (_metric(1.0 - front_foot_braking["value"], front_foot_braking["confidence"]), 0.15),
            ]
        )
        gather_line_stability = _combine_metrics(
            [
                (_metric(1.0 - foot_line_deviation["value"], foot_line_deviation["confidence"]), 0.40),
                (_metric(1.0 - lateral_trunk_lean["value"], lateral_trunk_lean["confidence"]), 0.25),
                (back_foot_stability, 0.20),
                (toe_alignment, 0.15),
            ]
        )
        pelvis_trunk_alignment = _combine_metrics(
            [
                (_metric(1.0 - hip_shoulder_mismatch["value"], hip_shoulder_mismatch["confidence"]), 0.45),
                (_metric(1.0 - trunk_rotation_snap["value"], trunk_rotation_snap["confidence"]), 0.35),
                (_metric(1.0 - lateral_trunk_lean["value"], lateral_trunk_lean["confidence"]), 0.20),
            ]
        )
        front_leg_support_score = _combine_metrics(
            [
                (_metric(1.0 - knee_brace_failure["value"], knee_brace_failure["confidence"]), 0.60),
                (knee_brace_proxy, 0.25),
                (_metric(1.0 - front_foot_braking["value"], front_foot_braking["confidence"]), 0.15),
            ]
        )
        trunk_drift_after_ffc = _combine_metrics(
            [
                (front_foot_braking, 0.45),
                (lateral_trunk_lean, 0.35),
                (_metric(1.0 - front_leg_support_score["value"], front_leg_support_score["confidence"]), 0.20),
            ]
        )
        transfer_efficiency_score = _combine_metrics(
            [
                (front_leg_support_score, 0.30),
                (_metric(1.0 - front_foot_braking["value"], front_foot_braking["confidence"]), 0.20),
                (_metric(1.0 - hip_shoulder_mismatch["value"], hip_shoulder_mismatch["confidence"]), 0.20),
                (_metric(1.0 - trunk_drift_after_ffc["value"], trunk_drift_after_ffc["confidence"]), 0.20),
                (_metric(chain_quality, chain_quality), 0.10),
            ]
        )

        sequence_pattern = str((((risk_lookup.get("hip_shoulder_mismatch") or {}).get("debug")) or {}).get("sequence_pattern") or "unknown").lower()
        shoulder_rotation_timing_value = {
            "hips_lead": 0.82,
            "in_sync": 0.68,
            "shoulders_lead": 0.22,
        }.get(sequence_pattern, 0.45)
        shoulder_rotation_timing = _metric(shoulder_rotation_timing_value, hip_shoulder_mismatch["confidence"])

        release_timing_stability = _combine_metrics(
            [
                (_metric(1.0 - hip_shoulder_mismatch["value"], hip_shoulder_mismatch["confidence"]), 0.40),
                (_metric(1.0 - trunk_rotation_snap["value"], trunk_rotation_snap["confidence"]), 0.20),
                (shoulder_rotation_timing, 0.25),
                (_metric(flow_state_score, flow_conf), 0.15),
            ]
        )
        terminal_impulse_score = _combine_metrics(
            [
                (front_foot_braking, 0.25),
                (trunk_rotation_snap, 0.25),
                (_metric(1.0 - transfer_efficiency_score["value"], transfer_efficiency_score["confidence"]), 0.20),
                (_metric(elbow_velocity, speed_metric_conf), 0.15),
                (_metric(1.0 - shoulder_rotation_timing["value"], shoulder_rotation_timing["confidence"]), 0.15),
            ]
        )
        distal_velocity_rescue = _combine_metrics(
            [
                (_metric(1.0 - release_timing_stability["value"], release_timing_stability["confidence"]), 0.45),
                (_metric(elbow_velocity, speed_metric_conf), 0.25),
                (_metric(wrist_arm_ratio, speed_metric_conf), 0.15),
                (hip_shoulder_mismatch, 0.15),
            ]
        )
        dissipation_burden_score = _combine_metrics(
            [
                (lateral_trunk_lean, 0.25),
                (trunk_rotation_snap, 0.25),
                (front_foot_braking, 0.20),
                (knee_brace_failure, 0.20),
                (foot_line_deviation, 0.10),
            ]
        )
        followthrough_asymmetry = _combine_metrics(
            [
                (lateral_trunk_lean, 0.60),
                (foot_line_deviation, 0.40),
            ]
        )
        trunk_fold_severity = _combine_metrics(
            [
                (trunk_rotation_snap, 0.60),
                (lateral_trunk_lean, 0.40),
            ]
        )
        chest_stack_over_landing = _combine_metrics(
            [
                (_metric(1.0 - trunk_drift_after_ffc["value"], trunk_drift_after_ffc["confidence"]), 0.75),
                (front_leg_support_score, 0.25),
            ]
        )

        metrics = {
            "event_chain_quality": _metric(chain_quality, chain_quality),
            "action_confidence": _metric(action_conf, action_conf),
            "runup_rhythm_stability": runup_rhythm_stability,
            "approach_momentum_score": approach_momentum,
            "gather_line_stability": gather_line_stability,
            "pelvis_trunk_alignment": pelvis_trunk_alignment,
            "front_leg_support_score": front_leg_support_score,
            "trunk_drift_after_ffc": trunk_drift_after_ffc,
            "chest_stack_over_landing": chest_stack_over_landing,
            "transfer_efficiency_score": transfer_efficiency_score,
            "shoulder_rotation_timing": shoulder_rotation_timing,
            "release_timing_stability": release_timing_stability,
            "terminal_impulse_score": terminal_impulse_score,
            "distal_velocity_rescue": distal_velocity_rescue,
            "dissipation_burden_score": dissipation_burden_score,
            "followthrough_asymmetry": followthrough_asymmetry,
            "trunk_fold_severity": trunk_fold_severity,
            "wrist_arm_ratio": _metric(wrist_arm_ratio, speed_metric_conf),
            "shoulder_body_ratio": _metric(shoulder_body_ratio, speed_metric_conf),
            "pelvis_body_ratio": _metric(pelvis_body_ratio, speed_metric_conf),
            "elbow_velocity_norm": _metric(elbow_velocity, speed_metric_conf),
            "sequence_pattern": {
                "value": sequence_pattern,
                "confidence": hip_shoulder_mismatch["confidence"],
                "available": bool(hip_shoulder_mismatch["confidence"] > 0.0),
            },
            "speed_available": {
                "value": bool(speed_available),
                "confidence": speed_metric_conf,
                "available": True,
            },
        }

        return metrics

    def _build_symptoms(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        symptoms_cfg = self._pack["symptoms"]
        symptom_scores = {
            "weak_approach_build": _combine_metrics(
                [
                    (_metric(1.0 - metrics["approach_momentum_score"]["value"], metrics["approach_momentum_score"]["confidence"]), 0.70),
                    (_metric(1.0 - metrics["runup_rhythm_stability"]["value"], metrics["runup_rhythm_stability"]["confidence"]), 0.30),
                ]
            ),
            "unstable_gather": _combine_metrics(
                [
                    (_metric(1.0 - metrics["gather_line_stability"]["value"], metrics["gather_line_stability"]["confidence"]), 0.55),
                    (_metric(1.0 - metrics["pelvis_trunk_alignment"]["value"], metrics["pelvis_trunk_alignment"]["confidence"]), 0.45),
                ]
            ),
            "front_leg_softening": _metric(
                1.0 - metrics["front_leg_support_score"]["value"],
                metrics["front_leg_support_score"]["confidence"],
            ),
            "late_trunk_drift": metrics["trunk_drift_after_ffc"],
            "arm_chase": _combine_metrics(
                [
                    (_metric(1.0 - metrics["release_timing_stability"]["value"], metrics["release_timing_stability"]["confidence"]), 0.60),
                    (metrics["distal_velocity_rescue"], 0.40),
                ]
            ),
            "high_terminal_thrust": metrics["terminal_impulse_score"],
            "asymmetric_dissipation": _combine_metrics(
                [
                    (metrics["dissipation_burden_score"], 0.65),
                    (metrics["followthrough_asymmetry"], 0.35),
                ]
            ),
        }

        symptoms: List[Dict[str, Any]] = []
        for symptom_id, cfg in symptoms_cfg.items():
            score_cfg = symptom_scores.get(symptom_id) or _metric(0.0, 0.0)
            score = _safe_float(score_cfg.get("value"), 0.0)
            confidence = _safe_float(score_cfg.get("confidence"), 0.0)
            symptoms.append(
                {
                    "id": symptom_id,
                    "title": cfg["title"],
                    "category": cfg["category"],
                    "phase": cfg["phase"],
                    "description": cfg["description"],
                    "score": _round3(score),
                    "confidence": _round3(confidence),
                    "present": bool(score >= 0.55 and confidence >= 0.15),
                    "severity": _symptom_severity(score),
                    "render_focus_regions": list(cfg["render_focus_regions"]),
                    "evidence_inputs": list(cfg["evidence_inputs"]),
                    "possible_mechanisms": list(cfg["possible_mechanisms"]),
                }
            )
        symptoms.sort(key=lambda item: item["score"], reverse=True)
        return symptoms

    def _build_evidence_flags(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            "approach_momentum_score_low": _metric(1.0 - metrics["approach_momentum_score"]["value"], metrics["approach_momentum_score"]["confidence"]),
            "terminal_impulse_score_high": metrics["terminal_impulse_score"],
            "transfer_efficiency_score_moderate_or_lower": _metric(1.0 - metrics["transfer_efficiency_score"]["value"], metrics["transfer_efficiency_score"]["confidence"]),
            "runup_rhythm_stability_below_clean": _metric(1.0 - metrics["runup_rhythm_stability"]["value"], metrics["runup_rhythm_stability"]["confidence"]),
            "gather_line_stability_low": _metric(1.0 - metrics["gather_line_stability"]["value"], metrics["gather_line_stability"]["confidence"]),
            "front_leg_support_score_low": _metric(1.0 - metrics["front_leg_support_score"]["value"], metrics["front_leg_support_score"]["confidence"]),
            "pelvis_trunk_alignment_unstable": _metric(1.0 - metrics["pelvis_trunk_alignment"]["value"], metrics["pelvis_trunk_alignment"]["confidence"]),
            "trunk_drift_after_ffc_elevated": metrics["trunk_drift_after_ffc"],
            "trunk_drift_after_ffc_high": metrics["trunk_drift_after_ffc"],
            "chest_stack_over_landing_delayed": _metric(1.0 - metrics["chest_stack_over_landing"]["value"], metrics["chest_stack_over_landing"]["confidence"]),
            "dissipation_burden_score_elevated": metrics["dissipation_burden_score"],
            "dissipation_burden_score_high": metrics["dissipation_burden_score"],
            "release_timing_stability_low": _metric(1.0 - metrics["release_timing_stability"]["value"], metrics["release_timing_stability"]["confidence"]),
            "shoulder_rotation_timing_late": _metric(1.0 - metrics["shoulder_rotation_timing"]["value"], metrics["shoulder_rotation_timing"]["confidence"]),
            "distal_velocity_rescue_present": metrics["distal_velocity_rescue"],
            "followthrough_asymmetry_present": metrics["followthrough_asymmetry"],
            "trunk_fold_severity_elevated": metrics["trunk_fold_severity"],
        }

    def _score_mechanisms(
        self,
        symptoms: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        symptom_lookup = {symptom["id"]: symptom for symptom in symptoms}
        evidence_lookup = self._build_evidence_flags(metrics)
        hypotheses: List[Dict[str, Any]] = []

        for mechanism_id, cfg in self._pack["mechanisms"].items():
            required_symptom_scores = [symptom_lookup[sid]["score"] for sid in cfg["required_symptoms"]]
            supporting_symptom_scores = [symptom_lookup[sid]["score"] for sid in cfg["supporting_symptoms"]]
            contradictory_symptom_scores = [symptom_lookup[sid]["score"] for sid in cfg["contradictory_symptoms"]]

            required_evidence_scores = [
                _safe_float((evidence_lookup.get(eid) or {}).get("value"), 0.0)
                for eid in cfg["required_evidence"]
            ]
            supporting_evidence_scores = [
                _safe_float((evidence_lookup.get(eid) or {}).get("value"), 0.0)
                for eid in cfg["supporting_evidence"]
            ]
            evidence_confidences = [
                _safe_float((evidence_lookup.get(eid) or {}).get("confidence"), 0.0)
                for eid in [*cfg["required_evidence"], *cfg["supporting_evidence"]]
            ]
            evidence_completeness = _average(evidence_confidences, default=0.0)

            support_score = _clip01(
                (0.40 * _average(required_symptom_scores, default=0.0))
                + (0.15 * _average(supporting_symptom_scores, default=0.0))
                + (0.30 * _average(required_evidence_scores, default=0.0))
                + (0.15 * _average(supporting_evidence_scores, default=0.0))
            )

            context_penalty, contradiction_notes = self._context_penalty(
                mechanism_id=mechanism_id,
                mechanism=cfg,
                metrics=metrics,
            )
            contradiction_penalty = _clip01(
                (0.30 * (1.0 - _average(required_symptom_scores, default=0.0)))
                + (0.25 * (1.0 - _average(required_evidence_scores, default=0.0)))
                + (0.15 * _average(contradictory_symptom_scores, default=0.0))
                + (0.15 * (1.0 - evidence_completeness))
                + (0.15 * context_penalty)
            )
            overall_confidence = _clip01(support_score - (0.85 * contradiction_penalty))

            hypotheses.append(
                {
                    "id": mechanism_id,
                    "title": cfg["title"],
                    "family": cfg["family"],
                    "summary": cfg["summary"],
                    "support_score": _round3(support_score),
                    "contradiction_penalty": _round3(contradiction_penalty),
                    "evidence_completeness": _round3(evidence_completeness),
                    "overall_confidence": _round3(overall_confidence),
                    "required_symptom_ids": list(cfg["required_symptoms"]),
                    "supporting_symptom_ids": list(cfg["supporting_symptoms"]),
                    "trajectory_ids": list(cfg["trajectory_ids"]),
                    "prescription_ids": list(cfg["prescription_ids"]),
                    "render_story_ids": list(cfg["render_story_ids"]),
                    "history_metrics_to_track": list(cfg["history_metrics_to_track"]),
                    "matched_symptom_ids": [
                        symptom_id
                        for symptom_id in [*cfg["required_symptoms"], *cfg["supporting_symptoms"]]
                        if symptom_lookup[symptom_id]["present"]
                    ],
                    "contradiction_notes": contradiction_notes,
                }
            )

        hypotheses.sort(key=lambda item: item["overall_confidence"], reverse=True)

        thresholds = self._pack["globals"]["match_thresholds"]
        top = hypotheses[0] if hypotheses else None
        runner_up = hypotheses[1] if len(hypotheses) > 1 else None
        ambiguous = bool(
            top
            and runner_up
            and top["overall_confidence"] >= thresholds["partial_match_min"]
            and runner_up["overall_confidence"] >= thresholds["partial_match_min"]
            and abs(top["overall_confidence"] - runner_up["overall_confidence"])
            <= thresholds["ambiguous_match_delta_max"]
        )

        if not top or top["overall_confidence"] < thresholds["weak_match_min"]:
            diagnosis_status = "no_match"
            primary = None
            secondary = []
        elif ambiguous:
            diagnosis_status = "ambiguous_match"
            primary = top
            secondary = [runner_up] if runner_up else []
        elif top["overall_confidence"] >= thresholds["confident_match_min"]:
            diagnosis_status = "confident_match"
            primary = top
            secondary = [
                item for item in hypotheses[1:3]
                if item["overall_confidence"] >= thresholds["weak_match_min"]
            ]
        elif top["overall_confidence"] >= thresholds["partial_match_min"]:
            diagnosis_status = "partial_match"
            primary = top
            secondary = [
                item for item in hypotheses[1:3]
                if item["overall_confidence"] >= thresholds["weak_match_min"]
            ]
        else:
            diagnosis_status = "weak_match"
            primary = top
            secondary = []

        selected_mechanisms = [item for item in [primary, *secondary] if item]
        trajectory_ids = []
        prescription_ids = []
        render_story_ids = []
        for item in selected_mechanisms:
            trajectory_ids.extend(item["trajectory_ids"])
            prescription_ids.extend(item["prescription_ids"])
            render_story_ids.extend(item["render_story_ids"])

        selection = {
            "diagnosis_status": diagnosis_status,
            "ambiguous": ambiguous,
            "primary_mechanism_id": primary["id"] if primary else None,
            "primary_mechanism_title": primary["title"] if primary else None,
            "overall_confidence": _round3(_safe_float(top["overall_confidence"], 0.0) if top else 0.0),
            "primary": primary,
            "secondary": secondary,
            "selected_mechanism_ids": [item["id"] for item in selected_mechanisms],
            "selected_trajectory_ids": list(dict.fromkeys(trajectory_ids)),
            "selected_prescription_ids": list(dict.fromkeys(prescription_ids)),
            "selected_render_story_ids": list(dict.fromkeys(render_story_ids)),
        }
        return hypotheses, selection

    def _context_penalty(
        self,
        *,
        mechanism_id: str,
        mechanism: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        penalties: List[float] = []
        chain_quality = _safe_float(metrics["event_chain_quality"]["value"], 0.0)
        sequence_pattern = str(metrics["sequence_pattern"]["value"] or "unknown").lower()
        approach_momentum = _safe_float(metrics["approach_momentum_score"]["value"], 0.0)
        terminal_impulse = _safe_float(metrics["terminal_impulse_score"]["value"], 0.0)
        front_leg_support = _safe_float(metrics["front_leg_support_score"]["value"], 0.0)
        dissipation_burden = _safe_float(metrics["dissipation_burden_score"]["value"], 0.0)

        if chain_quality < 0.25 and mechanism["family"] in {"block_bracing_deficit", "gather_deficit"}:
            penalties.append(0.75)
            notes.append("Weak event-chain quality reduces confidence in landing-dependent stories.")
        if mechanism_id == "low_build_forcing_late_rescue" and approach_momentum >= 0.7:
            penalties.append(0.7)
            notes.append("Approach build does not currently look low enough for a low-build primary story.")
        if mechanism_id == "low_build_forcing_late_rescue" and terminal_impulse < 0.45:
            penalties.append(0.65)
            notes.append("Terminal impulse is not high enough to support a late-rescue pattern.")
        if mechanism_id == "soft_block_with_trunk_carry" and front_leg_support >= 0.65:
            penalties.append(0.65)
            notes.append("Front-leg support does not look soft enough for a soft-block primary story.")
        if mechanism_id == "late_arm_acceleration_due_to_chain_delay" and sequence_pattern == "in_sync":
            penalties.append(0.7)
            notes.append("Hip-shoulder timing does not currently look delayed enough to support arm chase.")
        if mechanism_id == "high_terminal_impulse_with_high_dissipation_burden" and dissipation_burden < 0.5:
            penalties.append(0.65)
            notes.append("Dissipation burden is too low for a high-cost terminal impulse story.")
        return _average(penalties, default=0.0), notes

    def _build_kinetic_chain_payload(
        self,
        metrics: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        selection: Dict[str, Any],
        *,
        capture_quality: Dict[str, Any],
        archetype: Optional[Dict[str, Any]],
        history_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if capture_quality.get("status") == "UNUSABLE":
            return {
                "version": "kinetic_chain_v1",
                "knowledge_pack_version": self._pack["pack_version"],
                "diagnosis_status": selection["diagnosis_status"],
                "confidence": 0.0,
                "capture_quality": capture_quality,
                "archetype": None,
                "approach_build": None,
                "gather_and_organize": None,
                "transfer": None,
                "block": None,
                "release_generation": None,
                "dissipation": None,
                "pace_translation": {},
                "internal_metrics": {},
                "derived_indices": {},
                "mechanism_hypotheses": [],
                "historical_context": {
                    "prior_run_count": history_context["prior_run_count"],
                    "recent_dominant_mechanism_id": history_context["dominant_mechanism_id"],
                    "recent_dominant_archetype_id": history_context["dominant_archetype_id"],
                },
                "selected_render_story_ids": [],
            }
        approach = metrics["approach_momentum_score"]
        gather = _combine_metrics(
            [
                (metrics["gather_line_stability"], 0.55),
                (metrics["pelvis_trunk_alignment"], 0.45),
            ]
        )
        transfer = metrics["transfer_efficiency_score"]
        block = _combine_metrics(
            [
                (metrics["front_leg_support_score"], 0.60),
                (_metric(1.0 - metrics["trunk_drift_after_ffc"]["value"], metrics["trunk_drift_after_ffc"]["confidence"]), 0.40),
            ]
        )
        release_generation = _metric(
            metrics["terminal_impulse_score"]["value"],
            metrics["terminal_impulse_score"]["confidence"],
            label="late_thrust" if metrics["terminal_impulse_score"]["value"] >= 0.6 else "carried",
        )
        dissipation = _metric(
            metrics["dissipation_burden_score"]["value"],
            metrics["dissipation_burden_score"]["confidence"],
            label="high_load_concentration" if metrics["dissipation_burden_score"]["value"] >= 0.6 else "distributed",
        )
        leakage_before_block = _combine_metrics(
            [
                (_metric(1.0 - metrics["gather_line_stability"]["value"], metrics["gather_line_stability"]["confidence"]), 0.50),
                (_metric(1.0 - metrics["pelvis_trunk_alignment"]["value"], metrics["pelvis_trunk_alignment"]["confidence"]), 0.30),
                (_metric(1.0 - approach["value"], approach["confidence"]), 0.20),
            ]
        )
        leakage_at_block = _combine_metrics(
            [
                (_metric(1.0 - metrics["front_leg_support_score"]["value"], metrics["front_leg_support_score"]["confidence"]), 0.45),
                (metrics["trunk_drift_after_ffc"], 0.35),
                (_metric(1.0 - metrics["chest_stack_over_landing"]["value"], metrics["chest_stack_over_landing"]["confidence"]), 0.20),
            ]
        )
        late_arm_chase = _combine_metrics(
            [
                (metrics["distal_velocity_rescue"], 0.60),
                (_metric(1.0 - metrics["release_timing_stability"]["value"], metrics["release_timing_stability"]["confidence"]), 0.40),
            ]
        )
        distributed_load_index = _metric(
            1.0 - metrics["dissipation_burden_score"]["value"],
            metrics["dissipation_burden_score"]["confidence"],
        )
        late_chain_dependence_index = _combine_metrics(
            [
                (metrics["terminal_impulse_score"], 0.45),
                (late_arm_chase, 0.35),
                (_metric(1.0 - transfer["value"], transfer["confidence"]), 0.20),
            ]
        )

        pace_leakage = []
        if leakage_before_block["value"] >= 0.45:
            pace_leakage.append(
                {
                    "stage": "before_block",
                    "severity": _round3(leakage_before_block["value"]),
                    "reason": selection.get("primary_mechanism_id") or "gather_or_transfer_disorganization",
                }
            )
        if transfer["value"] < 0.55:
            pace_leakage.append(
                {
                    "stage": "transfer_and_block",
                    "severity": _round3(1.0 - transfer["value"]),
                    "reason": selection.get("primary_mechanism_id") or "transfer_efficiency_below_clean",
                }
            )
        if leakage_at_block["value"] >= 0.45 or block["value"] < 0.55:
            pace_leakage.append(
                {
                    "stage": "front_foot_block",
                    "severity": _round3(max(1.0 - block["value"], leakage_at_block["value"])),
                    "reason": selection.get("primary_mechanism_id") or "soft_block_pattern",
                }
            )

        return {
            "version": "kinetic_chain_v1",
            "knowledge_pack_version": self._pack["pack_version"],
            "diagnosis_status": selection["diagnosis_status"],
            "confidence": selection["overall_confidence"],
            "capture_quality": capture_quality,
            "archetype": archetype,
            "approach_build": self._phase_payload("approach_build", approach, metrics),
            "gather_and_organize": self._phase_payload("gather_and_organize", gather, metrics),
            "transfer": self._phase_payload("transfer", transfer, metrics),
            "block": self._phase_payload("block", block, metrics),
            "release_generation": self._phase_payload("release_generation", release_generation, metrics),
            "dissipation": self._phase_payload("dissipation", dissipation, metrics),
            "pace_translation": {
                "approach_momentum": approach["value"],
                "transfer_efficiency": transfer["value"],
                "terminal_impulse": metrics["terminal_impulse_score"]["value"],
                "leakage_before_block": leakage_before_block["value"],
                "leakage_at_block": leakage_at_block["value"],
                "late_arm_chase": late_arm_chase["value"],
                "dissipation_burden": metrics["dissipation_burden_score"]["value"],
                "distributed_load_index": distributed_load_index["value"],
                "late_chain_dependence_index": late_chain_dependence_index["value"],
                "pace_leakage": pace_leakage,
            },
            "internal_metrics": {
                "runup_build_score": approach["value"],
                "delivery_stride_transfer_score": transfer["value"],
                "front_leg_transfer_score": metrics["front_leg_support_score"]["value"],
                "late_thrust_score": metrics["terminal_impulse_score"]["value"],
                "arm_chase_score": late_arm_chase["value"],
                "dissipation_burden_score": metrics["dissipation_burden_score"]["value"],
                "pace_leakage_score": _round3(max(leakage_before_block["value"], leakage_at_block["value"])),
            },
            "derived_indices": {
                "pace_translation_efficiency": transfer["value"],
                "distributed_load_index": distributed_load_index["value"],
                "late_chain_dependence_index": late_chain_dependence_index["value"],
                "terminal_violence_index": metrics["terminal_impulse_score"]["value"],
            },
            "mechanism_hypotheses": [
                {
                    "id": item["id"],
                    "confidence": item["overall_confidence"],
                }
                for item in hypotheses[:3]
            ],
            "historical_context": {
                "prior_run_count": history_context["prior_run_count"],
                "recent_dominant_mechanism_id": history_context["dominant_mechanism_id"],
                "recent_dominant_archetype_id": history_context["dominant_archetype_id"],
            },
            "selected_render_story_ids": selection["selected_render_story_ids"],
        }

    def _phase_payload(
        self,
        phase_key: str,
        metric_cfg: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = _safe_float(metric_cfg.get("value"), 0.0)
        label = str(metric_cfg.get("label") or _metric_band(score))
        return {
            "score": _round3(score),
            "label": label,
            "notes": self._phase_notes(phase_key, score=score, metrics=metrics),
        }

    def _phase_notes(
        self,
        phase_key: str,
        *,
        score: float,
        metrics: Dict[str, Any],
    ) -> List[str]:
        notes: List[str] = []
        if phase_key == "approach_build":
            if score < 0.45:
                notes.append("Run-up build looks too modest to carry the delivery cleanly.")
            elif score < 0.65:
                notes.append("Approach intent is present, but the build is not fully convincing yet.")
            else:
                notes.append("Approach momentum looks usable and mostly connected.")
        elif phase_key == "gather_and_organize":
            if score < 0.45:
                notes.append("The body is arriving at the crease without enough organization for clean transfer.")
            elif score < 0.65:
                notes.append("Gather organization is mixed and still leaks a little before landing.")
            else:
                notes.append("Gather line and pelvis-trunk organization look fairly calm.")
        elif phase_key == "transfer":
            if score < 0.45:
                notes.append("Momentum is leaking before or at landing instead of moving through the chain.")
            elif score < 0.65:
                notes.append("Transfer is present but not clean enough yet.")
            else:
                notes.append("Transfer into release looks reasonably connected.")
        elif phase_key == "block":
            if score < 0.45:
                notes.append("Landing does not become a stable transfer point.")
            elif score < 0.65:
                notes.append("The landing base supports some transfer, but trunk carry is still visible.")
            else:
                notes.append("Landing support looks fairly stable for this clip.")
        elif phase_key == "release_generation":
            if metrics["terminal_impulse_score"]["value"] >= 0.6:
                notes.append("A large share of visible pace generation is happening late near release.")
            else:
                notes.append("Release looks less like a late rescue and more like carried momentum.")
        elif phase_key == "dissipation":
            if score >= 0.6:
                notes.append("The body appears to be paying a concentrated late-chain load cost after release.")
            else:
                notes.append("Dissipation looks more distributed and less abrupt.")
        return notes

    def _build_mechanism_explanation(
        self,
        symptoms: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]],
        selection: Dict[str, Any],
        *,
        archetype: Optional[Dict[str, Any]],
        account_role: Optional[str],
        history_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        wording = self._pack["wording"]
        selected_surface = self._wording_surface(account_role, wording)
        primary = selection.get("primary")
        secondary = selection.get("secondary") or []
        if not primary:
            top_symptom = symptoms[0] if symptoms else None
            surface_variants = self._unknown_surface_variants(wording)
            selected_variant = surface_variants[selected_surface]
            return {
                "version": "mechanism_explanation_v1",
                "knowledge_pack_version": self._pack["pack_version"],
                "diagnosis_status": selection["diagnosis_status"],
                "primary_symptom": top_symptom["title"] if top_symptom else "No strong symptom bundle yet",
                "primary_mechanism": selected_variant["primary_mechanism"],
                "secondary_contributors": [],
                "performance_impact": selected_variant["performance_impact"],
                "load_impact": selected_variant["load_impact"],
                "first_intervention": None,
                "coach_check": selected_variant["coach_check"],
                "selected_surface": selected_surface,
                "surface_variants": surface_variants,
                "selected_render_story_ids": [],
                "selected_history_binding_ids": [],
                "archetype": archetype,
                "history_story": history_context.get("history_story"),
            }

        primary_cfg = self._pack["mechanisms"][primary["id"]]
        present_symptoms = {symptom["id"]: symptom for symptom in symptoms if symptom["present"]}
        primary_symptom_title = None
        for symptom_id in primary_cfg["required_symptoms"]:
            if symptom_id in present_symptoms:
                primary_symptom_title = present_symptoms[symptom_id]["title"]
                break
        if not primary_symptom_title and symptoms:
            primary_symptom_title = symptoms[0]["title"]

        secondary_contributors = []
        for symptom_id in primary_cfg["supporting_symptoms"]:
            symptom = present_symptoms.get(symptom_id)
            if symptom:
                secondary_contributors.append(symptom["title"])
        for item in secondary:
            secondary_contributors.append(item["title"])
        secondary_contributors = list(dict.fromkeys(secondary_contributors))[:3]

        first_prescription_id = primary_cfg["prescription_ids"][0] if primary_cfg["prescription_ids"] else None
        first_prescription = self._pack["prescriptions"].get(first_prescription_id) if first_prescription_id else None
        surface_variants = self._surface_variants(
            wording=wording,
            diagnosis_status=selection["diagnosis_status"],
            primary_title=primary["title"],
            primary_summary=primary_cfg["summary"],
            performance_effects=primary_cfg["performance_effects"],
            load_effects=primary_cfg["load_effects"],
            first_intervention=first_prescription["primary_cue"] if first_prescription else None,
            coach_check=first_prescription["coach_check"] if first_prescription else None,
            history_story=history_context.get("history_story"),
        )
        selected_variant = surface_variants[selected_surface]

        return {
            "version": "mechanism_explanation_v1",
            "knowledge_pack_version": self._pack["pack_version"],
            "diagnosis_status": selection["diagnosis_status"],
            "primary_symptom": primary_symptom_title,
            "primary_mechanism": primary["title"],
            "primary_mechanism_summary": selected_variant["primary_mechanism_summary"],
            "secondary_contributors": secondary_contributors,
            "performance_impact": selected_variant["performance_impact"],
            "load_impact": selected_variant["load_impact"],
            "first_intervention": selected_variant["first_intervention"],
            "coach_check": selected_variant["coach_check"],
            "selected_surface": selected_surface,
            "surface_variants": surface_variants,
            "selected_render_story_ids": selection["selected_render_story_ids"],
            "selected_history_binding_ids": self._history_binding_ids(selection),
            "archetype": archetype,
            "history_story": selected_variant["history_story"],
        }

    def _select_archetype(
        self,
        *,
        metrics: Dict[str, Any],
        selection: Dict[str, Any],
        history_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        approach = _safe_float(metrics["approach_momentum_score"]["value"], 0.0)
        transfer = _safe_float(metrics["transfer_efficiency_score"]["value"], 0.0)
        terminal = _safe_float(metrics["terminal_impulse_score"]["value"], 0.0)
        dissipation = _safe_float(metrics["dissipation_burden_score"]["value"], 0.0)
        primary_mechanism_id = selection.get("primary_mechanism_id")
        diagnosis_status = str(selection.get("diagnosis_status") or "")

        archetype_id: Optional[str] = None
        if primary_mechanism_id == "soft_block_with_trunk_carry":
            archetype_id = "soft_block_leakage_bowler"
        elif primary_mechanism_id == "late_arm_acceleration_due_to_chain_delay":
            archetype_id = "arm_chase_bowler"
        elif primary_mechanism_id == "high_terminal_impulse_with_high_dissipation_burden":
            archetype_id = "late_thrust_pace_bowler"
        elif primary_mechanism_id == "low_build_forcing_late_rescue":
            if terminal >= 0.65:
                archetype_id = "late_thrust_pace_bowler"
            elif approach < 0.55 and transfer >= 0.65:
                archetype_id = "low_build_high_efficiency_bowler"
        elif approach >= 0.65 and transfer >= 0.65 and dissipation < 0.45:
            archetype_id = "efficient_transfer_bowler"
        elif approach < 0.55 and transfer >= 0.65 and terminal < 0.55:
            archetype_id = "low_build_high_efficiency_bowler"
        elif approach >= 0.65 and transfer < 0.55:
            archetype_id = "high_build_low_conversion_bowler"
        elif terminal >= 0.65 and dissipation >= 0.60:
            archetype_id = "late_thrust_pace_bowler"

        dominant_archetype_id = history_context.get("dominant_archetype_id")
        dominant_ratio = _safe_float(history_context.get("dominant_archetype_ratio"), 0.0)
        if (
            diagnosis_status in {"weak_match", "no_match"}
            and dominant_archetype_id
            and dominant_ratio >= 0.6
        ):
            archetype_id = str(dominant_archetype_id)
        elif not archetype_id and dominant_archetype_id and dominant_ratio >= 0.6:
            archetype_id = str(dominant_archetype_id)
        elif (
            archetype_id
            and dominant_archetype_id == archetype_id
            and dominant_ratio >= 0.5
        ):
            pass

        if not archetype_id:
            return None
        cfg = self._pack["archetypes"].get(archetype_id)
        if not cfg:
            return None
        history_support_runs = 0
        for entry in history_context.get("recent_entries") or []:
            if (entry.get("archetype_id") or "") == archetype_id:
                history_support_runs += 1
        return {
            "id": archetype_id,
            "title": cfg["title"],
            "short_label": cfg["short_label"],
            "summary": cfg["summary"],
            "history_story_template": cfg["history_story_template"],
            "coaching_priority_template": cfg["coaching_priority_template"],
            "selection_basis": (
                "historical_consensus"
                if dominant_archetype_id == archetype_id and dominant_ratio >= 0.5
                else "current_clip"
            ),
            "history_support_runs": history_support_runs,
        }

    def _build_prescription_plan(
        self,
        selection: Dict[str, Any],
        *,
        prescription_allowed: bool,
    ) -> Dict[str, Any]:
        if not prescription_allowed:
            return {
                "version": "prescription_plan_v1",
                "knowledge_pack_version": self._pack["pack_version"],
                "prescriptions": [],
                "primary_prescription_id": None,
                "suppressed": True,
            }
        prescriptions: List[Dict[str, Any]] = []
        for prescription_id in selection["selected_prescription_ids"]:
            cfg = self._pack["prescriptions"].get(prescription_id)
            if not cfg:
                continue
            prescriptions.append(
                {
                    "id": prescription_id,
                    "title": cfg["title"],
                    "goal": cfg["goal"],
                    "primary_cue": cfg["primary_cue"],
                    "why_this_first": cfg["why_this_first"],
                    "coach_check": cfg["coach_check"],
                    "reassess_after": cfg["reassess_after"],
                    "review_window_type": cfg["review_window_type"],
                    "followup_metric_targets": list(cfg["followup_metric_targets"]),
                    "avoid_for_now": list(cfg["avoid_for_now"]),
                }
            )
        return {
            "version": "prescription_plan_v1",
            "knowledge_pack_version": self._pack["pack_version"],
            "prescriptions": prescriptions,
            "primary_prescription_id": prescriptions[0]["id"] if prescriptions else None,
            "suppressed": False,
        }

    def _build_history_plan(
        self,
        selection: Dict[str, Any],
        *,
        metrics: Dict[str, Any],
        archetype: Optional[Dict[str, Any]],
        history_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        binding_ids = self._history_binding_ids(selection)
        bindings: List[Dict[str, Any]] = []
        for binding_id in binding_ids:
            cfg = self._pack["history_bindings"].get(binding_id)
            if not cfg:
                continue
            bindings.append(
                {
                    "id": binding_id,
                    "title": cfg["title"],
                    "primary_metric": cfg["primary_metric"],
                    "metrics": list(cfg["metrics"]),
                    "chart_summary": cfg["chart_summary"],
                    "followup_check_ids": list(cfg["followup_check_ids"]),
                }
            )

        binding_trends = self._binding_trends(
            binding_ids=binding_ids,
            metrics=metrics,
            history_context=history_context,
        )
        followup_checks: List[Dict[str, Any]] = []
        selected_check_ids: List[str] = []
        for trajectory_id in selection["selected_trajectory_ids"]:
            trajectory = self._pack["trajectories"].get(trajectory_id)
            if not trajectory:
                continue
            for check_id in trajectory["followup_signals"]:
                if check_id not in selected_check_ids:
                    selected_check_ids.append(check_id)
        for check_id in selected_check_ids:
            cfg = self._pack["followup_checks"].get(check_id)
            if not cfg:
                continue
            followup_checks.append(
                {
                    "id": check_id,
                    "title": cfg["title"],
                    "recommended_review_window": cfg["recommended_review_window"],
                    "history_graph_binding": cfg["history_graph_binding"],
                    "success_signals": list(cfg["success_signals"]),
                    "failure_signals": list(cfg["failure_signals"]),
                }
            )

        render_stories: List[Dict[str, Any]] = []
        for story_id in selection["selected_render_story_ids"]:
            cfg = self._pack["render_stories"].get(story_id)
            if not cfg:
                continue
            render_stories.append(
                {
                    "id": story_id,
                    "title": cfg["title"],
                    "phases": list(cfg["phases"]),
                    "focus_regions": list(cfg["focus_regions"]),
                }
            )

        return {
            "version": "history_plan_v1",
            "knowledge_pack_version": self._pack["pack_version"],
            "history_window_runs": self.history_window_runs,
            "prior_run_count": history_context["prior_run_count"],
            "history_story": history_context.get("history_story"),
            "coaching_priority": (
                archetype.get("coaching_priority_template")
                if isinstance(archetype, dict)
                else None
            ),
            "archetype_history": {
                "dominant_archetype_id": history_context.get("dominant_archetype_id"),
                "dominant_archetype_ratio": history_context.get("dominant_archetype_ratio"),
            },
            "mechanism_history": {
                "dominant_mechanism_id": history_context.get("dominant_mechanism_id"),
                "dominant_mechanism_ratio": history_context.get("dominant_mechanism_ratio"),
            },
            "history_bindings": bindings,
            "binding_trends": binding_trends,
            "followup_checks": followup_checks,
            "render_stories": render_stories,
        }

    def _build_coach_diagnosis(
        self,
        *,
        events: Dict[str, Any],
        risks: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        symptoms: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]],
        selection: Dict[str, Any],
        capture_quality: Dict[str, Any],
        render_reasoning: Dict[str, Any],
        mechanism_explanation: Dict[str, Any],
        prescription_plan: Dict[str, Any],
        history_plan: Dict[str, Any],
        archetype: Optional[Dict[str, Any]],
        history_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnosis_status = str(selection.get("diagnosis_status") or "no_match")
        state = self._presentation_state(diagnosis_status)
        primary = selection.get("primary") or {}
        if not isinstance(primary, dict):
            primary = {}

        visible_symptom = self._visible_symptom(symptoms)
        supporting_contributors = self._supporting_contributors(
            risks=risks,
            metrics=metrics,
            symptoms=symptoms,
            selection=selection,
        )
        upper_body = [
            item for item in supporting_contributors if item.get("body_group") == "upper_body"
        ]
        lower_body = [
            item for item in supporting_contributors if item.get("body_group") == "lower_body"
        ]
        phase_findings = self._phase_anchored_findings(
            events=events,
            visible_symptom=visible_symptom,
            contributors=supporting_contributors,
        )

        trajectory_ids = list(selection.get("selected_trajectory_ids") or [])
        trajectories = [
            self._pack["trajectories"][trajectory_id]
            for trajectory_id in trajectory_ids
            if trajectory_id in self._pack["trajectories"]
        ]
        near_term = trajectories[0]["performance_consequence"] if trajectories else None
        medium_term = trajectories[0]["repeatability_consequence"] if trajectories else None
        long_term = trajectories[0]["load_consequence"] if trajectories else None

        prescriptions = prescription_plan.get("prescriptions") or []
        primary_prescription = prescriptions[0] if prescriptions else {}
        followup_checks = history_plan.get("followup_checks") or []
        primary_followup = followup_checks[0] if followup_checks else {}
        history_bindings = history_plan.get("history_bindings") or []
        render_stories = history_plan.get("render_stories") or []
        primary_break_point = self._primary_break_point(
            primary=primary,
            visible_symptom=visible_symptom,
        )
        change_strategy = self._change_strategy(
            capture_quality=capture_quality,
            diagnosis_status=diagnosis_status,
            primary_prescription=primary_prescription,
            primary_followup=primary_followup,
            trajectories=trajectories,
        )

        what_is_ok = self._what_is_ok(
            metrics=metrics,
            archetype=archetype,
        )
        what_is_not_ok = self._what_is_not_ok(
            visible_symptom=visible_symptom,
            primary=primary,
            contributors=supporting_contributors,
        )
        compensations = self._compensation_patterns(symptoms)

        holdback = {
            "reason": (
                selection.get("no_match_reason")
                or mechanism_explanation.get("performance_impact")
                or None
            ),
            "top_candidates": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "overall_confidence": item.get("overall_confidence"),
                }
                for item in hypotheses[:2]
                if isinstance(item, dict)
            ],
        }

        return {
            "version": "coach_diagnosis_v1",
            "knowledge_pack_id": self._pack["pack_id"],
            "knowledge_pack_version": self._pack["pack_version"],
            "state": state,
            "diagnosis_status": diagnosis_status,
            "capture_quality_status": capture_quality.get("status"),
            "visible_symptom": visible_symptom,
            "primary_mechanism": (
                {
                    "id": primary.get("id"),
                    "title": primary.get("title"),
                    "family": primary.get("family"),
                    "summary": primary.get("summary"),
                    "overall_confidence": primary.get("overall_confidence"),
                    "matched_symptom_ids": list(primary.get("matched_symptom_ids") or []),
                    "history_story": mechanism_explanation.get("history_story"),
                }
                if primary
                else None
            ),
            "supporting_contributors": supporting_contributors,
            "upper_body_contributors": upper_body,
            "lower_body_contributors": lower_body,
            "compensations": compensations,
            "what_is_ok": what_is_ok,
            "what_is_not_ok": what_is_not_ok,
            "primary_break_point": primary_break_point,
            "near_term_effect": near_term,
            "medium_term_effect": medium_term,
            "long_term_outlook": long_term,
            "first_priority": (
                {
                    "prescription_id": primary_prescription.get("id"),
                    "title": primary_prescription.get("title"),
                    "goal": primary_prescription.get("goal"),
                    "primary_cue": primary_prescription.get("primary_cue"),
                    "why_this_first": primary_prescription.get("why_this_first"),
                }
                if primary_prescription
                else None
            ),
            "change_strategy": change_strategy,
            "do_not_change_yet": list(primary_prescription.get("avoid_for_now") or []),
            "improvement_check": (
                {
                    "coach_check": mechanism_explanation.get("coach_check"),
                    "check_id": primary_followup.get("id"),
                    "title": primary_followup.get("title"),
                    "recommended_review_window": primary_followup.get("recommended_review_window"),
                    "success_signals": list(primary_followup.get("success_signals") or []),
                    "failure_signals": list(primary_followup.get("failure_signals") or []),
                    "history_graph_binding": primary_followup.get("history_graph_binding"),
                    "followup_metric_targets": list(primary_prescription.get("followup_metric_targets") or []),
                }
            ),
            "phase_anchored_findings": phase_findings,
            "renderer_bindings": {
                "renderer_mode": render_reasoning.get("renderer_mode"),
                "selected_story_id": render_reasoning.get("selected_story_id"),
                "story_ids": list(selection.get("selected_render_story_ids") or []),
                "stories": render_stories,
            },
            "history_bindings": {
                "history_window_runs": history_plan.get("history_window_runs"),
                "prior_run_count": history_context.get("prior_run_count"),
                "dominant_mechanism_id": history_context.get("dominant_mechanism_id"),
                "dominant_archetype_id": history_context.get("dominant_archetype_id"),
                "bindings": history_bindings,
            },
            "archetype": (
                {
                    "id": archetype.get("id"),
                    "title": archetype.get("title"),
                    "short_label": archetype.get("short_label"),
                    "summary": archetype.get("summary"),
                }
                if archetype
                else None
            ),
            "holdback": holdback if state != "MATCH" else None,
        }

    def _visible_symptom(self, symptoms: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for symptom in symptoms:
            if symptom.get("present"):
                return {
                    "id": symptom.get("id"),
                    "title": symptom.get("title"),
                    "phase": str(symptom.get("phase") or "").upper(),
                    "summary": symptom.get("description"),
                    "severity": symptom.get("severity"),
                    "confidence": symptom.get("confidence"),
                }
        if symptoms:
            symptom = symptoms[0]
            return {
                "id": symptom.get("id"),
                "title": symptom.get("title"),
                "phase": str(symptom.get("phase") or "").upper(),
                "summary": symptom.get("description"),
                "severity": symptom.get("severity"),
                "confidence": symptom.get("confidence"),
            }
        return None

    def _supporting_contributors(
        self,
        *,
        risks: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        symptoms: List[Dict[str, Any]],
        selection: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        contributors: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        def add(item: Optional[Dict[str, Any]]) -> None:
            if not item:
                return
            item_id = str(item.get("id") or "").strip()
            if not item_id or item_id in seen_ids:
                return
            seen_ids.add(item_id)
            contributors.append(item)

        symptom_lookup = {
            str(symptom.get("id") or ""): symptom
            for symptom in symptoms
            if isinstance(symptom, dict)
        }
        primary = selection.get("primary") or {}
        if isinstance(primary, dict):
            for symptom_id in list(primary.get("supporting_symptom_ids") or []):
                symptom = symptom_lookup.get(str(symptom_id))
                if not symptom or not symptom.get("present"):
                    continue
                add(
                    {
                        "id": symptom.get("id"),
                        "title": symptom.get("title"),
                        "body_group": self._symptom_body_group(str(symptom.get("id") or "")),
                        "phase": str(symptom.get("phase") or "").upper(),
                        "role": "supporting_symptom",
                        "summary": symptom.get("description"),
                    }
                )

        for risk in risks:
            if not isinstance(risk, dict):
                continue
            risk_id = str(risk.get("risk_id") or "").strip()
            cfg = _RISK_CONTRIBUTOR_CATALOG.get(risk_id)
            if not cfg:
                continue
            signal = _safe_float(risk.get("signal_strength"), 0.0)
            if signal < 0.35:
                continue
            add(
                {
                    "id": risk_id,
                    "title": cfg["title"],
                    "body_group": cfg["body_group"],
                    "phase": cfg["phase"],
                    "role": "supporting_risk",
                    "signal_strength": _round3(signal),
                    "summary": cfg["summary"],
                }
            )

        for metric_name, cfg in _METRIC_CONTRIBUTOR_CATALOG.items():
            metric = metrics.get(metric_name) or {}
            if not isinstance(metric, dict):
                continue
            value = _safe_float(metric.get("value"), 0.0)
            confidence = _safe_float(metric.get("confidence"), 0.0)
            if confidence < 0.15:
                continue
            severity = (1.0 - value) if metric_name != "trunk_drift_after_ffc" else value
            if severity < 0.4:
                continue
            add(
                {
                    "id": metric_name,
                    "title": cfg["title"],
                    "body_group": cfg["body_group"],
                    "phase": cfg["phase"],
                    "role": "supporting_metric",
                    "signal_strength": _round3(severity),
                    "summary": cfg["summary"],
                }
            )

        return contributors[:6]

    def _phase_anchored_findings(
        self,
        *,
        events: Dict[str, Any],
        visible_symptom: Optional[Dict[str, Any]],
        contributors: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        if visible_symptom:
            findings.append(
                {
                    "id": visible_symptom.get("id"),
                    "title": visible_symptom.get("title"),
                    "phase": visible_symptom.get("phase"),
                    "role": "visible_symptom",
                    "summary": visible_symptom.get("summary"),
                }
            )
        for contributor in contributors[:4]:
            findings.append(
                {
                    "id": contributor.get("id"),
                    "title": contributor.get("title"),
                    "phase": contributor.get("phase"),
                    "role": contributor.get("role"),
                    "summary": contributor.get("summary"),
                }
            )

        anchor_lookup = {
            "BFC": "bfc",
            "FFC": "ffc",
            "UAH": "uah",
            "RELEASE": "release",
        }
        for finding in findings:
            phase = str(finding.get("phase") or "").upper()
            event_key = anchor_lookup.get(phase)
            event = (events or {}).get(event_key) or {}
            if isinstance(event, dict):
                finding["anchor_frame"] = event.get("frame")
                finding["anchor_confidence"] = _safe_float(event.get("confidence"), 0.0)
        return findings

    def _primary_break_point(
        self,
        *,
        primary: Dict[str, Any],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        phase_key = None
        if primary:
            mechanism_cfg = self._pack["mechanisms"].get(str(primary.get("id") or "")) or {}
            phases = list(mechanism_cfg.get("primary_phases") or [])
            if phases:
                phase_key = str(phases[0]).strip()
        if not phase_key and visible_symptom:
            phase_key = str(visible_symptom.get("phase") or "").strip()
        if not phase_key:
            return None
        return {
            "phase_id": phase_key,
            "title": _PHASE_LABELS.get(phase_key, phase_key.replace("_", " ").title()),
            "summary": self._break_point_summary(phase_key),
        }

    def _break_point_summary(self, phase_key: str) -> str:
        mapping = {
            "approach_build": "The chain is losing value before enough usable build reaches the crease.",
            "gather_and_organize": "The body is arriving without enough organization to set up clean transfer.",
            "transfer_and_block": "The main leak is around landing, where momentum is not becoming a stable transfer point.",
            "whip_and_release": "The chain is reaching release late enough that the distal segments have to rescue timing or pace.",
            "dissipation_and_recovery": "The action is paying for pace after release through a harsh or concentrated finish.",
            "BFC": "The chain is already showing a visible organization issue at back-foot contact.",
            "FFC": "The key leak is visible at front-foot contact.",
            "RELEASE": "The issue becomes clearest right at release.",
        }
        return mapping.get(
            phase_key,
            "This is the main place where the chain is no longer carrying momentum cleanly.",
        )

    def _change_strategy(
        self,
        *,
        capture_quality: Dict[str, Any],
        diagnosis_status: str,
        primary_prescription: Dict[str, Any],
        primary_followup: Dict[str, Any],
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if str(capture_quality.get("status") or "").upper() == "UNUSABLE":
            return {
                "change_size": "hold",
                "adoption_risk": "unknown",
                "why_smallest_useful_change": "Capture quality is too weak to justify a coaching change yet.",
                "expected_near_term_tradeoff": "Do not introduce a mechanical change until the anchors are clearer.",
                "next_review_window": "Retest with a clearer clip first.",
                "improvement_signal": "Cleaner BFC, FFC, and release anchors before a correction is introduced.",
            }

        if diagnosis_status in {"no_match", "ambiguous_match"}:
            return {
                "change_size": "micro",
                "adoption_risk": "low",
                "why_smallest_useful_change": "The system is holding back, so the safest move is to review anchors before making a bigger change.",
                "expected_near_term_tradeoff": "This should protect near-term performance because it avoids a premature rebuild cue.",
                "next_review_window": primary_followup.get("recommended_review_window")
                or "Next 3 deliveries, then next session.",
                "improvement_signal": "The next clip should make one story clearer before a stronger intervention is chosen.",
            }

        avoid_for_now = list(primary_prescription.get("avoid_for_now") or [])
        if avoid_for_now:
            change_size = "micro"
        elif trajectories:
            change_size = "moderate"
        else:
            change_size = "micro"

        adoption_risk = "low" if change_size == "micro" else "medium"
        if len(avoid_for_now) >= 3:
            adoption_risk = "high"

        expected_tradeoff = (
            "This should be small enough to protect near-term performance while still changing the main leak."
            if change_size == "micro"
            else "This may feel different for a short period, so it should be introduced outside the highest-pressure moments."
        )

        return {
            "change_size": change_size,
            "adoption_risk": adoption_risk,
            "why_smallest_useful_change": (
                primary_prescription.get("why_this_first")
                or "This is the smallest useful intervention before larger changes are considered."
            ),
            "expected_near_term_tradeoff": expected_tradeoff,
            "next_review_window": primary_followup.get("recommended_review_window")
            or primary_prescription.get("reassess_after"),
            "improvement_signal": (
                (primary_followup.get("success_signals") or [None])[0]
                or (primary_prescription.get("followup_metric_targets") or [None])[0]
            ),
        }

    def _what_is_ok(
        self,
        *,
        metrics: Dict[str, Any],
        archetype: Optional[Dict[str, Any]],
    ) -> List[str]:
        positives: List[str] = []
        if isinstance(archetype, dict):
            cfg = self._pack["archetypes"].get(str(archetype.get("id") or "")) or {}
            for item in list(cfg.get("expected_strengths") or []):
                text = str(item).strip()
                if text:
                    positives.append(text)
        metric_strengths = [
            ("transfer_efficiency_score", "Transfer into release is staying reasonably connected."),
            ("approach_momentum_score", "Approach intent is bringing usable momentum into the crease."),
            ("release_timing_stability", "Release timing looks calmer than the main leak would suggest."),
            ("front_leg_support_score", "Landing support is giving the action at least some usable base."),
        ]
        for metric_name, text in metric_strengths:
            metric = metrics.get(metric_name) or {}
            if not isinstance(metric, dict):
                continue
            if _safe_float(metric.get("value"), 0.0) >= 0.65:
                positives.append(text)
        return list(dict.fromkeys(positives))[:3]

    def _what_is_not_ok(
        self,
        *,
        visible_symptom: Optional[Dict[str, Any]],
        primary: Dict[str, Any],
        contributors: List[Dict[str, Any]],
    ) -> List[str]:
        issues: List[str] = []
        if visible_symptom and visible_symptom.get("summary"):
            issues.append(str(visible_symptom["summary"]))
        if primary.get("summary"):
            issues.append(str(primary["summary"]))
        for contributor in contributors[:3]:
            summary = str(contributor.get("summary") or "").strip()
            if summary:
                issues.append(summary)
        return list(dict.fromkeys(issues))[:4]

    def _compensation_patterns(self, symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compensation_ids = {
            "arm_chase",
            "high_terminal_thrust",
            "asymmetric_dissipation",
            "late_chain_load_dump",
        }
        patterns: List[Dict[str, Any]] = []
        for symptom in symptoms:
            if str(symptom.get("id") or "") not in compensation_ids:
                continue
            if not symptom.get("present"):
                continue
            patterns.append(
                {
                    "id": symptom.get("id"),
                    "title": symptom.get("title"),
                    "phase": str(symptom.get("phase") or "").upper(),
                    "summary": symptom.get("description"),
                }
            )
        return patterns[:3]

    def _symptom_body_group(self, symptom_id: str) -> str:
        upper = {"late_trunk_drift", "arm_chase"}
        lower = {"front_leg_softening", "unstable_gather"}
        if symptom_id in upper:
            return "upper_body"
        if symptom_id in lower:
            return "lower_body"
        return "whole_chain"

    def _history_binding_ids(self, selection: Dict[str, Any]) -> List[str]:
        selected_mechanism_ids = set(selection["selected_mechanism_ids"])
        selected_trajectory_ids = set(selection["selected_trajectory_ids"])
        binding_ids: List[str] = []
        for binding_id, cfg in self._pack["history_bindings"].items():
            if selected_mechanism_ids.intersection(cfg["mechanism_ids"]) or selected_trajectory_ids.intersection(cfg["trajectory_ids"]):
                binding_ids.append(binding_id)
        return binding_ids

    def _summarize_prior_results(self, prior_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        tracked_metric_keys = {
            "approach_momentum_score",
            "gather_line_stability",
            "pelvis_trunk_alignment",
            "front_leg_support_score",
            "trunk_drift_after_ffc",
            "transfer_efficiency_score",
            "release_timing_stability",
            "terminal_impulse_score",
            "distal_velocity_rescue",
            "dissipation_burden_score",
            "followthrough_asymmetry",
            "trunk_fold_severity",
            "chest_stack_over_landing",
            "shoulder_rotation_timing",
        }
        entries: List[Dict[str, Any]] = []
        for raw_entry in prior_results[: self.history_window_runs]:
            if not isinstance(raw_entry, dict):
                continue
            result_json = raw_entry.get("result_json") or {}
            if not isinstance(result_json, dict):
                continue
            deterministic = result_json.get("deterministic_expert_v1") or {}
            if not isinstance(deterministic, dict):
                continue
            selection = deterministic.get("selection") or {}
            metrics = deterministic.get("metrics") or {}
            archetype = deterministic.get("archetype_v1") or {}
            if not isinstance(selection, dict) or not isinstance(metrics, dict):
                continue
            entry_metrics: Dict[str, float] = {}
            for metric_key in tracked_metric_keys:
                metric_cfg = metrics.get(metric_key) or {}
                if isinstance(metric_cfg, dict) and isinstance(metric_cfg.get("value"), (int, float)):
                    entry_metrics[metric_key] = _round3(_safe_float(metric_cfg.get("value"), 0.0))
            entries.append(
                {
                    "run_id": raw_entry.get("run_id"),
                    "created_at": raw_entry.get("created_at"),
                    "diagnosis_status": selection.get("diagnosis_status"),
                    "primary_mechanism_id": selection.get("primary_mechanism_id"),
                    "archetype_id": archetype.get("id") if isinstance(archetype, dict) else None,
                    "metrics": entry_metrics,
                }
            )

        mechanism_counts = Counter(
            str(entry.get("primary_mechanism_id") or "")
            for entry in entries
            if str(entry.get("primary_mechanism_id") or "")
        )
        archetype_counts = Counter(
            str(entry.get("archetype_id") or "")
            for entry in entries
            if str(entry.get("archetype_id") or "")
        )
        dominant_mechanism_id = mechanism_counts.most_common(1)[0][0] if mechanism_counts else None
        dominant_archetype_id = archetype_counts.most_common(1)[0][0] if archetype_counts else None
        prior_run_count = len(entries)
        dominant_mechanism_ratio = (
            round(mechanism_counts[dominant_mechanism_id] / prior_run_count, 3)
            if dominant_mechanism_id and prior_run_count
            else 0.0
        )
        dominant_archetype_ratio = (
            round(archetype_counts[dominant_archetype_id] / prior_run_count, 3)
            if dominant_archetype_id and prior_run_count
            else 0.0
        )

        history_story = None
        if dominant_archetype_id:
            archetype_cfg = self._pack["archetypes"].get(dominant_archetype_id) or {}
            template = archetype_cfg.get("history_story_template")
            if isinstance(template, str) and template.strip():
                history_story = template

        return {
            "version": "expert_history_context_v1",
            "history_window_runs": self.history_window_runs,
            "prior_run_count": prior_run_count,
            "recent_entries": entries,
            "dominant_mechanism_id": dominant_mechanism_id,
            "dominant_mechanism_ratio": dominant_mechanism_ratio,
            "dominant_archetype_id": dominant_archetype_id,
            "dominant_archetype_ratio": dominant_archetype_ratio,
            "history_story": history_story,
        }

    def _binding_trends(
        self,
        *,
        binding_ids: List[str],
        metrics: Dict[str, Any],
        history_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        quick_window = int(self._pack["globals"]["history_window_defaults"]["quick_check_runs"])
        prior_entries = history_context.get("recent_entries") or []
        trends: List[Dict[str, Any]] = []
        for binding_id in binding_ids:
            cfg = self._pack["history_bindings"].get(binding_id)
            if not cfg:
                continue
            metric_trends: List[Dict[str, Any]] = []
            statuses: List[str] = []
            for metric_name in cfg["metrics"]:
                current_cfg = metrics.get(metric_name) or {}
                if not isinstance(current_cfg, dict) or not isinstance(current_cfg.get("value"), (int, float)):
                    continue
                current_value = _round3(_safe_float(current_cfg.get("value"), 0.0))
                previous_values = [
                    _safe_float((entry.get("metrics") or {}).get(metric_name), 0.0)
                    for entry in prior_entries
                    if isinstance((entry.get("metrics") or {}).get(metric_name), (int, float))
                ]
                quick_values = previous_values[:quick_window]
                quick_average = _round3(_average(quick_values, default=current_value))
                trend_average = _round3(_average(previous_values, default=current_value))
                delta_quick = _round3(current_value - quick_average)
                preferred_direction = self._preferred_direction(metric_name)
                status = self._trend_status(
                    metric_name=metric_name,
                    delta=delta_quick,
                    preferred_direction=preferred_direction,
                )
                statuses.append(status)
                metric_trends.append(
                    {
                        "metric": metric_name,
                        "preferred_direction": preferred_direction,
                        "current_value": current_value,
                        "quick_average": quick_average,
                        "trend_average": trend_average,
                        "delta_vs_quick_average": delta_quick,
                        "status": status,
                    }
                )
            binding_status = "steady"
            if any(status == "worse" for status in statuses):
                binding_status = "worse"
            elif any(status == "better" for status in statuses):
                binding_status = "better"
            trends.append(
                {
                    "id": binding_id,
                    "title": cfg["title"],
                    "status": binding_status,
                    "chart_summary": cfg["chart_summary"],
                    "recent_run_ids": [
                        entry.get("run_id")
                        for entry in prior_entries[:quick_window]
                        if entry.get("run_id")
                    ],
                    "metrics": metric_trends,
                }
            )
        return trends

    def _preferred_direction(self, metric_name: str) -> str:
        lower_is_better = {
            "trunk_drift_after_ffc",
            "terminal_impulse_score",
            "distal_velocity_rescue",
            "dissipation_burden_score",
            "followthrough_asymmetry",
            "trunk_fold_severity",
        }
        return "lower" if metric_name in lower_is_better else "higher"

    def _trend_status(
        self,
        *,
        metric_name: str,
        delta: float,
        preferred_direction: str,
    ) -> str:
        tolerance = 0.04
        if abs(delta) <= tolerance:
            return "steady"
        if preferred_direction == "lower":
            return "better" if delta < 0.0 else "worse"
        return "better" if delta > 0.0 else "worse"

    def _surface_variants(
        self,
        *,
        wording: Dict[str, Any],
        diagnosis_status: str,
        primary_title: str,
        primary_summary: str,
        performance_effects: List[str],
        load_effects: List[str],
        first_intervention: Optional[str],
        coach_check: Optional[str],
        history_story: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        surfaces = wording.get("surfaces") or {}
        status_leads = wording.get("status_leads") or {}
        variants: Dict[str, Dict[str, Any]] = {}
        for surface in surfaces.keys():
            surface_cfg = surfaces.get(surface) or {}
            lead_cfg = status_leads.get(diagnosis_status) or {}
            lead = str(lead_cfg.get(surface) or "").strip()
            prefix = f"{lead} " if lead else ""
            variants[surface] = {
                "primary_mechanism": f"{prefix}{primary_title}".strip(),
                "primary_mechanism_summary": (
                    f"{surface_cfg.get('primary_mechanism_prefix', wording['hedges']['primary_mechanism'])} {primary_summary.lower()}"
                ).strip(),
                "performance_impact": (
                    f"{surface_cfg.get('performance_prefix', wording['hedges']['performance'])} {' '.join(performance_effects).lower()}"
                ).strip(),
                "load_impact": (
                    f"{surface_cfg.get('load_prefix', wording['hedges']['load'])} {' '.join(load_effects).lower()}"
                ).strip(),
                "first_intervention": first_intervention,
                "coach_check": coach_check,
                "history_story": history_story,
            }
        return variants

    def _unknown_surface_variants(self, wording: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        legacy = wording["unknown_path"]
        unknown_surfaces = wording.get("unknown_path_surfaces") or {}
        variants: Dict[str, Dict[str, Any]] = {}
        for surface in (wording.get("surfaces") or {}).keys():
            surface_cfg = unknown_surfaces.get(surface) or {}
            variants[surface] = {
                "primary_mechanism": surface_cfg.get("primary_mechanism", legacy["primary_mechanism"]),
                "performance_impact": surface_cfg.get("performance_impact", legacy["performance_impact"]),
                "load_impact": surface_cfg.get("load_impact", legacy["load_impact"]),
                "coach_check": surface_cfg.get("coach_check", legacy["coach_check"]),
            }
        return variants

    def _wording_surface(self, account_role: Optional[str], wording: Dict[str, Any]) -> str:
        role = str(account_role or "").strip().lower()
        surfaces = wording.get("surfaces") or {}
        if role in surfaces:
            return role
        if role in {"coach", "reviewer", "clinician"} and "coach" in surfaces:
            return "coach"
        return "player"

    def _presentation_state(self, diagnosis_status: str) -> str:
        if diagnosis_status == "ambiguous_match":
            return "AMBIGUOUS"
        if diagnosis_status in {"confident_match", "partial_match", "weak_match"}:
            return "MATCH"
        return "NO_MATCH"

    def _build_presentation_payload(
        self,
        *,
        selection: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        capture_quality: Dict[str, Any],
        render_reasoning: Dict[str, Any],
        mechanism_explanation: Dict[str, Any],
        prescription_plan: Dict[str, Any],
        archetype: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        diagnosis_status = str(selection.get("diagnosis_status") or "no_match")
        state = self._presentation_state(diagnosis_status)
        payload: Dict[str, Any] = {
            "version": "presentation_payload_v1",
            "state": state,
            "diagnosis_status": diagnosis_status,
            "knowledge_pack_id": self._pack["pack_id"],
            "knowledge_pack_version": self._pack["pack_version"],
            "capture_quality_status": str(capture_quality.get("status") or "WEAK"),
            "renderer_mode": str(render_reasoning.get("renderer_mode") or "event_only"),
            "selected_surface": mechanism_explanation.get("selected_surface"),
            "selected_render_story_ids": list(selection.get("selected_render_story_ids") or []),
            "selected_history_binding_ids": list(
                mechanism_explanation.get("selected_history_binding_ids") or []
            ),
            "prescription_suppressed": bool(prescription_plan.get("suppressed")),
            "archetype": (
                {
                    "id": archetype.get("id"),
                    "title": archetype.get("title"),
                    "short_label": archetype.get("short_label"),
                }
                if archetype
                else None
            ),
        }
        if state == "MATCH":
            payload["match"] = {
                "primary_mechanism_id": selection.get("primary_mechanism_id"),
                "primary_mechanism": mechanism_explanation.get("primary_mechanism"),
                "primary_mechanism_summary": mechanism_explanation.get("primary_mechanism_summary"),
                "performance_impact": mechanism_explanation.get("performance_impact"),
                "load_impact": mechanism_explanation.get("load_impact"),
                "first_intervention": mechanism_explanation.get("first_intervention"),
                "coach_check": mechanism_explanation.get("coach_check"),
                "primary_prescription_id": prescription_plan.get("primary_prescription_id"),
                "coaching_priority": (prescription_plan.get("prescriptions") or [{}])[0].get("goal")
                if prescription_plan.get("prescriptions")
                else None,
            }
        elif state == "AMBIGUOUS":
            payload["ambiguous"] = {
                "top_candidates": [
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "overall_confidence": item.get("overall_confidence"),
                        "support_score": item.get("support_score"),
                        "contradiction_penalty": item.get("contradiction_penalty"),
                    }
                    for item in hypotheses[:2]
                    if isinstance(item, dict)
                ],
                "holdback_reason": mechanism_explanation.get("performance_impact"),
                "coach_check": mechanism_explanation.get("coach_check"),
            }
        else:
            payload["no_match"] = {
                "holdback_reason": mechanism_explanation.get("performance_impact"),
                "load_holdback_reason": mechanism_explanation.get("load_impact"),
                "coach_check": mechanism_explanation.get("coach_check"),
                "selected_render_story_ids": list(selection.get("selected_render_story_ids") or []),
            }
        return payload
