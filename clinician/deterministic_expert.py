from __future__ import annotations

from collections import Counter
import re
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
    acceptable_max: float = 0.25,
    workable_max: float = 0.50,
) -> Dict[str, Any]:
    risk = risk_lookup.get(risk_id) or {}
    signal = _clip01(_safe_float(risk.get("signal_strength"), 0.0))
    confidence = _clip01(max(_safe_float(risk.get("confidence"), 0.0), chain_quality * 0.45))
    adjusted = _banded_signal(signal, acceptable_max=acceptable_max, workable_max=workable_max)
    band = _acceptance_band(signal, acceptable_max=acceptable_max, workable_max=workable_max)
    metric = _metric(adjusted, confidence, label=band)
    metric["raw_value"] = _round3(signal)
    metric["acceptance_band"] = band
    return metric


def _basic_metric(
    basic_lookup: Dict[str, Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    basic = basic_lookup.get(key) or {}
    score = _status_score(str(basic.get("status") or "unknown"))
    confidence = _clip01(_safe_float(basic.get("confidence"), 0.0))
    return _metric(score, confidence)


def _acceptance_band(
    score: float,
    *,
    acceptable_max: float = 0.25,
    workable_max: float = 0.50,
) -> str:
    value = _clip01(score)
    if value <= acceptable_max:
        return "acceptable"
    if value <= workable_max:
        return "workable"
    return "problematic"


def _banded_signal(
    score: float,
    *,
    acceptable_max: float = 0.25,
    workable_max: float = 0.50,
) -> float:
    value = _clip01(score)
    if value <= acceptable_max:
        return 0.0
    if value <= workable_max:
        ratio = (value - acceptable_max) / max(1e-6, workable_max - acceptable_max)
        return _clip01(0.08 + (0.20 * ratio))
    ratio = (value - workable_max) / max(1e-6, 1.0 - workable_max)
    return _clip01(0.28 + (0.72 * ratio))


def _normalized_text(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
    return [token for token in cleaned.split() if len(token) > 2]


def _is_duplicate_story(candidate: str, existing: Iterable[str]) -> bool:
    candidate_tokens = set(_normalized_text(candidate))
    if not candidate_tokens:
        return True
    for item in existing:
        other_tokens = set(_normalized_text(item))
        if not other_tokens:
            continue
        overlap = len(candidate_tokens.intersection(other_tokens))
        smaller = max(1, min(len(candidate_tokens), len(other_tokens)))
        if overlap / smaller >= 0.6:
            return True
    return False


class DeterministicExpertSystem:
    def __init__(self, *, pack_version: Optional[str] = None):
        self._pack = load_knowledge_pack(pack_version)

    @property
    def history_window_runs(self) -> int:
        return int(self._pack["globals"]["history_window_defaults"]["trend_window_runs"])

    def _knowledge_evidence_items(self, target_type: str, target_id: Optional[str]) -> List[Dict[str, Any]]:
        normalized_target_id = str(target_id or "").strip()
        if not normalized_target_id:
            return []
        indexes = self._pack.get("runtime_indexes") or {}
        by_target = indexes.get("knowledge_evidence_by_target") or {}
        return list(((by_target.get(str(target_type)) or {}).get(normalized_target_id) or []))

    def _reconciliation_concept(self, target_type: str, target_id: Optional[str]) -> Optional[Dict[str, Any]]:
        normalized_target_id = str(target_id or "").strip()
        if not normalized_target_id:
            return None
        indexes = self._pack.get("runtime_indexes") or {}
        ref = f"{target_type}:{normalized_target_id}"
        concept = (indexes.get("reconciliation_target_concepts") or {}).get(ref)
        return dict(concept) if isinstance(concept, dict) else None

    def _knowledge_support_score(self, target_type: str, target_id: Optional[str]) -> float:
        tier_weight = {"A": 1.0, "B": 0.82, "C": 0.65, "INTERNAL": 0.55}
        consensus_weight = {"accepted": 1.0, "reviewed": 0.85, "draft": 0.65}
        kind_weight = {
            "biomechanics_truth": 1.0,
            "performance_relation": 0.92,
            "load_risk": 0.92,
            "coaching_translation": 0.78,
            "intervention_heuristic": 0.72,
        }
        evidence_items = self._knowledge_evidence_items(target_type, target_id)
        if not evidence_items:
            return 0.0
        weighted_scores: List[float] = []
        for item in evidence_items:
            weighted_scores.append(
                tier_weight.get(str(item.get("evidence_tier") or ""), 0.5)
                * consensus_weight.get(str(item.get("coach_consensus_status") or ""), 0.6)
                * kind_weight.get(str(item.get("evidence_kind") or ""), 0.65)
            )
        return _round3(_average(weighted_scores, default=0.0))

    def _role_detail_policy(self, account_role: Optional[str]) -> Dict[str, Any]:
        policies = (self._pack.get("coach_judgments") or {}).get("role_detail_policies") or {}
        role = str(account_role or "").strip().lower()
        if not role:
            return {}
        if role in policies:
            return dict(policies[role])
        if role in {"coach", "reviewer", "clinician"} and "coach" in policies:
            return dict(policies["coach"])
        if role == "parent" and "parent" in policies:
            return dict(policies["parent"])
        return dict(policies.get("player") or {})

    def _evidence_basis_for_target(self, target_type: str, target_id: Optional[str]) -> Optional[Dict[str, Any]]:
        normalized_target_id = str(target_id or "").strip()
        if not normalized_target_id:
            return None
        evidence_items = self._knowledge_evidence_items(target_type, normalized_target_id)
        concept = self._reconciliation_concept(target_type, normalized_target_id)
        if not evidence_items and not concept:
            return None
        return {
            "target_type": target_type,
            "target_id": normalized_target_id,
            "knowledge_support": self._knowledge_support_score(target_type, normalized_target_id),
            "evidence_items": [
                {
                    "id": item["id"],
                    "evidence_kind": item["evidence_kind"],
                    "evidence_tier": item["evidence_tier"],
                    "claim_summary": item["claim_summary"],
                    "source_ids": list(item["source_ids"]),
                    "coach_consensus_status": item["coach_consensus_status"],
                }
                for item in evidence_items
            ],
            "canonical_concept": concept,
        }

    def _simple_text(self, text: Optional[str]) -> Optional[str]:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return None
        wording = self._pack.get("wording") or {}
        simple_language = wording.get("simple_language") or {}
        exact_overrides = simple_language.get("exact_overrides") or {}
        if normalized in exact_overrides:
            return str(exact_overrides[normalized]).strip() or None
        rewritten = normalized
        for source, target in sorted(
            (simple_language.get("replacements") or {}).items(),
            key=lambda item: len(str(item[0] or "")),
            reverse=True,
        ):
            source_text = str(source or "").strip()
            target_text = str(target or "").strip()
            if not source_text or not target_text:
                continue
            rewritten = re.sub(
                re.escape(source_text),
                target_text,
                rewritten,
                flags=re.IGNORECASE,
            )
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]
        return rewritten or None

    def _cfg_simple_text(
        self,
        cfg: Optional[Dict[str, Any]],
        key: str,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        explicit = str(((cfg or {}).get(key) or "")).strip()
        if explicit:
            return explicit
        return self._simple_text(fallback)

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
        coach_diagnosis = self._filter_coach_diagnosis(
            account_role=account_role,
            coach_diagnosis=coach_diagnosis,
        )
        presentation_payload = self._build_presentation_payload(
            selection=selection,
            hypotheses=hypotheses,
            capture_quality=capture_quality,
            render_reasoning=render_reasoning,
            mechanism_explanation=mechanism_explanation,
            prescription_plan=prescription_plan,
            coach_diagnosis=coach_diagnosis,
            archetype=archetype,
        )
        frontend_surface = self._build_frontend_surface(
            account_role=account_role,
            coach_diagnosis=coach_diagnosis,
            presentation_payload=presentation_payload,
            render_reasoning=render_reasoning,
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
            "frontend_surface_v1": frontend_surface,
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
        risk_acceptance_cfg = (
            ((self._pack.get("globals") or {}).get("acceptance_bands") or {}).get("risk_signal")
            or {}
        )
        risk_acceptable_max = _safe_float(risk_acceptance_cfg.get("acceptable_max"), 0.25)
        risk_workable_max = _safe_float(risk_acceptance_cfg.get("workable_max"), 0.50)

        front_foot_braking = _risk_metric(
            risk_lookup,
            "front_foot_braking_shock",
            chain_quality=chain_quality,
            acceptable_max=risk_acceptable_max,
            workable_max=risk_workable_max,
        )
        knee_brace_failure = _risk_metric(
            risk_lookup,
            "knee_brace_failure",
            chain_quality=chain_quality,
            acceptable_max=risk_acceptable_max,
            workable_max=risk_workable_max,
        )
        trunk_rotation_snap = _risk_metric(
            risk_lookup,
            "trunk_rotation_snap",
            chain_quality=chain_quality,
            acceptable_max=risk_acceptable_max,
            workable_max=risk_workable_max,
        )
        hip_shoulder_mismatch = _risk_metric(
            risk_lookup,
            "hip_shoulder_mismatch",
            chain_quality=chain_quality,
            acceptable_max=risk_acceptable_max,
            workable_max=risk_workable_max,
        )
        lateral_trunk_lean = _risk_metric(
            risk_lookup,
            "lateral_trunk_lean",
            chain_quality=chain_quality,
            acceptable_max=risk_acceptable_max,
            workable_max=risk_workable_max,
        )
        foot_line_deviation = _risk_metric(
            risk_lookup,
            "foot_line_deviation",
            chain_quality=chain_quality,
            acceptable_max=risk_acceptable_max,
            workable_max=risk_workable_max,
        )

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
                (_metric(1.0 - front_foot_braking["value"], front_foot_braking["confidence"]), 0.10),
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
                (knee_brace_proxy, 0.35),
                (_metric(1.0 - front_foot_braking["value"], front_foot_braking["confidence"]), 0.05),
            ]
        )
        trunk_drift_after_ffc = _combine_metrics(
            [
                (front_foot_braking, 0.15),
                (lateral_trunk_lean, 0.55),
                (_metric(1.0 - front_leg_support_score["value"], front_leg_support_score["confidence"]), 0.30),
            ]
        )
        transfer_efficiency_score = _combine_metrics(
            [
                (front_leg_support_score, 0.35),
                (_metric(1.0 - front_foot_braking["value"], front_foot_braking["confidence"]), 0.05),
                (_metric(1.0 - hip_shoulder_mismatch["value"], hip_shoulder_mismatch["confidence"]), 0.25),
                (_metric(1.0 - trunk_drift_after_ffc["value"], trunk_drift_after_ffc["confidence"]), 0.25),
                (_metric(chain_quality, chain_quality), 0.10),
            ]
        )

        sequence_pattern = str((((risk_lookup.get("hip_shoulder_mismatch") or {}).get("debug")) or {}).get("sequence_pattern") or "unknown").lower()
        shoulder_rotation_timing_value = {
            "hips_lead": 0.82,
            "in_sync": 0.60,
            "shoulders_lead": 0.22,
        }.get(sequence_pattern, 0.45)
        shoulder_sequence_score = _metric(
            shoulder_rotation_timing_value,
            hip_shoulder_mismatch["confidence"],
        )
        shoulder_rotation_timing = _combine_metrics(
            [
                (shoulder_sequence_score, 0.40),
                (_metric(1.0 - hip_shoulder_mismatch["value"], hip_shoulder_mismatch["confidence"]), 0.20),
                (gather_line_stability, 0.20),
                (pelvis_trunk_alignment, 0.20),
            ]
        )

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
                    (_metric(1.0 - metrics["release_timing_stability"]["value"], metrics["release_timing_stability"]["confidence"]), 0.45),
                    (metrics["distal_velocity_rescue"], 0.30),
                    (_metric(1.0 - metrics["shoulder_rotation_timing"]["value"], metrics["shoulder_rotation_timing"]["confidence"]), 0.25),
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
                    "simple_title": self._cfg_simple_text(cfg, "simple_title", cfg["title"]),
                    "category": cfg["category"],
                    "phase": cfg["phase"],
                    "description": cfg["description"],
                    "simple_description": self._cfg_simple_text(
                        cfg,
                        "simple_description",
                        cfg["description"],
                    ),
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
            knowledge_support = self._knowledge_support_score("mechanism", mechanism_id)
            reconciliation_concept = self._reconciliation_concept("mechanism", mechanism_id)

            support_score = _clip01(
                (0.40 * _average(required_symptom_scores, default=0.0))
                + (0.15 * _average(supporting_symptom_scores, default=0.0))
                + (0.30 * _average(required_evidence_scores, default=0.0))
                + (0.15 * _average(supporting_evidence_scores, default=0.0))
                + (0.05 * knowledge_support)
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
                    "knowledge_support": knowledge_support,
                    "knowledge_evidence_ids": [
                        item["id"]
                        for item in self._knowledge_evidence_items("mechanism", mechanism_id)
                    ],
                    "canonical_concept": reconciliation_concept,
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
        same_reconciliation_story = bool(
            top
            and runner_up
            and isinstance(top.get("canonical_concept"), dict)
            and isinstance(runner_up.get("canonical_concept"), dict)
            and top["canonical_concept"].get("concept_id")
            == runner_up["canonical_concept"].get("concept_id")
        )
        if (
            not ambiguous
            and same_reconciliation_story
            and top
            and runner_up
            and top["overall_confidence"] >= thresholds["partial_match_min"]
            and runner_up["overall_confidence"] >= thresholds["weak_match_min"]
            and abs(top["overall_confidence"] - runner_up["overall_confidence"])
            <= thresholds["ambiguous_match_delta_max"] + 0.08
        ):
            ambiguous = True

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
            "reconciliation_story_id": (
                top["canonical_concept"].get("concept_id")
                if top and isinstance(top.get("canonical_concept"), dict)
                else None
            ),
            "reconciliation_note": (
                top["canonical_concept"].get("reconciliation_note")
                if ambiguous and top and isinstance(top.get("canonical_concept"), dict)
                else None
            ),
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
                "simple_primary_mechanism": self._simple_text(selected_variant["primary_mechanism"]),
                "secondary_contributors": [],
                "performance_impact": selected_variant["performance_impact"],
                "simple_performance_impact": self._simple_text(selected_variant["performance_impact"]),
                "load_impact": selected_variant["load_impact"],
                "simple_load_impact": self._simple_text(selected_variant["load_impact"]),
                "first_intervention": None,
                "coach_check": selected_variant["coach_check"],
                "simple_coach_check": self._simple_text(selected_variant["coach_check"]),
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
            "simple_primary_mechanism": self._cfg_simple_text(
                primary_cfg,
                "simple_title",
                primary["title"],
            ),
            "primary_mechanism_summary": selected_variant["primary_mechanism_summary"],
            "simple_primary_mechanism_summary": self._cfg_simple_text(
                primary_cfg,
                "simple_summary",
                selected_variant["primary_mechanism_summary"],
            ),
            "secondary_contributors": secondary_contributors,
            "performance_impact": selected_variant["performance_impact"],
            "simple_performance_impact": self._simple_text(selected_variant["performance_impact"]),
            "load_impact": selected_variant["load_impact"],
            "simple_load_impact": self._simple_text(selected_variant["load_impact"]),
            "first_intervention": selected_variant["first_intervention"],
            "coach_check": selected_variant["coach_check"],
            "simple_coach_check": self._simple_text(selected_variant["coach_check"]),
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
            "simple_summary": self._cfg_simple_text(cfg, "simple_summary", cfg["summary"]),
            "history_story_template": cfg["history_story_template"],
            "simple_history_story_template": self._cfg_simple_text(
                cfg,
                "simple_history_story_template",
                cfg["history_story_template"],
            ),
            "coaching_priority_template": cfg["coaching_priority_template"],
            "simple_coaching_priority_template": self._cfg_simple_text(
                cfg,
                "simple_coaching_priority_template",
                cfg["coaching_priority_template"],
            ),
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
                    "simple_title": self._cfg_simple_text(cfg, "simple_title", cfg["title"]),
                    "goal": cfg["goal"],
                    "simple_goal": self._cfg_simple_text(cfg, "simple_goal", cfg["goal"]),
                    "primary_cue": cfg["primary_cue"],
                    "simple_primary_cue": self._cfg_simple_text(
                        cfg,
                        "simple_primary_cue",
                        cfg["primary_cue"],
                    ),
                    "why_this_first": cfg["why_this_first"],
                    "simple_why_this_first": self._cfg_simple_text(
                        cfg,
                        "simple_why_this_first",
                        cfg["why_this_first"],
                    ),
                    "coach_check": cfg["coach_check"],
                    "simple_coach_check": self._cfg_simple_text(
                        cfg,
                        "simple_coach_check",
                        cfg["coach_check"],
                    ),
                    "reassess_after": cfg["reassess_after"],
                    "review_window_type": cfg["review_window_type"],
                    "followup_metric_targets": list(cfg["followup_metric_targets"]),
                    "avoid_for_now": list(cfg["avoid_for_now"]),
                    "change_reaction": dict(cfg["change_reaction"]),
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
                    "simple_title": self._cfg_simple_text(cfg, "simple_title", cfg["title"]),
                    "primary_metric": cfg["primary_metric"],
                    "metrics": list(cfg["metrics"]),
                    "chart_summary": cfg["chart_summary"],
                    "simple_chart_summary": self._cfg_simple_text(
                        cfg,
                        "simple_chart_summary",
                        cfg["chart_summary"],
                    ),
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
                    "simple_title": self._cfg_simple_text(cfg, "simple_title", cfg["title"]),
                    "recommended_review_window": cfg["recommended_review_window"],
                    "history_graph_binding": cfg["history_graph_binding"],
                    "success_signals": list(cfg["success_signals"]),
                    "simple_success_signals": [
                        self._simple_text(item) or str(item)
                        for item in list(cfg.get("simple_success_signals") or cfg["success_signals"])
                    ],
                    "failure_signals": list(cfg["failure_signals"]),
                    "simple_failure_signals": [
                        self._simple_text(item) or str(item)
                        for item in list(cfg.get("simple_failure_signals") or cfg["failure_signals"])
                    ],
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
            "simple_history_story": (
                archetype.get("simple_history_story_template")
                if isinstance(archetype, dict)
                else self._simple_text(history_context.get("history_story"))
            ),
            "coaching_priority": (
                archetype.get("coaching_priority_template")
                if isinstance(archetype, dict)
                else None
            ),
            "simple_coaching_priority": (
                archetype.get("simple_coaching_priority_template")
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
            events=events,
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
        primary_change_reaction = primary_prescription.get("change_reaction") or {}
        followup_checks = history_plan.get("followup_checks") or []
        primary_followup = followup_checks[0] if followup_checks else {}
        history_bindings = history_plan.get("history_bindings") or []
        render_stories = history_plan.get("render_stories") or []
        primary_break_point = self._primary_break_point(
            primary=primary,
            visible_symptom=visible_symptom,
        )
        acceptance_summary = self._acceptance_summary(
            metrics=metrics,
            contributors=supporting_contributors,
        )
        kinetic_chain_status = self._kinetic_chain_status(
            metrics=metrics,
            diagnosis_status=diagnosis_status,
            primary_break_point=primary_break_point,
            capture_quality=capture_quality,
            acceptance_summary=acceptance_summary,
        )
        root_cause = self._root_cause(
            diagnosis_status=diagnosis_status,
            capture_quality=capture_quality,
            primary=primary,
            visible_symptom=visible_symptom,
            contributors=supporting_contributors,
            primary_break_point=primary_break_point,
            kinetic_chain_status=kinetic_chain_status,
            mechanism_explanation=mechanism_explanation,
        )
        if kinetic_chain_status.get("id") == "connected":
            visible_symptom = None
            phase_findings = []
            primary_break_point = {
                "phase_id": None,
                "title": "No clear break point",
                "summary": "No single phase is breaking the chain enough to justify intervention right now.",
            }
        change_strategy = self._change_strategy(
            capture_quality=capture_quality,
            diagnosis_status=diagnosis_status,
            primary_prescription=primary_prescription,
            primary_change_reaction=primary_change_reaction,
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
        if kinetic_chain_status.get("id") == "connected":
            what_is_not_ok = []
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
        if kinetic_chain_status.get("id") == "connected":
            holdback["top_candidates"] = []

        easy_explanation = self._build_easy_explanation(
            {
                "state": state,
                "kinetic_chain_status": kinetic_chain_status,
                "root_cause": root_cause,
                "first_priority": (
                    {
                        "simple_primary_cue": primary_prescription.get("simple_primary_cue"),
                        "primary_cue": primary_prescription.get("primary_cue"),
                    }
                    if primary_prescription
                    else None
                ),
                "improvement_check": (
                    {
                        "simple_success_signals": list(primary_followup.get("simple_success_signals") or []),
                        "simple_coach_check": mechanism_explanation.get("simple_coach_check"),
                    }
                    if (primary_followup or mechanism_explanation)
                    else None
                ),
            }
        )

        return {
            "version": "coach_diagnosis_v1",
            "knowledge_pack_id": self._pack["pack_id"],
            "knowledge_pack_version": self._pack["pack_version"],
            "state": state,
            "diagnosis_status": diagnosis_status,
            "capture_quality_status": capture_quality.get("status"),
            "kinetic_chain_status": kinetic_chain_status,
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
            "root_cause": root_cause,
            "supporting_contributors": supporting_contributors,
            "upper_body_contributors": upper_body,
            "lower_body_contributors": lower_body,
            "compensations": compensations,
            "what_is_ok": what_is_ok,
            "what_is_not_ok": what_is_not_ok,
            "acceptance_summary": acceptance_summary,
            "key_metrics": acceptance_summary.get("key_metrics"),
            "primary_break_point": primary_break_point,
            "near_term_effect": near_term,
            "medium_term_effect": medium_term,
            "long_term_outlook": long_term,
            "first_priority": (
                {
                    "prescription_id": primary_prescription.get("id"),
                    "title": primary_prescription.get("title"),
                    "simple_title": primary_prescription.get("simple_title"),
                    "goal": primary_prescription.get("goal"),
                    "simple_goal": primary_prescription.get("simple_goal"),
                    "primary_cue": primary_prescription.get("primary_cue"),
                    "simple_primary_cue": primary_prescription.get("simple_primary_cue"),
                    "why_this_first": primary_prescription.get("why_this_first"),
                    "simple_why_this_first": primary_prescription.get("simple_why_this_first"),
                }
                if primary_prescription
                else None
            ),
            "change_strategy": change_strategy,
            "change_reaction": primary_change_reaction if primary_change_reaction else None,
            "do_not_change_yet": list(primary_prescription.get("avoid_for_now") or []),
            "improvement_check": (
                {
                    "coach_check": mechanism_explanation.get("coach_check"),
                    "simple_coach_check": (
                        primary_followup.get("simple_success_signals") or [mechanism_explanation.get("simple_coach_check")]
                    )[0]
                    if (primary_followup.get("simple_success_signals") or [mechanism_explanation.get("simple_coach_check")])
                    else None,
                    "check_id": primary_followup.get("id"),
                    "title": primary_followup.get("title"),
                    "simple_title": primary_followup.get("simple_title"),
                    "recommended_review_window": primary_followup.get("recommended_review_window"),
                    "success_signals": list(primary_followup.get("success_signals") or []),
                    "simple_success_signals": list(primary_followup.get("simple_success_signals") or []),
                    "failure_signals": list(primary_followup.get("failure_signals") or []),
                    "simple_failure_signals": list(primary_followup.get("simple_failure_signals") or []),
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
            "easy_explanation": easy_explanation,
            "evidence_basis": {
                "primary_mechanism": self._evidence_basis_for_target(
                    "mechanism",
                    primary.get("id"),
                ),
                "visible_symptom": self._evidence_basis_for_target(
                    "symptom",
                    (visible_symptom or {}).get("id"),
                ),
                "supporting_contributors": [
                    basis
                    for item in supporting_contributors
                    if item.get("role") != "supporting_symptom"
                    for basis in [self._evidence_basis_for_target("contributor", item.get("id"))]
                    if basis is not None
                ],
                "reconciliation_story_id": selection.get("reconciliation_story_id"),
                "reconciliation_note": selection.get("reconciliation_note"),
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
                    "simple_title": symptom.get("simple_title"),
                    "phase": str(symptom.get("phase") or "").upper(),
                    "summary": symptom.get("description"),
                    "simple_summary": symptom.get("simple_description"),
                    "severity": symptom.get("severity"),
                    "confidence": symptom.get("confidence"),
                }
        if symptoms:
            symptom = symptoms[0]
            return {
                "id": symptom.get("id"),
                "title": symptom.get("title"),
                "simple_title": symptom.get("simple_title"),
                "phase": str(symptom.get("phase") or "").upper(),
                "summary": symptom.get("description"),
                "simple_summary": symptom.get("simple_description"),
                "severity": symptom.get("severity"),
                "confidence": symptom.get("confidence"),
            }
        return None

    def _filter_coach_diagnosis(
        self,
        *,
        account_role: Optional[str],
        coach_diagnosis: Dict[str, Any],
    ) -> Dict[str, Any]:
        policy = self._role_detail_policy(account_role)
        if not policy:
            return coach_diagnosis
        filtered = dict(coach_diagnosis)
        filtered["detail_policy_id"] = policy.get("id")
        if not policy.get("include_supporting_contributors", False):
            filtered["supporting_contributors"] = []
        if not policy.get("include_body_group_breakdown", False):
            filtered["upper_body_contributors"] = []
            filtered["lower_body_contributors"] = []
        if not policy.get("include_compensations", False):
            filtered["compensations"] = []
        if not policy.get("include_what_is_ok", False):
            filtered["what_is_ok"] = []
        if not policy.get("include_what_is_not_ok", False):
            filtered["what_is_not_ok"] = []
        if not policy.get("include_phase_anchors", False):
            filtered["phase_anchored_findings"] = []
        if not policy.get("include_history_bindings", False):
            history_bindings = dict(filtered.get("history_bindings") or {})
            history_bindings["bindings"] = []
            filtered["history_bindings"] = history_bindings
        if not policy.get("include_change_reaction", False):
            filtered["change_reaction"] = None
        if not policy.get("include_evidence_basis", False):
            filtered["evidence_basis"] = None
        holdback = filtered.get("holdback")
        if (
            isinstance(holdback, dict)
            and not policy.get("include_holdback_candidates", False)
        ):
            filtered["holdback"] = {
                "reason": holdback.get("reason"),
                "top_candidates": [],
            }
        return filtered

    def _frontend_headline(self, coach_diagnosis: Dict[str, Any]) -> str:
        chain_status = coach_diagnosis.get("kinetic_chain_status") or {}
        break_point = coach_diagnosis.get("primary_break_point") or {}
        status_id = str(chain_status.get("id") or "")
        phase_id = str(break_point.get("phase_id") or "")
        if status_id == "connected":
            return "Action is connected"
        if phase_id == "transfer_and_block":
            return "Leak at landing"
        if phase_id in {"whip_and_release", "UAH", "RELEASE"}:
            return "Late release leak"
        if status_id == "workable_but_leaking":
            return "Action is workable but leaking"
        return str(chain_status.get("label") or "Kinetic chain review")

    def _frontend_summary_lines(self, coach_diagnosis: Dict[str, Any]) -> List[str]:
        chain_status = coach_diagnosis.get("kinetic_chain_status") or {}
        break_point = coach_diagnosis.get("primary_break_point") or {}
        lower = list(coach_diagnosis.get("lower_body_contributors") or [])
        upper = list(coach_diagnosis.get("upper_body_contributors") or [])
        first_priority = coach_diagnosis.get("first_priority") or {}
        status_id = str(chain_status.get("id") or "")
        phase_id = str(break_point.get("phase_id") or "")

        if status_id == "connected":
            return [
                "The action is moving well through the chain.",
                "No single phase is breaking enough to need a change right now.",
                "Keep repeating what is working before changing anything bigger.",
            ]

        lines: List[str] = []
        if phase_id == "transfer_and_block":
            lines.append("The action is still working, but it gets weak around landing.")
            if lower and upper:
                lines.append(
                    "The front side is not giving a strong base, so the upper body keeps moving through release."
                )
            elif lower:
                lines.append(
                    "The front side is not giving a strong base when the front foot lands."
                )
        elif phase_id in {"whip_and_release", "UAH", "RELEASE"}:
            lines.append("The action is getting rushed close to release.")
            lines.append("The upper body is having to rescue timing late instead of being carried there cleanly.")
        else:
            lines.append(
                "The action can still work, but one part of the chain is leaking."
            )

        cue = str(first_priority.get("primary_cue") or "").strip()
        if cue:
            lines.append(f"Start with one small change first: {cue}.")
        else:
            lines.append("Start with one small change first, not a big rebuild.")
        return lines[:3]

    def _frontend_chips(
        self,
        coach_diagnosis: Dict[str, Any],
        presentation_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chain_status = coach_diagnosis.get("kinetic_chain_status") or {}
        break_point = coach_diagnosis.get("primary_break_point") or {}
        change_strategy = coach_diagnosis.get("change_strategy") or {}
        return [
            {
                "id": "state",
                "label": str(chain_status.get("label") or presentation_payload.get("state") or "").strip(),
                "tone": str(chain_status.get("acceptance_band") or "workable"),
            },
            {
                "id": "break_point",
                "label": str(break_point.get("title") or "No clear break point").strip(),
                "tone": "neutral",
            },
            {
                "id": "change_size",
                "label": str(change_strategy.get("change_size") or "micro").replace("_", " ").title(),
                "tone": str(change_strategy.get("adoption_risk") or "low"),
            },
        ]

    def _build_easy_explanation(self, coach_diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        root_cause = coach_diagnosis.get("root_cause") or {}
        renderer_guidance = root_cause.get("renderer_guidance") or {}
        first_priority = coach_diagnosis.get("first_priority") or {}
        improvement_check = coach_diagnosis.get("improvement_check") or {}
        kinetic_chain_status = coach_diagnosis.get("kinetic_chain_status") or {}
        mechanism_cfg = (self._pack.get("mechanisms") or {}).get(str(root_cause.get("mechanism_id") or "")) or {}
        state_id = str(kinetic_chain_status.get("id") or "")

        if state_id == "connected":
            return {
                "headline": "Action is working well.",
                "what_to_notice": "The action stays connected through landing and release.",
                "why_it_happens": "The body is carrying movement through the action without one obvious leak.",
                "first_fix": None,
                "check_next": "Keep repeating this shape before changing anything bigger.",
            }

        return {
            "headline": self._cfg_simple_text(
                mechanism_cfg,
                "simple_title",
                root_cause.get("title"),
            )
            or coach_diagnosis.get("state"),
            "what_to_notice": renderer_guidance.get("simple_symptom_text")
            or self._simple_text(root_cause.get("summary")),
            "why_it_happens": self._cfg_simple_text(
                mechanism_cfg,
                "simple_summary",
                root_cause.get("why_it_is_happening"),
            ),
            "first_fix": first_priority.get("simple_primary_cue") or self._simple_text(first_priority.get("primary_cue")),
            "check_next": (
                (improvement_check.get("simple_success_signals") or [improvement_check.get("simple_coach_check")])[0]
                if (improvement_check.get("simple_success_signals") or [improvement_check.get("simple_coach_check")])
                else None
            ),
        }

    def _build_frontend_surface(
        self,
        *,
        account_role: Optional[str],
        coach_diagnosis: Dict[str, Any],
        presentation_payload: Dict[str, Any],
        render_reasoning: Dict[str, Any],
    ) -> Dict[str, Any]:
        role = str(account_role or "player").strip().lower() or "player"
        return {
            "version": "frontend_surface_v1",
            "role": role,
            "detail_policy_id": coach_diagnosis.get("detail_policy_id"),
            "state": coach_diagnosis.get("state") or presentation_payload.get("state"),
            "headline": self._frontend_headline(coach_diagnosis),
            "summary_lines": self._frontend_summary_lines(coach_diagnosis),
            "chips": self._frontend_chips(coach_diagnosis, presentation_payload),
            "hero": {
                "kinetic_chain_status": coach_diagnosis.get("kinetic_chain_status"),
                "visible_symptom": coach_diagnosis.get("visible_symptom"),
                "primary_break_point": coach_diagnosis.get("primary_break_point"),
                "primary_mechanism": coach_diagnosis.get("primary_mechanism"),
                "root_cause": coach_diagnosis.get("root_cause"),
            },
            "body": {
                "root_cause": coach_diagnosis.get("root_cause"),
                "what_is_ok": coach_diagnosis.get("what_is_ok"),
                "what_is_not_ok": coach_diagnosis.get("what_is_not_ok"),
                "upper_body_contributors": coach_diagnosis.get("upper_body_contributors"),
                "lower_body_contributors": coach_diagnosis.get("lower_body_contributors"),
                "phase_anchored_findings": coach_diagnosis.get("phase_anchored_findings"),
                "compensations": coach_diagnosis.get("compensations"),
                "acceptance_summary": coach_diagnosis.get("acceptance_summary"),
                "key_metrics": coach_diagnosis.get("key_metrics"),
            },
            "guidance": {
                "first_priority": coach_diagnosis.get("first_priority"),
                "do_not_change_yet": coach_diagnosis.get("do_not_change_yet"),
                "change_strategy": coach_diagnosis.get("change_strategy"),
                "improvement_check": coach_diagnosis.get("improvement_check"),
                "change_reaction": coach_diagnosis.get("change_reaction"),
            },
            "renderer": {
                "renderer_mode": render_reasoning.get("renderer_mode"),
                "selected_story_id": render_reasoning.get("selected_story_id"),
            },
            "history": coach_diagnosis.get("history_bindings"),
            "easy_explanation": coach_diagnosis.get("easy_explanation"),
            "holdback": coach_diagnosis.get("holdback"),
        }

    def _supporting_contributors(
        self,
        *,
        events: Dict[str, Any],
        risks: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        symptoms: List[Dict[str, Any]],
        selection: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        contributors: List[Dict[str, Any]] = []
        symptom_lookup = {
            str(symptom.get("id") or ""): symptom
            for symptom in symptoms
            if isinstance(symptom, dict)
        }
        selected_mechanism_ids = set(selection.get("selected_mechanism_ids") or [])
        primary_mechanism_id = str(selection.get("primary_mechanism_id") or "").strip()
        risk_contributors = {
            str(cfg.get("source_key") or contributor_id): cfg
            for contributor_id, cfg in self._pack["contributors"].items()
            if str(cfg.get("source_type") or "") == "risk"
        }
        metric_contributors = [
            (str(cfg.get("source_key") or contributor_id), cfg)
            for contributor_id, cfg in self._pack["contributors"].items()
            if str(cfg.get("source_type") or "") == "metric"
        ]
        event_contributors = [
            cfg
            for cfg in self._pack["contributors"].values()
            if str(cfg.get("source_type") or "") == "event"
        ]
        contributor_acceptance_cfg = (
            ((self._pack.get("globals") or {}).get("acceptance_bands") or {}).get("contributor_severity")
            or {}
        )
        contributor_acceptable_max = _safe_float(contributor_acceptance_cfg.get("acceptable_max"), 0.25)
        contributor_workable_max = _safe_float(contributor_acceptance_cfg.get("workable_max"), 0.50)

        def _candidate(
            *,
            item_id: str,
            title: str,
            simple_title: Optional[str],
            body_group: str,
            phase: str,
            role: str,
            summary: str,
            simple_summary: Optional[str],
            signal_strength: float,
            raw_signal_strength: Optional[float] = None,
            confidence: float,
            possible_mechanism_ids: Optional[List[str]] = None,
            target_type: str = "contributor",
        ) -> Dict[str, Any]:
            evidence_support = self._knowledge_support_score(target_type, item_id)
            canonical_concept = self._reconciliation_concept(target_type, item_id)
            possible_ids = set(possible_mechanism_ids or [])
            acceptance_band = _acceptance_band(
                raw_signal_strength if raw_signal_strength is not None else signal_strength,
                acceptable_max=contributor_acceptable_max,
                workable_max=contributor_workable_max,
            )
            mechanism_affinity = 0.0
            if primary_mechanism_id and primary_mechanism_id in possible_ids:
                mechanism_affinity = 0.16
            elif selected_mechanism_ids.intersection(possible_ids):
                mechanism_affinity = 0.08
            phase_anchor_boost = {
                "BFC": 0.08,
                "FFC": 0.08,
                "UAH": 0.08,
                "RELEASE": 0.08,
                "BFC_TO_FFC": 0.05,
                "FFC_TO_RELEASE": 0.05,
            }.get(phase, 0.0)
            ranking_score = _clip01(
                (0.52 * _clip01(signal_strength))
                + (0.18 * _clip01(confidence))
                + (0.18 * evidence_support)
                + mechanism_affinity
                + phase_anchor_boost
            )
            return {
                "id": item_id,
                "title": title,
                "simple_title": simple_title or self._simple_text(title),
                "body_group": body_group,
                "phase": phase,
                "role": role,
                "signal_strength": _round3(signal_strength),
                "raw_signal_strength": _round3(raw_signal_strength if raw_signal_strength is not None else signal_strength),
                "acceptance_band": acceptance_band,
                "evidence_support": evidence_support,
                "ranking_score": _round3(ranking_score),
                "summary": summary,
                "simple_summary": simple_summary or self._simple_text(summary),
                "knowledge_evidence_ids": [
                    item["id"] for item in self._knowledge_evidence_items(target_type, item_id)
                ],
                "canonical_concept": canonical_concept,
            }

        candidates: List[Dict[str, Any]] = []
        primary = selection.get("primary") or {}
        if isinstance(primary, dict):
            for symptom_id in list(primary.get("supporting_symptom_ids") or []):
                symptom = symptom_lookup.get(str(symptom_id))
                if not symptom or not symptom.get("present"):
                    continue
                candidates.append(
                    _candidate(
                        item_id=str(symptom.get("id") or ""),
                        title=str(symptom.get("title") or ""),
                        simple_title=str(symptom.get("simple_title") or ""),
                        body_group=self._symptom_body_group(str(symptom.get("id") or "")),
                        phase=str(symptom.get("phase") or "").upper(),
                        role="supporting_symptom",
                        summary=str(symptom.get("description") or ""),
                        simple_summary=str(symptom.get("simple_description") or ""),
                        signal_strength=_safe_float(symptom.get("score"), 0.0),
                        raw_signal_strength=_safe_float(symptom.get("score"), 0.0),
                        confidence=_safe_float(symptom.get("confidence"), 0.0),
                        possible_mechanism_ids=list(symptom.get("possible_mechanisms") or []),
                        target_type="symptom",
                    )
                )

        for risk in risks:
            if not isinstance(risk, dict):
                continue
            risk_id = str(risk.get("risk_id") or "").strip()
            cfg = risk_contributors.get(risk_id) or {}
            if not cfg:
                continue
            raw_signal = _safe_float(risk.get("signal_strength"), 0.0)
            confidence = _safe_float(risk.get("confidence"), 0.0)
            signal = _banded_signal(
                raw_signal,
                acceptable_max=contributor_acceptable_max,
                workable_max=contributor_workable_max,
            )
            if signal < 0.35:
                continue
            candidates.append(
                _candidate(
                    item_id=risk_id,
                    title=cfg["title"],
                    simple_title=self._cfg_simple_text(cfg, "simple_title", cfg["title"]),
                    body_group=cfg["body_group"],
                    phase=cfg["phase"],
                    role="supporting_risk",
                    summary=cfg["summary"],
                    simple_summary=self._cfg_simple_text(cfg, "simple_summary", cfg["summary"]),
                    signal_strength=signal,
                    raw_signal_strength=raw_signal,
                    confidence=confidence,
                    possible_mechanism_ids=list(cfg.get("possible_mechanism_ids") or []),
                )
            )

        for metric_name, cfg in metric_contributors:
            metric = metrics.get(metric_name) or {}
            if not isinstance(metric, dict):
                continue
            value = _safe_float(metric.get("value"), 0.0)
            confidence = _safe_float(metric.get("confidence"), 0.0)
            if confidence < 0.15:
                continue
            raw_severity = (1.0 - value) if metric_name != "trunk_drift_after_ffc" else value
            severity = _banded_signal(
                raw_severity,
                acceptable_max=contributor_acceptable_max,
                workable_max=contributor_workable_max,
            )
            if severity < 0.4:
                continue
            candidates.append(
                _candidate(
                    item_id=metric_name,
                    title=cfg["title"],
                    simple_title=self._cfg_simple_text(cfg, "simple_title", cfg["title"]),
                    body_group=cfg["body_group"],
                    phase=cfg["phase"],
                    role="supporting_metric",
                    summary=cfg["summary"],
                    simple_summary=self._cfg_simple_text(cfg, "simple_summary", cfg["summary"]),
                    signal_strength=severity,
                    raw_signal_strength=raw_severity,
                    confidence=confidence,
                    possible_mechanism_ids=list(cfg.get("possible_mechanism_ids") or []),
                )
            )

        for cfg in event_contributors:
            signal, confidence = self._event_contributor_signal(events=events, contributor_cfg=cfg)
            if signal < 0.4 or confidence < 0.15:
                continue
            candidates.append(
                _candidate(
                    item_id=str(cfg.get("id") or ""),
                    title=str(cfg.get("title") or ""),
                    simple_title=self._cfg_simple_text(cfg, "simple_title", str(cfg.get("title") or "")),
                    body_group=str(cfg.get("body_group") or "whole_chain"),
                    phase=str(cfg.get("phase") or "").upper(),
                    role="supporting_phase_anchor",
                    summary=str(cfg.get("summary") or ""),
                    simple_summary=self._cfg_simple_text(cfg, "simple_summary", str(cfg.get("summary") or "")),
                    signal_strength=signal,
                    confidence=confidence,
                    possible_mechanism_ids=list(cfg.get("possible_mechanism_ids") or []),
                )
            )

        candidates.sort(key=lambda item: item["ranking_score"], reverse=True)
        prepared_candidates: List[Dict[str, Any]] = []
        concept_scores: Dict[str, float] = {}
        seen_candidate_ids: set[str] = set()
        for candidate in candidates:
            item_id = str(candidate.get("id") or "").strip()
            if not item_id or item_id in seen_candidate_ids:
                continue
            concept = candidate.get("canonical_concept") or {}
            concept_id = str(concept.get("concept_id") or "").strip()
            relation = str(concept.get("relation") or "").strip()
            adjusted_score = _safe_float(candidate.get("ranking_score"), 0.0)
            if concept_id and relation == "similar" and concept_id in concept_scores:
                adjusted_score = max(0.0, adjusted_score - 0.08)
            if adjusted_score < 0.45:
                continue
            candidate["ranking_score"] = _round3(adjusted_score)
            if concept_id and concept_id not in concept_scores:
                concept_scores[concept_id] = adjusted_score
            seen_candidate_ids.add(item_id)
            prepared_candidates.append(candidate)

        selected_ids: set[str] = set()

        def _pick_first(predicate) -> None:
            for candidate in prepared_candidates:
                item_id = str(candidate.get("id") or "").strip()
                if not item_id or item_id in selected_ids:
                    continue
                if predicate(candidate):
                    selected_ids.add(item_id)
                    contributors.append(candidate)
                    return

        _pick_first(lambda _candidate: True)
        _pick_first(lambda candidate: str(candidate.get("body_group") or "") == "upper_body")
        _pick_first(lambda candidate: str(candidate.get("body_group") or "") == "lower_body")
        _pick_first(lambda candidate: str(candidate.get("phase") or "") in {"BFC", "FFC", "BFC_TO_FFC"})

        for candidate in prepared_candidates:
            item_id = str(candidate.get("id") or "").strip()
            if not item_id or item_id in selected_ids:
                continue
            selected_ids.add(item_id)
            contributors.append(candidate)
            if len(contributors) >= 6:
                break

        return contributors[:6]

    def _event_contributor_signal(
        self,
        *,
        events: Dict[str, Any],
        contributor_cfg: Dict[str, Any],
    ) -> Tuple[float, float]:
        phase = str(contributor_cfg.get("phase") or "").strip().upper()
        event_lookup = {
            "BFC": "bfc",
            "FFC": "ffc",
            "UAH": "uah",
            "RELEASE": "release",
        }
        event_key = event_lookup.get(phase)
        if not event_key:
            return 0.0, 0.0
        event = (events or {}).get(event_key) or {}
        if not isinstance(event, dict):
            return 0.0, 0.0
        source_key = str(contributor_cfg.get("source_key") or "").strip()
        if not source_key:
            return 0.0, 0.0

        def _extract_raw(container: Dict[str, Any]) -> Any:
            for key in (source_key, source_key.lower(), source_key.upper()):
                if key in container:
                    return container.get(key)
            return None

        raw_value = _extract_raw(event)
        if raw_value is None:
            for nested_key in ("signals", "flags", "debug", "observations", "derived"):
                nested = event.get(nested_key) or {}
                if isinstance(nested, dict):
                    raw_value = _extract_raw(nested)
                    if raw_value is not None:
                        break
        if raw_value is None:
            return 0.0, 0.0

        confidence = _safe_float(event.get("confidence"), 0.0)
        if isinstance(raw_value, dict):
            confidence = max(confidence, _safe_float(raw_value.get("confidence"), confidence))
            if "value" in raw_value:
                raw_value = raw_value.get("value")
            elif "score" in raw_value:
                raw_value = raw_value.get("score")
            elif "present" in raw_value:
                raw_value = raw_value.get("present")

        if isinstance(raw_value, bool):
            signal = 0.72 if raw_value else 0.0
        elif isinstance(raw_value, (int, float)):
            signal = _clip01(abs(float(raw_value)))
        else:
            text = str(raw_value or "").strip().lower()
            if text in {"true", "yes", "present", "left", "right", "tilted"}:
                signal = 0.68
            else:
                signal = 0.0
        return _round3(signal), _round3(max(confidence, 0.35 if signal > 0.0 else 0.0))

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
        break_cfg = self._pack["coach_judgments"]["break_points"].get(phase_key) or {}
        return {
            "phase_id": phase_key,
            "title": break_cfg.get(
                "title",
                _PHASE_LABELS.get(phase_key, phase_key.replace("_", " ").title()),
            ),
            "summary": break_cfg.get("summary") or self._break_point_summary(phase_key),
        }

    def _break_point_summary(self, phase_key: str) -> str:
        label = _PHASE_LABELS.get(phase_key, phase_key.replace("_", " ").title())
        return f"{label} is the main place where the chain is no longer carrying momentum cleanly."

    def _preferred_root_cause_body_group(self, phase_id: str) -> Optional[str]:
        normalized = str(phase_id or "").strip()
        if normalized == "transfer_and_block":
            return "lower_body"
        if normalized in {"whip_and_release", "UAH", "RELEASE"}:
            return "upper_body"
        if normalized in {
            "approach_build",
            "gather_and_organize",
            "dissipation_and_recovery",
        }:
            return "whole_chain"
        return None

    def _root_cause_driver(
        self,
        *,
        contributors: List[Dict[str, Any]],
        preferred_body_group: Optional[str],
        preferred_ids: Optional[List[str]] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        excluded = exclude_ids or set()
        ranked = [
            item
            for item in contributors
            if str(item.get("id") or "").strip() not in excluded
        ]
        problematic = [
            item
            for item in ranked
            if str(item.get("acceptance_band") or "") == "problematic"
        ]
        candidates = problematic or ranked
        preferred = [
            str(item_id or "").strip()
            for item_id in (preferred_ids or [])
            if str(item_id or "").strip()
        ]

        if preferred:
            for preferred_id in preferred:
                for item in candidates:
                    if str(item.get("id") or "").strip() == preferred_id:
                        return {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "role": item.get("role"),
                            "body_group": item.get("body_group"),
                            "phase": item.get("phase"),
                            "summary": item.get("summary"),
                            "acceptance_band": item.get("acceptance_band"),
                        }

        if preferred_body_group:
            for item in candidates:
                if str(item.get("body_group") or "") == preferred_body_group:
                    return {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "role": item.get("role"),
                        "body_group": item.get("body_group"),
                        "phase": item.get("phase"),
                        "summary": item.get("summary"),
                        "acceptance_band": item.get("acceptance_band"),
                    }

        for item in candidates:
            return {
                "id": item.get("id"),
                "title": item.get("title"),
                "role": item.get("role"),
                "body_group": item.get("body_group"),
                "phase": item.get("phase"),
                "summary": item.get("summary"),
                "acceptance_band": item.get("acceptance_band"),
            }
        return None

    def _root_cause_load_watch_label(self, body_group: Optional[str]) -> str:
        normalized = str(body_group or "").strip()
        if normalized == "lower_body":
            return "Front knee / leg chain"
        if normalized == "upper_body":
            return "Lower back / side trunk"
        return "Whole chain load watch"

    def _root_cause_compensation(
        self,
        *,
        mechanism_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        contributors: List[Dict[str, Any]],
        exclude_ids: Optional[set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        excluded = set(exclude_ids or set())
        if isinstance(primary_driver, dict):
            driver_id = str(primary_driver.get("id") or "").strip()
            if driver_id:
                excluded.add(driver_id)
        mechanism_cfg = (self._pack.get("mechanisms") or {}).get(str(mechanism_id or "")) or {}
        contributor_cfg = (
            (self._pack.get("contributors") or {}).get(str((primary_driver or {}).get("id") or ""))
            or {}
        )
        preferred_ids = [
            str(item_id or "").strip()
            for item_id in (
                list(mechanism_cfg.get("common_compensation_ids") or [])
                + list(contributor_cfg.get("common_compensation_ids") or [])
            )
            if str(item_id or "").strip() and str(item_id or "").strip() not in excluded
        ]
        compensation = self._root_cause_driver(
            contributors=contributors,
            preferred_body_group=None,
            preferred_ids=preferred_ids,
            exclude_ids=excluded,
        )
        if compensation:
            compensation = dict(compensation)
            compensation["relationship"] = "downstream_compensation"
        return compensation

    def _root_cause_symptom_text(
        self,
        *,
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        compensation_summary = str((compensation or {}).get("summary") or "").strip()
        if compensation_summary:
            return compensation_summary
        visible_summary = str((visible_symptom or {}).get("summary") or "").strip()
        if visible_summary:
            return visible_summary
        driver_summary = str((primary_driver or {}).get("summary") or "").strip()
        return driver_summary or None

    def _root_cause_load_watch_text(
        self,
        *,
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        primary_label = self._root_cause_load_watch_label(
            (primary_driver or {}).get("body_group"),
        )
        compensation_label = self._root_cause_load_watch_label(
            (compensation or {}).get("body_group"),
        )
        if compensation and compensation_label != primary_label:
            return f"{primary_label}\n{compensation_label}"
        return primary_label

    def _root_cause_simple_symptom_text(
        self,
        *,
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        primary_id = str((primary_driver or {}).get("id") or "").strip()
        compensation_id = str((compensation or {}).get("id") or "").strip()
        symptom_id = str((visible_symptom or {}).get("id") or "").strip()

        if primary_id in {"front_leg_support_score", "knee_brace_failure"}:
            if compensation_id == "lateral_trunk_lean":
                return "Front leg doesn't hold strong at landing, then the body falls away."
            return "Front leg doesn't hold strong at landing."
        if primary_id == "foot_line_deviation":
            return "Front foot lands a bit across the line."
        if primary_id in {"shoulder_rotation_timing", "hip_shoulder_mismatch"}:
            if compensation_id in {"trunk_rotation_snap", "lateral_trunk_lean"}:
                return "Top half gets late, then has to rush near release."
            return "Top half gets late near release."
        if primary_id == "trunk_rotation_snap":
            return "Top half turns too sharply near release."

        if compensation_id == "lateral_trunk_lean":
            return "Body falls away at release."
        if compensation_id == "trunk_rotation_snap":
            return "Body turns too sharply near release."
        if symptom_id == "front_leg_softening":
            return "Front leg gets soft at landing."
        if symptom_id in {"arm_chase", "release_window_instability"}:
            return "The arm has to rush late."
        return "This is the main thing to notice."

    def _root_cause_simple_load_watch_text(
        self,
        *,
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        primary_group = str((primary_driver or {}).get("body_group") or "").strip()
        compensation_group = str((compensation or {}).get("body_group") or "").strip()

        if primary_group == "lower_body" and compensation_group == "upper_body":
            return "Front leg works hard.\nLower back works hard too."
        if primary_group == "lower_body":
            return "Front leg works harder here."
        if primary_group == "upper_body":
            return "Lower back works harder here."
        if compensation_group == "upper_body":
            return "Lower back works harder here."
        return "This is where the body works harder."

    def _root_cause_anchor_risk_ids(
        self,
        *,
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        ffc_risk_ids = {
            "knee_brace_failure",
            "foot_line_deviation",
            "front_foot_braking_shock",
        }
        release_risk_ids = {
            "lateral_trunk_lean",
            "hip_shoulder_mismatch",
            "trunk_rotation_snap",
            "front_foot_braking_shock",
        }
        anchors = {"ffc": None, "release": None}
        for item in (primary_driver, compensation):
            item_id = str((item or {}).get("id") or "").strip()
            if not item_id:
                continue
            if anchors["ffc"] is None and item_id in ffc_risk_ids:
                anchors["ffc"] = item_id
            if anchors["release"] is None and item_id in release_risk_ids:
                anchors["release"] = item_id
        symptom_id = str((visible_symptom or {}).get("id") or "").strip()
        if anchors["ffc"] is None and symptom_id == "front_leg_softening":
            anchors["ffc"] = "knee_brace_failure"
        if anchors["ffc"] is None and symptom_id == "underused_transfer_window":
            anchors["ffc"] = "foot_line_deviation"
        if anchors["release"] is None and symptom_id == "late_trunk_drift":
            anchors["release"] = "lateral_trunk_lean"
        if anchors["release"] is None and symptom_id in {
            "arm_chase",
            "release_window_instability",
        }:
            anchors["release"] = "hip_shoulder_mismatch"
        return anchors

    def _root_cause_region_priority(
        self,
        *,
        risk_id: Optional[str],
    ) -> List[str]:
        normalized = str(risk_id or "").strip()
        if normalized == "knee_brace_failure":
            return ["knee", "shin", "groin"]
        if normalized == "foot_line_deviation":
            return ["shin", "knee", "groin"]
        if normalized == "front_foot_braking_shock":
            return ["knee", "shin", "groin"]
        if normalized == "lateral_trunk_lean":
            return ["side_trunk", "upper_trunk", "lumbar"]
        if normalized == "hip_shoulder_mismatch":
            return ["lumbar", "side_trunk", "upper_trunk"]
        if normalized == "trunk_rotation_snap":
            return ["lumbar", "side_trunk", "upper_trunk"]
        return []

    def _renderer_storyboard_cue(
        self,
        *,
        story_id: Optional[str],
        phase_key: str,
    ) -> Optional[str]:
        story_cfg = (self._pack.get("render_stories") or {}).get(str(story_id or "")) or {}
        if not story_cfg:
            return None
        phase_aliases = {
            "ffc": {"front_foot_contact", "transfer_and_block", "ffc", "bfc_to_ffc"},
            "release": {"release", "whip_and_release", "ffc_to_release"},
        }
        valid_phases = phase_aliases.get(str(phase_key or "").strip().lower(), set())
        for item in list(story_cfg.get("storyboard") or []):
            if not isinstance(item, dict):
                continue
            raw_phase = str(item.get("phase") or "").strip().lower()
            cue = str(item.get("cue") or "").strip()
            if cue and raw_phase in valid_phases:
                return cue
        return None

    def _root_cause_phase_proof_headline(
        self,
        *,
        phase_key: str,
        risk_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        normalized_phase = str(phase_key or "").strip().lower()
        normalized_risk = str(risk_id or "").strip()
        primary_id = str((primary_driver or {}).get("id") or "").strip()
        compensation_id = str((compensation or {}).get("id") or "").strip()

        if normalized_phase == "ffc":
            if normalized_risk in {"knee_brace_failure", "front_leg_support_score"} or primary_id in {
                "knee_brace_failure",
                "front_leg_support_score",
            }:
                return "Front leg doesn't hold strong at landing."
            if normalized_risk == "foot_line_deviation" or primary_id == "foot_line_deviation":
                return "Front foot lands a bit across the line."
            if normalized_risk == "front_foot_braking_shock":
                return "Landing hits too sharply here."
        if normalized_phase == "release":
            if normalized_risk == "lateral_trunk_lean" or compensation_id == "lateral_trunk_lean":
                return "Then the body falls away through release."
            if normalized_risk == "hip_shoulder_mismatch" or compensation_id == "hip_shoulder_mismatch":
                return "Then hips and shoulders stop working together."
            if normalized_risk == "trunk_rotation_snap" or compensation_id == "trunk_rotation_snap":
                return "Then the top half turns too sharply."
        return None

    def _root_cause_phase_proof_body(
        self,
        *,
        phase_key: str,
        risk_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        normalized_phase = str(phase_key or "").strip().lower()
        normalized_risk = str(risk_id or "").strip()
        primary_group = str((primary_driver or {}).get("body_group") or "").strip()

        if normalized_phase == "ffc":
            if normalized_risk in {"knee_brace_failure", "front_leg_support_score"}:
                return "The landing leg is not becoming a clear transfer point."
            if normalized_risk == "foot_line_deviation":
                return "The landing foot is not lining force up cleanly toward target."
            if normalized_risk == "front_foot_braking_shock":
                return "The front foot is taking force sharply instead of turning it into calm transfer."
        if normalized_phase == "release":
            if normalized_risk == "lateral_trunk_lean":
                if primary_group == "lower_body":
                    return "Because the landing base stays soft, the trunk keeps travelling past it."
                return "The trunk keeps travelling instead of stacking cleanly into release."
            if normalized_risk == "hip_shoulder_mismatch":
                return "The top and bottom halves are not arriving together cleanly into release."
            if normalized_risk == "trunk_rotation_snap":
                return "The top half has to turn sharply because timing is arriving late."
        return None

    def _root_cause_phase_proof_step(
        self,
        *,
        phase_key: str,
        risk_id: Optional[str],
        story_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        normalized_phase = str(phase_key or "").strip().lower()
        normalized_risk = str(risk_id or "").strip()
        if normalized_phase not in {"ffc", "release"} or not normalized_risk:
            return None

        title = "Where It Starts" if normalized_phase == "ffc" else "What Happens Next"
        step_role = "where_it_starts" if normalized_phase == "ffc" else "compensation"
        headline = self._root_cause_phase_proof_headline(
            phase_key=normalized_phase,
            risk_id=normalized_risk,
            primary_driver=primary_driver,
            compensation=compensation,
        )
        body = self._root_cause_phase_proof_body(
            phase_key=normalized_phase,
            risk_id=normalized_risk,
            primary_driver=primary_driver,
            compensation=compensation,
        )
        proof_line = self._renderer_storyboard_cue(
            story_id=story_id,
            phase_key=normalized_phase,
        )
        if not headline and not body and not proof_line:
            return None
        return {
            "step_role": step_role,
            "title": title,
            "headline": headline,
            "body": body,
            "proof_line": proof_line,
        }

    def _root_cause_phase_targets(
        self,
        *,
        story_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        anchor_risk_ids = self._root_cause_anchor_risk_ids(
            primary_driver=primary_driver,
            compensation=compensation,
            visible_symptom=visible_symptom,
        )

        def _target_for_phase(phase_key: str) -> Optional[Dict[str, Any]]:
            risk_id = str(anchor_risk_ids.get(phase_key) or "").strip()
            if not risk_id:
                return None
            source_role = None
            source_item: Optional[Dict[str, Any]] = None
            if str((primary_driver or {}).get("id") or "").strip() == risk_id:
                source_role = "primary_driver"
                source_item = primary_driver
            elif str((compensation or {}).get("id") or "").strip() == risk_id:
                source_role = "compensation"
                source_item = compensation
            else:
                source_role = "phase_anchor"
                source_item = primary_driver if phase_key == "ffc" else compensation
            body_group = str((source_item or {}).get("body_group") or "").strip() or None
            return {
                "risk_id": risk_id,
                "source_role": source_role,
                "body_group": body_group,
                "load_watch_label": self._root_cause_load_watch_label(body_group),
                "region_priority": self._root_cause_region_priority(risk_id=risk_id),
                "proof_step": self._root_cause_phase_proof_step(
                    phase_key=phase_key,
                    risk_id=risk_id,
                    story_id=story_id,
                    primary_driver=primary_driver,
                    compensation=compensation,
                ),
            }

        targets: Dict[str, Dict[str, Any]] = {}
        for phase_key in ("ffc", "release"):
            target = _target_for_phase(phase_key)
            if target:
                targets[phase_key] = target
        return targets

    def _focus_regions_for_target(self, target_type: str, target_id: Optional[str]) -> List[str]:
        normalized_id = str(target_id or "").strip()
        if not normalized_id:
            return []
        if target_type == "contributor":
            cfg = (self._pack.get("contributors") or {}).get(normalized_id) or {}
            return list(cfg.get("renderer_focus_regions") or [])
        if target_type == "symptom":
            cfg = (self._pack.get("symptoms") or {}).get(normalized_id) or {}
            return list(cfg.get("render_focus_regions") or [])
        return []

    def _root_cause_target_type_for_driver(self, driver: Optional[Dict[str, Any]]) -> str:
        role = str((driver or {}).get("role") or "").strip()
        if role == "supporting_symptom":
            return "symptom"
        return "contributor"

    def _root_cause_biomechanics_basis(
        self,
        *,
        mechanism_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        mechanism_cfg = (self._pack.get("mechanisms") or {}).get(str(mechanism_id or "")) or {}
        candidates: List[Tuple[str, str]] = []
        if mechanism_id:
            candidates.append(("mechanism", mechanism_id))
        if isinstance(primary_driver, dict):
            candidates.append(
                (
                    self._root_cause_target_type_for_driver(primary_driver),
                    str(primary_driver.get("id") or ""),
                )
            )
        symptom_id = str((visible_symptom or {}).get("id") or "").strip()
        if symptom_id:
            candidates.append(("symptom", symptom_id))

        selected_basis: Optional[Dict[str, Any]] = None
        selected_target: Optional[Tuple[str, str]] = None
        for target_type, target_id in candidates:
            if not target_id:
                continue
            basis = self._evidence_basis_for_target(target_type, target_id)
            if basis:
                selected_basis = basis
                selected_target = (target_type, target_id)
                break

        if not selected_basis:
            return None

        evidence_items = list(selected_basis.get("evidence_items") or [])
        primary_claim = (
            str((evidence_items[0] or {}).get("claim_summary") or "").strip()
            if evidence_items
            else ""
        )
        supporting_claims = [
            str(item.get("claim_summary") or "").strip()
            for item in evidence_items[1:3]
            if str(item.get("claim_summary") or "").strip()
        ]
        canonical = selected_basis.get("canonical_concept") or {}
        primary_evidence = evidence_items[0] if evidence_items else {}
        principle_claims = [
            str(item).strip()
            for item in list(mechanism_cfg.get("biomechanics_principles") or [])
            if str(item).strip()
        ]
        proof_lines = [
            str(item).strip()
            for item in list(mechanism_cfg.get("cue_ready_proof_lines") or [])
            if str(item).strip()
        ]

        return {
            "target_type": selected_target[0] if selected_target else None,
            "target_id": selected_target[1] if selected_target else None,
            "knowledge_support": selected_basis.get("knowledge_support"),
            "canonical_concept": {
                "concept_id": canonical.get("concept_id"),
                "title": canonical.get("title"),
                "reconciliation_note": canonical.get("reconciliation_note"),
            }
            if canonical
            else None,
            "primary_claim": primary_claim or None,
            "supporting_claims": supporting_claims,
            "evidence_kind": primary_evidence.get("evidence_kind"),
            "evidence_tier": primary_evidence.get("evidence_tier"),
            "principle_claims": principle_claims,
            "cue_ready_proof_lines": proof_lines,
        }

    def _root_cause_renderer_guidance(
        self,
        *,
        mechanism_id: Optional[str],
        primary_driver: Optional[Dict[str, Any]],
        compensation: Optional[Dict[str, Any]],
        visible_symptom: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        mechanism_cfg = (self._pack.get("mechanisms") or {}).get(str(mechanism_id or "")) or {}
        story_ids = list(mechanism_cfg.get("render_story_ids") or [])
        story_id = str(story_ids[0] or "").strip() if story_ids else ""
        proof_lines = [
            str(item).strip()
            for item in list(mechanism_cfg.get("cue_ready_proof_lines") or [])
            if str(item).strip()
        ]
        anchor_risk_ids = self._root_cause_anchor_risk_ids(
            primary_driver=primary_driver,
            compensation=compensation,
            visible_symptom=visible_symptom,
        )
        symptom_text = self._root_cause_symptom_text(
            primary_driver=primary_driver,
            compensation=compensation,
            visible_symptom=visible_symptom,
        )
        load_watch_text = self._root_cause_load_watch_text(
            primary_driver=primary_driver,
            compensation=compensation,
        )
        simple_symptom_text = self._root_cause_simple_symptom_text(
            primary_driver=primary_driver,
            compensation=compensation,
            visible_symptom=visible_symptom,
        )
        simple_load_watch_text = self._root_cause_simple_load_watch_text(
            primary_driver=primary_driver,
            compensation=compensation,
        )
        phase_targets = self._root_cause_phase_targets(
            story_id=story_id or None,
            primary_driver=primary_driver,
            compensation=compensation,
            visible_symptom=visible_symptom,
        )
        if story_ids:
            story_cfg = (self._pack.get("render_stories") or {}).get(story_id) or {}
            focus_regions = list(story_cfg.get("focus_regions") or [])
            cue_points: List[str] = []
            for cue in proof_lines:
                if cue and cue not in cue_points:
                    cue_points.append(cue)
            for cue in [
                str(item.get("cue") or "").strip()
                for item in list(story_cfg.get("storyboard") or [])[:3]
                if str(item.get("cue") or "").strip()
            ]:
                if cue and cue not in cue_points:
                    cue_points.append(cue)
            return {
                "story_id": story_id or None,
                "title": story_cfg.get("title"),
                "focus_regions": focus_regions,
                "phases": list(story_cfg.get("phases") or []),
                "cue_points": cue_points[:4],
                "anchor_risk_ids": anchor_risk_ids,
                "phase_targets": phase_targets,
                "warning_hotspots_allowed": True,
                "symptom_text": symptom_text,
                "load_watch_text": load_watch_text,
                "simple_symptom_text": simple_symptom_text,
                "simple_load_watch_text": simple_load_watch_text,
            }

        if isinstance(primary_driver, dict):
            target_type = self._root_cause_target_type_for_driver(primary_driver)
            focus_regions = self._focus_regions_for_target(
                target_type,
                primary_driver.get("id"),
            )
            if focus_regions:
                return {
                    "story_id": None,
                    "title": primary_driver.get("title"),
                    "focus_regions": focus_regions,
                    "phases": [primary_driver.get("phase")] if primary_driver.get("phase") else [],
                    "cue_points": list(
                        dict.fromkeys(
                            [
                                cue
                                for cue in [
                                    *proof_lines,
                                    str(primary_driver.get("summary") or "").strip(),
                                ]
                                if cue
                            ]
                        )
                    )[:4],
                    "anchor_risk_ids": anchor_risk_ids,
                    "phase_targets": phase_targets,
                    "warning_hotspots_allowed": True,
                    "symptom_text": symptom_text,
                    "load_watch_text": load_watch_text,
                    "simple_symptom_text": simple_symptom_text,
                    "simple_load_watch_text": simple_load_watch_text,
                }

        symptom_id = str((visible_symptom or {}).get("id") or "").strip()
        if symptom_id:
            focus_regions = self._focus_regions_for_target("symptom", symptom_id)
            if focus_regions:
                return {
                    "story_id": None,
                    "title": (visible_symptom or {}).get("title"),
                    "focus_regions": focus_regions,
                    "phases": [str((visible_symptom or {}).get("phase") or "").strip()],
                    "cue_points": list(
                        dict.fromkeys(
                            [
                                cue
                                for cue in [
                                    *proof_lines,
                                    str((visible_symptom or {}).get("summary") or "").strip(),
                                ]
                                if cue
                            ]
                        )
                    )[:4],
                    "anchor_risk_ids": anchor_risk_ids,
                    "phase_targets": phase_targets,
                    "warning_hotspots_allowed": True,
                    "symptom_text": symptom_text,
                    "load_watch_text": load_watch_text,
                    "simple_symptom_text": simple_symptom_text,
                    "simple_load_watch_text": simple_load_watch_text,
                }
        return None

    def _root_cause(
        self,
        *,
        diagnosis_status: str,
        capture_quality: Dict[str, Any],
        primary: Dict[str, Any],
        visible_symptom: Optional[Dict[str, Any]],
        contributors: List[Dict[str, Any]],
        primary_break_point: Optional[Dict[str, Any]],
        kinetic_chain_status: Dict[str, Any],
        mechanism_explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        capture_status = str(capture_quality.get("status") or "").upper()
        break_phase_id = str((primary_break_point or {}).get("phase_id") or "").strip()
        break_title = str((primary_break_point or {}).get("title") or "").strip()
        preferred_body_group = self._preferred_root_cause_body_group(break_phase_id)
        mechanism_cfg = (self._pack.get("mechanisms") or {}).get(str(primary.get("id") or "")) or {}
        primary_driver = self._root_cause_driver(
            contributors=contributors,
            preferred_body_group=preferred_body_group,
            preferred_ids=list(mechanism_cfg.get("primary_driver_ids") or []),
        )
        primary_driver_id = (
            str(primary_driver.get("id") or "").strip()
            if isinstance(primary_driver, dict)
            else ""
        )
        compensation = self._root_cause_compensation(
            mechanism_id=str(primary.get("id") or "").strip() or None,
            primary_driver=primary_driver,
            contributors=contributors,
            exclude_ids={primary_driver_id} if primary_driver_id else set(),
        )
        compensation_id = (
            str(compensation.get("id") or "").strip()
            if isinstance(compensation, dict)
            else ""
        )
        secondary_driver = self._root_cause_driver(
            contributors=contributors,
            preferred_body_group=None,
            exclude_ids={
                item_id
                for item_id in {primary_driver_id, compensation_id}
                if item_id
            },
        )
        mechanism_id = str(primary.get("id") or "").strip() or None
        mechanism_title = str(primary.get("title") or "").strip()
        mechanism_summary = str(
            primary.get("summary")
            or mechanism_explanation.get("primary_mechanism_summary")
            or ""
        ).strip()
        performance_impact = str(mechanism_explanation.get("performance_impact") or "").strip()
        load_impact = str(mechanism_explanation.get("load_impact") or "").strip()
        visible_summary = str((visible_symptom or {}).get("summary") or "").strip()
        biomechanics_basis = self._root_cause_biomechanics_basis(
            mechanism_id=mechanism_id,
            primary_driver=primary_driver,
            visible_symptom=visible_symptom,
        )
        renderer_guidance = self._root_cause_renderer_guidance(
            mechanism_id=mechanism_id,
            primary_driver=primary_driver,
            compensation=compensation,
            visible_symptom=visible_symptom,
        )

        if capture_status == "UNUSABLE":
            return {
                "status": "not_interpretable",
                "mechanism_id": None,
                "title": "Root cause not interpretable yet",
                "summary": "Capture quality is too weak to call a root cause from this clip.",
                "why_it_is_happening": "The clip does not show the chain clearly enough to justify a root-cause story yet.",
                "where_it_starts": None,
                "primary_driver": None,
                "compensation": None,
                "secondary_driver": None,
                "chain_story": "ActionLab should improve the clip evidence first before claiming why the chain is breaking.",
                "performance_impact": None,
                "load_impact": None,
                "biomechanics_basis": None,
                "renderer_guidance": None,
            }

        if str(kinetic_chain_status.get("id") or "") == "connected":
            return {
                "status": "no_clear_problem",
                "mechanism_id": None,
                "title": "No clear root cause",
                "summary": "No single phase is breaking the chain enough to justify a root-cause intervention right now.",
                "why_it_is_happening": "The action looks connected enough that ActionLab should not force a pathology story from this clip.",
                "where_it_starts": None,
                "primary_driver": None,
                "compensation": None,
                "secondary_driver": None,
                "chain_story": "Momentum is carrying through the action cleanly enough that there is no obvious break point to explain away.",
                "performance_impact": None,
                "load_impact": None,
                "biomechanics_basis": None,
                "renderer_guidance": None,
            }

        if not primary:
            summary = "The clip shows a leak, but ActionLab is not yet confident enough to call one primary root cause."
            why_lines = [
                "The action is showing a problem pattern, but the evidence is still short of a confident mechanism match."
            ]
            if break_title:
                why_lines.append(
                    f"The best current clue is around {break_title}, where the chain is starting to look less connected."
                )
            if isinstance(primary_driver, dict):
                why_lines.append(
                    f"{primary_driver['title']} is the strongest current driver signal: {primary_driver['summary']}"
                )

            chain_lines: List[str] = []
            if break_title:
                chain_lines.append(f"The leak appears to start around {break_title}.")
            if isinstance(primary_driver, dict):
                driver_summary = str(primary_driver.get("summary") or "").strip()
                if driver_summary:
                    chain_lines.append(driver_summary)
            elif visible_summary:
                chain_lines.append(visible_summary)
            if performance_impact and not _is_duplicate_story(performance_impact, chain_lines):
                chain_lines.append(performance_impact)
            if load_impact and not _is_duplicate_story(load_impact, chain_lines):
                chain_lines.append(load_impact)

            return {
                "status": "holdback",
                "mechanism_id": None,
                "title": "Root cause still being narrowed",
                "summary": summary,
                "why_it_is_happening": " ".join(why_lines),
                "where_it_starts": (
                    {
                        "phase_id": break_phase_id or None,
                        "title": break_title,
                    }
                    if break_title
                    else None
                ),
                "primary_driver": primary_driver,
                "compensation": compensation,
                "secondary_driver": secondary_driver,
                "chain_story": " ".join(line for line in chain_lines if line),
                "performance_impact": performance_impact or None,
                "load_impact": load_impact or None,
                "biomechanics_basis": biomechanics_basis,
                "renderer_guidance": renderer_guidance,
            }

        why_it_is_happening = (
            mechanism_summary
            or f"{mechanism_title} is the main root-cause story on this clip."
        )
        chain_lines = []
        if break_title:
            chain_lines.append(f"It starts at {break_title}.")
        chain_lines.append(why_it_is_happening)
        if isinstance(primary_driver, dict):
            driver_summary = str(primary_driver.get("summary") or "").strip()
            if driver_summary and not _is_duplicate_story(driver_summary, chain_lines):
                chain_lines.append(driver_summary)
        if isinstance(compensation, dict):
            compensation_line = (
                f"{compensation['title']} is the visible compensation that shows up after the main leak starts."
            )
            if not _is_duplicate_story(compensation_line, chain_lines):
                chain_lines.append(compensation_line)
        elif visible_summary and not _is_duplicate_story(visible_summary, chain_lines):
            chain_lines.append(visible_summary)
        if (
            isinstance(secondary_driver, dict)
            and str(secondary_driver.get("body_group") or "") != str((primary_driver or {}).get("body_group") or "")
        ):
            secondary_line = (
                f"{secondary_driver['title']} shows how the leak is carrying into the "
                f"{str(secondary_driver.get('body_group') or 'rest of the chain').replace('_', ' ')}."
            )
            if not _is_duplicate_story(secondary_line, chain_lines):
                chain_lines.append(secondary_line)
        if performance_impact and not _is_duplicate_story(performance_impact, chain_lines):
            chain_lines.append(performance_impact)

        return {
            "status": "clear",
            "mechanism_id": mechanism_id,
            "title": mechanism_title or "Primary root cause",
            "summary": why_it_is_happening,
            "why_it_is_happening": why_it_is_happening,
            "where_it_starts": (
                {
                    "phase_id": break_phase_id or None,
                    "title": break_title,
                }
                if break_title
                else None
            ),
            "primary_driver": primary_driver,
            "compensation": compensation,
            "secondary_driver": secondary_driver,
            "chain_story": " ".join(line for line in chain_lines if line),
            "performance_impact": performance_impact or None,
            "load_impact": load_impact or None,
            "biomechanics_basis": biomechanics_basis,
            "renderer_guidance": renderer_guidance,
        }

    def _change_strategy(
        self,
        *,
        capture_quality: Dict[str, Any],
        diagnosis_status: str,
        primary_prescription: Dict[str, Any],
        primary_change_reaction: Dict[str, Any],
        primary_followup: Dict[str, Any],
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if str(capture_quality.get("status") or "").upper() == "UNUSABLE":
            return {
                "change_size": "hold",
                "adoption_risk": "unknown",
                "why_smallest_useful_change": "Capture quality is too weak to justify a coaching change yet.",
                "selection_window_safety": "Do not change the action from this clip alone.",
                "match_pressure_risk": "unknown",
                "adoption_rationale": "The safest choice is to improve evidence first, not to coach a change from unclear anchors.",
                "expected_near_term_tradeoff": "Do not introduce a mechanical change until the anchors are clearer.",
                "next_review_window": "Retest with a clearer clip first.",
                "improvement_signal": "Cleaner BFC, FFC, and release anchors before a correction is introduced.",
            }

        if diagnosis_status in {"no_match", "ambiguous_match"}:
            return {
                "change_size": "micro",
                "adoption_risk": "low",
                "why_smallest_useful_change": "The system is holding back, so the safest move is to review anchors before making a bigger change.",
                "selection_window_safety": "Safe because it avoids a rebuild cue while the story is still uncertain.",
                "match_pressure_risk": "low",
                "adoption_rationale": "Bowlers usually accept anchor-checking more easily than a change that might threaten short-term performance.",
                "expected_near_term_tradeoff": "This should protect near-term performance because it avoids a premature rebuild cue.",
                "next_review_window": primary_followup.get("recommended_review_window")
                or "Next 3 deliveries, then next session.",
                "improvement_signal": "The next clip should make one story clearer before a stronger intervention is chosen.",
            }

        avoid_for_now = list(primary_prescription.get("avoid_for_now") or [])
        match_pressure_risk = str(
            primary_change_reaction.get("match_pressure_risk") or ""
        ).strip().lower()
        if avoid_for_now:
            change_size = "micro"
        elif trajectories:
            change_size = "moderate"
        else:
            change_size = "micro"

        adoption_risk = (
            match_pressure_risk
            if match_pressure_risk in {"low", "medium", "high"}
            else ("low" if change_size == "micro" else "medium")
        )
        if len(avoid_for_now) >= 3 and adoption_risk != "high":
            adoption_risk = "medium"

        expected_tradeoff = (
            (
                (primary_change_reaction.get("near_term_negative") or [None])[0]
                or "This should be small enough to protect near-term performance while still changing the main leak."
            )
            if change_size == "micro"
            else (
                (primary_change_reaction.get("near_term_negative") or [None])[0]
                or "This may feel different for a short period, so it should be introduced outside the highest-pressure moments."
            )
        )

        return {
            "change_size": change_size,
            "adoption_risk": adoption_risk,
            "why_smallest_useful_change": (
                primary_prescription.get("why_this_first")
                or "This is the smallest useful intervention before larger changes are considered."
            ),
            "selection_window_safety": primary_change_reaction.get("selection_window_safety"),
            "match_pressure_risk": adoption_risk,
            "adoption_rationale": primary_change_reaction.get("adoption_rationale"),
            "expected_near_term_tradeoff": expected_tradeoff,
            "next_review_window": primary_followup.get("recommended_review_window")
            or primary_prescription.get("reassess_after"),
            "improvement_signal": (
                (primary_followup.get("success_signals") or [None])[0]
                or (primary_prescription.get("followup_metric_targets") or [None])[0]
            ),
        }

    def _acceptance_cfg(self, key: str) -> Dict[str, float]:
        cfg = (((self._pack.get("globals") or {}).get("acceptance_bands") or {}).get(key) or {})
        return {
            "acceptable_max": _safe_float(cfg.get("acceptable_max"), 0.25),
            "workable_max": _safe_float(cfg.get("workable_max"), 0.50),
        }

    def _metric_acceptance_band(
        self,
        metric: Dict[str, Any],
        *,
        inverse_good: bool = False,
        band_key: str = "contributor_severity",
    ) -> str:
        cfg = self._acceptance_cfg(band_key)
        raw_value = metric.get("raw_value")
        base_value = _safe_float(raw_value if isinstance(raw_value, (int, float)) else metric.get("value"), 0.0)
        severity = 1.0 - base_value if inverse_good else base_value
        return _acceptance_band(
            severity,
            acceptable_max=cfg["acceptable_max"],
            workable_max=cfg["workable_max"],
        )

    def _key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        metric_defs = {
            "front_leg_support_score": {"title": "Front-leg support", "inverse_good": True},
            "trunk_drift_after_ffc": {"title": "Late trunk drift", "inverse_good": False},
            "transfer_efficiency_score": {"title": "Transfer efficiency", "inverse_good": True},
            "pelvis_trunk_alignment": {"title": "Pelvis-trunk alignment", "inverse_good": True},
            "shoulder_rotation_timing": {"title": "Shoulder rotation timing", "inverse_good": True},
        }
        out: Dict[str, Dict[str, Any]] = {}
        for metric_id, cfg in metric_defs.items():
            metric = metrics.get(metric_id) or {}
            if not isinstance(metric, dict):
                continue
            out[metric_id] = {
                "id": metric_id,
                "title": cfg["title"],
                "score": _round3(_safe_float(metric.get("value"), 0.0)),
                "confidence": _round3(_safe_float(metric.get("confidence"), 0.0)),
                "acceptance_band": self._metric_acceptance_band(
                    metric,
                    inverse_good=bool(cfg["inverse_good"]),
                ),
            }
        return out

    def _acceptance_summary(
        self,
        *,
        metrics: Dict[str, Any],
        contributors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        key_metrics = self._key_metrics(metrics)
        counts = {"acceptable": 0, "workable": 0, "problematic": 0}
        for metric in key_metrics.values():
            band = str(metric.get("acceptance_band") or "workable")
            if band in counts:
                counts[band] += 1
        for contributor in contributors:
            band = str(contributor.get("acceptance_band") or "").strip()
            if band in counts:
                counts[band] += 1
        overall = "acceptable"
        if counts["problematic"] > 0:
            overall = "problematic"
        elif counts["workable"] > 0:
            overall = "workable"
        return {
            "overall_band": overall,
            "counts": counts,
            "key_metrics": key_metrics,
        }

    def _kinetic_chain_status(
        self,
        *,
        metrics: Dict[str, Any],
        diagnosis_status: str,
        primary_break_point: Dict[str, Any],
        capture_quality: Dict[str, Any],
        acceptance_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        chain_statuses = (self._pack.get("coach_judgments") or {}).get("chain_statuses") or {}
        if str(capture_quality.get("status") or "").upper() == "UNUSABLE":
            status_id = "not_interpretable_with_confidence"
        else:
            break_phase_id = str(primary_break_point.get("phase_id") or "")
            overall_band = str(acceptance_summary.get("overall_band") or "workable")
            release_cost = _safe_float((metrics.get("dissipation_burden_score") or {}).get("value"), 0.0)
            release_gap = 1.0 - _safe_float((metrics.get("release_timing_stability") or {}).get("value"), 0.0)
            if diagnosis_status == "no_match":
                if overall_band == "acceptable":
                    status_id = "connected"
                elif release_cost >= 0.62 and release_gap <= 0.45:
                    status_id = "effective_but_expensive"
                else:
                    status_id = "workable_but_leaking"
            elif diagnosis_status == "ambiguous_match":
                if overall_band == "acceptable":
                    status_id = "connected"
                elif release_cost >= 0.62 and release_gap <= 0.45:
                    status_id = "effective_but_expensive"
                else:
                    status_id = "workable_but_leaking"
            elif break_phase_id == "transfer_and_block":
                status_id = "breaking_at_transfer"
            elif break_phase_id in {"whip_and_release", "UAH", "RELEASE"}:
                status_id = "breaking_at_release_timing"
            elif release_cost >= 0.62:
                status_id = "effective_but_expensive"
            else:
                status_id = "workable_but_leaking"

        cfg = chain_statuses.get(status_id) or {}
        return {
            "id": status_id,
            "label": cfg.get("title"),
            "summary": cfg.get("summary"),
            "coach_prompt": cfg.get("coach_prompt"),
            "acceptance_band": acceptance_summary.get("overall_band"),
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
        strength_signals = (self._pack.get("coach_judgments") or {}).get("strength_signals") or {}
        for metric_name, signal_cfg in strength_signals.items():
            metric = metrics.get(metric_name) or {}
            if not isinstance(metric, dict):
                continue
            min_value = _safe_float(signal_cfg.get("min_value"), 0.65)
            if _safe_float(metric.get("value"), 0.0) >= min_value:
                positives.append(str(signal_cfg.get("summary") or "").strip())
        return list(dict.fromkeys(positives))[:3]

    def _what_is_not_ok(
        self,
        *,
        visible_symptom: Optional[Dict[str, Any]],
        primary: Dict[str, Any],
        contributors: List[Dict[str, Any]],
    ) -> List[str]:
        issues: List[str] = []
        visible_summary = str((visible_symptom or {}).get("summary") or "").strip()
        if visible_summary and not _is_duplicate_story(visible_summary, issues):
            issues.append(visible_summary)

        grouped: Dict[str, str] = {}
        for contributor in contributors:
            if str(contributor.get("acceptance_band") or "") != "problematic":
                continue
            body_group = str(contributor.get("body_group") or "whole_chain")
            summary = str(contributor.get("summary") or "").strip()
            if not summary or body_group in grouped or _is_duplicate_story(summary, issues):
                continue
            grouped[body_group] = summary

        for body_group in ("lower_body", "upper_body", "whole_chain"):
            summary = grouped.get(body_group)
            if summary:
                issues.append(summary)

        primary_summary = str(primary.get("summary") or "").strip()
        if primary_summary and not _is_duplicate_story(primary_summary, issues):
            issues.append(primary_summary)

        return issues[:3]

    def _compensation_patterns(self, symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compensation_ids = set(
            ((self._pack.get("coach_judgments") or {}).get("compensation_symptom_ids") or [])
        )
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
        mapping = (self._pack.get("coach_judgments") or {}).get("symptom_body_groups") or {}
        return str(mapping.get(symptom_id) or "whole_chain")

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
        coach_diagnosis: Dict[str, Any],
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
                "root_cause": coach_diagnosis.get("root_cause"),
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
                "root_cause": coach_diagnosis.get("root_cause"),
            }
        else:
            payload["no_match"] = {
                "holdback_reason": mechanism_explanation.get("performance_impact"),
                "load_holdback_reason": mechanism_explanation.get("load_impact"),
                "coach_check": mechanism_explanation.get("coach_check"),
                "root_cause": coach_diagnosis.get("root_cause"),
                "selected_render_story_ids": list(selection.get("selected_render_story_ids") or []),
            }
        return payload
