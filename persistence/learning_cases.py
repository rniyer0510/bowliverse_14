from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.clinician.knowledge_pack import load_knowledge_pack
from app.common.logger import get_logger
from app.persistence.models import LearningCase, LearningCaseCluster
from app.persistence.session import SessionLocal

logger = get_logger(__name__)

LEARNING_CASE_EVENT_NAME = "actionlab.learning_case.v1"
_OPEN_CLUSTER_STATUSES = {"OPEN", "CLUSTERED", "QUEUED", "UNDER_REVIEW"}

_CASE_RULES = {
    "capture_quality_unusable": {
        "case_type": "LOW_CONFIDENCE",
        "priority_key": "single_run_low_confidence",
        "priority_fallback": "D",
        "suggested_gap_type": "capture_quality",
        "renderer_mode": "event_only",
        "trigger_reason": (
            "Capture quality was unusable, so deterministic mechanism scoring was short-circuited."
        ),
    },
    "no_match": {
        "case_type": "NO_MATCH",
        "priority_key": "no_match_recurring",
        "priority_fallback": "A",
        "suggested_gap_type": "missing_mechanism",
        "renderer_mode": "event_only",
        "trigger_reason": (
            "No mechanism exceeded the weak-match threshold, so the deterministic engine held back a root-cause selection."
        ),
    },
    "ambiguous_match": {
        "case_type": "AMBIGUOUS_MATCH",
        "priority_key": "ambiguous_recurring",
        "priority_fallback": "B",
        "suggested_gap_type": "weak_distinction_rules",
        "renderer_mode": "partial_evidence",
        "trigger_reason": (
            "Top mechanisms remained inside the ambiguity threshold, so the deterministic engine could not separate them cleanly."
        ),
    },
    "weak_match": {
        "case_type": "LOW_CONFIDENCE",
        "priority_key": "single_run_low_confidence",
        "priority_fallback": "D",
        "suggested_gap_type": "weak_evidence",
        "renderer_mode": "partial_evidence",
        "trigger_reason": (
            "A candidate mechanism survived scoring, but confidence stayed below the partial-match threshold."
        ),
    },
}

_COACH_FLAG_RULES = {
    "wrong_mechanism": {
        "priority_key": "coach_feedback_wrong_mechanism",
        "priority_fallback": "B",
        "suggested_gap_type": "coach_wrong_mechanism",
        "trigger_reason": "Coach marked the selected mechanism as incorrect.",
    },
    "wrong_prescription": {
        "priority_key": "coach_feedback_wrong_prescription",
        "priority_fallback": "C",
        "suggested_gap_type": "coach_wrong_prescription",
        "trigger_reason": "Coach marked the prescribed first intervention as incorrect.",
    },
    "right_mechanism_wrong_wording": {
        "priority_key": "coach_feedback_wording",
        "priority_fallback": "D",
        "suggested_gap_type": "coach_wording_revision",
        "trigger_reason": "Coach agreed with the mechanism but flagged the wording as misleading.",
    },
    "renderer_story_misleading": {
        "priority_key": "coach_feedback_renderer",
        "priority_fallback": "E",
        "suggested_gap_type": "coach_renderer_story",
        "trigger_reason": "Coach marked the walkthrough or render story as misleading.",
    },
    "capture_quality_bad": {
        "priority_key": "coach_feedback_capture_quality",
        "priority_fallback": "C",
        "suggested_gap_type": "coach_capture_quality",
        "trigger_reason": "Coach marked the clip quality as too poor for reliable interpretation.",
    },
}


def build_learning_case_event(
    *,
    result: Dict[str, Any],
    account_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    deterministic = _deterministic_payload(result)
    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        return None
    capture_quality = deterministic.get("capture_quality_v1") or {}
    if not isinstance(capture_quality, dict):
        capture_quality = {}
    diagnosis_status = str(selection.get("diagnosis_status") or "").strip().lower()
    if str(capture_quality.get("status") or "").upper() == "UNUSABLE":
        diagnosis_status = "capture_quality_unusable"
    rule = _CASE_RULES.get(diagnosis_status)
    if not rule:
        return None

    symptoms = _symptoms_from_result(result)
    hypotheses = _hypotheses_from_result(result)
    top_hypothesis = _top_hypothesis(selection=selection, hypotheses=hypotheses)
    prescription_plan = deterministic.get("prescription_plan_v1") or {}
    if not isinstance(prescription_plan, dict):
        prescription_plan = {}

    return _base_event_payload(
        result=result,
        account_id=account_id,
        source_type="runtime_gap",
        case_type=rule["case_type"],
        priority=_priority_from_pack(
            rule["priority_key"],
            fallback=rule["priority_fallback"],
        ),
        suggested_gap_type=rule["suggested_gap_type"],
        trigger_reason=rule["trigger_reason"],
        renderer_mode=rule["renderer_mode"],
        symptoms=symptoms,
        hypotheses=hypotheses,
        chosen_mechanism=selection.get("primary_mechanism_id"),
        confidence_breakdown=_confidence_breakdown(top_hypothesis),
        contradictions_triggered=list(top_hypothesis.get("contradiction_notes") or []),
        prescription_id=prescription_plan.get("primary_prescription_id"),
        followup_outcome="NOT_YET_DUE",
    )


def build_coach_feedback_learning_case_event(
    *,
    result: Dict[str, Any],
    account_id: Optional[str],
    coach_flag_type: str,
    notes: Optional[str] = None,
    flagged_mechanism_id: Optional[str] = None,
    flagged_prescription_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    rule = _COACH_FLAG_RULES.get(str(coach_flag_type or "").strip())
    if not rule:
        return None

    deterministic = _deterministic_payload(result)
    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}
    symptoms = _symptoms_from_result(result)
    hypotheses = _hypotheses_from_result(result)
    chosen_mechanism = (
        _safe_str(flagged_mechanism_id)
        or _safe_str(selection.get("primary_mechanism_id"))
    )
    prescription_id = _safe_str(flagged_prescription_id)
    if not prescription_id:
        prescription_plan = deterministic.get("prescription_plan_v1") or {}
        if isinstance(prescription_plan, dict):
            prescription_id = _safe_str(prescription_plan.get("primary_prescription_id"))

    event_payload = _base_event_payload(
        result=result,
        account_id=account_id,
        source_type="coach_feedback",
        case_type="COACH_FEEDBACK",
        priority=_priority_from_pack(
            rule["priority_key"],
            fallback=rule["priority_fallback"],
        ),
        suggested_gap_type=rule["suggested_gap_type"],
        trigger_reason=rule["trigger_reason"],
        renderer_mode=_renderer_mode_from_result(result),
        symptoms=symptoms,
        hypotheses=hypotheses,
        chosen_mechanism=chosen_mechanism,
        confidence_breakdown=_confidence_breakdown(_top_hypothesis(selection=selection, hypotheses=hypotheses)),
        contradictions_triggered=[],
        prescription_id=prescription_id,
        followup_outcome="NOT_YET_DUE",
    )
    event_payload["coach_flag_type"] = coach_flag_type
    if notes:
        event_payload["coach_flag_notes"] = str(notes).strip()
    return event_payload


def build_prescription_non_response_learning_case_event(
    *,
    assigned_result: Dict[str, Any],
    latest_result: Dict[str, Any],
    followup: Dict[str, Any],
    account_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    response_status = str(followup.get("response_status") or "").strip().upper()
    if response_status not in {"NO_CLEAR_CHANGE", "WORSENING"}:
        return None

    deterministic = _deterministic_payload(assigned_result)
    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}
    symptoms = _symptoms_from_result(assigned_result)
    hypotheses = _hypotheses_from_result(assigned_result)
    prescription_id = _safe_str(followup.get("prescription_id"))
    trigger_reason = (
        f"Prescription {prescription_id or 'unknown_prescription'} produced a {response_status.lower()} follow-up outcome inside its review window."
    )
    event_payload = _base_event_payload(
        result=latest_result or assigned_result,
        account_id=account_id,
        source_type="prescription_followup",
        case_type="PRESCRIPTION_NON_RESPONSE",
        priority=_priority_from_pack(
            "prescription_non_response",
            fallback="C",
        ),
        suggested_gap_type="prescription_non_response",
        trigger_reason=trigger_reason,
        renderer_mode=_renderer_mode_from_result(latest_result or assigned_result),
        symptoms=symptoms,
        hypotheses=hypotheses,
        chosen_mechanism=_safe_str(selection.get("primary_mechanism_id")),
        confidence_breakdown=_confidence_breakdown(_top_hypothesis(selection=selection, hypotheses=hypotheses)),
        contradictions_triggered=[],
        prescription_id=prescription_id,
        followup_outcome=response_status,
    )
    event_payload["prescription_assigned_at_run_id"] = followup.get("prescription_assigned_at_run_id")
    event_payload["actual_direction_of_change"] = dict(followup.get("actual_direction_of_change") or {})
    event_payload["expected_direction_of_change"] = dict(followup.get("expected_direction_of_change") or {})
    return event_payload


def symptom_bundle_hash(symptoms: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for symptom in symptoms or []:
        if not isinstance(symptom, dict):
            continue
        symptom_id = str(symptom.get("id") or "").strip()
        severity = str(symptom.get("severity") or "").strip().lower()
        score = _safe_float(symptom.get("score"), 0.0)
        confidence = _safe_float(symptom.get("confidence"), 0.0)
        if not symptom_id:
            continue
        if severity not in {"low", "moderate", "high"}:
            continue
        if score < 0.35 or confidence < 0.15:
            continue
        parts.append(f"{symptom_id}:{severity}")
    canonical = "|".join(sorted(set(parts))) or "no_detected_symptoms"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def write_learning_case(
    *,
    event_payload: Dict[str, Any],
    db: Optional[Session] = None,
) -> Dict[str, str]:
    owns_session = db is None
    db = db or SessionLocal()
    try:
        learning_case_id = uuid.UUID(str(event_payload["event_id"]))
        run_id = uuid.UUID(str(event_payload["run_id"]))
        player_id = uuid.UUID(str(event_payload["player_id"]))
        account_id_raw = event_payload.get("account_id")
        account_id = uuid.UUID(str(account_id_raw)) if account_id_raw else None

        cluster = _get_or_create_cluster(event_payload=event_payload, run_id=run_id, db=db)

        row = LearningCase(
            learning_case_id=learning_case_id,
            run_id=run_id,
            player_id=player_id,
            account_id=account_id,
            learning_case_cluster_id=cluster.learning_case_cluster_id,
            event_name=LEARNING_CASE_EVENT_NAME,
            knowledge_pack_id=_safe_str(event_payload.get("knowledge_pack_id")),
            knowledge_pack_version=str(event_payload["knowledge_pack_version"]),
            source_type=str(event_payload.get("source_type") or "runtime_gap"),
            case_type=str(event_payload["case_type"]),
            priority=str(event_payload["priority"]),
            status="CLUSTERED",
            suggested_gap_type=str(event_payload["suggested_gap_type"]),
            trigger_reason=str(event_payload["trigger_reason"]),
            symptom_bundle_hash=str(event_payload["symptom_bundle_hash"]),
            chosen_mechanism_id=_safe_str(event_payload.get("chosen_mechanism")),
            renderer_mode=str(event_payload["renderer_mode"]),
            prescription_id=_safe_str(event_payload.get("prescription_id")),
            followup_outcome=str(event_payload.get("followup_outcome") or "NOT_YET_DUE"),
            clip_metadata=dict(event_payload.get("clip_metadata") or {}),
            detected_symptoms=list(event_payload.get("detected_symptoms") or []),
            candidate_mechanisms=list(event_payload.get("candidate_mechanisms") or []),
            confidence_breakdown=dict(event_payload.get("confidence_breakdown") or {}),
            contradictions_triggered=list(event_payload.get("contradictions_triggered") or []),
            event_payload=dict(event_payload),
        )

        db.add(row)
        db.flush()

        if cluster.representative_learning_case_id is None:
            cluster.representative_learning_case_id = row.learning_case_id
        cluster.updated_at = datetime.utcnow()
        event_payload["learning_case_cluster_id"] = str(cluster.learning_case_cluster_id)

        if owns_session:
            db.commit()
        else:
            db.flush()
        logger.info(
            "[learning_case] stored learning_case_id=%s cluster_id=%s run_id=%s case_type=%s priority=%s",
            learning_case_id,
            cluster.learning_case_cluster_id,
            run_id,
            row.case_type,
            row.priority,
        )
        return {
            "learning_case_id": str(learning_case_id),
            "learning_case_cluster_id": str(cluster.learning_case_cluster_id),
        }
    except Exception:
        if owns_session:
            db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def increment_cluster_coach_flag_count(
    *,
    learning_case_cluster_id: Optional[str],
    db: Session,
) -> None:
    if not learning_case_cluster_id:
        return
    try:
        cluster_uuid = uuid.UUID(str(learning_case_cluster_id))
    except Exception:
        return
    cluster = db.get(LearningCaseCluster, cluster_uuid)
    if cluster is None:
        return
    cluster.coach_flag_count = int(cluster.coach_flag_count or 0) + 1
    cluster.updated_at = datetime.utcnow()
    db.flush()


def _base_event_payload(
    *,
    result: Dict[str, Any],
    account_id: Optional[str],
    source_type: str,
    case_type: str,
    priority: str,
    suggested_gap_type: str,
    trigger_reason: str,
    renderer_mode: str,
    symptoms: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    chosen_mechanism: Optional[str],
    confidence_breakdown: Dict[str, Any],
    contradictions_triggered: List[str],
    prescription_id: Optional[str],
    followup_outcome: str,
) -> Dict[str, Any]:
    deterministic = _deterministic_payload(result)
    detected_symptoms = _detected_symptoms(symptoms)
    return {
        "event_name": LEARNING_CASE_EVENT_NAME,
        "event_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_pack_id": deterministic.get("knowledge_pack_id"),
        "knowledge_pack_version": (
            deterministic.get("knowledge_pack_version")
            or _safe_str(result.get("knowledge_pack_version"))
            or load_knowledge_pack()["pack_version"]
        ),
        "run_id": result.get("run_id"),
        "player_id": ((result.get("input") or {}).get("player_id")),
        "account_id": account_id,
        "source_type": source_type,
        "case_type": case_type,
        "priority": priority,
        "status": "CLUSTERED",
        "symptom_bundle_hash": symptom_bundle_hash(symptoms),
        "clip_metadata": _clip_metadata(result),
        "detected_symptoms": [item["id"] for item in detected_symptoms],
        "candidate_mechanisms": _candidate_mechanisms(hypotheses),
        "chosen_mechanism": chosen_mechanism,
        "confidence_breakdown": dict(confidence_breakdown or {}),
        "contradictions_triggered": list(contradictions_triggered or []),
        "renderer_mode": renderer_mode,
        "prescription_id": prescription_id,
        "followup_outcome": followup_outcome,
        "trigger_reason": trigger_reason,
        "suggested_gap_type": suggested_gap_type,
    }


def _get_or_create_cluster(
    *,
    event_payload: Dict[str, Any],
    run_id: uuid.UUID,
    db: Session,
) -> LearningCaseCluster:
    cluster_key = _cluster_key_hash(event_payload)
    cluster = (
        db.query(LearningCaseCluster)
        .filter(
            LearningCaseCluster.cluster_key_hash == cluster_key,
            LearningCaseCluster.status.in_(tuple(_OPEN_CLUSTER_STATUSES)),
        )
        .first()
    )
    now = datetime.utcnow()
    if cluster is None:
        cluster = LearningCaseCluster(
            learning_case_cluster_id=uuid.uuid4(),
            cluster_key_hash=cluster_key,
            knowledge_pack_id=_safe_str(event_payload.get("knowledge_pack_id")),
            knowledge_pack_version=str(event_payload["knowledge_pack_version"]),
            source_type=str(event_payload.get("source_type") or "runtime_gap"),
            case_type=str(event_payload["case_type"]),
            priority=str(event_payload["priority"]),
            status="OPEN",
            suggested_gap_type=str(event_payload["suggested_gap_type"]),
            trigger_reason=str(event_payload["trigger_reason"]),
            symptom_bundle_hash=str(event_payload["symptom_bundle_hash"]),
            renderer_mode=_safe_str(event_payload.get("renderer_mode")),
            chosen_mechanism_id=_safe_str(event_payload.get("chosen_mechanism")),
            prescription_id=_safe_str(event_payload.get("prescription_id")),
            candidate_mechanism_ids=[
                item["id"]
                for item in list(event_payload.get("candidate_mechanisms") or [])
                if isinstance(item, dict) and item.get("id")
            ],
            cluster_payload={
                "source_type": event_payload.get("source_type"),
                "case_type": event_payload.get("case_type"),
                "suggested_gap_type": event_payload.get("suggested_gap_type"),
                "candidate_mechanisms": list(event_payload.get("candidate_mechanisms") or []),
            },
            case_count=1,
            coach_flag_count=0,
            first_run_id=run_id,
            latest_run_id=run_id,
            created_at=now,
            updated_at=now,
        )
        db.add(cluster)
        db.flush()
        return cluster

    cluster.case_count = int(cluster.case_count or 0) + 1
    cluster.latest_run_id = run_id
    cluster.updated_at = now
    cluster.priority = _higher_priority(cluster.priority, str(event_payload["priority"]))
    if not cluster.renderer_mode and event_payload.get("renderer_mode"):
        cluster.renderer_mode = _safe_str(event_payload.get("renderer_mode"))
    if not cluster.prescription_id and event_payload.get("prescription_id"):
        cluster.prescription_id = _safe_str(event_payload.get("prescription_id"))
    db.flush()
    return cluster


def _cluster_key_hash(event_payload: Dict[str, Any]) -> str:
    candidate_ids = sorted(
        {
            str(item.get("id")).strip()
            for item in list(event_payload.get("candidate_mechanisms") or [])
            if isinstance(item, dict) and item.get("id")
        }
    )
    parts = [
        str(event_payload.get("source_type") or "runtime_gap").strip(),
        str(event_payload.get("knowledge_pack_version") or "").strip(),
        str(event_payload.get("case_type") or "").strip(),
        str(event_payload.get("suggested_gap_type") or "").strip(),
        str(event_payload.get("symptom_bundle_hash") or "").strip(),
        str(event_payload.get("renderer_mode") or "").strip(),
        str(event_payload.get("prescription_id") or "").strip(),
        str(event_payload.get("chosen_mechanism") or "").strip(),
        ",".join(candidate_ids),
    ]
    canonical = "|".join(parts)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _priority_from_pack(priority_key: str, *, fallback: str) -> str:
    try:
        globals_cfg = load_knowledge_pack()["globals"]
    except Exception:
        return fallback
    defaults = globals_cfg.get("cluster_priority_defaults") or {}
    value = str(defaults.get(priority_key) or fallback).strip().upper()
    return value if value in {"A", "B", "C", "D", "E"} else fallback


def _renderer_mode_from_result(result: Dict[str, Any]) -> str:
    render_reasoning = result.get("render_reasoning_v1") or {}
    if isinstance(render_reasoning, dict):
        explicit_mode = _safe_str(render_reasoning.get("renderer_mode"))
        if explicit_mode:
            return explicit_mode
    deterministic = _deterministic_payload(result)
    selection = deterministic.get("selection") or {}
    diagnosis_status = str((selection or {}).get("diagnosis_status") or "").strip().lower()
    if diagnosis_status == "no_match":
        return "event_only"
    if diagnosis_status in {"ambiguous_match", "weak_match"}:
        return "partial_evidence"
    return "full_causal_story"


def _deterministic_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    deterministic = result.get("deterministic_expert_v1") or {}
    return deterministic if isinstance(deterministic, dict) else {}


def _symptoms_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    symptoms = _deterministic_payload(result).get("symptoms") or []
    return symptoms if isinstance(symptoms, list) else []


def _hypotheses_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    hypotheses = _deterministic_payload(result).get("mechanism_hypotheses") or []
    return hypotheses if isinstance(hypotheses, list) else []


def _top_hypothesis(
    *,
    selection: Dict[str, Any],
    hypotheses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primary = selection.get("primary") or {}
    if isinstance(primary, dict) and primary:
        return dict(primary)
    if hypotheses and isinstance(hypotheses[0], dict):
        return dict(hypotheses[0])
    return {}


def _detected_symptoms(symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detected: List[Dict[str, Any]] = []
    for symptom in symptoms or []:
        if not isinstance(symptom, dict):
            continue
        score = _safe_float(symptom.get("score"), 0.0)
        confidence = _safe_float(symptom.get("confidence"), 0.0)
        severity = str(symptom.get("severity") or "").strip().lower()
        if score < 0.35 or confidence < 0.15:
            continue
        if severity not in {"low", "moderate", "high"}:
            continue
        detected.append(
            {
                "id": str(symptom.get("id") or ""),
                "severity": severity,
            }
        )
    return detected


def _candidate_mechanisms(hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in hypotheses[:3]:
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "id": item.get("id"),
                "confidence": item.get("overall_confidence"),
                "support_score": item.get("support_score"),
                "contradiction_penalty": item.get("contradiction_penalty"),
                "evidence_completeness": item.get("evidence_completeness"),
            }
        )
    return candidates


def _confidence_breakdown(hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "support_score": _safe_float(hypothesis.get("support_score"), 0.0),
        "contradiction_penalty": _safe_float(hypothesis.get("contradiction_penalty"), 0.0),
        "evidence_completeness": _safe_float(hypothesis.get("evidence_completeness"), 0.0),
        "overall_confidence": _safe_float(hypothesis.get("overall_confidence"), 0.0),
    }


def _clip_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    video = result.get("video") or {}
    input_cfg = result.get("input") or {}
    action = result.get("action") or {}
    return {
        "fps": _safe_float(video.get("fps"), 0.0),
        "total_frames": _safe_int(video.get("total_frames")),
        "hand": _safe_str(input_cfg.get("hand")),
        "age_group": _safe_str(input_cfg.get("age_group")),
        "season": _safe_int(input_cfg.get("season")),
        "action_type": _safe_str(action.get("action_type")),
        "action_intent": _safe_str(action.get("intent")),
    }


def _higher_priority(current: Optional[str], incoming: str) -> str:
    order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    current_norm = str(current or incoming).strip().upper()
    incoming_norm = str(incoming).strip().upper()
    if order.get(incoming_norm, 99) < order.get(current_norm, 99):
        return incoming_norm
    return current_norm if current_norm in order else incoming_norm


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


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None
