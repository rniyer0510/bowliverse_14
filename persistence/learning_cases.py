from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.common.logger import get_logger
from app.persistence.models import LearningCase
from app.persistence.session import SessionLocal

logger = get_logger(__name__)

LEARNING_CASE_EVENT_NAME = "actionlab.learning_case.v1"

_CASE_RULES = {
    "no_match": {
        "case_type": "NO_MATCH",
        "priority": "A",
        "suggested_gap_type": "missing_mechanism",
        "renderer_mode": "event_only",
        "trigger_reason": (
            "No mechanism exceeded the weak-match threshold, so the deterministic engine held back a root-cause selection."
        ),
    },
    "ambiguous_match": {
        "case_type": "AMBIGUOUS_MATCH",
        "priority": "B",
        "suggested_gap_type": "weak_distinction_rules",
        "renderer_mode": "partial_evidence",
        "trigger_reason": (
            "Top mechanisms remained inside the ambiguity threshold, so the deterministic engine could not separate them cleanly."
        ),
    },
    "weak_match": {
        "case_type": "LOW_CONFIDENCE",
        "priority": "D",
        "suggested_gap_type": "weak_evidence",
        "renderer_mode": "partial_evidence",
        "trigger_reason": (
            "A candidate mechanism survived scoring, but confidence stayed below the partial-match threshold."
        ),
    },
}


def build_learning_case_event(
    *,
    result: Dict[str, Any],
    account_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    deterministic = result.get("deterministic_expert_v1") or {}
    if not isinstance(deterministic, dict):
        return None

    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        return None
    diagnosis_status = str(selection.get("diagnosis_status") or "").strip().lower()
    rule = _CASE_RULES.get(diagnosis_status)
    if not rule:
        return None

    symptoms = deterministic.get("symptoms") or []
    if not isinstance(symptoms, list):
        symptoms = []
    hypotheses = deterministic.get("mechanism_hypotheses") or []
    if not isinstance(hypotheses, list):
        hypotheses = []
    top_hypothesis = _top_hypothesis(selection=selection, hypotheses=hypotheses)

    event_id = str(uuid.uuid4())
    detected_symptoms = _detected_symptoms(symptoms)
    clip_metadata = _clip_metadata(result)
    prescription_plan = deterministic.get("prescription_plan_v1") or {}
    if not isinstance(prescription_plan, dict):
        prescription_plan = {}

    return {
        "event_name": LEARNING_CASE_EVENT_NAME,
        "event_id": event_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_pack_id": deterministic.get("knowledge_pack_id"),
        "knowledge_pack_version": deterministic.get("knowledge_pack_version"),
        "run_id": result.get("run_id"),
        "player_id": ((result.get("input") or {}).get("player_id")),
        "account_id": account_id,
        "case_type": rule["case_type"],
        "priority": rule["priority"],
        "status": "OPEN",
        "symptom_bundle_hash": symptom_bundle_hash(symptoms),
        "clip_metadata": clip_metadata,
        "detected_symptoms": [item["id"] for item in detected_symptoms],
        "candidate_mechanisms": _candidate_mechanisms(hypotheses),
        "chosen_mechanism": selection.get("primary_mechanism_id"),
        "confidence_breakdown": _confidence_breakdown(top_hypothesis),
        "contradictions_triggered": list(top_hypothesis.get("contradiction_notes") or []),
        "renderer_mode": rule["renderer_mode"],
        "prescription_id": prescription_plan.get("primary_prescription_id"),
        "followup_outcome": "NOT_YET_DUE",
        "trigger_reason": rule["trigger_reason"],
        "suggested_gap_type": rule["suggested_gap_type"],
    }


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
) -> str:
    owns_session = db is None
    db = db or SessionLocal()
    try:
        learning_case_id = uuid.UUID(str(event_payload["event_id"]))
        run_id = uuid.UUID(str(event_payload["run_id"]))
        player_id = uuid.UUID(str(event_payload["player_id"]))
        account_id_raw = event_payload.get("account_id")
        account_id = uuid.UUID(str(account_id_raw)) if account_id_raw else None

        row = LearningCase(
            learning_case_id=learning_case_id,
            run_id=run_id,
            player_id=player_id,
            account_id=account_id,
            event_name=LEARNING_CASE_EVENT_NAME,
            knowledge_pack_id=_safe_str(event_payload.get("knowledge_pack_id")),
            knowledge_pack_version=str(event_payload["knowledge_pack_version"]),
            case_type=str(event_payload["case_type"]),
            priority=str(event_payload["priority"]),
            status=str(event_payload.get("status") or "OPEN"),
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
        if owns_session:
            db.commit()
        else:
            db.flush()
        logger.info(
            "[learning_case] stored learning_case_id=%s run_id=%s case_type=%s priority=%s",
            learning_case_id,
            run_id,
            row.case_type,
            row.priority,
        )
        return str(learning_case_id)
    except Exception:
        if owns_session:
            db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


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
    return {
        "fps": _safe_float(video.get("fps"), 0.0),
        "total_frames": _safe_int(video.get("total_frames")),
        "hand": _safe_str(input_cfg.get("hand")),
        "age_group": _safe_str(input_cfg.get("age_group")),
        "season": _safe_int(input_cfg.get("season")),
    }


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
