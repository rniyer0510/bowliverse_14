from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.persistence.knowledge_pack_monitoring import resolve_open_rollback_alerts_for_candidate
from app.persistence.models import (
    KnowledgePackReleaseCandidate,
    KnowledgePackReleaseEvent,
    LearningCase,
    LearningCaseCluster,
)

ELIGIBLE_CLUSTER_STATUSES = {"RESOLVED"}
RELEASE_ACTIONS = {
    "record_schema_validation",
    "record_referential_integrity",
    "record_regression_pass",
    "promote_dev",
    "promote_staging",
    "record_staging_evaluation",
    "approve_production",
    "promote_production",
    "reject",
    "rollback",
}


def create_release_candidate(
    *,
    knowledge_pack_id: str,
    base_pack_version: str,
    candidate_pack_version: str,
    summary: str,
    created_by_account_id: Optional[str],
    cluster_rows: List[LearningCaseCluster],
    case_rows: List[LearningCase],
    change_summary: Optional[Dict[str, Any]],
    tests_added: Optional[List[str]],
    reinterpret_run_ids: Optional[List[str]],
    supersedes_pack_version: Optional[str],
    db: Session,
) -> KnowledgePackReleaseCandidate:
    if not cluster_rows:
        raise ValueError("At least one reviewed learning-case cluster is required")
    invalid = [
        str(row.learning_case_cluster_id)
        for row in cluster_rows
        if str(getattr(row, "status", "") or "").upper() not in ELIGIBLE_CLUSTER_STATUSES
    ]
    if invalid:
        raise ValueError(
            f"All motivating clusters must be RESOLVED before release-candidate creation: {', '.join(invalid)}"
        )
    existing = (
        db.query(KnowledgePackReleaseCandidate)
        .filter(KnowledgePackReleaseCandidate.candidate_pack_version == str(candidate_pack_version).strip())
        .first()
    )
    if existing:
        raise ValueError(f"Release candidate already exists for pack version {candidate_pack_version}")

    now = datetime.utcnow()
    candidate = KnowledgePackReleaseCandidate(
        knowledge_pack_release_candidate_id=uuid.uuid4(),
        knowledge_pack_id=str(knowledge_pack_id).strip(),
        base_pack_version=str(base_pack_version).strip(),
        candidate_pack_version=str(candidate_pack_version).strip(),
        supersedes_pack_version=_safe_str(supersedes_pack_version),
        status="DRAFT",
        current_environment=None,
        summary=str(summary).strip(),
        change_summary_json=dict(change_summary or {}),
        motivating_cluster_ids=[str(row.learning_case_cluster_id) for row in cluster_rows],
        motivating_case_ids=[str(row.learning_case_id) for row in case_rows if getattr(row, "learning_case_id", None)],
        tests_added=_clean_str_list(tests_added),
        reinterpret_run_ids=_clean_str_list(reinterpret_run_ids),
        schema_validated=False,
        referential_integrity_validated=False,
        regression_suite_passed=False,
        staging_evaluation_passed=False,
        approval_granted=False,
        created_by_account_id=_parse_uuid(created_by_account_id),
        updated_by_account_id=_parse_uuid(created_by_account_id),
        promoted_at=None,
        created_at=now,
        updated_at=now,
    )
    db.add(candidate)
    db.flush()

    _record_release_event(
        candidate_row=candidate,
        action="create",
        account_id=created_by_account_id,
        from_status="NONE",
        to_status=candidate.status,
        from_environment=None,
        to_environment=candidate.current_environment,
        notes=summary,
        metadata={
            "motivating_cluster_ids": list(candidate.motivating_cluster_ids or []),
            "motivating_case_ids": list(candidate.motivating_case_ids or []),
            "tests_added": list(candidate.tests_added or []),
            "reinterpret_run_ids": list(candidate.reinterpret_run_ids or []),
        },
        db=db,
    )
    return candidate


def apply_release_action(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    action: str,
    account_id: Optional[str],
    notes: Optional[str],
    metadata: Optional[Dict[str, Any]],
    db: Session,
) -> KnowledgePackReleaseEvent:
    action_norm = _normalize_action(action)
    from_status = str(candidate_row.status or "DRAFT")
    from_environment = _safe_str(candidate_row.current_environment)
    now = datetime.utcnow()

    if action_norm == "record_schema_validation":
        candidate_row.schema_validated = True
    elif action_norm == "record_referential_integrity":
        candidate_row.referential_integrity_validated = True
    elif action_norm == "record_regression_pass":
        _require(candidate_row.schema_validated and candidate_row.referential_integrity_validated, "Schema and referential integrity must pass before regression is recorded")
        candidate_row.regression_suite_passed = True
    elif action_norm == "promote_dev":
        _require(candidate_row.schema_validated and candidate_row.referential_integrity_validated, "Schema and referential integrity must pass before dev promotion")
        candidate_row.status = "IN_DEV"
        candidate_row.current_environment = "dev"
    elif action_norm == "promote_staging":
        _require(candidate_row.regression_suite_passed, "Regression suite must pass before staging promotion")
        candidate_row.status = "IN_STAGING"
        candidate_row.current_environment = "staging"
    elif action_norm == "record_staging_evaluation":
        _require(str(candidate_row.current_environment or "") == "staging", "Candidate must be in staging before staging evaluation is recorded")
        candidate_row.staging_evaluation_passed = True
    elif action_norm == "approve_production":
        _require(candidate_row.staging_evaluation_passed, "Staging evaluation must pass before production approval")
        candidate_row.approval_granted = True
        candidate_row.status = "APPROVED"
    elif action_norm == "promote_production":
        _require(candidate_row.approval_granted, "Production approval must be granted before production promotion")
        _require(str(candidate_row.current_environment or "") == "staging", "Candidate must be in staging before production promotion")
        candidate_row.status = "PROMOTED"
        candidate_row.current_environment = "production"
        candidate_row.promoted_at = now
    elif action_norm == "reject":
        candidate_row.status = "REJECTED"
    elif action_norm == "rollback":
        _require(str(candidate_row.current_environment or "") == "production", "Only production candidates can be rolled back")
        candidate_row.status = "ROLLED_BACK"
        candidate_row.current_environment = "staging"
        resolve_open_rollback_alerts_for_candidate(
            candidate_id=candidate_row.knowledge_pack_release_candidate_id,
            db=db,
            resolution_status="ROLLED_BACK",
        )

    candidate_row.updated_at = now
    candidate_row.updated_by_account_id = _parse_uuid(account_id)

    event = _record_release_event(
        candidate_row=candidate_row,
        action=action_norm,
        account_id=account_id,
        from_status=from_status,
        to_status=str(candidate_row.status),
        from_environment=from_environment,
        to_environment=_safe_str(candidate_row.current_environment),
        notes=notes,
        metadata=dict(metadata or {}),
        db=db,
    )
    db.flush()
    return event


def _record_release_event(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    action: str,
    account_id: Optional[str],
    from_status: str,
    to_status: str,
    from_environment: Optional[str],
    to_environment: Optional[str],
    notes: Optional[str],
    metadata: Optional[Dict[str, Any]],
    db: Session,
) -> KnowledgePackReleaseEvent:
    event = KnowledgePackReleaseEvent(
        knowledge_pack_release_event_id=uuid.uuid4(),
        knowledge_pack_release_candidate_id=candidate_row.knowledge_pack_release_candidate_id,
        account_id=_parse_uuid(account_id),
        action=action,
        from_status=from_status,
        to_status=to_status,
        from_environment=_safe_str(from_environment),
        to_environment=_safe_str(to_environment),
        notes=_safe_str(notes),
        metadata_json=dict(metadata or {}),
        created_at=datetime.utcnow(),
    )
    db.add(event)
    db.flush()
    return event


def _normalize_action(action: str) -> str:
    action_norm = str(action or "").strip().lower()
    if action_norm not in RELEASE_ACTIONS:
        raise ValueError(f"Unsupported release action: {action}")
    return action_norm


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _clean_str_list(values: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    for value in values or []:
        text = _safe_str(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _parse_uuid(value: Any) -> Optional[uuid.UUID]:
    text = _safe_str(value)
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except Exception:
        return None
