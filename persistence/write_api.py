"""
Explicit WRITE APIs (Phase-I + Coach Notes)

NOTE:
Player creation & updates are handled via read_api.py
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid

from app.persistence.session import get_db
from app.persistence.learning_cases import (
    apply_cluster_review_action,
    apply_learning_case_review_action,
    build_coach_feedback_learning_case_event,
    increment_cluster_coach_flag_count,
    write_learning_case,
)
from app.persistence.knowledge_pack_releases import (
    apply_release_action,
    create_release_candidate,
)
from app.persistence.knowledge_pack_regressions import run_release_candidate_regression
from app.persistence.models import (
    AnalysisRun,
    AnalysisResultRaw,
    AccountPlayerLink,
    CoachFlag,
    KnowledgePackRegressionRun,
    KnowledgePackReleaseCandidate,
    KnowledgePackReleaseEvent,
    LearningCase,
    LearningCaseCluster,
    LearningCaseReviewEvent,
)
from app.common.logger import get_logger
from app.common.auth import get_current_account

logger = get_logger(__name__)

router = APIRouter()


class CoachNotesUpdate(BaseModel):
    """Request model for updating coach notes - notes can be empty string"""
    coach_notes: str  # Can be empty string to clear notes


class CoachFlagCreate(BaseModel):
    flag_type: str
    notes: str = ""
    flagged_mechanism_id: str | None = None
    flagged_prescription_id: str | None = None


class LearningCaseReviewActionCreate(BaseModel):
    action: str
    notes: str = ""


class KnowledgePackReleaseCandidateCreate(BaseModel):
    knowledge_pack_id: str
    base_pack_version: str
    candidate_pack_version: str
    summary: str
    motivating_cluster_ids: list[str]
    change_summary: dict = {}
    tests_added: list[str] = []
    reinterpret_run_ids: list[str] = []
    supersedes_pack_version: str | None = None


class KnowledgePackReleaseActionCreate(BaseModel):
    action: str
    notes: str = ""
    metadata: dict = {}


@router.patch("/analysis/{run_id}/coach-notes")
async def update_coach_notes(
    run_id: str,
    update: CoachNotesUpdate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db)
):
    """
    Update coach notes for a specific analysis run.
    
    Notes can be empty string to clear existing notes.
    
    Args:
        run_id: UUID of the analysis run
        update: Coach notes update payload
        db: Database session
    
    Returns:
        Updated analysis run with coach notes
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    
    # Find the analysis run
    analysis_run = db.query(AnalysisRun).filter(
        AnalysisRun.run_id == run_uuid
    ).first()
    
    if not analysis_run:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # Enforce account ownership/link access.
    link = (
        db.query(AccountPlayerLink)
        .filter(
            AccountPlayerLink.account_id == current_account.account_id,
            AccountPlayerLink.player_id == analysis_run.player_id,
        )
        .first()
    )
    if not link:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update coach notes (can be empty string)
    analysis_run.coach_notes = update.coach_notes if update.coach_notes else None
    
    try:
        db.commit()
        db.refresh(analysis_run)
        logger.info(f"Coach notes updated for run_id={run_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update coach notes: {e}")
        raise HTTPException(status_code=500, detail="Failed to update coach notes")
    
    return {
        "run_id": str(analysis_run.run_id),
        "coach_notes": analysis_run.coach_notes or "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/analysis/{run_id}/coach-notes")
async def get_coach_notes(
    run_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db)
):
    """
    Get coach notes for a specific analysis run.
    
    Returns empty string if no notes exist.
    
    Args:
        run_id: UUID of the analysis run
        db: Database session
    
    Returns:
        Coach notes for the analysis run (empty string if none)
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    
    analysis_run = db.query(AnalysisRun).filter(
        AnalysisRun.run_id == run_uuid
    ).first()
    
    if not analysis_run:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # Enforce account ownership/link access.
    link = (
        db.query(AccountPlayerLink)
        .filter(
            AccountPlayerLink.account_id == current_account.account_id,
            AccountPlayerLink.player_id == analysis_run.player_id,
        )
        .first()
    )
    if not link:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "run_id": str(analysis_run.run_id),
        "coach_notes": analysis_run.coach_notes or ""
    }


@router.post("/analysis/{run_id}/coach-flags")
async def create_coach_flag(
    run_id: str,
    payload: CoachFlagCreate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    if str(getattr(current_account, "role", "")).lower() != "coach":
        raise HTTPException(status_code=403, detail="Coach feedback is only available to coach accounts")

    analysis_run = db.query(AnalysisRun).filter(AnalysisRun.run_id == run_uuid).first()
    if not analysis_run:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    link = (
        db.query(AccountPlayerLink)
        .filter(
            AccountPlayerLink.account_id == current_account.account_id,
            AccountPlayerLink.player_id == analysis_run.player_id,
        )
        .first()
    )
    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    flag_type = str(payload.flag_type or "").strip()
    if flag_type not in {
        "wrong_mechanism",
        "wrong_prescription",
        "right_mechanism_wrong_wording",
        "renderer_story_misleading",
        "capture_quality_bad",
    }:
        raise HTTPException(status_code=400, detail="Unsupported coach flag type")

    existing = (
        db.query(CoachFlag)
        .filter(
            CoachFlag.run_id == run_uuid,
            CoachFlag.account_id == current_account.account_id,
            CoachFlag.flag_type == flag_type,
            CoachFlag.flagged_mechanism_id == (payload.flagged_mechanism_id.strip() if payload.flagged_mechanism_id else None),
            CoachFlag.flagged_prescription_id == (payload.flagged_prescription_id.strip() if payload.flagged_prescription_id else None),
        )
        .first()
    )
    if existing:
        existing.notes = payload.notes.strip() or None
        existing.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        return _coach_flag_response(existing, deduped=True)

    raw = db.query(AnalysisResultRaw).filter(AnalysisResultRaw.run_id == run_uuid).first()
    result_json = raw.result_json if raw and isinstance(raw.result_json, dict) else _minimal_result_for_run(analysis_run)
    event_payload = build_coach_feedback_learning_case_event(
        result=result_json,
        account_id=str(current_account.account_id),
        coach_flag_type=flag_type,
        notes=payload.notes,
        flagged_mechanism_id=payload.flagged_mechanism_id,
        flagged_prescription_id=payload.flagged_prescription_id,
    )

    learning_case_id = None
    learning_case_cluster_id = None
    if event_payload:
        stored = write_learning_case(event_payload=event_payload, db=db)
        learning_case_id = stored["learning_case_id"]
        learning_case_cluster_id = stored["learning_case_cluster_id"]

    deterministic = ((result_json or {}).get("deterministic_expert_v1") or {})
    coach_flag = CoachFlag(
        coach_flag_id=uuid.uuid4(),
        run_id=analysis_run.run_id,
        player_id=analysis_run.player_id,
        account_id=current_account.account_id,
        knowledge_pack_id=(deterministic.get("knowledge_pack_id") if isinstance(deterministic, dict) else None),
        knowledge_pack_version=(deterministic.get("knowledge_pack_version") if isinstance(deterministic, dict) else None),
        flag_type=flag_type,
        notes=payload.notes.strip() or None,
        flagged_mechanism_id=payload.flagged_mechanism_id.strip() if payload.flagged_mechanism_id else None,
        flagged_prescription_id=payload.flagged_prescription_id.strip() if payload.flagged_prescription_id else None,
        learning_case_id=uuid.UUID(learning_case_id) if learning_case_id else None,
        learning_case_cluster_id=uuid.UUID(learning_case_cluster_id) if learning_case_cluster_id else None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(coach_flag)
    db.flush()
    increment_cluster_coach_flag_count(
        learning_case_cluster_id=learning_case_cluster_id,
        db=db,
    )
    db.commit()
    db.refresh(coach_flag)
    return _coach_flag_response(coach_flag, deduped=False)


@router.get("/analysis/{run_id}/coach-flags")
async def list_coach_flags(
    run_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    analysis_run = db.query(AnalysisRun).filter(AnalysisRun.run_id == run_uuid).first()
    if not analysis_run:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    link = (
        db.query(AccountPlayerLink)
        .filter(
            AccountPlayerLink.account_id == current_account.account_id,
            AccountPlayerLink.player_id == analysis_run.player_id,
        )
        .first()
    )
    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    rows = (
        db.query(CoachFlag)
        .filter(CoachFlag.run_id == run_uuid)
        .order_by(CoachFlag.created_at.desc())
        .all()
    )
    return {
        "run_id": run_id,
        "items": [_coach_flag_response(row, deduped=False) for row in rows],
    }


@router.post("/learning-case-clusters/{learning_case_cluster_id}/review-actions")
async def create_learning_case_cluster_review_action(
    learning_case_cluster_id: str,
    payload: LearningCaseReviewActionCreate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_coach_reviewer(current_account)
    cluster_uuid = _parse_uuid_param(learning_case_cluster_id, "learning_case_cluster_id")
    cluster_row = (
        db.query(LearningCaseCluster)
        .filter(LearningCaseCluster.learning_case_cluster_id == cluster_uuid)
        .first()
    )
    if not cluster_row:
        raise HTTPException(status_code=404, detail="Learning case cluster not found")

    case_rows = (
        db.query(LearningCase)
        .filter(LearningCase.learning_case_cluster_id == cluster_uuid)
        .order_by(LearningCase.created_at.asc())
        .all()
    )
    if not case_rows:
        raise HTTPException(status_code=404, detail="Learning case cluster has no cases")

    _ensure_learning_case_access(
        current_account=current_account,
        player_id=case_rows[0].player_id,
        db=db,
    )

    try:
        event = apply_cluster_review_action(
            cluster_row=cluster_row,
            case_rows=case_rows,
            action=payload.action,
            account_id=str(current_account.account_id),
            notes=payload.notes,
            metadata={"target_type": "cluster"},
            db=db,
        )
        db.commit()
        db.refresh(cluster_row)
        db.refresh(event)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise

    return {
        "learning_case_cluster_id": str(cluster_row.learning_case_cluster_id),
        "status": cluster_row.status,
        "case_count": len(case_rows),
        "review_event": _learning_case_review_event_response(event),
    }


@router.get("/learning-case-clusters/{learning_case_cluster_id}/review-actions")
async def list_learning_case_cluster_review_actions(
    learning_case_cluster_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    cluster_uuid = _parse_uuid_param(learning_case_cluster_id, "learning_case_cluster_id")
    cluster_row = (
        db.query(LearningCaseCluster)
        .filter(LearningCaseCluster.learning_case_cluster_id == cluster_uuid)
        .first()
    )
    if not cluster_row:
        raise HTTPException(status_code=404, detail="Learning case cluster not found")

    case_row = (
        db.query(LearningCase)
        .filter(LearningCase.learning_case_cluster_id == cluster_uuid)
        .order_by(LearningCase.created_at.asc())
        .first()
    )
    if not case_row:
        raise HTTPException(status_code=404, detail="Learning case cluster has no cases")

    _ensure_learning_case_access(
        current_account=current_account,
        player_id=case_row.player_id,
        db=db,
    )

    rows = (
        db.query(LearningCaseReviewEvent)
        .filter(LearningCaseReviewEvent.learning_case_cluster_id == cluster_uuid)
        .order_by(LearningCaseReviewEvent.created_at.desc())
        .all()
    )
    return {
        "learning_case_cluster_id": learning_case_cluster_id,
        "items": [_learning_case_review_event_response(row) for row in rows],
    }


@router.post("/learning-cases/{learning_case_id}/review-actions")
async def create_learning_case_review_action(
    learning_case_id: str,
    payload: LearningCaseReviewActionCreate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_coach_reviewer(current_account)
    case_uuid = _parse_uuid_param(learning_case_id, "learning_case_id")
    case_row = (
        db.query(LearningCase)
        .filter(LearningCase.learning_case_id == case_uuid)
        .first()
    )
    if not case_row:
        raise HTTPException(status_code=404, detail="Learning case not found")

    _ensure_learning_case_access(
        current_account=current_account,
        player_id=case_row.player_id,
        db=db,
    )

    if case_row.learning_case_cluster_id is None:
        raise HTTPException(status_code=400, detail="Learning case is not attached to a cluster")

    cluster_row = (
        db.query(LearningCaseCluster)
        .filter(LearningCaseCluster.learning_case_cluster_id == case_row.learning_case_cluster_id)
        .first()
    )
    if not cluster_row:
        raise HTTPException(status_code=404, detail="Learning case cluster not found")

    sibling_case_rows = (
        db.query(LearningCase)
        .filter(LearningCase.learning_case_cluster_id == case_row.learning_case_cluster_id)
        .order_by(LearningCase.created_at.asc())
        .all()
    )

    try:
        event = apply_learning_case_review_action(
            cluster_row=cluster_row,
            case_row=case_row,
            sibling_case_rows=sibling_case_rows,
            action=payload.action,
            account_id=str(current_account.account_id),
            notes=payload.notes,
            metadata={"target_type": "case"},
            db=db,
        )
        db.commit()
        db.refresh(case_row)
        db.refresh(cluster_row)
        db.refresh(event)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise

    return {
        "learning_case_id": str(case_row.learning_case_id),
        "learning_case_cluster_id": str(cluster_row.learning_case_cluster_id),
        "status": case_row.status,
        "cluster_status": cluster_row.status,
        "review_event": _learning_case_review_event_response(event),
    }


@router.get("/learning-cases/{learning_case_id}/review-actions")
async def list_learning_case_review_actions(
    learning_case_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    case_uuid = _parse_uuid_param(learning_case_id, "learning_case_id")
    case_row = (
        db.query(LearningCase)
        .filter(LearningCase.learning_case_id == case_uuid)
        .first()
    )
    if not case_row:
        raise HTTPException(status_code=404, detail="Learning case not found")

    _ensure_learning_case_access(
        current_account=current_account,
        player_id=case_row.player_id,
        db=db,
    )

    rows = (
        db.query(LearningCaseReviewEvent)
        .filter(LearningCaseReviewEvent.learning_case_id == case_uuid)
        .order_by(LearningCaseReviewEvent.created_at.desc())
        .all()
    )
    return {
        "learning_case_id": learning_case_id,
        "items": [_learning_case_review_event_response(row) for row in rows],
    }


@router.post("/knowledge-pack-release-candidates")
async def create_knowledge_pack_release_candidate(
    payload: KnowledgePackReleaseCandidateCreate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_coach_reviewer(current_account)
    cluster_ids = []
    for raw_id in payload.motivating_cluster_ids:
        cluster_ids.append(_parse_uuid_param(raw_id, "motivating_cluster_id"))
    if not cluster_ids:
        raise HTTPException(status_code=400, detail="At least one motivating_cluster_id is required")

    cluster_rows = (
        db.query(LearningCaseCluster)
        .filter(LearningCaseCluster.learning_case_cluster_id.in_(cluster_ids))
        .all()
    )
    if len(cluster_rows) != len(set(cluster_ids)):
        raise HTTPException(status_code=404, detail="One or more motivating learning-case clusters were not found")

    case_rows = (
        db.query(LearningCase)
        .filter(LearningCase.learning_case_cluster_id.in_(cluster_ids))
        .order_by(LearningCase.created_at.asc())
        .all()
    )

    try:
        candidate = create_release_candidate(
            knowledge_pack_id=payload.knowledge_pack_id,
            base_pack_version=payload.base_pack_version,
            candidate_pack_version=payload.candidate_pack_version,
            summary=payload.summary,
            created_by_account_id=str(current_account.account_id),
            cluster_rows=cluster_rows,
            case_rows=case_rows,
            change_summary=payload.change_summary,
            tests_added=payload.tests_added,
            reinterpret_run_ids=payload.reinterpret_run_ids,
            supersedes_pack_version=payload.supersedes_pack_version,
            db=db,
        )
        db.commit()
        db.refresh(candidate)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise

    return _knowledge_pack_release_candidate_response(candidate)


@router.post("/knowledge-pack-release-candidates/{release_candidate_id}/actions")
async def create_knowledge_pack_release_action(
    release_candidate_id: str,
    payload: KnowledgePackReleaseActionCreate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_coach_reviewer(current_account)
    release_candidate_uuid = _parse_uuid_param(release_candidate_id, "release_candidate_id")
    candidate = (
        db.query(KnowledgePackReleaseCandidate)
        .filter(KnowledgePackReleaseCandidate.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .first()
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Knowledge-pack release candidate not found")

    try:
        event = apply_release_action(
            candidate_row=candidate,
            action=payload.action,
            account_id=str(current_account.account_id),
            notes=payload.notes,
            metadata=payload.metadata,
            db=db,
        )
        db.commit()
        db.refresh(candidate)
        db.refresh(event)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise

    return {
        "release_candidate": _knowledge_pack_release_candidate_response(candidate),
        "release_event": _knowledge_pack_release_event_response(event),
    }


@router.post("/knowledge-pack-release-candidates/{release_candidate_id}/run-regression-suite")
async def run_knowledge_pack_release_regression_suite(
    release_candidate_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    _require_coach_reviewer(current_account)
    release_candidate_uuid = _parse_uuid_param(release_candidate_id, "release_candidate_id")
    candidate = (
        db.query(KnowledgePackReleaseCandidate)
        .filter(KnowledgePackReleaseCandidate.knowledge_pack_release_candidate_id == release_candidate_uuid)
        .first()
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Knowledge-pack release candidate not found")

    try:
        regression_run = run_release_candidate_regression(
            candidate_row=candidate,
            account_id=str(current_account.account_id),
            db=db,
        )
        db.commit()
        db.refresh(candidate)
        db.refresh(regression_run)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise

    return {
        "release_candidate": _knowledge_pack_release_candidate_response(candidate),
        "regression_run": _knowledge_pack_regression_run_response(regression_run),
    }


def _minimal_result_for_run(analysis_run: AnalysisRun) -> dict:
    return {
        "run_id": str(analysis_run.run_id),
        "input": {
            "player_id": str(analysis_run.player_id),
            "hand": analysis_run.handedness,
            "age_group": analysis_run.age_group,
            "season": analysis_run.season,
        },
        "video": {
            "fps": analysis_run.fps,
            "total_frames": analysis_run.total_frames,
        },
    }


def _require_coach_reviewer(current_account) -> None:
    if str(getattr(current_account, "role", "")).lower() != "coach":
        raise HTTPException(status_code=403, detail="Learning-case review is only available to coach accounts")


def _ensure_learning_case_access(*, current_account, player_id: uuid.UUID, db: Session) -> None:
    link = (
        db.query(AccountPlayerLink)
        .filter(
            AccountPlayerLink.account_id == current_account.account_id,
            AccountPlayerLink.player_id == player_id,
        )
        .first()
    )
    if not link:
        raise HTTPException(status_code=403, detail="Access denied")


def _parse_uuid_param(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name} format")


def _coach_flag_response(row: CoachFlag, *, deduped: bool) -> dict:
    return {
        "coach_flag_id": str(row.coach_flag_id),
        "run_id": str(row.run_id),
        "player_id": str(row.player_id),
        "flag_type": row.flag_type,
        "notes": row.notes or "",
        "flagged_mechanism_id": row.flagged_mechanism_id,
        "flagged_prescription_id": row.flagged_prescription_id,
        "learning_case_id": str(row.learning_case_id) if row.learning_case_id else None,
        "learning_case_cluster_id": (
            str(row.learning_case_cluster_id)
            if row.learning_case_cluster_id
            else None
        ),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "deduped": deduped,
    }


def _learning_case_review_event_response(row: LearningCaseReviewEvent) -> dict:
    return {
        "learning_case_review_event_id": str(row.learning_case_review_event_id),
        "learning_case_cluster_id": str(row.learning_case_cluster_id),
        "learning_case_id": str(row.learning_case_id) if row.learning_case_id else None,
        "account_id": str(row.account_id) if row.account_id else None,
        "action": row.action,
        "from_status": row.from_status,
        "to_status": row.to_status,
        "notes": row.notes or "",
        "metadata": dict(row.metadata_json or {}),
        "created_at": row.created_at,
    }


def _knowledge_pack_release_candidate_response(row: KnowledgePackReleaseCandidate) -> dict:
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


def _knowledge_pack_release_event_response(row: KnowledgePackReleaseEvent) -> dict:
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


def _knowledge_pack_regression_run_response(row: KnowledgePackRegressionRun) -> dict:
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
