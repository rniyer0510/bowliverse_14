"""
Explicit WRITE APIs (Phase-I + Coach Notes)

NOTE:
Player creation & updates are handled via read_api.py
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import uuid

from app.persistence.session import get_db
from app.persistence.models import AnalysisRun
from app.common.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class CoachNotesUpdate(BaseModel):
    """Request model for updating coach notes - notes can be empty string"""
    coach_notes: str  # Can be empty string to clear notes


@router.patch("/analysis/{run_id}/coach-notes")
async def update_coach_notes(
    run_id: str,
    update: CoachNotesUpdate,
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
    
    # Update coach notes (can be empty string)
    analysis_run.coach_notes = update.coach_notes if update.coach_notes else None
    
    try:
        db.commit()
        db.refresh(analysis_run)
        logger.info(f"Coach notes updated for run_id={run_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update coach notes: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return {
        "run_id": str(analysis_run.run_id),
        "coach_notes": analysis_run.coach_notes or "",
        "updated_at": analysis_run.created_at.isoformat()
    }


@router.get("/analysis/{run_id}/coach-notes")
async def get_coach_notes(
    run_id: str,
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
    
    return {
        "run_id": str(analysis_run.run_id),
        "coach_notes": analysis_run.coach_notes or ""
    }
