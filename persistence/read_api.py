"""
Read-only APIs for ActionLab persistence (Phase-I)

SAFE:
- No writes
- No schema changes
- No orchestrator coupling
"""

from fastapi import APIRouter, HTTPException
from typing import List

from app.persistence.session import SessionLocal
from app.persistence.models import (
    Account,
    Player,
    AccountPlayerLink,
    AnalysisRun,
    AnalysisResultRaw,
)

router = APIRouter()


def _get_current_account(db):
    """
    Phase-I assumption:
    Single-device → use most recently created account
    """
    account = (
        db.query(Account)
        .order_by(Account.created_at.desc())
        .first()
    )
    if not account:
        raise HTTPException(status_code=404, detail="No account found")
    return account


@router.get("/players")
def list_players():
    """
    List players linked to current account
    """
    db = SessionLocal()
    try:
        account = _get_current_account(db)

        links = (
            db.query(AccountPlayerLink)
            .filter_by(account_id=account.account_id)
            .all()
        )

        players = []
        for link in links:
            players.append({
                "player_id": str(link.player_id),
                "player_name": link.player_name,
                "link_type": link.link_type,
            })

        return {
            "account": {
                "role": account.role,
                "name": account.name,
            },
            "players": players,
        }

    finally:
        db.close()


@router.get("/players/{player_id}/latest")
def latest_analysis(player_id: str):
    """
    Latest analysis for a player
    """
    db = SessionLocal()
    try:
        run = (
            db.query(AnalysisRun)
            .filter_by(player_id=player_id)
            .order_by(AnalysisRun.created_at.desc())
            .first()
        )
        if not run:
            raise HTTPException(status_code=404, detail="No analysis found")

        raw = (
            db.query(AnalysisResultRaw)
            .filter_by(run_id=run.run_id)
            .first()
        )

        return {
            "run_id": str(run.run_id),
            "created_at": run.created_at,
            "result": raw.result_json if raw else None,
        }

    finally:
        db.close()


@router.get("/players/{player_id}/history")
def analysis_history(player_id: str, limit: int = 20):
    """
    Analysis history for a player
    """
    db = SessionLocal()
    try:
        runs = (
            db.query(AnalysisRun)
            .filter_by(player_id=player_id)
            .order_by(AnalysisRun.created_at.desc())
            .limit(limit)
            .all()
        )

        history = []
        for run in runs:
            history.append({
                "run_id": str(run.run_id),
                "created_at": run.created_at,
                "fps": run.fps,
                "total_frames": run.total_frames,
            })

        return {
            "player_id": player_id,
            "count": len(history),
            "history": history,
        }

    finally:
        db.close()
