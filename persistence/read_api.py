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

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

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

    players = []
    for link in links:
        player = db.query(Player).filter_by(player_id=link.player_id).first()

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

    return {
        "items": [
            {
                "run_id": str(r.run_id),
                "created_at": r.created_at,
                "season": r.season,
                "age_group": r.age_group,
                "schema_version": r.schema_version,
                "fps": r.fps,
                "total_frames": r.total_frames,
            }
            for r in runs
        ]
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
        "result": raw.result_json if raw else None,
    }

