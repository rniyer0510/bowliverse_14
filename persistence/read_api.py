"""
ActionLab persistence READ + WRITE APIs (Phase-I)

Supports:
- Player creation with age_group + season
- Player profile updates (age_group / season)
- Player listing
- Analysis history
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.persistence.session import SessionLocal
from app.persistence.models import (
    Account,
    Player,
    AccountPlayerLink,
    AnalysisRun,
    AnalysisResultRaw,
)

router = APIRouter()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_or_create_current_account(db) -> Account:
    account = (
        db.query(Account)
        .order_by(Account.created_at.desc())
        .first()
    )
    if account:
        return account

    account = Account(role="coach", name="Default")
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


# ---------------------------------------------------------------------
# Models (Pydantic v2 compatible)
# ---------------------------------------------------------------------

class PlayerCreateRequest(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    handedness: str = Field(..., pattern="^(R|L)$")
    age_group: str = Field(..., pattern="^(U10|U14|U16|U19|ADULT)$")
    season: int = Field(..., ge=2000, le=2100)


class PlayerProfileUpdate(BaseModel):
    age_group: Optional[str] = Field(None, pattern="^(U10|U14|U16|U19|ADULT)$")
    season: Optional[int] = Field(None, ge=2000, le=2100)


# ---------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------

@router.get("/players")
def list_players():
    db = SessionLocal()
    try:
        account = _get_or_create_current_account(db)

        links = (
            db.query(AccountPlayerLink)
            .filter_by(account_id=account.account_id)
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
    finally:
        db.close()


@router.post("/players", status_code=201)
def create_player(payload: PlayerCreateRequest):
    db = SessionLocal()
    try:
        account = _get_or_create_current_account(db)

        existing = (
            db.query(AccountPlayerLink)
            .filter_by(
                account_id=account.account_id,
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
            primary_owner_account_id=account.account_id,
            created_by_account_id=account.account_id,
            handedness=payload.handedness,
            age_group=payload.age_group,
            season=payload.season,
        )
        db.add(player)
        db.flush()

        db.add(AccountPlayerLink(
            account_id=account.account_id,
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

    finally:
        db.close()


@router.patch("/players/{player_id}/profile")
def update_player_profile(player_id: str, payload: PlayerProfileUpdate):
    db = SessionLocal()
    try:
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

    finally:
        db.close()


@router.get("/players/{player_id}/latest")
def latest_analysis(player_id: str):
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
