"""
ActionLab persistence APIs (Phase-I)

Background
- The deployed backend currently exposes only GET routes for players, which
  causes the mobile app's "Add Player" to fail with:
    POST /players -> 405 Method Not Allowed

Fix (minimal)
- Add a small POST /players endpoint to create a Player and link it to the
  "current" account.

Phase-I assumptions retained
- Single-device usage: the "current" account is the most recently created.
- If no account exists yet (fresh DB), we auto-create a default account.
- No coupling to orchestrator; only persistence tables are touched.
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
    """
    Return the most recently created account.

    Phase-I UX improvement:
    - On fresh installs, allow APIs to work without a manual bootstrap step.
    """
    account = (
        db.query(Account)
        .order_by(Account.created_at.desc())
        .first()
    )
    if account:
        return account

    # Fresh DB: create a sensible default "coach" account so multiple players
    # can be linked and managed from the device.
    account = Account(role="coach", name="Default")
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


# ---------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------

@router.get("/players")
def list_players():
    """List players linked to the current account."""
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


class PlayerCreateRequest(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    handedness: str = Field(..., min_length=1, max_length=2)
    bowling_type: Optional[str] = None  # accepted for forward-compat (not stored Phase-I)


@router.post("/players", status_code=201)
@router.post("/players/", status_code=201)
def create_player(payload: PlayerCreateRequest):
    """
    Create a new player and link it to the current account.

    Behavior:
    - If a player with the same name is already linked to the current account,
      return the existing link (idempotent-ish UX).
    """
    db = SessionLocal()
    try:
        account = _get_or_create_current_account(db)

        name = payload.player_name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="player_name is required")

        hand = payload.handedness.strip().upper()
        if hand not in ("R", "L"):
            raise HTTPException(status_code=422, detail="handedness must be 'R' or 'L'")

        # Reuse existing linked player by name (prevents duplicates from UI retries)
        existing = (
            db.query(AccountPlayerLink)
            .filter_by(account_id=account.account_id, player_name=name)
            .first()
        )
        if existing:
            return {
                "player_id": str(existing.player_id),
                "player_name": existing.player_name,
                "link_type": existing.link_type,
                "already_exists": True,
            }

        player = Player(
            primary_owner_account_id=account.account_id,
            created_by_account_id=account.account_id,
            handedness=hand,
        )
        db.add(player)
        db.flush()  # ensures player.player_id is available

        link = AccountPlayerLink(
            account_id=account.account_id,
            player_id=player.player_id,
            link_type="owner",
            player_name=name,
        )
        db.add(link)

        db.commit()

        return {
            "player_id": str(player.player_id),
            "player_name": name,
            "link_type": link.link_type,
        }

    finally:
        db.close()


# ---------------------------------------------------------------------
# Analysis (READ ONLY)
# ---------------------------------------------------------------------

@router.get("/players/{player_id}/latest")
def latest_analysis(player_id: str):
    """Latest analysis for a player."""
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
    """Analysis history for a player."""
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

