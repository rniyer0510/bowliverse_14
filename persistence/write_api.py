"""
Write APIs for ActionLab persistence (Phase-I)

Scope:
- Player creation
- Coach / Parent / Self supported
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.persistence.session import SessionLocal
from app.persistence.models import Account, Player, AccountPlayerLink

# ------------------------------------------------------------
# Router (THIS WAS MISSING)
# ------------------------------------------------------------
router = APIRouter()


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
class PlayerCreate(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    handedness: str = Field(..., min_length=1, max_length=2)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_or_create_account(db):
    account = (
        db.query(Account)
        .order_by(Account.created_at.desc())
        .first()
    )
    if account:
        return account

    # Phase-I default account
    account = Account(role="coach", name="Default")
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@router.post("/players", status_code=201)
@router.post("/players/", status_code=201)
def create_player(payload: PlayerCreate):
    db = SessionLocal()
    try:
        account = _get_or_create_account(db)

        name = payload.player_name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="player_name required")

        hand = payload.handedness.upper()
        if hand not in ("R", "L"):
            raise HTTPException(status_code=422, detail="handedness must be R or L")

        # Link type based on account role
        role = (account.role or "").lower()
        if role == "coach":
            link_type = "coach"
        elif role == "parent":
            link_type = "parent"
        else:
            link_type = "self"

        # Prevent duplicate player names per account
        existing = (
            db.query(AccountPlayerLink)
            .filter_by(
                account_id=account.account_id,
                player_name=name,
            )
            .first()
        )
        if existing:
            return {
                "player_id": str(existing.player_id),
                "player_name": existing.player_name,
                "link_type": existing.link_type,
                "already_exists": True,
            }

        # Create player
        player = Player(
            primary_owner_account_id=account.account_id,
            created_by_account_id=account.account_id,
            handedness=hand,
        )
        db.add(player)
        db.flush()  # ensures player_id is available

        link = AccountPlayerLink(
            account_id=account.account_id,
            player_id=player.player_id,
            player_name=name,
            link_type=link_type,
        )
        db.add(link)

        db.commit()

        return {
            "player_id": str(player.player_id),
            "player_name": name,
            "link_type": link_type,
        }

    finally:
        db.close()

