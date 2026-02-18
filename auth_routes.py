from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from app.persistence.session import get_db
from app.persistence.models import User, Account, Player, AccountPlayerLink
from app.common.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_account,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_current_season():
    return datetime.utcnow().year


def validate_season(season: int):
    current = get_current_season()
    if season < current - 1 or season > current + 1:
        raise HTTPException(
            status_code=400,
            detail="Season must be within ±1 of current year",
        )
    return season


def normalize_handedness(value: str):
    value = value.strip().upper()
    if value in ["RIGHT", "R"]:
        return "R"
    if value in ["LEFT", "L"]:
        return "L"
    raise HTTPException(status_code=400, detail="Invalid handedness")


# ------------------------------------------------------------
# Register
# ------------------------------------------------------------
@router.post("/register")
def register(data: dict, db: Session = Depends(get_db)):

    username = data.get("username", "").strip().lower()
    password = data.get("password")
    role = data.get("role", "").strip().lower()
    full_name = data.get("full_name", "").strip()

    if role not in ["player", "coach", "parent"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    if not username or not password or not full_name:
        raise HTTPException(status_code=400, detail="Missing required fields")

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # ------------------------------------------------------------
    # Create Account
    # ------------------------------------------------------------
    account = Account(role=role, name=full_name)
    db.add(account)
    db.flush()

    # ------------------------------------------------------------
    # Create User
    # ------------------------------------------------------------
    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        account_id=account.account_id,
    )
    db.add(user)

    # ------------------------------------------------------------
    # If Player → Create Player + Link
    # ------------------------------------------------------------
    if role == "player":

        handedness = data.get("handedness")
        age_group = data.get("age_group")
        season = data.get("season")

        if not handedness or not age_group:
            raise HTTPException(
                status_code=400,
                detail="Missing player fields",
            )

        handedness = normalize_handedness(handedness)

        if season:
            season = validate_season(int(season))
        else:
            season = get_current_season()

        player = Player(
            primary_owner_account_id=account.account_id,
            created_by_account_id=account.account_id,
            handedness=handedness,
            age_group=age_group,
            season=season,
        )

        db.add(player)
        db.flush()

        link = AccountPlayerLink(
            account_id=account.account_id,
            player_id=player.player_id,
            link_type="self",
            player_name=full_name,
        )

        db.add(link)

    db.commit()

    token = create_access_token(user)

    return {"access_token": token}


# ------------------------------------------------------------
# Login
# ------------------------------------------------------------
@router.post("/login")
def login(data: dict, db: Session = Depends(get_db)):

    username = data.get("username", "").strip().lower()
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing credentials")

    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user)

    return {"access_token": token}


# ------------------------------------------------------------
# Get Current Authenticated Account
# ------------------------------------------------------------
@router.get("/me")
def get_me(current_account=Depends(get_current_account)):
    """
    Returns identity of the currently authenticated account.
    Used by frontend to validate session.
    """

    return {
        "account_id": str(current_account.account_id),
        "role": current_account.role,
        "name": current_account.name,
    }

