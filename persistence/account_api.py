from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.persistence.session import SessionLocal
from app.persistence.models import Account

router = APIRouter()

class AccountContext(BaseModel):
    role: str

@router.post("/account/context")
def set_account_context(payload: AccountContext):
    db = SessionLocal()
    try:
        account = (
            db.query(Account)
            .order_by(Account.created_at.desc())
            .first()
        )
        if not account:
            raise HTTPException(status_code=404, detail="No account found")

        role = payload.role.lower()
        if role not in ("player", "coach", "parent"):
            raise HTTPException(status_code=422, detail="Invalid role")

        account.role = role
        db.commit()

        return {"status": "ok", "role": role}
    finally:
        db.close()

