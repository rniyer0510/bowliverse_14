from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from sqlalchemy.orm import Session

from app.persistence.session import get_db
from app.persistence.models import Account, User
from app.common.auth import get_current_account, get_current_user

router = APIRouter()

class AccountContext(BaseModel):
    role: str

@router.post("/account/context")
def set_account_context(
    payload: AccountContext,
    current_account: Account = Depends(get_current_account),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    role = payload.role.lower()
    if role not in ("player", "coach", "parent", "reviewer", "clinician"):
        raise HTTPException(status_code=422, detail="Invalid role")

    # Lock role switching at runtime to prevent self-privilege drift.
    if role != current_account.role:
        raise HTTPException(status_code=403, detail="Role switching is not allowed")

    return {
        "status": "ok",
        "account_id": str(current_account.account_id),
        "role": current_account.role,
    }
