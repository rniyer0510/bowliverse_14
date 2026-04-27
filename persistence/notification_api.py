from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.common.auth import get_current_account
from app.persistence.models import DeviceRegistration, NotificationEvent
from app.persistence.notifications import register_device, unregister_device
from app.persistence.session import get_db

router = APIRouter()


class DeviceRegisterPayload(BaseModel):
    platform: str
    push_provider: str
    push_token: str
    device_label: str | None = None
    app_version: str | None = None
    locale: str | None = None
    timezone: str | None = None


class DeviceUnregisterPayload(BaseModel):
    push_token: str


@router.post("/devices/register")
def register_device_endpoint(
    payload: DeviceRegisterPayload,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    try:
        row = register_device(
            account_id=str(current_account.account_id),
            platform=payload.platform,
            push_provider=payload.push_provider,
            push_token=payload.push_token,
            device_label=payload.device_label,
            app_version=payload.app_version,
            locale=payload.locale,
            timezone=payload.timezone,
            db=db,
        )
        db.commit()
        db.refresh(row)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise
    return _device_registration_response(row)


@router.post("/devices/unregister")
def unregister_device_endpoint(
    payload: DeviceUnregisterPayload,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    try:
        row = unregister_device(
            account_id=str(current_account.account_id),
            push_token=payload.push_token,
            db=db,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Device registration not found")
        db.commit()
        db.refresh(row)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    return _device_registration_response(row)


@router.get("/devices")
def list_registered_devices(
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(DeviceRegistration)
        .filter(DeviceRegistration.account_id == current_account.account_id)
        .order_by(DeviceRegistration.updated_at.desc())
        .all()
    )
    return {"items": [_device_registration_response(row) for row in rows]}


@router.get("/notifications")
def list_notification_events(
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(NotificationEvent)
        .filter(NotificationEvent.account_id == current_account.account_id)
        .order_by(NotificationEvent.created_at.desc())
        .limit(50)
        .all()
    )
    return {"items": [_notification_event_response(row) for row in rows]}


def _device_registration_response(row: DeviceRegistration) -> dict:
    return {
        "device_registration_id": str(row.device_registration_id),
        "account_id": str(row.account_id),
        "platform": row.platform,
        "push_provider": row.push_provider,
        "push_token": row.push_token,
        "device_label": row.device_label,
        "app_version": row.app_version,
        "locale": row.locale,
        "timezone": row.timezone,
        "is_active": bool(row.is_active),
        "last_seen_at": row.last_seen_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _notification_event_response(row: NotificationEvent) -> dict:
    return {
        "notification_event_id": str(row.notification_event_id),
        "account_id": str(row.account_id),
        "event_type": row.event_type,
        "status": row.status,
        "active_device_count": int(row.active_device_count or 0),
        "payload": dict(row.payload_json or {}),
        "error_detail": row.error_detail,
        "sent_at": row.sent_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
