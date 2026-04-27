from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.common.logger import get_logger
from app.persistence.models import AccountPlayerLink, DeviceRegistration, NotificationEvent
from app.persistence.session import SessionLocal

logger = get_logger(__name__)


def register_device(
    *,
    account_id: str,
    platform: str,
    push_provider: str,
    push_token: str,
    device_label: Optional[str],
    app_version: Optional[str],
    locale: Optional[str],
    timezone: Optional[str],
    db: Session,
) -> DeviceRegistration:
    account_uuid = uuid.UUID(str(account_id))
    token = _required_text(push_token, "push_token")
    now = datetime.utcnow()
    row = (
        db.query(DeviceRegistration)
        .filter(DeviceRegistration.push_token == token)
        .first()
    )
    if row is None:
        row = DeviceRegistration(
            device_registration_id=uuid.uuid4(),
            account_id=account_uuid,
            platform=_normalize_platform(platform),
            push_provider=_normalize_provider(push_provider),
            push_token=token,
            device_label=_clean_text(device_label),
            app_version=_clean_text(app_version),
            locale=_clean_text(locale),
            timezone=_clean_text(timezone),
            is_active=True,
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    else:
        row.account_id = account_uuid
        row.platform = _normalize_platform(platform)
        row.push_provider = _normalize_provider(push_provider)
        row.device_label = _clean_text(device_label)
        row.app_version = _clean_text(app_version)
        row.locale = _clean_text(locale)
        row.timezone = _clean_text(timezone)
        row.is_active = True
        row.last_seen_at = now
        row.updated_at = now
    db.flush()
    return row


def unregister_device(
    *,
    account_id: str,
    push_token: str,
    db: Session,
) -> Optional[DeviceRegistration]:
    token = _required_text(push_token, "push_token")
    row = (
        db.query(DeviceRegistration)
        .filter(
            DeviceRegistration.account_id == uuid.UUID(str(account_id)),
            DeviceRegistration.push_token == token,
        )
        .first()
    )
    if row is None:
        return None
    row.is_active = False
    row.updated_at = datetime.utcnow()
    db.flush()
    return row


def create_analysis_completed_notification(
    *,
    account_id: str,
    result: Dict[str, Any],
    db: Session,
) -> NotificationEvent:
    payload = build_analysis_completed_payload(
        account_id=account_id,
        result=result,
        db=db,
    )
    active_device_count = (
        db.query(DeviceRegistration)
        .filter(
            DeviceRegistration.account_id == uuid.UUID(str(account_id)),
            DeviceRegistration.is_active.is_(True),
        )
        .count()
    )
    now = datetime.utcnow()
    event = NotificationEvent(
        notification_event_id=uuid.uuid4(),
        account_id=uuid.UUID(str(account_id)),
        event_type="analysis_completed",
        status="PENDING" if active_device_count > 0 else "SKIPPED_NO_DEVICE",
        active_device_count=active_device_count,
        payload_json=payload,
        error_detail=None,
        sent_at=None,
        created_at=now,
        updated_at=now,
    )
    db.add(event)
    db.flush()
    return event


def persist_analysis_completed_notification_best_effort(
    *,
    account_id: str,
    result: Dict[str, Any],
) -> None:
    db = SessionLocal()
    try:
        create_analysis_completed_notification(
            account_id=account_id,
            result=result,
            db=db,
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(
            "[notification] account_id=%s event_type=analysis_completed persisted=false error=%s",
            account_id,
            exc,
        )
    finally:
        db.close()


def build_analysis_completed_payload(
    *,
    account_id: str,
    result: Dict[str, Any],
    db: Session,
) -> Dict[str, Any]:
    run_id = _clean_text(result.get("run_id"))
    input_cfg = result.get("input") or {}
    player_id = _clean_text(input_cfg.get("player_id"))
    player_name = None
    if player_id:
        link = (
            db.query(AccountPlayerLink)
            .filter(
                AccountPlayerLink.account_id == uuid.UUID(str(account_id)),
                AccountPlayerLink.player_id == uuid.UUID(player_id),
            )
            .first()
        )
        player_name = _clean_text(getattr(link, "player_name", None))

    return {
        "type": "analysis_completed",
        "job_id": None,
        "run_id": run_id,
        "player_id": player_id,
        "player_name": player_name,
    }


def _normalize_platform(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ios", "android", "web"}:
        return normalized
    return "unknown"


def _normalize_provider(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"fcm", "apns"}:
        return normalized
    return "unknown"


def _required_text(value: Any, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _clean_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None
