from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    Text,
    Date,
    TIMESTAMP,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# =====================================================
# Base
# =====================================================

class Base(DeclarativeBase):
    pass


# =====================================================
# Identity
# =====================================================

class Account(Base):
    __tablename__ = "account"

    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    role: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('player','coach','parent')",
            name="ck_account_role",
        ),
    )


class Player(Base):
    __tablename__ = "player"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    primary_owner_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )

    created_by_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )

    handedness: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    date_of_birth: Mapped[Optional[Date]] = mapped_column(Date, nullable=True)

    # -------------------------------------------------
    # Age grouping (coach-controlled, season-based)
    # -------------------------------------------------
    age_group: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="SENIOR",
    )

    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=2025,
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "age_group IN ('U10','U14','U16','U19','SENIOR')",
            name="ck_player_age_group",
        ),
    )


class AccountPlayerLink(Base):
    __tablename__ = "account_player_link"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("player.player_id"),
        nullable=False,
    )

    link_type: Mapped[str] = mapped_column(String, nullable=False)

    # Relationship-level label
    player_name: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "link_type IN ('owner','self','coach','parent','child','trainee')",
            name="ck_account_player_link_type",
        ),
    )


# =====================================================
# Analysis Core (facts only)
# =====================================================

class AnalysisRun(Base):
    __tablename__ = "analysis_run"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("player.player_id"),
        nullable=False,
    )

    schema_version: Mapped[str] = mapped_column(String, nullable=False)
    device_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    handedness: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    # ← NEW FIELD: Coach's Notes
    coach_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )


class EventAnchor(Base):
    __tablename__ = "event_anchor"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(String, nullable=False)
    frame: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    method: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('BFC','FFC','UAH','RELEASE')",
            name="ck_event_anchor_type",
        ),
    )


class BiomechSignal(Base):
    __tablename__ = "biomech_signal"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )

    signal_key: Mapped[str] = mapped_column(String, nullable=False)
    event_anchor: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    value: Mapped[float] = mapped_column(Float, nullable=False)
    units: Mapped[str] = mapped_column(String, nullable=False)

    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_flag: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    baseline_eligible: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )


class RiskMeasurement(Base):
    __tablename__ = "risk_measurement"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )

    risk_id: Mapped[str] = mapped_column(String, nullable=False)
    signal_strength: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anchor_event: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    window_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    window_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class VisualEvidence(Base):
    __tablename__ = "visual_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )

    risk_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    frame: Mapped[int] = mapped_column(Integer, nullable=False)
    anchor: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    visual_confidence: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)


class AnalysisResultRaw(Base):
    __tablename__ = "analysis_result_raw"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        primary_key=True,
    )

    result_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
