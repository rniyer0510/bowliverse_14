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


def _current_season() -> int:
    return datetime.utcnow().year


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
        default=_current_season,
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
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    deterministic_diagnosis_status: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    deterministic_primary_mechanism_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    deterministic_archetype_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )

    handedness: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    # -------------------------------------------------
    # Snapshot fields (Option A – frozen per run)
    # -------------------------------------------------
    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    age_group: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
    )

    # Coach notes (editable later)
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


class LearningCase(Base):
    __tablename__ = "learning_case"

    learning_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("player.player_id"),
        nullable=False,
    )

    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )

    event_name: Mapped[str] = mapped_column(String, nullable=False)
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    case_type: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="OPEN")
    suggested_gap_type: Mapped[str] = mapped_column(String, nullable=False)
    trigger_reason: Mapped[str] = mapped_column(Text, nullable=False)
    symptom_bundle_hash: Mapped[str] = mapped_column(String, nullable=False)
    chosen_mechanism_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    renderer_mode: Mapped[str] = mapped_column(String, nullable=False)
    prescription_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    followup_outcome: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="NOT_YET_DUE",
    )
    clip_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    detected_symptoms: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    candidate_mechanisms: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    confidence_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    contradictions_triggered: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    event_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "case_type IN ('NO_MATCH','AMBIGUOUS_MATCH','LOW_CONFIDENCE')",
            name="ck_learning_case_type",
        ),
        CheckConstraint(
            "priority IN ('A','B','C','D','E')",
            name="ck_learning_case_priority",
        ),
        CheckConstraint(
            "status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')",
            name="ck_learning_case_status",
        ),
        CheckConstraint(
            "followup_outcome IN ('NOT_YET_DUE','IMPROVING','NO_CLEAR_CHANGE','WORSENING','INSUFFICIENT_DATA')",
            name="ck_learning_case_followup_outcome",
        ),
    )


# ------------------------------------------------------------
# Phase 1 Auth - User Model
# ------------------------------------------------------------

class User(Base):
    __tablename__ = "user"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )


class LoginAudit(Base):
    __tablename__ = "login_audit"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user.user_id"),
        nullable=True,
    )

    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )

    username: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    login_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    ip_address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
