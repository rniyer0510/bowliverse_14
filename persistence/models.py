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


class AnalysisExplanationTrace(Base):
    __tablename__ = "analysis_explanation_trace"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        primary_key=True,
    )
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    diagnosis_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    primary_mechanism_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    matched_symptom_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    candidate_mechanisms: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    supporting_evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    contradictions_triggered: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    selected_trajectory_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    selected_prescription_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    selected_render_story_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    selected_history_binding_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    explanation_trace_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )


class LearningCaseCluster(Base):
    __tablename__ = "learning_case_cluster"

    learning_case_cluster_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    cluster_key_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
    )
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="runtime_gap")
    case_type: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="OPEN")
    suggested_gap_type: Mapped[str] = mapped_column(String, nullable=False)
    trigger_reason: Mapped[str] = mapped_column(Text, nullable=False)
    symptom_bundle_hash: Mapped[str] = mapped_column(String, nullable=False)
    renderer_mode: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    chosen_mechanism_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prescription_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    candidate_mechanism_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    cluster_payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    coach_flag_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=True,
    )
    latest_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=True,
    )
    representative_learning_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case.learning_case_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('runtime_gap','coach_feedback','prescription_followup')",
            name="ck_learning_case_cluster_source_type",
        ),
        CheckConstraint(
            "case_type IN ('NO_MATCH','AMBIGUOUS_MATCH','LOW_CONFIDENCE','COACH_FEEDBACK','PRESCRIPTION_NON_RESPONSE')",
            name="ck_learning_case_cluster_type",
        ),
        CheckConstraint(
            "priority IN ('A','B','C','D','E')",
            name="ck_learning_case_cluster_priority",
        ),
        CheckConstraint(
            "status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')",
            name="ck_learning_case_cluster_status",
        ),
    )


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
    learning_case_cluster_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case_cluster.learning_case_cluster_id"),
        nullable=True,
    )

    event_name: Mapped[str] = mapped_column(String, nullable=False)
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="runtime_gap")
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
            "source_type IN ('runtime_gap','coach_feedback','prescription_followup')",
            name="ck_learning_case_source_type",
        ),
        CheckConstraint(
            "case_type IN ('NO_MATCH','AMBIGUOUS_MATCH','LOW_CONFIDENCE','COACH_FEEDBACK','PRESCRIPTION_NON_RESPONSE')",
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


class LearningCaseReviewEvent(Base):
    __tablename__ = "learning_case_review_event"

    learning_case_review_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    learning_case_cluster_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case_cluster.learning_case_cluster_id"),
        nullable=False,
    )
    learning_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case.learning_case_id"),
        nullable=True,
    )
    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    from_status: Mapped[str] = mapped_column(String, nullable=False)
    to_status: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "action IN ('triage','queue','start_review','resolve','reject','reopen','supersede','expire')",
            name="ck_learning_case_review_event_action",
        ),
        CheckConstraint(
            "from_status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')",
            name="ck_learning_case_review_event_from_status",
        ),
        CheckConstraint(
            "to_status IN ('OPEN','CLUSTERED','QUEUED','UNDER_REVIEW','RESOLVED','SUPERSEDED','EXPIRED','REJECTED')",
            name="ck_learning_case_review_event_to_status",
        ),
    )


class KnowledgePackReleaseCandidate(Base):
    __tablename__ = "knowledge_pack_release_candidate"

    knowledge_pack_release_candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    knowledge_pack_id: Mapped[str] = mapped_column(String, nullable=False)
    base_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    candidate_pack_version: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    supersedes_pack_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="DRAFT")
    current_environment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    change_summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    motivating_cluster_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    motivating_case_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    tests_added: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    reinterpret_run_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    schema_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    referential_integrity_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    regression_suite_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    staging_evaluation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approval_granted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )
    updated_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )
    promoted_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('DRAFT','IN_DEV','IN_STAGING','APPROVED','PROMOTED','REJECTED','ROLLED_BACK')",
            name="ck_knowledge_pack_release_candidate_status",
        ),
        CheckConstraint(
            "current_environment IS NULL OR current_environment IN ('dev','staging','production')",
            name="ck_knowledge_pack_release_candidate_environment",
        ),
    )


class KnowledgePackReleaseEvent(Base):
    __tablename__ = "knowledge_pack_release_event"

    knowledge_pack_release_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    knowledge_pack_release_candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_pack_release_candidate.knowledge_pack_release_candidate_id"),
        nullable=False,
    )
    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    from_status: Mapped[str] = mapped_column(String, nullable=False)
    to_status: Mapped[str] = mapped_column(String, nullable=False)
    from_environment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    to_environment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "action IN ('create','record_schema_validation','record_referential_integrity','record_regression_pass','promote_dev','promote_staging','record_staging_evaluation','approve_production','promote_production','reject','rollback')",
            name="ck_knowledge_pack_release_event_action",
        ),
        CheckConstraint(
            "from_status IN ('NONE','DRAFT','IN_DEV','IN_STAGING','APPROVED','PROMOTED','REJECTED','ROLLED_BACK')",
            name="ck_knowledge_pack_release_event_from_status",
        ),
        CheckConstraint(
            "to_status IN ('DRAFT','IN_DEV','IN_STAGING','APPROVED','PROMOTED','REJECTED','ROLLED_BACK')",
            name="ck_knowledge_pack_release_event_to_status",
        ),
        CheckConstraint(
            "from_environment IS NULL OR from_environment IN ('dev','staging','production')",
            name="ck_knowledge_pack_release_event_from_environment",
        ),
        CheckConstraint(
            "to_environment IS NULL OR to_environment IN ('dev','staging','production')",
            name="ck_knowledge_pack_release_event_to_environment",
        ),
    )


class KnowledgePackRegressionRun(Base):
    __tablename__ = "knowledge_pack_regression_run"

    knowledge_pack_regression_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    knowledge_pack_release_candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_pack_release_candidate.knowledge_pack_release_candidate_id"),
        nullable=False,
    )
    baseline_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    candidate_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="COMPLETED")
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expected_change_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stable_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    passed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validated_regression_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validated_regression_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    expected_change_success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expected_change_success_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('COMPLETED','FAILED')",
            name="ck_knowledge_pack_regression_run_status",
        ),
    )


class KnowledgePackRegressionCaseResult(Base):
    __tablename__ = "knowledge_pack_regression_case_result"

    knowledge_pack_regression_case_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    knowledge_pack_regression_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_pack_regression_run.knowledge_pack_regression_run_id"),
        nullable=False,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )
    learning_case_cluster_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case_cluster.learning_case_cluster_id"),
        nullable=True,
    )
    learning_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case.learning_case_id"),
        nullable=True,
    )
    expected_behavior: Mapped[str] = mapped_column(String, nullable=False)
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    baseline_pack_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    candidate_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    baseline_diagnosis_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    candidate_diagnosis_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    baseline_primary_mechanism_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    candidate_primary_mechanism_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    baseline_renderer_mode: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    candidate_renderer_mode: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "expected_behavior IN ('CHANGE','PRESERVE')",
            name="ck_knowledge_pack_regression_case_result_expected_behavior",
        ),
        CheckConstraint(
            "outcome IN ('PASS','FAIL')",
            name="ck_knowledge_pack_regression_case_result_outcome",
        ),
    )


class KnowledgePackMonitoringSnapshot(Base):
    __tablename__ = "knowledge_pack_monitoring_snapshot"

    knowledge_pack_monitoring_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    knowledge_pack_release_candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_pack_release_candidate.knowledge_pack_release_candidate_id"),
        nullable=False,
    )
    baseline_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    candidate_pack_version: Mapped[str] = mapped_column(String, nullable=False)
    baseline_window_start: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    baseline_window_end: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    candidate_window_start: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    candidate_window_end: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    sufficient_data: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rollback_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    baseline_metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    candidate_metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    regression_metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    alert_rules_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )


class KnowledgePackRollbackAlert(Base):
    __tablename__ = "knowledge_pack_rollback_alert"

    knowledge_pack_rollback_alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    knowledge_pack_release_candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_pack_release_candidate.knowledge_pack_release_candidate_id"),
        nullable=False,
    )
    knowledge_pack_monitoring_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_pack_monitoring_snapshot.knowledge_pack_monitoring_snapshot_id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="OPEN")
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    triggered_rules_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('OPEN','ACKNOWLEDGED','DISMISSED','ROLLED_BACK')",
            name="ck_knowledge_pack_rollback_alert_status",
        ),
    )


class DeviceRegistration(Base):
    __tablename__ = "device_registration"

    device_registration_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )
    platform: Mapped[str] = mapped_column(String, nullable=False)
    push_provider: Mapped[str] = mapped_column(String, nullable=False)
    push_token: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    device_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    app_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    locale: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timezone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_seen_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "platform IN ('ios','android','web','unknown')",
            name="ck_device_registration_platform",
        ),
        CheckConstraint(
            "push_provider IN ('fcm','apns','unknown')",
            name="ck_device_registration_push_provider",
        ),
    )


class NotificationEvent(Base):
    __tablename__ = "notification_event"

    notification_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")
    active_device_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sent_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('analysis_completed','analysis_failed','profile_country_required')",
            name="ck_notification_event_type",
        ),
        CheckConstraint(
            "status IN ('PENDING','SKIPPED_NO_DEVICE','SENT','FAILED')",
            name="ck_notification_event_status",
        ),
    )


class CoachFlag(Base):
    __tablename__ = "coach_flag"

    coach_flag_id: Mapped[uuid.UUID] = mapped_column(
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
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("account.account_id"),
        nullable=False,
    )
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    flag_type: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    flagged_mechanism_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    flagged_prescription_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    learning_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case.learning_case_id"),
        nullable=True,
    )
    learning_case_cluster_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case_cluster.learning_case_cluster_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "flag_type IN ('wrong_mechanism','wrong_prescription','right_mechanism_wrong_wording','renderer_story_misleading','capture_quality_bad')",
            name="ck_coach_flag_type",
        ),
    )


class PrescriptionFollowup(Base):
    __tablename__ = "prescription_followup"

    prescription_followup_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    prescription_assigned_at_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=False,
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("player.player_id"),
        nullable=False,
    )
    knowledge_pack_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    knowledge_pack_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prescription_id: Mapped[str] = mapped_column(String, nullable=False)
    review_window_type: Mapped[str] = mapped_column(String, nullable=False)
    followup_metrics: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    expected_direction_of_change: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    actual_direction_of_change: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    response_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="NOT_YET_DUE",
    )
    valid_followup_run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latest_followup_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_run.run_id"),
        nullable=True,
    )
    learning_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_case.learning_case_id"),
        nullable=True,
    )
    window_due_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "review_window_type IN ('next_3_runs','next_session','next_2_weeks')",
            name="ck_prescription_followup_review_window_type",
        ),
        CheckConstraint(
            "response_status IN ('NOT_YET_DUE','IMPROVING','NO_CLEAR_CHANGE','WORSENING','INSUFFICIENT_DATA')",
            name="ck_prescription_followup_response_status",
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
