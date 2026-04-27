from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.persistence.models import (
    AnalysisResultRaw,
    AnalysisRun,
    CoachFlag,
    KnowledgePackMonitoringSnapshot,
    KnowledgePackRegressionRun,
    KnowledgePackReleaseCandidate,
    KnowledgePackRollbackAlert,
    PrescriptionFollowup,
)
from app.persistence.read_api import _coverage_metrics_payload

MIN_MONITORED_RUNS = 3
NO_MATCH_RELATIVE_INCREASE_MAX = 0.20
NO_MATCH_ABSOLUTE_INCREASE_MIN = 0.02
AMBIGUITY_RELATIVE_INCREASE_MAX = 0.15
AMBIGUITY_ABSOLUTE_INCREASE_MIN = 0.02
VALIDATED_REGRESSION_RATE_MAX = 0.05
COACH_FLAG_RELATIVE_INCREASE_MAX = 0.20
COACH_FLAG_ABSOLUTE_INCREASE_MIN = 0.05


def run_post_promotion_monitoring(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    account_id: Optional[str],
    db: Session,
    now: Optional[datetime] = None,
) -> Tuple[KnowledgePackMonitoringSnapshot, Optional[KnowledgePackRollbackAlert]]:
    if str(candidate_row.status or "") != "PROMOTED":
        raise ValueError("Monitoring checks can only run for promoted release candidates")
    if not candidate_row.promoted_at:
        raise ValueError("Promoted release candidate is missing promoted_at")

    now = now or datetime.utcnow()
    candidate_window_start = candidate_row.promoted_at
    candidate_window_end = now
    if candidate_window_end <= candidate_window_start:
        raise ValueError("Monitoring window has not started yet")

    window_duration = candidate_window_end - candidate_window_start
    baseline_window_end = candidate_window_start
    baseline_window_start = baseline_window_end - window_duration

    baseline_metrics = _collect_window_metrics(
        pack_version=str(candidate_row.base_pack_version),
        window_start=baseline_window_start,
        window_end=baseline_window_end,
        db=db,
    )
    candidate_metrics = _collect_window_metrics(
        pack_version=str(candidate_row.candidate_pack_version),
        window_start=candidate_window_start,
        window_end=candidate_window_end,
        db=db,
    )
    regression_metrics = _latest_regression_metrics(candidate_row=candidate_row, db=db)

    evaluation = evaluate_monitoring_snapshot(
        baseline_metrics=baseline_metrics["overall"],
        candidate_metrics=candidate_metrics["overall"],
        regression_metrics=regression_metrics,
    )

    snapshot = KnowledgePackMonitoringSnapshot(
        knowledge_pack_monitoring_snapshot_id=uuid.uuid4(),
        knowledge_pack_release_candidate_id=candidate_row.knowledge_pack_release_candidate_id,
        baseline_pack_version=str(candidate_row.base_pack_version),
        candidate_pack_version=str(candidate_row.candidate_pack_version),
        baseline_window_start=baseline_window_start,
        baseline_window_end=baseline_window_end,
        candidate_window_start=candidate_window_start,
        candidate_window_end=candidate_window_end,
        sufficient_data=bool(evaluation["sufficient_data"]),
        alert_triggered=bool(evaluation["alert_triggered"]),
        rollback_recommended=bool(evaluation["rollback_recommended"]),
        baseline_metrics_json=dict(baseline_metrics),
        candidate_metrics_json=dict(candidate_metrics),
        regression_metrics_json=dict(regression_metrics),
        alert_rules_json=dict(evaluation),
        created_by_account_id=_parse_uuid(account_id),
        created_at=datetime.utcnow(),
    )
    db.add(snapshot)
    db.flush()

    alert = None
    if snapshot.alert_triggered:
        alert = _create_or_refresh_rollback_alert(
            candidate_row=candidate_row,
            snapshot_row=snapshot,
            evaluation=evaluation,
            db=db,
        )
    return snapshot, alert


def evaluate_monitoring_snapshot(
    *,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    regression_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_runs = _safe_int(baseline_metrics.get("total_runs"))
    candidate_runs = _safe_int(candidate_metrics.get("total_runs"))
    sufficient_data = baseline_runs >= MIN_MONITORED_RUNS and candidate_runs >= MIN_MONITORED_RUNS

    no_match = _rule_evaluation(
        metric_name="no_match_rate",
        baseline_value=_safe_float(baseline_metrics.get("no_match_rate")),
        candidate_value=_safe_float(candidate_metrics.get("no_match_rate")),
        relative_threshold=NO_MATCH_RELATIVE_INCREASE_MAX,
        absolute_delta_min=NO_MATCH_ABSOLUTE_INCREASE_MIN,
        sufficient_data=sufficient_data,
    )
    ambiguity = _rule_evaluation(
        metric_name="ambiguity_rate",
        baseline_value=_safe_float(baseline_metrics.get("ambiguity_rate")),
        candidate_value=_safe_float(candidate_metrics.get("ambiguity_rate")),
        relative_threshold=AMBIGUITY_RELATIVE_INCREASE_MAX,
        absolute_delta_min=AMBIGUITY_ABSOLUTE_INCREASE_MIN,
        sufficient_data=sufficient_data,
    )
    coach_flag = _rule_evaluation(
        metric_name="coach_flag_rate",
        baseline_value=_safe_float(baseline_metrics.get("coach_flag_rate")),
        candidate_value=_safe_float(candidate_metrics.get("coach_flag_rate")),
        relative_threshold=COACH_FLAG_RELATIVE_INCREASE_MAX,
        absolute_delta_min=COACH_FLAG_ABSOLUTE_INCREASE_MIN,
        sufficient_data=sufficient_data,
    )

    validated_regression_rate = _safe_float(regression_metrics.get("validated_regression_rate"))
    regression_rule = {
        "metric": "validated_regression_rate",
        "baseline_value": None,
        "candidate_value": validated_regression_rate,
        "delta": None,
        "relative_increase": None,
        "threshold": VALIDATED_REGRESSION_RATE_MAX,
        "triggered": validated_regression_rate > VALIDATED_REGRESSION_RATE_MAX,
        "reason": (
            f"validated_regression_rate exceeded {VALIDATED_REGRESSION_RATE_MAX:.2f}"
            if validated_regression_rate > VALIDATED_REGRESSION_RATE_MAX
            else "validated_regression_rate stayed within threshold"
        ),
    }

    rules = [no_match, ambiguity, coach_flag, regression_rule]
    triggered_rules = [rule for rule in rules if rule["triggered"]]
    return {
        "sufficient_data": sufficient_data,
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "alert_triggered": bool(triggered_rules),
        "rollback_recommended": bool(triggered_rules),
        "triggered_rule_count": len(triggered_rules),
        "triggered_rules": triggered_rules,
        "rules": rules,
    }


def resolve_open_rollback_alerts_for_candidate(
    *,
    candidate_id: uuid.UUID,
    db: Session,
    resolution_status: str = "ROLLED_BACK",
) -> int:
    resolution = str(resolution_status or "").strip().upper()
    if resolution not in {"ROLLED_BACK", "DISMISSED"}:
        raise ValueError("resolution_status must be ROLLED_BACK or DISMISSED")
    rows = (
        db.query(KnowledgePackRollbackAlert)
        .filter(
            KnowledgePackRollbackAlert.knowledge_pack_release_candidate_id == candidate_id,
            KnowledgePackRollbackAlert.status.in_(("OPEN", "ACKNOWLEDGED")),
        )
        .all()
    )
    now = datetime.utcnow()
    for row in rows:
        row.status = resolution
        row.resolved_at = now
        row.updated_at = now
    db.flush()
    return len(rows)


def _collect_window_metrics(
    *,
    pack_version: str,
    window_start: datetime,
    window_end: datetime,
    db: Session,
) -> Dict[str, Any]:
    runs = (
        db.query(AnalysisRun)
        .filter(
            AnalysisRun.knowledge_pack_version == pack_version,
            AnalysisRun.created_at >= window_start,
            AnalysisRun.created_at < window_end,
        )
        .order_by(AnalysisRun.created_at.desc())
        .all()
    )
    run_ids = [row.run_id for row in runs]
    raw_rows = []
    if run_ids:
        raw_rows = (
            db.query(AnalysisResultRaw)
            .filter(AnalysisResultRaw.run_id.in_(tuple(run_ids)))
            .all()
        )
    raw_by_run_id = {row.run_id: row for row in raw_rows}
    followups = (
        db.query(PrescriptionFollowup)
        .filter(
            PrescriptionFollowup.knowledge_pack_version == pack_version,
            PrescriptionFollowup.updated_at >= window_start,
            PrescriptionFollowup.updated_at < window_end,
        )
        .all()
    )
    coach_flags = (
        db.query(CoachFlag)
        .filter(
            CoachFlag.knowledge_pack_version == pack_version,
            CoachFlag.created_at >= window_start,
            CoachFlag.created_at < window_end,
        )
        .all()
    )
    return _coverage_metrics_payload(
        runs=runs,
        raw_by_run_id=raw_by_run_id,
        followups=followups,
        coach_flags=coach_flags,
    )


def _latest_regression_metrics(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    db: Session,
) -> Dict[str, Any]:
    regression_run = (
        db.query(KnowledgePackRegressionRun)
        .filter(
            KnowledgePackRegressionRun.knowledge_pack_release_candidate_id
            == candidate_row.knowledge_pack_release_candidate_id
        )
        .order_by(KnowledgePackRegressionRun.created_at.desc())
        .first()
    )
    if not regression_run:
        return {
            "status": "missing",
            "validated_regression_rate": 0.0,
            "total_cases": 0,
            "failed_cases": 0,
        }
    return {
        "status": regression_run.status,
        "validated_regression_rate": _safe_float(regression_run.validated_regression_rate),
        "total_cases": _safe_int(regression_run.total_cases),
        "failed_cases": _safe_int(regression_run.failed_cases),
        "knowledge_pack_regression_run_id": str(regression_run.knowledge_pack_regression_run_id),
    }


def _create_or_refresh_rollback_alert(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    snapshot_row: KnowledgePackMonitoringSnapshot,
    evaluation: Dict[str, Any],
    db: Session,
) -> KnowledgePackRollbackAlert:
    existing = (
        db.query(KnowledgePackRollbackAlert)
        .filter(
            KnowledgePackRollbackAlert.knowledge_pack_release_candidate_id
            == candidate_row.knowledge_pack_release_candidate_id,
            KnowledgePackRollbackAlert.status.in_(("OPEN", "ACKNOWLEDGED")),
        )
        .order_by(KnowledgePackRollbackAlert.updated_at.desc())
        .first()
    )
    triggered_rules = list(evaluation.get("triggered_rules") or [])
    summary = _alert_summary(triggered_rules)
    now = datetime.utcnow()
    if existing:
        existing.knowledge_pack_monitoring_snapshot_id = snapshot_row.knowledge_pack_monitoring_snapshot_id
        existing.status = "OPEN"
        existing.summary = summary
        existing.triggered_rules_json = {"triggered_rules": triggered_rules}
        existing.updated_at = now
        existing.resolved_at = None
        db.flush()
        return existing

    alert = KnowledgePackRollbackAlert(
        knowledge_pack_rollback_alert_id=uuid.uuid4(),
        knowledge_pack_release_candidate_id=candidate_row.knowledge_pack_release_candidate_id,
        knowledge_pack_monitoring_snapshot_id=snapshot_row.knowledge_pack_monitoring_snapshot_id,
        status="OPEN",
        summary=summary,
        triggered_rules_json={"triggered_rules": triggered_rules},
        resolved_at=None,
        created_at=now,
        updated_at=now,
    )
    db.add(alert)
    db.flush()
    return alert


def _rule_evaluation(
    *,
    metric_name: str,
    baseline_value: float,
    candidate_value: float,
    relative_threshold: float,
    absolute_delta_min: float,
    sufficient_data: bool,
) -> Dict[str, Any]:
    delta = round(candidate_value - baseline_value, 3)
    relative_increase = _relative_increase(candidate_value, baseline_value)
    triggered = (
        sufficient_data
        and delta > absolute_delta_min
        and relative_increase > relative_threshold
    )
    return {
        "metric": metric_name,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "relative_increase": relative_increase,
        "threshold": relative_threshold,
        "absolute_delta_min": absolute_delta_min,
        "triggered": triggered,
        "reason": (
            f"{metric_name} exceeded the allowed post-promotion drift threshold"
            if triggered
            else f"{metric_name} stayed within the allowed drift threshold"
        ),
    }


def _relative_increase(candidate_value: float, baseline_value: float) -> float:
    if baseline_value <= 0.0:
        return float("inf") if candidate_value > 0.0 else 0.0
    return round((candidate_value - baseline_value) / baseline_value, 3)


def _alert_summary(triggered_rules: list[dict]) -> str:
    metrics = [str(rule.get("metric") or "unknown") for rule in triggered_rules]
    if not metrics:
        return "No rollback alert was triggered."
    metric_list = ", ".join(metrics)
    return f"Post-promotion monitoring detected rollback risk for: {metric_list}."


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _parse_uuid(value: Any) -> Optional[uuid.UUID]:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return uuid.UUID(text)
        except Exception:
            return None
    return None
