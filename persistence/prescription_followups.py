from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.clinician.knowledge_pack import load_knowledge_pack
from app.common.logger import get_logger
from app.persistence.learning_cases import (
    build_prescription_non_response_learning_case_event,
    write_learning_case,
)
from app.persistence.models import AnalysisRun, AnalysisResultRaw, PrescriptionFollowup
from app.persistence.session import SessionLocal

logger = get_logger(__name__)

SESSION_GAP_MINUTES = 90
FOLLOWUP_EPSILON = 0.02


def sync_prescription_followups_for_run(
    *,
    run_id: str,
    db: Optional[Session] = None,
) -> Dict[str, int]:
    owns_session = db is None
    db = db or SessionLocal()
    try:
        current = (
            db.query(AnalysisRun, AnalysisResultRaw)
            .join(AnalysisResultRaw, AnalysisResultRaw.run_id == AnalysisRun.run_id)
            .filter(AnalysisRun.run_id == run_id)
            .first()
        )
        if not current:
            return {"created": 0, "updated": 0, "non_response_cases": 0}

        run_row, raw_row = current
        result_json = raw_row.result_json if isinstance(raw_row.result_json, dict) else {}
        created = _ensure_followup_assignments(
            run_row=run_row,
            result_json=result_json,
            db=db,
        )
        updated, non_response_cases = _evaluate_open_followups(
            current_run=run_row,
            current_result=result_json,
            db=db,
        )
        if owns_session:
            db.commit()
        else:
            db.flush()
        return {
            "created": created,
            "updated": updated,
            "non_response_cases": non_response_cases,
        }
    except Exception:
        if owns_session:
            db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def _ensure_followup_assignments(
    *,
    run_row: AnalysisRun,
    result_json: Dict[str, Any],
    db: Session,
) -> int:
    deterministic = (result_json.get("deterministic_expert_v1") or {})
    if not isinstance(deterministic, dict):
        deterministic = {}
    prescription_plan = result_json.get("prescription_plan_v1") or deterministic.get("prescription_plan_v1") or {}
    if not isinstance(prescription_plan, dict):
        prescription_plan = {}
    prescriptions = prescription_plan.get("prescriptions") or []
    if not isinstance(prescriptions, list):
        prescriptions = []

    created = 0
    globals_cfg = load_knowledge_pack()["globals"]
    followup_defaults = globals_cfg.get("followup_defaults") or {}
    default_window_type = str(followup_defaults.get("default_window_type") or "next_3_runs")
    knowledge_pack_id = deterministic.get("knowledge_pack_id")
    knowledge_pack_version = deterministic.get("knowledge_pack_version")

    for prescription in prescriptions:
        if not isinstance(prescription, dict):
            continue
        prescription_id = str(prescription.get("id") or "").strip()
        if not prescription_id:
            continue
        existing = (
            db.query(PrescriptionFollowup)
            .filter(
                PrescriptionFollowup.prescription_assigned_at_run_id == run_row.run_id,
                PrescriptionFollowup.prescription_id == prescription_id,
            )
            .first()
        )
        if existing:
            continue

        review_window_type = str(
            prescription.get("review_window_type")
            or default_window_type
        ).strip()
        targets = _parse_followup_targets(list(prescription.get("followup_metric_targets") or []))
        due_at = (
            run_row.created_at + timedelta(days=14)
            if review_window_type == "next_2_weeks"
            else None
        )
        row = PrescriptionFollowup(
            prescription_followup_id=uuid.uuid4(),
            prescription_assigned_at_run_id=run_row.run_id,
            player_id=run_row.player_id,
            knowledge_pack_id=str(knowledge_pack_id) if knowledge_pack_id else None,
            knowledge_pack_version=str(knowledge_pack_version) if knowledge_pack_version else None,
            prescription_id=prescription_id,
            review_window_type=review_window_type,
            followup_metrics=[target["metric"] for target in targets],
            expected_direction_of_change={target["metric"]: target["direction"] for target in targets},
            actual_direction_of_change={},
            response_status="NOT_YET_DUE",
            valid_followup_run_count=0,
            window_closed=False,
            latest_followup_run_id=None,
            learning_case_id=None,
            window_due_at=due_at,
            resolved_at=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(row)
        created += 1
    if created:
        db.flush()
    return created


def _evaluate_open_followups(
    *,
    current_run: AnalysisRun,
    current_result: Dict[str, Any],
    db: Session,
) -> Tuple[int, int]:
    open_rows = (
        db.query(PrescriptionFollowup)
        .filter(
            PrescriptionFollowup.player_id == current_run.player_id,
            PrescriptionFollowup.prescription_assigned_at_run_id != current_run.run_id,
            PrescriptionFollowup.response_status == "NOT_YET_DUE",
        )
        .all()
    )
    if not open_rows:
        return 0, 0

    updated = 0
    non_response_cases = 0
    globals_cfg = load_knowledge_pack()["globals"]
    followup_defaults = globals_cfg.get("followup_defaults") or {}
    min_valid_runs_by_window = dict(followup_defaults.get("min_valid_runs") or {})
    insufficient_data_status = str(
        followup_defaults.get("insufficient_data_status") or "INSUFFICIENT_DATA"
    )

    for followup in open_rows:
        assigned = (
            db.query(AnalysisRun, AnalysisResultRaw)
            .join(AnalysisResultRaw, AnalysisResultRaw.run_id == AnalysisRun.run_id)
            .filter(AnalysisRun.run_id == followup.prescription_assigned_at_run_id)
            .first()
        )
        if not assigned:
            continue
        assigned_run, assigned_raw = assigned
        assigned_result = assigned_raw.result_json if isinstance(assigned_raw.result_json, dict) else {}

        later_rows = (
            db.query(AnalysisRun, AnalysisResultRaw)
            .join(AnalysisResultRaw, AnalysisResultRaw.run_id == AnalysisRun.run_id)
            .filter(
                AnalysisRun.player_id == followup.player_id,
                AnalysisRun.created_at > assigned_run.created_at,
            )
            .order_by(AnalysisRun.created_at.asc())
            .all()
        )
        relevant_runs = _select_relevant_followup_runs(
            assigned_run=assigned_run,
            later_rows=later_rows,
            review_window_type=followup.review_window_type,
        )
        metrics_summary = _evaluate_metric_directions(
            assigned_result=assigned_result,
            relevant_runs=relevant_runs,
            expected_direction_of_change=dict(followup.expected_direction_of_change or {}),
        )
        valid_run_count = metrics_summary["valid_followup_run_count"]
        min_valid_runs = int(min_valid_runs_by_window.get(followup.review_window_type) or 1)
        window_closed = _window_closed(
            assigned_run=assigned_run,
            current_run=current_run,
            relevant_runs=relevant_runs,
            review_window_type=followup.review_window_type,
            min_valid_runs=min_valid_runs,
            current_result=current_result,
            window_due_at=followup.window_due_at,
        )
        response_status = _resolve_followup_status(
            metrics_summary=metrics_summary,
            valid_run_count=valid_run_count,
            min_valid_runs=min_valid_runs,
            window_closed=window_closed,
            insufficient_data_status=insufficient_data_status,
        )

        if (
            response_status == followup.response_status
            and valid_run_count == int(followup.valid_followup_run_count or 0)
            and bool(window_closed) == bool(followup.window_closed)
        ):
            continue

        followup.actual_direction_of_change = dict(metrics_summary["actual_direction_of_change"])
        followup.valid_followup_run_count = valid_run_count
        followup.window_closed = bool(window_closed)
        followup.latest_followup_run_id = (
            relevant_runs[-1][0].run_id if relevant_runs else None
        )
        followup.response_status = response_status
        followup.updated_at = datetime.utcnow()
        if response_status != "NOT_YET_DUE":
            followup.resolved_at = datetime.utcnow()

        if response_status in {"NO_CLEAR_CHANGE", "WORSENING"} and followup.learning_case_id is None:
            event_payload = build_prescription_non_response_learning_case_event(
                assigned_result=assigned_result,
                latest_result=current_result,
                followup={
                    "prescription_assigned_at_run_id": str(followup.prescription_assigned_at_run_id),
                    "prescription_id": followup.prescription_id,
                    "response_status": response_status,
                    "expected_direction_of_change": dict(followup.expected_direction_of_change or {}),
                    "actual_direction_of_change": dict(metrics_summary["actual_direction_of_change"]),
                },
            )
            if event_payload:
                stored = write_learning_case(event_payload=event_payload, db=db)
                followup.learning_case_id = uuid.UUID(stored["learning_case_id"])
                non_response_cases += 1

        updated += 1

    if updated:
        db.flush()
    return updated, non_response_cases


def _select_relevant_followup_runs(
    *,
    assigned_run: AnalysisRun,
    later_rows: List[Tuple[AnalysisRun, AnalysisResultRaw]],
    review_window_type: str,
) -> List[Tuple[AnalysisRun, Dict[str, Any]]]:
    selected: List[Tuple[AnalysisRun, Dict[str, Any]]] = []
    if review_window_type == "next_3_runs":
        for run_row, raw_row in later_rows[:3]:
            selected.append((run_row, raw_row.result_json if isinstance(raw_row.result_json, dict) else {}))
        return selected

    if review_window_type == "next_session":
        for run_row, raw_row in later_rows:
            if _is_new_session(assigned_run.created_at, run_row.created_at):
                selected.append((run_row, raw_row.result_json if isinstance(raw_row.result_json, dict) else {}))
                break
        return selected

    due_at = assigned_run.created_at + timedelta(days=14)
    for run_row, raw_row in later_rows:
        if run_row.created_at <= due_at:
            selected.append((run_row, raw_row.result_json if isinstance(raw_row.result_json, dict) else {}))
    return selected


def _evaluate_metric_directions(
    *,
    assigned_result: Dict[str, Any],
    relevant_runs: List[Tuple[AnalysisRun, Dict[str, Any]]],
    expected_direction_of_change: Dict[str, str],
) -> Dict[str, Any]:
    actual: Dict[str, Dict[str, Any]] = {}
    valid_followup_run_count = 0

    for metric_name, direction in expected_direction_of_change.items():
        baseline = _extract_metric_value(assigned_result, metric_name)
        samples = []
        contributing_run_ids = []
        for run_row, result_json in relevant_runs:
            value = _extract_metric_value(result_json, metric_name)
            if value is None:
                continue
            samples.append(value)
            contributing_run_ids.append(str(run_row.run_id))
        if samples:
            valid_followup_run_count = max(valid_followup_run_count, len(samples))

        average_value = sum(samples) / len(samples) if samples else None
        delta = (
            round(float(average_value) - float(baseline), 4)
            if baseline is not None and average_value is not None
            else None
        )
        normalized_direction = _normalize_direction(direction)
        actual[metric_name] = {
            "expected_direction": normalized_direction,
            "baseline": baseline,
            "followup_average": round(float(average_value), 4) if average_value is not None else None,
            "delta": delta,
            "contributing_run_ids": contributing_run_ids,
            "status": _metric_direction_status(delta, normalized_direction),
        }

    return {
        "actual_direction_of_change": actual,
        "valid_followup_run_count": valid_followup_run_count,
    }


def _resolve_followup_status(
    *,
    metrics_summary: Dict[str, Any],
    valid_run_count: int,
    min_valid_runs: int,
    window_closed: bool,
    insufficient_data_status: str,
) -> str:
    if valid_run_count < min_valid_runs:
        return insufficient_data_status if window_closed else "NOT_YET_DUE"

    statuses = [
        str(metric.get("status") or "")
        for metric in dict(metrics_summary.get("actual_direction_of_change") or {}).values()
        if isinstance(metric, dict)
    ]
    usable = [status for status in statuses if status in {"improving", "worsening", "flat"}]
    if not usable:
        return insufficient_data_status if window_closed else "NOT_YET_DUE"

    improving = sum(1 for status in usable if status == "improving")
    worsening = sum(1 for status in usable if status == "worsening")
    flat = sum(1 for status in usable if status == "flat")

    if worsening > improving and worsening >= flat:
        return "WORSENING"
    if improving > worsening and improving >= flat:
        return "IMPROVING"
    return "NO_CLEAR_CHANGE"


def _window_closed(
    *,
    assigned_run: AnalysisRun,
    current_run: AnalysisRun,
    relevant_runs: List[Tuple[AnalysisRun, Dict[str, Any]]],
    review_window_type: str,
    min_valid_runs: int,
    current_result: Dict[str, Any],
    window_due_at,
) -> bool:
    if review_window_type == "next_3_runs":
        return len(relevant_runs) >= 3
    if review_window_type == "next_session":
        return len(relevant_runs) >= min_valid_runs or _is_new_session(assigned_run.created_at, current_run.created_at)
    if review_window_type == "next_2_weeks":
        if len(relevant_runs) >= min_valid_runs:
            return True
        if window_due_at is None:
            return False
        return current_run.created_at >= window_due_at
    return False


def _parse_followup_targets(targets: List[str]) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for raw in targets or []:
        text = str(raw or "").strip()
        if not text:
            continue
        pieces = text.rsplit(" ", 1)
        if len(pieces) == 2:
            metric_name, direction = pieces
        else:
            metric_name, direction = pieces[0], "up"
        parsed.append(
            {
                "metric": metric_name.strip(),
                "direction": _normalize_direction(direction),
            }
        )
    return parsed


def _normalize_direction(direction: str) -> str:
    normalized = str(direction or "").strip().lower()
    if normalized in {"up", "increase", "higher"}:
        return "up"
    if normalized in {"down", "decrease", "lower", "calmer"}:
        return "down"
    return "up"


def _metric_direction_status(delta: Optional[float], direction: str) -> str:
    if delta is None:
        return "insufficient_data"
    if direction == "up":
        if delta >= FOLLOWUP_EPSILON:
            return "improving"
        if delta <= -FOLLOWUP_EPSILON:
            return "worsening"
        return "flat"
    if delta <= -FOLLOWUP_EPSILON:
        return "improving"
    if delta >= FOLLOWUP_EPSILON:
        return "worsening"
    return "flat"


def _extract_metric_value(result_json: Dict[str, Any], metric_name: str) -> Optional[float]:
    deterministic = (result_json.get("deterministic_expert_v1") or {})
    if not isinstance(deterministic, dict):
        deterministic = {}
    metrics = deterministic.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    metric = metrics.get(metric_name)
    if isinstance(metric, dict):
        value = metric.get("value")
        if isinstance(value, (int, float)):
            return float(value)

    kinetic_chain = result_json.get("kinetic_chain_v1") or {}
    if isinstance(kinetic_chain, dict):
        internal_metrics = kinetic_chain.get("internal_metrics") or {}
        if isinstance(internal_metrics, dict):
            fallback = internal_metrics.get(metric_name)
            if isinstance(fallback, (int, float)):
                return float(fallback)
    return None


def _is_new_session(assigned_at: datetime, candidate_at: datetime) -> bool:
    if candidate_at.date() != assigned_at.date():
        return True
    return (candidate_at - assigned_at) >= timedelta(minutes=SESSION_GAP_MINUTES)
