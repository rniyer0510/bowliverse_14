from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.clinician.deterministic_expert import DeterministicExpertSystem
from app.persistence.knowledge_pack_releases import apply_release_action
from app.persistence.models import (
    AnalysisResultRaw,
    AnalysisRun,
    KnowledgePackRegressionCaseResult,
    KnowledgePackRegressionRun,
    KnowledgePackReleaseCandidate,
    LearningCase,
    LearningCaseCluster,
)

STATUS_RANK = {
    "no_match": 0,
    "weak_match": 1,
    "ambiguous_match": 2,
    "partial_match": 3,
    "confident_match": 4,
}


def run_release_candidate_regression(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    account_id: Optional[str],
    db: Session,
) -> KnowledgePackRegressionRun:
    case_specs = _load_regression_case_specs(candidate_row=candidate_row, db=db)
    if not case_specs:
        raise ValueError("No regression cases are available for this release candidate")

    engine = DeterministicExpertSystem(pack_version=str(candidate_row.candidate_pack_version))
    regression_run = KnowledgePackRegressionRun(
        knowledge_pack_regression_run_id=uuid.uuid4(),
        knowledge_pack_release_candidate_id=candidate_row.knowledge_pack_release_candidate_id,
        baseline_pack_version=str(candidate_row.base_pack_version),
        candidate_pack_version=str(candidate_row.candidate_pack_version),
        status="COMPLETED",
        total_cases=0,
        expected_change_cases=0,
        stable_cases=0,
        passed_cases=0,
        failed_cases=0,
        validated_regression_count=0,
        validated_regression_rate=0.0,
        expected_change_success_count=0,
        expected_change_success_rate=0.0,
        summary_json={},
        created_by_account_id=_parse_uuid(account_id),
        created_at=datetime.utcnow(),
    )
    db.add(regression_run)
    db.flush()

    case_results: List[KnowledgePackRegressionCaseResult] = []
    expected_change_passes = 0
    validated_regressions = 0

    for spec in case_specs:
        baseline_result = spec["raw_result"]
        prior_results = _load_prior_results(
            run_row=spec["run_row"],
            history_window_runs=engine.history_window_runs,
            db=db,
        )
        candidate_result = _rerun_deterministic_payload(
            raw_result=baseline_result,
            candidate_pack_version=str(candidate_row.candidate_pack_version),
            prior_results=prior_results,
        )
        evaluation = evaluate_regression_case(
            baseline_result=baseline_result,
            candidate_result=candidate_result,
            expected_behavior=spec["expected_behavior"],
        )
        row = KnowledgePackRegressionCaseResult(
            knowledge_pack_regression_case_result_id=uuid.uuid4(),
            knowledge_pack_regression_run_id=regression_run.knowledge_pack_regression_run_id,
            run_id=spec["run_row"].run_id,
            learning_case_cluster_id=spec.get("learning_case_cluster_id"),
            learning_case_id=spec.get("learning_case_id"),
            expected_behavior=spec["expected_behavior"],
            outcome=evaluation["outcome"],
            baseline_pack_version=_safe_str(spec.get("baseline_pack_version")),
            candidate_pack_version=str(candidate_row.candidate_pack_version),
            baseline_diagnosis_status=evaluation["baseline"]["diagnosis_status"],
            candidate_diagnosis_status=evaluation["candidate"]["diagnosis_status"],
            baseline_primary_mechanism_id=evaluation["baseline"]["primary_mechanism_id"],
            candidate_primary_mechanism_id=evaluation["candidate"]["primary_mechanism_id"],
            baseline_renderer_mode=evaluation["baseline"]["renderer_mode"],
            candidate_renderer_mode=evaluation["candidate"]["renderer_mode"],
            reason=evaluation["reason"],
            result_json={
                "baseline": evaluation["baseline"],
                "candidate": evaluation["candidate"],
                "candidate_result": candidate_result,
            },
            created_at=datetime.utcnow(),
        )
        db.add(row)
        case_results.append(row)

        regression_run.total_cases += 1
        if spec["expected_behavior"] == "CHANGE":
            regression_run.expected_change_cases += 1
            if evaluation["outcome"] == "PASS":
                expected_change_passes += 1
        else:
            regression_run.stable_cases += 1
            if evaluation["outcome"] == "FAIL":
                validated_regressions += 1
        if evaluation["outcome"] == "PASS":
            regression_run.passed_cases += 1
        else:
            regression_run.failed_cases += 1

    regression_run.expected_change_success_count = expected_change_passes
    regression_run.validated_regression_count = validated_regressions
    regression_run.expected_change_success_rate = _rate(
        expected_change_passes,
        regression_run.expected_change_cases,
    )
    regression_run.validated_regression_rate = _rate(
        validated_regressions,
        regression_run.stable_cases,
    )
    regression_run.status = "COMPLETED" if regression_run.failed_cases == 0 else "FAILED"
    regression_run.summary_json = {
        "overall_passed": regression_run.failed_cases == 0,
        "candidate_pack_version": str(candidate_row.candidate_pack_version),
        "baseline_pack_version": str(candidate_row.base_pack_version),
        "expected_change_cases": regression_run.expected_change_cases,
        "stable_cases": regression_run.stable_cases,
        "validated_regression_count": regression_run.validated_regression_count,
        "validated_regression_rate": regression_run.validated_regression_rate,
        "expected_change_success_rate": regression_run.expected_change_success_rate,
    }

    candidate_row.regression_suite_passed = regression_run.failed_cases == 0
    candidate_row.updated_at = datetime.utcnow()
    candidate_row.updated_by_account_id = _parse_uuid(account_id)
    db.flush()

    if candidate_row.regression_suite_passed:
        apply_release_action(
            candidate_row=candidate_row,
            action="record_regression_pass",
            account_id=account_id,
            notes="Automated regression suite completed successfully.",
            metadata={
                "knowledge_pack_regression_run_id": str(regression_run.knowledge_pack_regression_run_id),
                "total_cases": regression_run.total_cases,
                "validated_regression_rate": regression_run.validated_regression_rate,
                "expected_change_success_rate": regression_run.expected_change_success_rate,
            },
            db=db,
        )
    return regression_run


def evaluate_regression_case(
    *,
    baseline_result: Dict[str, Any],
    candidate_result: Dict[str, Any],
    expected_behavior: str,
) -> Dict[str, Any]:
    baseline = _selection_summary(baseline_result)
    candidate = _selection_summary(candidate_result)
    expected = str(expected_behavior or "").strip().upper()

    if expected == "CHANGE":
        improved = (
            candidate["status_rank"] > baseline["status_rank"]
            or (
                baseline["primary_mechanism_id"] != candidate["primary_mechanism_id"]
                and candidate["primary_mechanism_id"] is not None
            )
            or (
                baseline["renderer_rank"] < candidate["renderer_rank"]
                and candidate["renderer_mode"] is not None
            )
        )
        outcome = "PASS" if improved else "FAIL"
        reason = (
            "Candidate pack materially changed the interpretation for a curated change case."
            if improved
            else "Candidate pack did not materially change the interpretation for a curated change case."
        )
    else:
        stable = (
            baseline["diagnosis_status"] == candidate["diagnosis_status"]
            and baseline["primary_mechanism_id"] == candidate["primary_mechanism_id"]
            and baseline["renderer_mode"] == candidate["renderer_mode"]
        )
        outcome = "PASS" if stable else "FAIL"
        reason = (
            "Candidate pack preserved the baseline interpretation for a stable regression case."
            if stable
            else "Candidate pack changed the interpretation for a stable regression case."
        )

    return {
        "outcome": outcome,
        "reason": reason,
        "baseline": baseline,
        "candidate": candidate,
    }


def _load_regression_case_specs(
    *,
    candidate_row: KnowledgePackReleaseCandidate,
    db: Session,
) -> List[Dict[str, Any]]:
    motivating_case_ids = {
        _parse_uuid(case_id)
        for case_id in list(candidate_row.motivating_case_ids or [])
        if _parse_uuid(case_id) is not None
    }
    motivating_cluster_ids = {
        _parse_uuid(cluster_id)
        for cluster_id in list(candidate_row.motivating_cluster_ids or [])
        if _parse_uuid(cluster_id) is not None
    }
    change_cases: List[LearningCase] = []
    if motivating_case_ids:
        change_cases.extend(
            db.query(LearningCase)
            .filter(LearningCase.learning_case_id.in_(tuple(motivating_case_ids)))
            .all()
        )
    elif motivating_cluster_ids:
        change_cases.extend(
            db.query(LearningCase)
            .filter(LearningCase.learning_case_cluster_id.in_(tuple(motivating_cluster_ids)))
            .all()
        )

    reinterpret_run_ids = {
        _parse_uuid(run_id)
        for run_id in list(candidate_row.reinterpret_run_ids or [])
        if _parse_uuid(run_id) is not None
    }
    if reinterpret_run_ids:
        extra_cases = (
            db.query(LearningCase)
            .filter(LearningCase.run_id.in_(tuple(reinterpret_run_ids)))
            .all()
        )
        change_cases.extend(extra_cases)

    seen_run_ids: set[uuid.UUID] = set()
    specs: List[Dict[str, Any]] = []
    for row in change_cases:
        if row.run_id in seen_run_ids:
            continue
        artifact = _load_run_artifact(run_id=row.run_id, db=db)
        if not artifact:
            continue
        specs.append(
            {
                "run_row": artifact["run_row"],
                "raw_result": artifact["raw_result"],
                "expected_behavior": "CHANGE",
                "learning_case_id": row.learning_case_id,
                "learning_case_cluster_id": row.learning_case_cluster_id,
                "baseline_pack_version": artifact["baseline_pack_version"],
            }
        )
        seen_run_ids.add(row.run_id)

    for run_id in reinterpret_run_ids:
        if run_id in seen_run_ids:
            continue
        artifact = _load_run_artifact(run_id=run_id, db=db)
        if not artifact:
            continue
        specs.append(
            {
                "run_row": artifact["run_row"],
                "raw_result": artifact["raw_result"],
                "expected_behavior": "CHANGE",
                "learning_case_id": None,
                "learning_case_cluster_id": None,
                "baseline_pack_version": artifact["baseline_pack_version"],
            }
        )
        seen_run_ids.add(run_id)

    resolved_cluster_ids = [
        row.learning_case_cluster_id
        for row in db.query(LearningCaseCluster)
        .filter(LearningCaseCluster.status == "RESOLVED")
        .all()
        if row.learning_case_cluster_id not in motivating_cluster_ids
    ]
    if resolved_cluster_ids:
        stable_cases = (
            db.query(LearningCase)
            .filter(LearningCase.learning_case_cluster_id.in_(tuple(resolved_cluster_ids)))
            .order_by(LearningCase.created_at.asc())
            .all()
        )
        for row in stable_cases:
            if row.run_id in seen_run_ids:
                continue
            artifact = _load_run_artifact(run_id=row.run_id, db=db)
            if not artifact:
                continue
            specs.append(
                {
                    "run_row": artifact["run_row"],
                    "raw_result": artifact["raw_result"],
                    "expected_behavior": "PRESERVE",
                    "learning_case_id": row.learning_case_id,
                    "learning_case_cluster_id": row.learning_case_cluster_id,
                    "baseline_pack_version": artifact["baseline_pack_version"],
                }
            )
            seen_run_ids.add(row.run_id)
    return specs


def _load_run_artifact(*, run_id: uuid.UUID, db: Session) -> Optional[Dict[str, Any]]:
    run_row = db.query(AnalysisRun).filter(AnalysisRun.run_id == run_id).first()
    raw_row = db.query(AnalysisResultRaw).filter(AnalysisResultRaw.run_id == run_id).first()
    raw_result = raw_row.result_json if raw_row and isinstance(raw_row.result_json, dict) else None
    if run_row is None or raw_result is None:
        return None
    return {
        "run_row": run_row,
        "raw_result": raw_result,
        "baseline_pack_version": (
            (raw_result.get("deterministic_expert_v1") or {}).get("knowledge_pack_version")
            or getattr(run_row, "knowledge_pack_version", None)
        ),
    }


def _load_prior_results(
    *,
    run_row: AnalysisRun,
    history_window_runs: int,
    db: Session,
) -> List[Dict[str, Any]]:
    rows = (
        db.query(AnalysisRun, AnalysisResultRaw)
        .join(AnalysisResultRaw, AnalysisResultRaw.run_id == AnalysisRun.run_id)
        .filter(
            AnalysisRun.player_id == run_row.player_id,
            AnalysisRun.created_at < run_row.created_at,
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(history_window_runs)
        .all()
    )
    prior_results: List[Dict[str, Any]] = []
    for prior_run, raw_row in rows:
        result_json = raw_row.result_json if raw_row and isinstance(raw_row.result_json, dict) else None
        if result_json is None:
            continue
        prior_results.append(
            {
                "run_id": str(prior_run.run_id),
                "result_json": result_json,
            }
        )
    return prior_results


def _rerun_deterministic_payload(
    *,
    raw_result: Dict[str, Any],
    candidate_pack_version: str,
    prior_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    engine = DeterministicExpertSystem(pack_version=candidate_pack_version)
    return engine.build(
        events=dict(raw_result.get("events") or {}),
        action=dict(raw_result.get("action") or {}),
        risks=list(raw_result.get("risks") or []),
        basics=dict(raw_result.get("basics") or {}),
        interpretation=dict(raw_result.get("interpretation") or {}),
        estimated_release_speed=dict(raw_result.get("estimated_release_speed") or {}),
        prior_results=list(prior_results or []),
        account_role=_safe_str((((raw_result.get("auth") or {}).get("account_role")))),
    )


def _selection_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    deterministic = result.get("deterministic_expert_v1") or result
    if not isinstance(deterministic, dict):
        deterministic = {}
    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}
    render_reasoning = deterministic.get("render_reasoning_v1") or result.get("render_reasoning_v1") or {}
    if not isinstance(render_reasoning, dict):
        render_reasoning = {}
    diagnosis_status = _safe_str(selection.get("diagnosis_status"))
    primary_mechanism_id = _safe_str(selection.get("primary_mechanism_id"))
    renderer_mode = _safe_str(render_reasoning.get("renderer_mode"))
    return {
        "diagnosis_status": diagnosis_status,
        "primary_mechanism_id": primary_mechanism_id,
        "renderer_mode": renderer_mode,
        "status_rank": STATUS_RANK.get(str(diagnosis_status or ""), -1),
        "renderer_rank": _renderer_rank(renderer_mode),
    }


def _renderer_rank(mode: Optional[str]) -> int:
    order = {"event_only": 0, "partial_evidence": 1, "full_causal_story": 2}
    return order.get(str(mode or ""), -1)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 3)


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _parse_uuid(value: Any) -> Optional[uuid.UUID]:
    text = _safe_str(value)
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except Exception:
        return None
