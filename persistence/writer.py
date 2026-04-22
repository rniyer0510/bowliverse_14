import uuid
from typing import Any, Dict, Optional

from app.persistence.session import SessionLocal
from app.persistence.models import (
    AnalysisRun,
    AnalysisResultRaw,
    AnalysisExplanationTrace,
    EventAnchor,
    BiomechSignal,
    RiskMeasurement,
    Player,
)
from app.common.logger import get_logger
from sqlalchemy.orm import Session

logger = get_logger(__name__)

EVENT_MAP = {
    "release": "RELEASE",
    "uah": "UAH",
    "ffc": "FFC",
    "bfc": "BFC",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _coerce_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value.strip():
        return uuid.UUID(value.strip())
    raise ValueError("run_id must be a valid UUID (uuid.UUID or UUID string)")


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    try:
        return int(str(x))
    except Exception:
        return None


def _deterministic_summary(result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    deterministic = (result.get("deterministic_expert_v1") or {})
    if not isinstance(deterministic, dict):
        deterministic = {}

    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}

    archetype = deterministic.get("archetype_v1") or {}
    if not isinstance(archetype, dict):
        archetype = {}

    return {
        "knowledge_pack_id": (
            deterministic.get("knowledge_pack_id")
            if isinstance(deterministic.get("knowledge_pack_id"), str)
            else None
        ),
        "knowledge_pack_version": (
            deterministic.get("knowledge_pack_version")
            if isinstance(deterministic.get("knowledge_pack_version"), str)
            else None
        ),
        "diagnosis_status": (
            selection.get("diagnosis_status")
            if isinstance(selection.get("diagnosis_status"), str)
            else None
        ),
        "primary_mechanism_id": (
            selection.get("primary_mechanism_id")
            if isinstance(selection.get("primary_mechanism_id"), str)
            else None
        ),
        "archetype_id": (
            archetype.get("id")
            if isinstance(archetype.get("id"), str)
            else None
        ),
    }


def _deterministic_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    deterministic = result.get("deterministic_expert_v1") or {}
    if not isinstance(deterministic, dict):
        deterministic = {}

    selection = deterministic.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}

    symptoms = deterministic.get("symptoms") or []
    if not isinstance(symptoms, list):
        symptoms = []

    hypotheses = deterministic.get("mechanism_hypotheses") or []
    if not isinstance(hypotheses, list):
        hypotheses = []

    primary = selection.get("primary") or {}
    if not isinstance(primary, dict):
        primary = {}

    explanation = result.get("mechanism_explanation_v1") or {}
    if not isinstance(explanation, dict):
        explanation = {}

    matched_symptom_ids = [
        str(symptom.get("id"))
        for symptom in symptoms
        if isinstance(symptom, dict) and symptom.get("present") and symptom.get("id")
    ]
    supporting_evidence = {
        "winner_required_symptom_ids": list(primary.get("required_symptom_ids") or []),
        "winner_supporting_symptom_ids": list(primary.get("supporting_symptom_ids") or []),
        "winner_matched_symptom_ids": list(primary.get("matched_symptom_ids") or []),
        "primary_summary": primary.get("summary"),
    }
    candidate_mechanisms = []
    for hypothesis in hypotheses[:5]:
        if not isinstance(hypothesis, dict):
            continue
        candidate_mechanisms.append(
            {
                "id": hypothesis.get("id"),
                "title": hypothesis.get("title"),
                "overall_confidence": hypothesis.get("overall_confidence"),
                "support_score": hypothesis.get("support_score"),
                "contradiction_penalty": hypothesis.get("contradiction_penalty"),
                "evidence_completeness": hypothesis.get("evidence_completeness"),
            }
        )

    trace_json = {
        "capture_quality": dict(deterministic.get("capture_quality_v1") or {}),
        "matched_symptom_ids": matched_symptom_ids,
        "candidate_mechanisms": candidate_mechanisms,
        "supporting_evidence": supporting_evidence,
        "contradictions_triggered": list(primary.get("contradiction_notes") or []),
        "selected_trajectory_ids": list(selection.get("selected_trajectory_ids") or []),
        "selected_prescription_ids": list(selection.get("selected_prescription_ids") or []),
        "selected_render_story_ids": list(selection.get("selected_render_story_ids") or []),
        "selected_history_binding_ids": list(explanation.get("selected_history_binding_ids") or []),
    }

    return {
        "knowledge_pack_id": deterministic.get("knowledge_pack_id") if isinstance(deterministic.get("knowledge_pack_id"), str) else None,
        "knowledge_pack_version": deterministic.get("knowledge_pack_version") if isinstance(deterministic.get("knowledge_pack_version"), str) else None,
        "diagnosis_status": selection.get("diagnosis_status") if isinstance(selection.get("diagnosis_status"), str) else None,
        "primary_mechanism_id": selection.get("primary_mechanism_id") if isinstance(selection.get("primary_mechanism_id"), str) else None,
        "matched_symptom_ids": matched_symptom_ids,
        "candidate_mechanisms": candidate_mechanisms,
        "supporting_evidence": supporting_evidence,
        "contradictions_triggered": list(primary.get("contradiction_notes") or []),
        "selected_trajectory_ids": list(selection.get("selected_trajectory_ids") or []),
        "selected_prescription_ids": list(selection.get("selected_prescription_ids") or []),
        "selected_render_story_ids": list(selection.get("selected_render_story_ids") or []),
        "selected_history_binding_ids": list(explanation.get("selected_history_binding_ids") or []),
        "explanation_trace_json": trace_json,
    }


# ------------------------------------------------------------
# Main Persistence
# ------------------------------------------------------------

def write_analysis(result: dict, db: Optional[Session] = None, **kwargs) -> str:
    """
    Persist a completed ActionLab analysis.

    V14+ Architecture:
    - player_id is passed explicitly from orchestrator
    - NO legacy name-based player resolution
    - Authorization already enforced at orchestrator level
    """

    owns_session = db is None
    db = db or SessionLocal()

    try:
        # --------------------------------------------------------
        # Canonical run_id
        # --------------------------------------------------------
        run_id_raw = kwargs.get("run_id")
        if not run_id_raw:
            raise ValueError("run_id must be provided by orchestrator")

        run_id = _coerce_uuid(run_id_raw)

        # --------------------------------------------------------
        # USE explicit player_id from result.input
        # --------------------------------------------------------
        input_player_id = (result.get("input") or {}).get("player_id")

        if not input_player_id:
            raise ValueError("player_id missing in result input")

        try:
            player_id = uuid.UUID(str(input_player_id))
        except Exception:
            raise ValueError("Invalid player_id format")

        # --------------------------------------------------------
        # Snapshot Player
        # --------------------------------------------------------
        player = db.get(Player, player_id)
        if not player:
            raise ValueError("Player not found for snapshot")

        snapshot_season = player.season
        snapshot_age_group = player.age_group

        override_season = kwargs.get("season")
        override_age_group = kwargs.get("age_group")
        if override_season is not None:
            snapshot_season = int(override_season)
        if override_age_group is not None:
            snapshot_age_group = str(override_age_group).upper()

        # --------------------------------------------------------
        # schema_version
        # --------------------------------------------------------
        schema_version = (
            result.get("schema_version")
            or result.get("schema")
            or (result.get("meta", {}) or {}).get("schema_version")
        )

        if not schema_version:
            schema_version = "actionlab.v14"

        # --------------------------------------------------------
        # Create AnalysisRun
        # --------------------------------------------------------
        deterministic_summary = _deterministic_summary(result)
        run = AnalysisRun(
            run_id=run_id,
            player_id=player_id,
            schema_version=str(schema_version),
            knowledge_pack_id=deterministic_summary["knowledge_pack_id"],
            knowledge_pack_version=deterministic_summary["knowledge_pack_version"],
            deterministic_diagnosis_status=deterministic_summary["diagnosis_status"],
            deterministic_primary_mechanism_id=deterministic_summary["primary_mechanism_id"],
            deterministic_archetype_id=deterministic_summary["archetype_id"],
            handedness=(result.get("input", {}) or {}).get("hand"),
            fps=_as_float((result.get("video", {}) or {}).get("fps")),
            total_frames=_as_int((result.get("video", {}) or {}).get("total_frames")),
            season=int(snapshot_season),
            age_group=str(snapshot_age_group),
            coach_notes=None,
        )

        db.add(run)
        db.flush()

        # --------------------------------------------------------
        # Event Anchors
        # --------------------------------------------------------
        events = result.get("events", {}) or {}

        for key, event_type in EVENT_MAP.items():
            ev = events.get(key) or {}
            frame_i = _as_int(ev.get("frame"))
            if frame_i is None:
                continue

            db.add(
                EventAnchor(
                    run_id=run_id,
                    event_type=event_type,
                    frame=frame_i,
                    confidence=_as_float(ev.get("confidence")),
                    method=ev.get("method"),
                )
            )

        # --------------------------------------------------------
        # Biomechanical Signals
        # --------------------------------------------------------
        elbow = result.get("elbow", {}) or {}
        elbow_conf = _as_float(elbow.get("confidence"))
        if elbow_conf is not None:
            db.add(
                BiomechSignal(
                    run_id=run_id,
                    signal_key="elbow_confidence",
                    value=elbow_conf,
                    units="ratio",
                    baseline_eligible=True,
                )
            )

        action = result.get("action", {}) or {}
        action_conf = _as_float(action.get("confidence"))
        if action_conf is not None:
            db.add(
                BiomechSignal(
                    run_id=run_id,
                    signal_key="action_confidence",
                    value=action_conf,
                    units="ratio",
                    baseline_eligible=True,
                )
            )

        # --------------------------------------------------------
        # Risk Measurements
        # --------------------------------------------------------
        for r in (result.get("risks") or []):
            if not isinstance(r, dict):
                continue

            risk_id = r.get("risk_id")
            if not risk_id:
                continue

            strength = _as_float(r.get("signal_strength"))
            if strength is None:
                logger.warning(
                    f"[persistence] skipping risk without signal_strength: {risk_id}"
                )
                continue

            visual = r.get("visual", {}) or {}
            vw = r.get("visual_window", {}) or {}

            db.add(
                RiskMeasurement(
                    run_id=run_id,
                    risk_id=str(risk_id),
                    signal_strength=strength,
                    confidence=_as_float(r.get("confidence")),
                    anchor_event=visual.get("anchor"),
                    window_start=_as_int(vw.get("start")),
                    window_end=_as_int(vw.get("end")),
                )
            )

        # --------------------------------------------------------
        # Raw JSON Storage
        # --------------------------------------------------------
        if isinstance(result, dict):
            result.setdefault("run_id", str(run_id))

        db.add(
            AnalysisResultRaw(
                run_id=run_id,
                result_json=result,
            )
        )

        explanation_trace = _deterministic_trace(result)
        db.add(
            AnalysisExplanationTrace(
                run_id=run_id,
                knowledge_pack_id=explanation_trace["knowledge_pack_id"],
                knowledge_pack_version=explanation_trace["knowledge_pack_version"],
                diagnosis_status=explanation_trace["diagnosis_status"],
                primary_mechanism_id=explanation_trace["primary_mechanism_id"],
                matched_symptom_ids=explanation_trace["matched_symptom_ids"],
                candidate_mechanisms=explanation_trace["candidate_mechanisms"],
                supporting_evidence=explanation_trace["supporting_evidence"],
                contradictions_triggered=explanation_trace["contradictions_triggered"],
                selected_trajectory_ids=explanation_trace["selected_trajectory_ids"],
                selected_prescription_ids=explanation_trace["selected_prescription_ids"],
                selected_render_story_ids=explanation_trace["selected_render_story_ids"],
                selected_history_binding_ids=explanation_trace["selected_history_binding_ids"],
                explanation_trace_json=explanation_trace["explanation_trace_json"],
            )
        )

        if owns_session:
            db.commit()
        else:
            db.flush()
        logger.info(f"Analysis persisted: run_id={run_id}")

        return str(run_id)

    except Exception as e:
        logger.warning(f"Persistence failed (rollback): {e}")
        if owns_session:
            db.rollback()
        raise

    finally:
        if owns_session:
            db.close()
