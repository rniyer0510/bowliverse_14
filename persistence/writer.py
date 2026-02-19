import uuid
from typing import Any, Dict, Optional

from app.persistence.session import SessionLocal
from app.persistence.models import (
    AnalysisRun,
    AnalysisResultRaw,
    EventAnchor,
    BiomechSignal,
    RiskMeasurement,
    Player,
)
from app.persistence.resolver import resolve_account
from app.common.logger import get_logger

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


# ------------------------------------------------------------
# Main Persistence
# ------------------------------------------------------------

def write_analysis(result: dict, **kwargs) -> str:
    """
    Persist a completed ActionLab analysis.

    V14+ Architecture:
    - player_id is passed explicitly from orchestrator
    - NO legacy name-based player resolution
    - Authorization already enforced at orchestrator level
    """

    db = SessionLocal()

    try:
        # --------------------------------------------------------
        # Canonical run_id
        # --------------------------------------------------------
        run_id_raw = kwargs.get("run_id")
        if not run_id_raw:
            raise ValueError("run_id must be provided by orchestrator")

        run_id = _coerce_uuid(run_id_raw)

        actor = kwargs.get("actor", {}) or {}

        # --------------------------------------------------------
        # Resolve account only (identity layer)
        # --------------------------------------------------------
        account = resolve_account(db, actor)

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
        run = AnalysisRun(
            run_id=run_id,
            player_id=player_id,
            schema_version=str(schema_version),
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

        db.commit()
        logger.info(f"Analysis persisted: run_id={run_id}")

        return str(run_id)

    except Exception as e:
        logger.warning(f"Persistence failed (rollback): {e}")
        db.rollback()
        raise

    finally:
        db.close()

