from app.persistence.session import SessionLocal
from app.persistence.models import (
    AnalysisRun,
    AnalysisResultRaw,
    EventAnchor,
    BiomechSignal,
    RiskMeasurement,
    Player,
)
from app.persistence.resolver import resolve_account, resolve_player
from app.common.logger import get_logger

logger = get_logger(__name__)

EVENT_MAP = {
    "release": "RELEASE",
    "uah": "UAH",
    "ffc": "FFC",
    "bfc": "BFC",
}


def write_analysis(result: dict, **kwargs) -> str:
    """
    Persist a completed ActionLab analysis.

    REQUIREMENT:
        run_id MUST be provided by orchestrator.
        DB must NOT generate its own UUID.

    Returns:
        str: canonical run_id
    """

    db = SessionLocal()

    try:
        # --------------------------------------------------------
        # Validate canonical run_id from orchestrator
        # --------------------------------------------------------
        run_id = kwargs.get("run_id")
        if not run_id:
            raise ValueError("run_id must be provided by orchestrator")

        actor = kwargs.get("actor", {}) or {}

        account = resolve_account(db, actor)
        player_id = resolve_player(db, account, actor)

        # --------------------------------------------------------
        # Snapshot season + age_group from Player
        # --------------------------------------------------------
        player = db.get(Player, player_id)
        if not player:
            raise ValueError("Player not found for snapshot")

        snapshot_season = player.season
        snapshot_age_group = player.age_group

        # --------------------------------------------------------
        # Create AnalysisRun (USE provided run_id)
        # --------------------------------------------------------
        run = AnalysisRun(
            run_id=run_id,   # 🔥 Canonical ID from orchestrator
            player_id=player_id,
            schema_version=result.get("schema"),
            handedness=result.get("input", {}).get("hand"),
            fps=result.get("video", {}).get("fps"),
            total_frames=result.get("video", {}).get("total_frames"),
            season=snapshot_season,
            age_group=snapshot_age_group,
            coach_notes=None,
        )

        db.add(run)
        db.flush()

        # --------------------------------------------------------
        # Event Anchors
        # --------------------------------------------------------
        for key, event_type in EVENT_MAP.items():
            ev = result.get("events", {}).get(key)
            if not ev or ev.get("frame") is None:
                continue

            db.add(EventAnchor(
                run_id=run_id,
                event_type=event_type,
                frame=ev["frame"],
                confidence=ev.get("confidence"),
                method=ev.get("method"),
            ))

        # --------------------------------------------------------
        # Biomechanical Signals (minimal baseline signals)
        # --------------------------------------------------------
        elbow = result.get("elbow", {})
        if elbow:
            db.add(BiomechSignal(
                run_id=run_id,
                signal_key="elbow_confidence",
                value=elbow.get("confidence"),
                units="ratio",
                baseline_eligible=True,
            ))

        action = result.get("action", {})
        if action:
            db.add(BiomechSignal(
                run_id=run_id,
                signal_key="action_confidence",
                value=action.get("confidence"),
                units="ratio",
                baseline_eligible=True,
            ))

        knee = result.get("basics", {}).get("knee_brace_proxy", {})
        dbg = knee.get("debug", {}) or {}
        if dbg.get("pelvis_drop") is not None:
            db.add(BiomechSignal(
                run_id=run_id,
                signal_key="knee_brace_pelvis_drop",
                value=dbg["pelvis_drop"],
                units="normalized",
                confidence=knee.get("confidence"),
                event_anchor="FFC",
                baseline_eligible=True,
            ))

        # --------------------------------------------------------
        # Risk Measurements
        # --------------------------------------------------------
        for r in result.get("risks", []) or []:
            risk_id = r.get("risk_id")
            if not risk_id:
                continue

            db.add(RiskMeasurement(
                run_id=run_id,
                risk_id=risk_id,
                signal_strength=r.get("signal_strength"),
                confidence=r.get("confidence"),
                anchor_event=r.get("visual", {}).get("anchor"),
                window_start=r.get("visual_window", {}).get("start"),
                window_end=r.get("visual_window", {}).get("end"),
            ))

        # --------------------------------------------------------
        # Store full JSON (immutable record)
        # --------------------------------------------------------
        db.add(AnalysisResultRaw(
            run_id=run_id,
            result_json=result,  # Already contains run_id
        ))

        db.commit()

        logger.info(f"Analysis persisted: run_id={run_id}")

        return run_id

    except Exception as e:
        logger.warning(f"Persistence failed: {e}")
        db.rollback()
        raise

    finally:
        db.close()

