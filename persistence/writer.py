from app.persistence.session import SessionLocal
from app.persistence.models import (
    AnalysisRun,
    AnalysisResultRaw,
    EventAnchor,
    BiomechSignal,
    RiskMeasurement,
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


def write_analysis(result: dict, **kwargs):
    db = SessionLocal()

    try:
        actor = kwargs.get("actor", {}) or {}

        account = resolve_account(db, actor)
        player_id = resolve_player(db, account, actor)

        run = AnalysisRun(
            player_id=player_id,
            schema_version=result.get("schema"),
            handedness=result.get("input", {}).get("hand"),
            fps=result.get("video", {}).get("fps"),
            total_frames=result.get("video", {}).get("total_frames"),
        )
        db.add(run)
        db.flush()

        # Events
        for key, event_type in EVENT_MAP.items():
            ev = result.get("events", {}).get(key)
            if not ev or ev.get("frame") is None:
                continue

            db.add(EventAnchor(
                run_id=run.run_id,
                event_type=event_type,
                frame=ev["frame"],
                confidence=ev.get("confidence"),
                method=ev.get("method"),
            ))

        # Biomech signals (minimal, expandable)
        elbow = result.get("elbow", {})
        if elbow:
            db.add(BiomechSignal(
                run_id=run.run_id,
                signal_key="elbow_confidence",
                value=elbow.get("confidence"),
                units="ratio",
                baseline_eligible=True,
            ))

        action = result.get("action", {})
        if action:
            db.add(BiomechSignal(
                run_id=run.run_id,
                signal_key="action_confidence",
                value=action.get("confidence"),
                units="ratio",
                baseline_eligible=True,
            ))

        knee = result.get("basics", {}).get("knee_brace_proxy", {})
        dbg = knee.get("debug", {}) or {}
        if dbg.get("pelvis_drop") is not None:
            db.add(BiomechSignal(
                run_id=run.run_id,
                signal_key="knee_brace_pelvis_drop",
                value=dbg["pelvis_drop"],
                units="normalized",
                confidence=knee.get("confidence"),
                event_anchor="FFC",
                baseline_eligible=True,
            ))

        # Risks
        for r in result.get("risks", []) or []:
            db.add(RiskMeasurement(
                run_id=run.run_id,
                risk_id=r["risk_id"],
                signal_strength=r.get("signal_strength"),
                confidence=r.get("confidence"),
                anchor_event=r.get("visual", {}).get("anchor"),
                window_start=r.get("visual_window", {}).get("start"),
                window_end=r.get("visual_window", {}).get("end"),
            ))

        db.add(AnalysisResultRaw(run_id=run.run_id, result_json=result))
        db.commit()

    except Exception as e:
        logger.warning(f"Persistence skipped: {e}")
        db.rollback()
    finally:
        db.close()
