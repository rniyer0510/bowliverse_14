from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json

from app.common.logger import get_logger
from app.io.loader import load_video

# Events
from app.workers.events.release_uah import detect_release_uah
from app.workers.events.ffc_bfc import detect_ffc_bfc

# Elbow (LOCKED – do not touch)
from app.workers.elbow.elbow_signal import compute_elbow_signal
from app.workers.elbow.elbow_legality import evaluate_elbow_legality

# Action (LOCKED)
from app.workers.action.action_classifier import classify_action

# Risk (v14 – cricket-native, baseball-style mechanics)
from app.workers.risk.risk_worker import run_risk_worker

# Interpretation
from app.interpretation.interpret_risks import interpret_risks

# Basics
from app.workers.efficiency.basic_coaching import analyze_basics

# Clinician
from app.clinician.interpreter import ClinicianInterpreter

# Persistence
from app.persistence.writer import write_analysis
from app.persistence.session import SessionLocal
from app.persistence.models import Player


# ------------------------------------------------------------
# App init
# ------------------------------------------------------------
app = FastAPI()
logger = get_logger(__name__)
clinician_engine = ClinicianInterpreter()

# ------------------------------------------------------------
# 🔹 Serve visual evidence
# ------------------------------------------------------------
VISUALS_DIR = "/tmp/actionlab_frames"

if os.path.isdir(VISUALS_DIR):
    app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")
    logger.info(f"Mounted visual evidence directory: {VISUALS_DIR}")
else:
    logger.warning(f"Visual evidence directory not found: {VISUALS_DIR}")


@app.post("/analyze")
def analyze(
    request: Request,
    file: UploadFile = File(...),
    player_id: str = Form(...),
    bowler_type: str = Form(None),
    actor: str = Form(None),
):
    """
    ActionLab V14 – Orchestrator
    Player-centric analysis (handedness resolved from Player profile)
    """
    logger.info("Analyze request received")

    analysis_run_id = f"analysis_{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------
    # Actor (strict)
    # ------------------------------------------------------------
    actor_obj = {}
    if actor is not None:
        try:
            parsed = json.loads(actor)
            if not isinstance(parsed, dict):
                raise ValueError
            actor_obj = parsed
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid actor JSON")

    # ------------------------------------------------------------
    # Resolve player + handedness
    # ------------------------------------------------------------
    db = SessionLocal()
    try:
        player = (
            db.query(Player)
            .filter(Player.player_id == player_id)
            .first()
        )
        if not player:
            raise HTTPException(
                status_code=404,
                detail="Player not found"
            )

        if not player.handedness:
            raise HTTPException(
                status_code=400,
                detail="Player handedness not set"
            )

        hand = player.handedness.upper()
        logger.info(f"[context] player={player_id} hand={hand}")

    finally:
        db.close()

    # ------------------------------------------------------------
    # Load video + pose
    # ------------------------------------------------------------
    video, pose_frames, _ = load_video(file)

    try:
        fps_val = float(video.get("fps") or 0.0)
    except Exception:
        fps_val = 0.0

    # ------------------------------------------------------------
    # Events: Release → UAH
    # ------------------------------------------------------------
    events = detect_release_uah(
        pose_frames=pose_frames,
        hand=hand,
        fps=fps_val,
    )

    # ------------------------------------------------------------
    # FFC/BFC
    # ------------------------------------------------------------
    release_frame = (events.get("release") or {}).get("frame")
    uah_frame = (events.get("uah") or {}).get("frame")

    if release_frame is not None:
        foot_events = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand=hand,
            release_frame=release_frame,
            uah_frame=uah_frame,
            fps=fps_val,
        )
        if foot_events:
            events.update(foot_events)
    else:
        logger.error("Release frame missing — cannot run FFC/BFC")

    bfc_frame = (events.get("bfc") or {}).get("frame")
    ffc_frame = (events.get("ffc") or {}).get("frame")

    # ------------------------------------------------------------
    # Action
    # ------------------------------------------------------------
    action = classify_action(
        pose_frames=pose_frames,
        hand=hand,
        bfc_frame=bfc_frame,
        ffc_frame=ffc_frame,
    )

    # ------------------------------------------------------------
    # Risks
    # ------------------------------------------------------------
    risks = run_risk_worker(
        pose_frames=pose_frames,
        video=video,
        events=events,
        action=action,
        run_id=analysis_run_id,
    )

    # ------------------------------------------------------------
    # Basics
    # ------------------------------------------------------------
    basics = analyze_basics(
        pose_frames=pose_frames,
        hand=hand,
        events=events,
        action=action,
    )

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------
    interpretation = interpret_risks(risks)

    # ------------------------------------------------------------
    # Elbow (LOCKED)
    # ------------------------------------------------------------
    elbow_signal = compute_elbow_signal(
        pose_frames=pose_frames,
        hand=hand,
    )

    elbow = evaluate_elbow_legality(
        elbow_signal=elbow_signal,
        events=events,
        fps=fps_val,
        pose_frames=pose_frames,
    )

    # ------------------------------------------------------------
    # Clinician
    # ------------------------------------------------------------
    clinician = clinician_engine.build(
        elbow=elbow,
        risks=risks,
        interpretation=interpretation,
    )

    # ------------------------------------------------------------
    # Response
    # ------------------------------------------------------------
    result = {
        "schema": "actionlab.v14",
        "input": {
            "player_id": player_id,
            "hand": hand,
        },
        "video": {
            "fps": video.get("fps"),
            "total_frames": video.get("total_frames"),
            "file_path": video.get("path"),
        },
        "events": events,
        "elbow": elbow,
        "action": action,
        "risks": risks,
        "basics": basics,
        "interpretation": interpretation,
        "clinician": clinician,
    }

    # ------------------------------------------------------------
    # Persistence (best-effort)
    # ------------------------------------------------------------
    try:
        write_analysis(
            result=result,
            file_path=video.get("path"),
            bowler_type=bowler_type,
            actor=actor_obj,
        )
    except Exception as e:
        logger.warning(f"Persistence skipped: {e}")

    return result


# ------------------------------------------------------------
# Read  & Write APIs
# ------------------------------------------------------------
# ------------------------------------------------------------
# Persistence APIs
# ------------------------------------------------------------
from app.persistence.read_api import router as persistence_read_router
from app.persistence.write_api import router as persistence_write_router
from app.persistence.account_api import router as account_router

app.include_router(persistence_read_router)
app.include_router(persistence_write_router)
app.include_router(account_router)

