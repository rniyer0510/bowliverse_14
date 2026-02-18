from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
import os
import uuid

from app.common.logger import get_logger
from app.common.auth import get_current_account
from app.io.loader import load_video

# Events
from app.workers.events.release_uah import detect_release_uah
from app.workers.events.ffc_bfc import detect_ffc_bfc

# Elbow
from app.workers.elbow.elbow_signal import compute_elbow_signal
from app.workers.elbow.elbow_legality import evaluate_elbow_legality

# Action
from app.workers.action.action_classifier import classify_action

# Risk
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
from app.persistence.models import Player, AccountPlayerLink

# Routers
from app.persistence.read_api import router as persistence_read_router
from app.persistence.write_api import router as persistence_write_router
from app.persistence.account_api import router as account_router
from app.auth_routes import router as auth_router


# ------------------------------------------------------------
# App Init
# ------------------------------------------------------------
app = FastAPI()

from app.persistence.models import Base
from app.persistence.session import engine

Base.metadata.create_all(bind=engine)

logger = get_logger(__name__)
clinician_engine = ClinicianInterpreter()


# ------------------------------------------------------------
# Serve Visual Evidence
# ------------------------------------------------------------
VISUALS_DIR = "/tmp/actionlab_frames"

if os.path.isdir(VISUALS_DIR):
    app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")
    logger.info(f"Mounted visual evidence directory: {VISUALS_DIR}")
else:
    logger.warning(f"Visual evidence directory not found: {VISUALS_DIR}")


# ------------------------------------------------------------
# Analyze Endpoint (Protected + Account Scoped)
# ------------------------------------------------------------
@app.post("/analyze")
def analyze(
    request: Request,
    file: UploadFile = File(...),
    player_id: str = Form(...),
    bowler_type: str = Form(None),
    actor: str = Form(None),  # ignored for identity
    current_account=Depends(get_current_account),
):
    """
    ActionLab V14 – Full Pipeline
    Auth Protected + Player Scoped
    """

    run_id = str(uuid.uuid4())

    # ------------------------------------------------------------
    # Secure Actor Injection
    # ------------------------------------------------------------
    actor_obj = {
        "account_id": str(current_account.account_id),
        "role": current_account.role,
    }

    # ------------------------------------------------------------
    # Enforce Player Ownership
    # ------------------------------------------------------------
    db = SessionLocal()
    try:
        # Step 1: Check link (ownership)
        link = (
            db.query(AccountPlayerLink)
            .filter(
                AccountPlayerLink.account_id == current_account.account_id,
                AccountPlayerLink.player_id == player_id,
            )
            .first()
        )

        if not link:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this player",
            )

        # Step 2: Fetch player
        player = (
            db.query(Player)
            .filter(Player.player_id == player_id)
            .first()
        )

        if not player:
            raise HTTPException(
                status_code=404,
                detail="Player not found",
            )

        if not player.handedness:
            raise HTTPException(
                status_code=400,
                detail="Player handedness not set",
            )

        hand = player.handedness.upper()

    finally:
        db.close()

    # ------------------------------------------------------------
    # Load Video
    # ------------------------------------------------------------
    video, pose_frames, _ = load_video(file)

    try:
        fps_val = float(video.get("fps") or 0.0)
    except Exception:
        fps_val = 0.0

    # ------------------------------------------------------------
    # Events
    # ------------------------------------------------------------
    events = detect_release_uah(
        pose_frames=pose_frames,
        hand=hand,
        fps=fps_val,
    )

    # ------------------------------------------------------------
    # FFC / BFC
    # ------------------------------------------------------------
    release_frame = (events.get("release") or {}).get("frame")
    delivery_window = events.get("delivery_window")

    if release_frame is not None and delivery_window is not None:
        foot_events = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand=hand,
            release_frame=release_frame,
            delivery_window=tuple(delivery_window),
        )
        if foot_events:
            events.update(foot_events)

    bfc_frame = (events.get("bfc") or {}).get("frame")
    ffc_frame = (events.get("ffc") or {}).get("frame")

    # ------------------------------------------------------------
    # Action Classification
    # ------------------------------------------------------------
    action = classify_action(
        pose_frames=pose_frames,
        hand=hand,
        bfc_frame=bfc_frame,
        ffc_frame=ffc_frame,
    )

    # ------------------------------------------------------------
    # Risk Worker
    # ------------------------------------------------------------
    risks = run_risk_worker(
        pose_frames=pose_frames,
        video=video,
        events=events,
        action=action,
        run_id=run_id,
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
    # Elbow
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
    # Clinician Layer
    # ------------------------------------------------------------
    clinician = clinician_engine.build(
        elbow=elbow,
        risks=risks,
        interpretation=interpretation,
    )

    # ------------------------------------------------------------
    # Build Response
    # ------------------------------------------------------------
    result = {
        "run_id": run_id,
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
    # Persist
    # ------------------------------------------------------------
    try:
        write_analysis(
            result=result,
            run_id=run_id,
            file_path=video.get("path"),
            bowler_type=bowler_type,
            actor=actor_obj,
        )
    except Exception as e:
        logger.warning(f"Persistence failed: {e}")

    return result


# ------------------------------------------------------------
# Routers
# ------------------------------------------------------------
app.include_router(auth_router)
app.include_router(persistence_read_router)
app.include_router(persistence_write_router)
app.include_router(account_router)

