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

# Interpretation (KEYED; no English strings)
from app.interpretation.interpret_risks import interpret_risks

# Basic coaching diagnostics (NON-RISK)
from app.workers.efficiency.basic_coaching import analyze_basics

# Clinician (YAML-driven explanations)
from app.clinician.interpreter import ClinicianInterpreter

# Persistence (best-effort, side-effect only)
from app.persistence.writer import write_analysis


# ------------------------------------------------------------
# App init
# ------------------------------------------------------------
app = FastAPI()
logger = get_logger(__name__)
clinician_engine = ClinicianInterpreter()

# ------------------------------------------------------------
# 🔹 Serve visual evidence over HTTP
# ------------------------------------------------------------
VISUALS_DIR = "/tmp/actionlab_frames"

if os.path.isdir(VISUALS_DIR):
    app.mount(
        "/visuals",
        StaticFiles(directory=VISUALS_DIR),
        name="visuals",
    )
    logger.info(f"Mounted visual evidence directory: {VISUALS_DIR}")
else:
    logger.warning(f"Visual evidence directory not found: {VISUALS_DIR}")


@app.post("/analyze")
def analyze(
    request: Request,
    file: UploadFile = File(...),
    hand: str = Form(...),
    bowler_type: str = Form(None),  # retained for backward compatibility
    actor: str = Form(None),        # actor JSON string
):
    """
    ActionLab V14 – Orchestrator
    """
    logger.info("Analyze request received")

    # ------------------------------------------------------------
    # ✅ UNIQUE ANALYSIS RUN ID (PER REQUEST)
    # ------------------------------------------------------------
    analysis_run_id = f"analysis_{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------
    # ✅ ACTOR (STRICT PARSING — NO SILENT FALLBACK)
    # ------------------------------------------------------------
    actor_obj = {}

    if actor is not None:
        try:
            parsed = json.loads(actor)
        except Exception as e:
            logger.error(f"[actor] invalid JSON: {actor}")
            raise HTTPException(
                status_code=400,
                detail="Invalid actor JSON"
            )

        if not isinstance(parsed, dict):
            logger.error(f"[actor] must be a JSON object: {parsed}")
            raise HTTPException(
                status_code=400,
                detail="actor must be a JSON object"
            )

        actor_obj = parsed
        logger.info(f"[actor] parsed successfully: {actor_obj}")

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
    # FFC/BFC (STRICTLY anchored off RELEASE)
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
        logger.error("[Orchestrator] Release frame missing — cannot run FFC/BFC")

    bfc_frame = (events.get("bfc") or {}).get("frame")
    ffc_frame = (events.get("ffc") or {}).get("frame")

    # ------------------------------------------------------------
    # Action classification
    # ------------------------------------------------------------
    action = classify_action(
        pose_frames=pose_frames,
        hand=hand,
        bfc_frame=bfc_frame,
        ffc_frame=ffc_frame,
    )

    # ------------------------------------------------------------
    # Risk computation (EVENT-DRIVEN VISUALS)
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
    # Clinician layer
    # ------------------------------------------------------------
    clinician = clinician_engine.build(
        elbow=elbow,
        risks=risks,
        interpretation=interpretation,
    )

    # ------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------
    result = {
        "schema": "actionlab.v14",
        "input": {"hand": hand},
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
    # Persistence (best-effort, actor-aware)
    # ------------------------------------------------------------
    try:
        write_analysis(
            result=result,
            file_path=video.get("path"),
            hand=hand,
            bowler_type=bowler_type,
            actor=actor_obj,
        )
    except Exception as e:
        logger.warning(f"Persistence skipped: {e}")

    return result

# ------------------------------------------------------------
# 📖 Read-only persistence APIs (Phase-I)
# ------------------------------------------------------------
from app.persistence.read_api import router as persistence_read_router
app.include_router(persistence_read_router)
