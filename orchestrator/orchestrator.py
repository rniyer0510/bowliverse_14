from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import os
import time
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

env_name = os.getenv("ACTIONLAB_ENV", "").lower()
default_auto_create = "false" if env_name in {"prod", "production"} else "true"
AUTO_CREATE_SCHEMA = (
    os.getenv("ACTIONLAB_AUTO_CREATE_SCHEMA", default_auto_create).lower() == "true"
)
if AUTO_CREATE_SCHEMA:
    Base.metadata.create_all(bind=engine)

logger = get_logger(__name__)
clinician_engine = ClinicianInterpreter()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    client_ip = request.client.host if request.client else "-"
    path = request.url.path

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            f"[request] request_id={request_id} method={request.method} "
            f"path={path} status=500 duration_ms={duration_ms:.1f} "
            f"client_ip={client_ip}"
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id

    # Avoid log flooding from static visual evidence fetches.
    if not path.startswith("/visuals/"):
        logger.info(
            f"[request] request_id={request_id} method={request.method} "
            f"path={path} status={response.status_code} "
            f"duration_ms={duration_ms:.1f} client_ip={client_ip}"
        )

    return response


def _delete_temp_video_safely(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        # Best effort cleanup only.
        pass


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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    player_id: str = Form(...),
    bowler_type: str = Form(None),
    age_group: str = Form(None),
    season: int = Form(None),
    actor: str = Form(None),  # ignored for identity
    current_account=Depends(get_current_account),
):
    """
    ActionLab V14 – Full Pipeline
    Auth Protected + Player Scoped
    """

    run_id = str(uuid.uuid4())
    request_id = getattr(request.state, "request_id", "-")

    logger.info(
        f"[analyze:start] request_id={request_id} run_id={run_id} "
        f"account_id={current_account.account_id} player_id={player_id}"
    )

    content_type = (file.content_type or "").lower()
    # Keep this permissive for low-end/older devices that often send wrong MIME.
    # Block only clearly non-binary content types.
    if content_type.startswith("text/") or content_type in {
        "application/json",
        "application/xml",
    }:
        logger.warning(
            f"[analyze:rejected] request_id={request_id} run_id={run_id} "
            f"reason=unsupported_media_type content_type={content_type}"
        )
        raise HTTPException(status_code=415, detail="Unsupported media type")

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
        effective_age_group = player.age_group
        effective_season = player.season

        if age_group is not None:
            normalized_age_group = age_group.strip().upper()
            allowed_age_groups = {"U10", "U14", "U16", "U19", "SENIOR"}
            if normalized_age_group not in allowed_age_groups:
                raise HTTPException(status_code=400, detail="Invalid age_group")
            effective_age_group = normalized_age_group

        if season is not None:
            current_year = datetime.utcnow().year
            if season < current_year - 1 or season > current_year + 1:
                raise HTTPException(
                    status_code=400,
                    detail="Season must be within ±1 of current year",
                )
            effective_season = season

    finally:
        db.close()

    video_temp_path = None
    cleanup_scheduled = False

    try:
        # ------------------------------------------------------------
        # Load Video
        # ------------------------------------------------------------
        try:
            video, pose_frames, _ = load_video(file)
        except RuntimeError:
            raise HTTPException(
                status_code=400,
                detail="Could not read uploaded video. Please upload a playable video file.",
            )

        video_temp_path = video.get("path")

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
                "age_group": effective_age_group,
                "season": effective_season,
            },
            "video": {
                "fps": video.get("fps"),
                "total_frames": video.get("total_frames"),
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
                age_group=effective_age_group,
                season=effective_season,
            )
        except Exception as e:
            logger.error(
                f"[analyze:persistence_failed] request_id={request_id} "
                f"run_id={run_id} error={e}"
            )
            raise HTTPException(status_code=500, detail="Analysis persistence failed")

        logger.info(
            f"[analyze:success] request_id={request_id} run_id={run_id} "
            f"player_id={player_id} risks={len(risks)}"
        )

        if video_temp_path:
            background_tasks.add_task(_delete_temp_video_safely, video_temp_path)
            cleanup_scheduled = True

        return result
    finally:
        if video_temp_path and not cleanup_scheduled:
            _delete_temp_video_safely(video_temp_path)


# ------------------------------------------------------------
# Routers
# ------------------------------------------------------------
app.include_router(auth_router)
app.include_router(persistence_read_router)
app.include_router(persistence_write_router)
app.include_router(account_router)
