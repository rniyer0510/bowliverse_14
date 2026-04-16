from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import os
import time
import uuid
from typing import Optional, Tuple

from app.common.logger import get_logger
from app.common.auth import get_current_account
from app.io.loader import load_video
from app.workers.screening.video_screen import run_preanalysis_screen
from app.workers.speed.release_speed import estimate_release_speed
from app.workers.render.coach_video_renderer import render_skeleton_video, RENDER_DIR
from app.workers.release_shape import build_release_shape_skeleton

# Events
from app.workers.events.release_uah import detect_release_uah
from app.workers.events.ffc_bfc import detect_ffc_bfc
from app.workers.events.event_confidence import chain_quality

# Elbow
from app.workers.elbow.compute_elbow_signal import compute_elbow_signal
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


def _compact_header(value: str, limit: int = 120) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _platform_hint(request: Request) -> str:
    explicit = (request.headers.get("x-client-platform") or "").strip().lower()
    if explicit:
        return explicit
    user_agent = (request.headers.get("user-agent") or "").lower()
    if "iphone" in user_agent or "ios" in user_agent:
        return "ios"
    if "android" in user_agent:
        return "android"
    return "unknown"


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


def _reject_with_code(
    *,
    request_id: str,
    run_id: str,
    code: str,
    detail: str,
    extra: str = "",
) -> None:
    extra_suffix = f" {extra}" if extra else ""
    logger.warning(
        f"[analyze:rejected] request_id={request_id} run_id={run_id} "
        f"reason={code}{extra_suffix}"
    )
    raise HTTPException(
        status_code=400,
        detail={
            "code": code,
            "message": detail,
        },
    )


def _reject_for_screening_failure(
    *,
    request_id: str,
    run_id: str,
    screening: dict,
) -> None:
    issues = screening.get("blocking_issues") or []
    issue = issues[0] if issues else {}
    code = str(issue.get("code") or "screening_failed")
    detail = str(issue.get("detail") or "Video failed pre-analysis screening.")
    _reject_with_code(
        request_id=request_id,
        run_id=run_id,
        code=code,
        detail=detail,
    )


def _gate_speed_estimate(
    *,
    estimated_release_speed: dict,
    event_chain: dict,
    events: dict,
) -> dict:
    if not estimated_release_speed.get("available"):
        return estimated_release_speed

    ordered = bool((event_chain or {}).get("ordered"))
    chain_quality_score = float((event_chain or {}).get("quality") or 0.0)
    release_confidence = float(((events or {}).get("release") or {}).get("confidence") or 0.0)
    speed_confidence = float(estimated_release_speed.get("confidence") or 0.0)

    weak_anchor_names = []
    weak_fallback_methods = {"ultimate_fallback", "single_foot_fallback", "no_foot_data_fallback"}
    for name in ("ffc", "bfc"):
        event = (events or {}).get(name) or {}
        method = str(event.get("method") or "")
        confidence = float(event.get("confidence") or 0.0)
        if method in weak_fallback_methods and confidence <= 0.20:
            weak_anchor_names.append(name)

    if not ordered:
        return {
            **estimated_release_speed,
            "available": False,
            "display_policy": "suppress",
            "display": None,
            "value_kph": None,
            "confidence": 0.0,
            "reason": "event_chain_unordered",
        }

    can_show_low_confidence = (
        ordered
        and release_confidence >= 0.50
        and speed_confidence >= 0.50
        and chain_quality_score >= 0.20
    )

    if chain_quality_score < 0.35 and can_show_low_confidence:
        return {
            **estimated_release_speed,
            "available": True,
            "display_policy": "show_low_confidence",
            "reason": "low_event_chain_quality",
        }

    if chain_quality_score < 0.35:
        return {
            **estimated_release_speed,
            "available": False,
            "display_policy": "suppress",
            "display": None,
            "value_kph": None,
            "confidence": 0.0,
            "reason": "low_event_chain_quality",
        }

    if weak_anchor_names and can_show_low_confidence:
        return {
            **estimated_release_speed,
            "available": True,
            "display_policy": "show_low_confidence",
            "reason": f"weak_{'_'.join(weak_anchor_names)}_anchor",
        }

    if weak_anchor_names:
        return {
            **estimated_release_speed,
            "available": False,
            "display_policy": "suppress",
            "display": None,
            "value_kph": None,
            "confidence": 0.0,
            "reason": f"weak_{'_'.join(weak_anchor_names)}_anchor",
        }

    return estimated_release_speed


def _walkthrough_render_window(
    *,
    events: dict,
    total_frames: int,
) -> Tuple[int, Optional[int]]:
    bfc_frame = (events.get("bfc") or {}).get("frame")
    ffc_frame = (events.get("ffc") or {}).get("frame")
    release_frame = (events.get("release") or {}).get("frame")

    anchors = [
        int(frame)
        for frame in (bfc_frame, ffc_frame, release_frame)
        if isinstance(frame, int) and frame >= 0
    ]
    if not anchors:
        fallback_end = min(total_frames, 180) if total_frames else 180
        return 0, fallback_end

    start = max(0, min(anchors) - 30)
    end = min(total_frames, max(anchors) + 28) if total_frames else max(anchors) + 28
    if end <= start:
        return start, None
    return start, end


def _build_walkthrough_render(
    *,
    run_id: str,
    video: dict,
    pose_frames: list,
    events: dict,
    hand: str,
    action: dict,
    elbow: dict,
    risks: list,
    estimated_release_speed: dict,
    report_story: Optional[dict],
) -> dict:
    video_path = video.get("path")
    if not video_path or not os.path.exists(video_path):
        return {"available": False, "reason": "missing_video_path"}

    total_frames = int(video.get("total_frames") or len(pose_frames) or 0)
    start_frame, end_frame = _walkthrough_render_window(
        events=events,
        total_frames=total_frames,
    )
    output_path = os.path.join(RENDERS_DIR, f"{run_id}_walkthrough.mp4")

    try:
        render_result = render_skeleton_video(
            video_path=video_path,
            pose_frames=pose_frames,
            events=events,
            hand=hand,
            action=action,
            elbow=elbow,
            risks=risks,
            estimated_release_speed=estimated_release_speed,
            report_story=report_story,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    except Exception as exc:
        logger.warning(
            f"[analyze:walkthrough_render_failed] run_id={run_id} error={exc}"
        )
        return {"available": False, "reason": "render_failed"}

    if not render_result.get("available"):
        return {
            "available": False,
            "reason": render_result.get("reason") or "render_unavailable",
        }

    return {
        **render_result,
        "renderer_version": "coach_video_renderer_v1",
        "artifact_type": "walkthrough_mp4",
        "url": f"/renders/{os.path.basename(output_path)}",
    }


# ------------------------------------------------------------
# Serve Visual Evidence
# ------------------------------------------------------------
VISUALS_DIR = "/tmp/actionlab_frames"
RENDERS_DIR = RENDER_DIR

if os.path.isdir(VISUALS_DIR):
    app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")
    logger.info(f"Mounted visual evidence directory: {VISUALS_DIR}")
else:
    logger.warning(f"Visual evidence directory not found: {VISUALS_DIR}")

os.makedirs(RENDERS_DIR, exist_ok=True)
app.mount("/renders", StaticFiles(directory=RENDERS_DIR), name="renders")
logger.info(f"Mounted walkthrough render directory: {RENDERS_DIR}")


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
        f"account_id={current_account.account_id} player_id={player_id} "
        f"platform={_platform_hint(request)} "
        f"content_type={file.content_type or '-'} "
        f"filename={file.filename or '-'} "
        f"user_agent={_compact_header(request.headers.get('user-agent') or '-')}"
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

    actor_obj = {
        "account_id": str(current_account.account_id),
        "role": current_account.role,
    }

    try:
        player_uuid = uuid.UUID(player_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid player_id format",
        )

    db = SessionLocal()
    try:
        link = (
            db.query(AccountPlayerLink)
            .filter(
                AccountPlayerLink.account_id == current_account.account_id,
                AccountPlayerLink.player_id == player_uuid,
            )
            .first()
        )

        if not link:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this player",
            )

        player = (
            db.query(Player)
            .filter(Player.player_id == player_uuid)
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
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "invalid_age_group",
                        "message": "Invalid age_group",
                    },
                )
            effective_age_group = normalized_age_group

        if season is not None:
            current_year = datetime.utcnow().year
            if season < current_year - 1 or season > current_year + 1:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "invalid_season",
                        "message": "Season must be within ±1 of current year",
                    },
                )
            effective_season = season

    finally:
        db.close()

    video_temp_path = None
    cleanup_scheduled = False

    try:
        try:
            video, pose_frames, _ = load_video(file)
        except RuntimeError:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_video",
                    "message": (
                        "Could not read uploaded video. Please upload a playable "
                        "video file."
                    ),
                },
            )

        video_temp_path = video.get("path")
        temp_file_size = None
        if video_temp_path and os.path.exists(video_temp_path):
            try:
                temp_file_size = os.path.getsize(video_temp_path)
            except Exception:
                temp_file_size = None

        try:
            fps_val = float(video.get("fps") or 0.0)
        except Exception:
            fps_val = 0.0

        logger.info(
            f"[analyze:video] request_id={request_id} run_id={run_id} "
            f"platform={_platform_hint(request)} "
            f"fps={fps_val:.3f} width={video.get('width')} height={video.get('height')} "
            f"frames={video.get('total_frames')} "
            f"file_size_bytes={temp_file_size if temp_file_size is not None else '-'}"
        )

        screening = run_preanalysis_screen(
            video=video,
            pose_frames=pose_frames,
            hand=hand,
        )
        if not screening.get("passed"):
            _reject_for_screening_failure(
                request_id=request_id,
                run_id=run_id,
                screening=screening,
            )

        events = detect_release_uah(
            pose_frames=pose_frames,
            hand=hand,
            fps=fps_val,
        )

        release_frame = (events.get("release") or {}).get("frame")
        delivery_window = events.get("delivery_window")

        if release_frame is not None and delivery_window is not None:
            foot_events = detect_ffc_bfc(
                pose_frames=pose_frames,
                hand=hand,
                release_frame=release_frame,
                delivery_window=tuple(delivery_window),
                fps=fps_val,
            )
            if foot_events:
                events.update(foot_events)

        bfc_frame = (events.get("bfc") or {}).get("frame")
        ffc_frame = (events.get("ffc") or {}).get("frame")
        uah_frame = (events.get("uah") or {}).get("frame")
        release_confidence = float((events.get("release") or {}).get("confidence") or 0.0)
        uah_confidence = float((events.get("uah") or {}).get("confidence") or 0.0)
        ffc_confidence = float((events.get("ffc") or {}).get("confidence") or 0.0)
        bfc_confidence = float((events.get("bfc") or {}).get("confidence") or 0.0)
        events["event_chain"] = chain_quality(
            bfc_frame=bfc_frame,
            ffc_frame=ffc_frame,
            uah_frame=uah_frame,
            release_frame=release_frame,
            bfc_confidence=bfc_confidence,
            ffc_confidence=ffc_confidence,
            uah_confidence=uah_confidence,
            release_confidence=release_confidence,
        )

        action = classify_action(
            pose_frames=pose_frames,
            hand=hand,
            bfc_frame=bfc_frame,
            ffc_frame=ffc_frame,
        )

        risks = run_risk_worker(
            pose_frames=pose_frames,
            video=video,
            events=events,
            action=action,
            run_id=run_id,
        )

        basics = analyze_basics(
            pose_frames=pose_frames,
            hand=hand,
            events=events,
            action=action,
        )

        interpretation = interpret_risks(risks)

        elbow_signal = compute_elbow_signal(
            pose_frames=pose_frames,
            hand=hand,
        )

        elbow = evaluate_elbow_legality(
            elbow_signal=elbow_signal,
            events=events,
            fps=fps_val,
            pose_frames=pose_frames,
            hand=hand,
        )

        estimated_release_speed = estimate_release_speed(
            pose_frames=pose_frames,
            events=events,
            video=video,
            hand=hand,
        )
        estimated_release_speed = _gate_speed_estimate(
            estimated_release_speed=estimated_release_speed,
            event_chain=events.get("event_chain") or {},
            events=events,
        )
        speed_debug = estimated_release_speed.get("debug") or {}
        logger.info(
            f"[analyze:speed] request_id={request_id} run_id={run_id} "
            f"platform={_platform_hint(request)} "
            f"release_frame={(events.get('release') or {}).get('frame')} "
            f"peak_frame={(events.get('peak') or {}).get('frame')} "
            f"uah_frame={(events.get('uah') or {}).get('frame')} "
            f"display={estimated_release_speed.get('display') or '-'} "
            f"value_kph={estimated_release_speed.get('value_kph') if estimated_release_speed.get('available') else '-'} "
            f"confidence={estimated_release_speed.get('confidence')} "
            f"method={estimated_release_speed.get('method')} "
            f"reason={estimated_release_speed.get('reason') or '-'} "
            f"wrist_arm_ratio={speed_debug.get('wrist_arm_ratio', '-')} "
            f"shoulder_body_ratio={speed_debug.get('shoulder_body_ratio', '-')} "
            f"pelvis_body_ratio={speed_debug.get('pelvis_body_ratio', '-')} "
            f"elbow_extension_velocity={speed_debug.get('elbow_extension_velocity_deg_per_sec', '-')} "
            f"arm_length_cv={speed_debug.get('arm_length_cv', '-')} "
            f"wrist_window_cv={speed_debug.get('wrist_window_cv', '-')}"
        )

        clinician = clinician_engine.build(
            elbow=elbow,
            risks=risks,
            interpretation=interpretation,
            action=action,
        )
        release_shape = build_release_shape_skeleton(
            pose_frames=pose_frames,
            events=events,
            hand=hand,
            action=action,
        )

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
            "estimated_release_speed": estimated_release_speed,
            "action": action,
            "release_shape": release_shape,
            "risks": risks,
            "basics": basics,
            "interpretation": interpretation,
            "clinician": clinician,
        }
        result["visual_walkthrough"] = _build_walkthrough_render(
            run_id=run_id,
            video=video,
            pose_frames=pose_frames,
            events=events,
            hand=hand,
            action=action,
            elbow=elbow,
            risks=risks,
            estimated_release_speed=estimated_release_speed,
            report_story=(clinician or {}).get("report_story_v1"),
        )

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
        except Exception:
            logger.exception(
                f"[analyze:persistence_failed] request_id={request_id} "
                f"run_id={run_id}"
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


app.include_router(auth_router)
app.include_router(persistence_read_router)
app.include_router(persistence_write_router)
app.include_router(account_router)
