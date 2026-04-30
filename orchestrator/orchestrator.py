from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.common.logger import get_logger
from app.common.auth import get_current_account
from app.common.stable_cache import get_player_profile, remember_player_profile
from app.io.loader import cleanup_stale_temp_uploads, load_video
from app.workers.screening.video_screen import run_preanalysis_screen
from app.workers.speed.release_speed import estimate_release_speed
from app.workers.render.coach_video_renderer import render_skeleton_video, RENDER_DIR
from app.workers.render.render_storage import (
    cleanup_old_renders,
    download_render_artifact,
    normalize_render_filename,
    render_retention_days,
    upload_render_artifact,
)

# Events
from app.workers.events.release_uah import detect_release_uah
from app.workers.events.ffc_bfc import detect_ffc_bfc
from app.workers.events.event_confidence import chain_quality, annotate_detection_contract
from app.workers.events.signal_cache import build_signal_cache

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
from app.clinician.deterministic_expert import DeterministicExpertSystem
from app.clinician.loader import validate_known_yaml_files
from app.clinician.knowledge_pack import validate_default_knowledge_pack
from app.clinician.interpreter import ClinicianInterpreter

# Persistence
from app.persistence.learning_cases import (
    build_learning_case_event,
    write_learning_case,
)
from app.persistence.notifications import (
    persist_analysis_completed_notification_best_effort,
)
from app.persistence.prescription_followups import sync_prescription_followups_for_run
from app.persistence.writer import write_analysis
from app.persistence.session import SessionLocal
from app.persistence.models import (
    Player,
    AccountPlayerLink,
    AnalysisRun,
    AnalysisResultRaw,
)

# Routers
from app.persistence.read_api import router as persistence_read_router
from app.persistence.write_api import router as persistence_write_router
from app.persistence.account_api import router as account_router
from app.persistence.notification_api import router as notification_router
from app.auth_routes import router as auth_router


# ------------------------------------------------------------
# App Init
# ------------------------------------------------------------
app = FastAPI()

from app.persistence.models import Base
from app.persistence.session import engine

AUTO_CREATE_SCHEMA = (
    os.getenv("ACTIONLAB_AUTO_CREATE_SCHEMA", "false").lower() == "true"
)
if AUTO_CREATE_SCHEMA:
    logger = get_logger(__name__)
    logger.warning(
        "[schema] ACTIONLAB_AUTO_CREATE_SCHEMA=true; using create_all for local/bootstrap only",
    )
    Base.metadata.create_all(bind=engine)
logger = get_logger(__name__)
clinician_engine: Optional[ClinicianInterpreter] = None
deterministic_expert_engine: Optional[DeterministicExpertSystem] = None
RENDERS_DIR = RENDER_DIR

WALKTHROUGH_PAUSE_SECONDS = 5.0
WALKTHROUGH_SLOW_MOTION_FACTOR = 5.0
WALKTHROUGH_END_SUMMARY_SECONDS = 2.5
WALKTHROUGH_RENDERER_VERSION = "coach_video_renderer_v2_2026_04_24"

@app.on_event("startup")
def _startup_housekeeping() -> None:
    global clinician_engine, deterministic_expert_engine
    validate_known_yaml_files()
    validate_default_knowledge_pack()
    clinician_engine = ClinicianInterpreter()
    deterministic_expert_engine = DeterministicExpertSystem()
    render_cleanup = cleanup_old_renders(
        RENDER_DIR,
        retention_days=render_retention_days(),
    )
    logger.info(
        "[render_storage] active_dir=%s cleanup_scanned=%s cleanup_removed=%s retention_days=%s",
        RENDER_DIR,
        render_cleanup.get("scanned", 0),
        render_cleanup.get("removed", 0),
        render_retention_days(),
    )
    temp_cleanup = cleanup_stale_temp_uploads()
    logger.info(
        "[loader] temp_upload_cleanup_scanned=%s removed=%s",
        temp_cleanup.get("scanned", 0),
        temp_cleanup.get("removed", 0),
    )


def _compact_header(value: str, limit: int = 120) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _request_header(request: Request, name: str) -> str:
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    try:
        return str(headers.get(name) or "")
    except Exception:
        return ""


def _platform_hint(request: Request) -> str:
    explicit = _request_header(request, "x-client-platform").strip().lower()
    if explicit:
        return explicit
    user_agent = _request_header(request, "user-agent").lower()
    if "iphone" in user_agent or "ios" in user_agent:
        return "ios"
    if "android" in user_agent:
        return "android"
    return "unknown"


def _public_asset_url(path: str) -> str:
    asset_path = str(path or "").strip()
    if not asset_path:
        return asset_path
    if asset_path.startswith("http://") or asset_path.startswith("https://"):
        return asset_path
    if not asset_path.startswith("/"):
        asset_path = f"/{asset_path}"
    base_url = (os.getenv("ACTIONLAB_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        return asset_path
    return f"{base_url}{asset_path}"


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
    except Exception as exc:
        logger.warning("[cleanup:temp_video_failed] path=%s error=%s", path, exc)


def _persist_analysis_result(
    *,
    request_id: str,
    run_id: str,
    result: dict,
    video: dict,
    bowler_type: Optional[str],
    actor_obj: dict,
    effective_age_group: str,
    effective_season: int,
) -> dict:
    last_error = None
    for attempt in (1, 2):
        db = SessionLocal()
        try:
            write_analysis(
                db=db,
                result=result,
                run_id=run_id,
                file_path=video.get("path"),
                bowler_type=bowler_type,
                actor=actor_obj,
                age_group=effective_age_group,
                season=effective_season,
            )
            db.commit()
            if attempt > 1:
                logger.info(
                    "[analyze:persistence_recovered] request_id=%s run_id=%s attempt=%s",
                    request_id,
                    run_id,
                    attempt,
                )
            return {
                "persisted": True,
                "attempts": attempt,
            }
        except Exception as exc:
            db.rollback()
            last_error = exc
            logger.error(
                "[analyze:persistence_failed] request_id=%s run_id=%s attempt=%s error=%s",
                request_id,
                run_id,
                attempt,
                exc,
            )
        finally:
            db.close()

    warnings = list(result.get("warnings") or [])
    warnings.append(
        {
            "code": "analysis_not_persisted",
            "detail": "Analysis completed, but saving the result failed. Please retry or contact support if this keeps happening.",
        }
    )
    result["warnings"] = warnings
    result["persistence"] = {
        "persisted": False,
        "attempts": 2,
        "error": str(last_error) if last_error is not None else "unknown",
    }
    return result["persistence"]


def _get_clinician_engine() -> ClinicianInterpreter:
    global clinician_engine
    if clinician_engine is None:
        validate_known_yaml_files()
        validate_default_knowledge_pack()
        clinician_engine = ClinicianInterpreter()
    return clinician_engine


def _get_deterministic_expert_engine() -> DeterministicExpertSystem:
    global deterministic_expert_engine
    if deterministic_expert_engine is None:
        validate_default_knowledge_pack()
        deterministic_expert_engine = DeterministicExpertSystem()
    return deterministic_expert_engine


def _load_recent_expert_history(
    *,
    db,
    player_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    rows = (
        db.query(
            AnalysisRun.run_id,
            AnalysisRun.created_at,
            AnalysisResultRaw.result_json,
        )
        .join(AnalysisResultRaw, AnalysisResultRaw.run_id == AnalysisRun.run_id)
        .filter(AnalysisRun.player_id == player_id)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
        .all()
    )
    history: List[Dict[str, Any]] = []
    for run_id, created_at, result_json in rows:
        if not isinstance(result_json, dict):
            continue
        history.append(
            {
                "run_id": str(run_id),
                "created_at": created_at,
                "result_json": result_json,
            }
        )
    return history


def _persist_learning_case_best_effort(
    *,
    request_id: str,
    run_id: str,
    result: Dict[str, Any],
    account_id: Optional[str],
) -> None:
    try:
        event_payload = build_learning_case_event(
            result=result,
            account_id=account_id,
        )
        if not event_payload:
            logger.info(
                "[learning_case] request_id=%s run_id=%s skipped=no_gap_trigger",
                request_id,
                run_id,
            )
            return
        write_learning_case(event_payload=event_payload)
    except Exception as exc:
        logger.error(
            "[learning_case] request_id=%s run_id=%s persisted=false error=%s",
            request_id,
            run_id,
            exc,
        )


def _sync_prescription_followups_best_effort(
    *,
    request_id: str,
    run_id: str,
) -> None:
    try:
        summary = sync_prescription_followups_for_run(run_id=run_id)
        logger.info(
            "[prescription_followup] request_id=%s run_id=%s created=%s updated=%s non_response_cases=%s",
            request_id,
            run_id,
            summary.get("created", 0),
            summary.get("updated", 0),
            summary.get("non_response_cases", 0),
        )
    except Exception as exc:
        logger.error(
            "[prescription_followup] request_id=%s run_id=%s persisted=false error=%s",
            request_id,
            run_id,
            exc,
        )


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
    playback_mode: Optional[dict] = None,
) -> dict:
    if not estimated_release_speed.get("available"):
        return estimated_release_speed

    playback_mode = playback_mode or {}
    if str(playback_mode.get("mode") or "").strip().lower() == "likely_slow_motion":
        return {
            **estimated_release_speed,
            "available": False,
            "display_policy": "suppress",
            "display": None,
            "value_kph": None,
            "confidence": 0.0,
            "reason": "slow_motion_playback_detected",
        }

    ordered = bool((event_chain or {}).get("ordered"))
    chain_quality = float((event_chain or {}).get("quality") or 0.0)
    release_confidence = float(((events or {}).get("release") or {}).get("confidence") or 0.0)
    speed_confidence = float(estimated_release_speed.get("confidence") or 0.0)
    speed_debug = dict(estimated_release_speed.get("debug") or {})
    soft_release_recovery_mode = bool(speed_debug.get("soft_release_recovery_mode"))
    overall_wrist_visibility = float(speed_debug.get("overall_wrist_visibility") or 0.0)
    shoulder_body_ratio = float(speed_debug.get("shoulder_body_ratio") or 0.0)

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
        and chain_quality >= 0.20
    )

    weak_speed_evidence = (
        soft_release_recovery_mode
        and (
            overall_wrist_visibility < 0.20
            or shoulder_body_ratio > 0.80
        )
    )

    if weak_speed_evidence and can_show_low_confidence:
        return {
            **estimated_release_speed,
            "available": True,
            "display_policy": "show_low_confidence",
            "reason": "weak_release_speed_evidence",
        }

    if weak_speed_evidence:
        return {
            **estimated_release_speed,
            "available": False,
            "display_policy": "suppress",
            "display": None,
            "value_kph": None,
            "confidence": 0.0,
            "reason": "weak_release_speed_evidence",
        }

    if chain_quality < 0.35 and can_show_low_confidence:
        return {
            **estimated_release_speed,
            "available": True,
            "display_policy": "show_low_confidence",
            "reason": "low_event_chain_quality",
        }

    if chain_quality < 0.35:
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


def _reconcile_anchor_order(
    *,
    events: Dict[str, Any],
    fps: float,
) -> Dict[str, Any]:
    """
    Resolve tiny anchor-order inversions after all event detectors have run.

    On noisy or behind-camera clips, UAH and FFC can land within a frame or two
    of each other. If UAH is only slightly earlier than FFC, snap it forward to
    the FFC instant instead of letting the whole chain go unordered.
    """
    try:
        fps_f = max(1.0, float(fps or 30.0))
    except Exception:
        fps_f = 30.0

    tol = max(1, int(round(0.04 * fps_f)))  # about 40 ms
    uah = events.get("uah") or {}
    ffc = events.get("ffc") or {}
    uah_frame = uah.get("frame")
    ffc_frame = ffc.get("frame")

    if uah_frame is None or ffc_frame is None:
        return events

    try:
        uah_i = int(uah_frame)
        ffc_i = int(ffc_frame)
    except Exception:
        return events

    if uah_i < ffc_i and (ffc_i - uah_i) <= tol:
        reconciled = dict(uah)
        reconciled["frame"] = ffc_i
        reconciled["method"] = f"{uah.get('method') or 'uah'}_ffc_reconciled"
        reconciled["confidence"] = round(
            max(0.0, float(uah.get("confidence") or 0.0) - 0.05),
            2,
        )
        events["uah"] = reconciled

    return events


def _safe_frame(event: Optional[Dict[str, Any]]) -> Optional[int]:
    try:
        if not isinstance(event, dict):
            return None
        frame = event.get("frame")
        return None if frame is None else int(frame)
    except Exception:
        return None


def _safe_confidence(event: Optional[Dict[str, Any]]) -> float:
    try:
        if not isinstance(event, dict):
            return 0.0
        return float(event.get("confidence") or 0.0)
    except Exception:
        return 0.0


def _mean_visible_weight(raw_vis, weights, start: int, end: int, *, min_vis: float = 0.20) -> float:
    try:
        raw = raw_vis[start:end]
        wts = weights[start:end]
    except Exception:
        return 0.0
    mask = raw >= float(min_vis)
    if not np.any(mask):
        return 0.0
    return float(np.mean(wts[mask]))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _local_peak_frame(
    signal: Any,
    start: int,
    end: int,
    *,
    mode: str,
) -> Optional[int]:
    try:
        window = np.asarray(signal[start:end], dtype=float)
    except Exception:
        return None
    valid = np.isfinite(window)
    if not np.any(valid):
        return None
    if mode == "peak_min":
        filled = np.where(valid, window, np.inf)
        return int(start + int(np.argmin(filled)))
    filled = np.where(valid, window, -np.inf)
    return int(start + int(np.argmax(filled)))


def _anchor_spacing_score(events: Dict[str, Any]) -> float:
    bfc = _safe_frame(events.get("bfc"))
    ffc = _safe_frame(events.get("ffc"))
    uah = _safe_frame(events.get("uah"))
    release = _safe_frame(events.get("release"))
    if None in {bfc, ffc, uah, release}:
        return 0.0

    score = 1.0
    if not (bfc <= ffc <= uah <= release):
        score -= 0.70

    gaps = (
        (ffc - bfc, 2, 8, 14, 0.30),
        (uah - ffc, 1, 6, 10, 0.45),
        (release - uah, 1, 6, 10, 0.35),
    )
    for gap, ideal_min, ideal_max, hard_max, penalty in gaps:
        if gap < ideal_min:
            score -= penalty * (1.0 if gap <= 0 else 0.5)
        elif gap > hard_max:
            score -= penalty * 0.75
        elif gap > ideal_max:
            score -= penalty * 0.35

    return max(0.0, min(1.0, round(score, 3)))


def _detect_anchor_hypothesis(
    *,
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
    delivery_window: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    events = detect_release_uah(
        pose_frames=pose_frames,
        hand=hand,
        fps=fps,
        delivery_window=delivery_window,
    )
    release_frame = _safe_frame(events.get("release"))
    hypothesis_window = events.get("delivery_window")
    if release_frame is not None and hypothesis_window is not None:
        foot_events = detect_ffc_bfc(
            pose_frames=pose_frames,
            hand=hand,
            release_frame=release_frame,
            delivery_window=tuple(hypothesis_window),
            fps=fps,
        )
        if foot_events:
            events.update(foot_events)

    events = _reconcile_anchor_order(
        events=events,
        fps=fps,
    )
    events["event_chain"] = chain_quality(
        bfc_frame=_safe_frame(events.get("bfc")),
        ffc_frame=_safe_frame(events.get("ffc")),
        uah_frame=_safe_frame(events.get("uah")),
        release_frame=_safe_frame(events.get("release")),
        bfc_confidence=_safe_confidence(events.get("bfc")),
        ffc_confidence=_safe_confidence(events.get("ffc")),
        uah_confidence=_safe_confidence(events.get("uah")),
        release_confidence=_safe_confidence(events.get("release")),
    )
    return annotate_detection_contract(events)


def _score_anchor_hypothesis(
    *,
    pose_frames: List[Dict[str, Any]],
    hand: str,
    fps: float,
    events: Dict[str, Any],
    delivery_window: Optional[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    n_frames = len(pose_frames)
    cache = build_signal_cache(
        pose_frames=pose_frames,
        hand=hand,
        fps=fps,
    )
    release_window = (events.get("release") or {}).get("window") or []
    try:
        start = max(0, min(n_frames - 1, int(release_window[0])))
        end = max(start + 1, min(n_frames, int(release_window[1]) + 1))
    except Exception:
        start = 0
        end = max(1, n_frames)

    wrist_vis = _mean_visible_weight(
        cache["wrist_vis_raw"],
        cache["wrist_vis_weight"],
        start,
        end,
    )
    elbow_vis = _mean_visible_weight(
        cache["bowling_elbow_vis_raw"],
        cache["bowling_elbow_vis_weight"],
        start,
        end,
    )
    nb_elbow_vis = _mean_visible_weight(
        cache["nb_elbow_vis_raw"],
        cache["nb_elbow_vis_weight"],
        start,
        end,
    )
    visibility_score = max(0.0, min(1.0, (wrist_vis + elbow_vis) / 2.0))
    spacing_score = _anchor_spacing_score(events)
    release_conf = _safe_confidence(events.get("release"))
    uah_conf = _safe_confidence(events.get("uah"))
    release_frame = _safe_frame(events.get("release"))
    release_hint = None
    try:
        release_hint = int((delivery_window or {}).get("release_hint"))
    except Exception:
        release_hint = None

    nb_peak_frame = _local_peak_frame(
        cache.get("nb_elbow_y"),
        start,
        end,
        mode="peak_min",
    )
    shoulder_peak_frame = _local_peak_frame(
        cache.get("shoulder_angular_velocity"),
        start,
        end,
        mode="peak_max",
    )

    hint_alignment = 0.0
    if release_hint is not None and release_frame is not None:
        hint_band = max(6.0, float(fps) * 0.35)
        hint_alignment = _clamp01(1.0 - (abs(int(release_frame) - int(release_hint)) / hint_band))

    nb_release_alignment = 0.0
    if nb_peak_frame is not None and release_frame is not None:
        nb_band = max(4.0, float(fps) * 0.18)
        nb_release_alignment = _clamp01(
            1.0 - (abs(int(release_frame) - int(nb_peak_frame)) / nb_band)
        )

    nb_shoulder_alignment = 0.0
    if nb_peak_frame is not None and shoulder_peak_frame is not None:
        shoulder_band = max(4.0, float(fps) * 0.22)
        nb_shoulder_alignment = _clamp01(
            1.0 - (abs(int(nb_peak_frame) - int(shoulder_peak_frame)) / shoulder_band)
        )

    behind_camera_non_bowling_arm_score = hint_alignment * (
        (0.55 * nb_elbow_vis)
        + (0.25 * nb_release_alignment)
        + (0.20 * nb_shoulder_alignment)
    )
    uah_frame = _safe_frame(events.get("uah"))
    context_origin = start
    try:
        context_origin = max(0, int((delivery_window or {}).get("analysis_start", start)))
    except Exception:
        context_origin = start
    min_uah_runway = max(3.0, float(fps) * 0.10)
    min_release_runway = max(6.0, float(fps) * 0.20)
    uah_runway_score = 0.0
    if uah_frame is not None:
        uah_runway_score = _clamp01((float(uah_frame) - float(context_origin)) / min_uah_runway)
    release_runway_score = 0.0
    if release_frame is not None:
        release_runway_score = _clamp01((float(release_frame) - float(context_origin)) / min_release_runway)
    pre_release_context_score = round((0.70 * uah_runway_score) + (0.30 * release_runway_score), 3)

    score = (
        0.45 * spacing_score
        + 0.25 * release_conf
        + 0.15 * uah_conf
        + 0.15 * visibility_score
        + 0.45 * behind_camera_non_bowling_arm_score
    )
    behind_camera_hint = bool(hint_alignment >= 0.45)
    uncertain_checks: List[str] = []
    if release_hint is None:
        uncertain_checks.append("release_hint_unavailable")
    elif hint_alignment < 0.45:
        uncertain_checks.append("rear_view_release_hint_not_confirmed")
    if nb_peak_frame is None:
        uncertain_checks.append("non_bowling_arm_peak_unavailable")
    if shoulder_peak_frame is None:
        uncertain_checks.append("shoulder_peak_unavailable")
    if uah_runway_score < 0.75 or pre_release_context_score < 0.55:
        uncertain_checks.append("pre_release_context_insufficient")
    debug = {
        "hand": hand,
        "score": round(score, 3),
        "spacing_score": round(spacing_score, 3),
        "release_confidence": round(release_conf, 3),
        "uah_confidence": round(uah_conf, 3),
        "visibility_score": round(visibility_score, 3),
        "wrist_visibility": round(wrist_vis, 3),
        "elbow_visibility": round(elbow_vis, 3),
        "non_bowling_elbow_visibility": round(nb_elbow_vis, 3),
        "release_frame": release_frame,
        "uah_frame": uah_frame,
        "ffc_frame": _safe_frame(events.get("ffc")),
        "bfc_frame": _safe_frame(events.get("bfc")),
        "release_hint_frame": release_hint,
        "release_hint_alignment": round(hint_alignment, 3),
        "nb_elbow_peak_frame": nb_peak_frame,
        "shoulder_peak_frame": shoulder_peak_frame,
        "nb_release_alignment": round(nb_release_alignment, 3),
        "nb_shoulder_alignment": round(nb_shoulder_alignment, 3),
        "uah_runway_score": round(uah_runway_score, 3),
        "release_runway_score": round(release_runway_score, 3),
        "behind_camera_hint": behind_camera_hint,
        "behind_camera_non_bowling_arm_score": round(behind_camera_non_bowling_arm_score, 3),
        "pre_release_context_score": pre_release_context_score,
        "uncertain_checks": uncertain_checks,
    }
    return score, debug


def _playback_mode_v1(*, video: Dict[str, Any]) -> Dict[str, Any]:
    try:
        fps = float(video.get("fps") or 0.0)
    except Exception:
        fps = 0.0
    try:
        total_frames = int(video.get("total_frames") or 0)
    except Exception:
        total_frames = 0
    if fps <= 0.0 or total_frames <= 0:
        return {
            "mode": "unknown",
            "reason": "insufficient_video_metadata",
            "duration_seconds": None,
            "effective_fps": None,
            "effective_fps_source": "unknown",
        }
    duration_seconds = float(total_frames) / float(fps)
    if duration_seconds >= 12.0 and total_frames >= 240:
        return {
            "mode": "likely_slow_motion",
            "reason": "implausibly_long_single_delivery",
            "duration_seconds": round(duration_seconds, 3),
            "effective_fps": None,
            "effective_fps_source": "unknown",
        }
    if fps <= 40.0 and duration_seconds >= 8.0 and total_frames >= 240:
        return {
            "mode": "likely_slow_motion",
            "reason": "implausibly_long_single_delivery",
            "duration_seconds": round(duration_seconds, 3),
            "effective_fps": None,
            "effective_fps_source": "unknown",
        }
    return {
        "mode": "real_time_or_high_fps",
        "reason": "video_duration_plausible",
        "duration_seconds": round(duration_seconds, 3),
        "effective_fps": round(float(fps), 3),
        "effective_fps_source": "raw_video_fps",
    }


def _resolve_analysis_hand(
    *,
    pose_frames: List[Dict[str, Any]],
    fps: float,
    delivery_window: Optional[Dict[str, Any]],
    preferred_hand: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    preferred_raw = str(preferred_hand or "").strip().upper()
    preferred_valid = preferred_raw in {"R", "L"}
    preferred = preferred_raw if preferred_valid else "R"

    hypotheses: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {"preferred_hand": preferred, "hypotheses": {}}
    for hand in ("R", "L"):
        events = _detect_anchor_hypothesis(
            pose_frames=pose_frames,
            hand=hand,
            fps=fps,
            delivery_window=delivery_window,
        )
        score, details = _score_anchor_hypothesis(
            pose_frames=pose_frames,
            hand=hand,
            fps=fps,
            events=events,
            delivery_window=delivery_window,
        )
        hypotheses[hand] = {"events": events, "score": score, "details": details}
        debug["hypotheses"][hand] = details

    r_score = float(hypotheses["R"]["score"])
    l_score = float(hypotheses["L"]["score"])
    evidence_best = "R" if r_score >= l_score else "L"
    margin = abs(r_score - l_score)

    if preferred_valid:
        chosen = preferred
        debug["resolution_reason"] = "profile_hand_locked"
        debug["clip_evidence_best_hand"] = evidence_best
        if evidence_best != preferred:
            chosen_details = debug["hypotheses"].get(chosen) or {}
            uncertain = chosen_details.setdefault("uncertain_checks", [])
            if "clip_evidence_disagrees_with_profile_hand" not in uncertain:
                uncertain.append("clip_evidence_disagrees_with_profile_hand")
    else:
        chosen = evidence_best
        debug["resolution_reason"] = "profile_hand_missing_fallback"

    debug["resolved_hand"] = chosen
    debug["score_margin"] = round(margin, 3)
    return chosen, hypotheses[chosen]["events"], debug


def _walkthrough_render_window(
    *,
    events: dict,
    total_frames: int,
    playback_mode: Optional[dict] = None,
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

    slow_motion_mode = str((playback_mode or {}).get("mode") or "").strip().lower() == "likely_slow_motion"
    if slow_motion_mode:
        start_pad = 96
        end_pad = 18
    else:
        start_pad = 30
        end_pad = 28

    start = max(0, min(anchors) - start_pad)
    end = min(total_frames, max(anchors) + end_pad) if total_frames else max(anchors) + end_pad
    if end <= start:
        return start, None
    return start, end


def _deterministic_render_story_context(
    deterministic_expert: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    deterministic_expert = deterministic_expert or {}
    coach_diagnosis = (deterministic_expert.get("coach_diagnosis_v1") or {})
    root_cause = (coach_diagnosis.get("root_cause") or {})
    chain_status = (coach_diagnosis.get("kinetic_chain_status") or {})
    if not coach_diagnosis:
        return None

    root_cause_status = str(root_cause.get("status") or "").strip().lower()
    chain_status_id = str(chain_status.get("id") or "").strip().lower()
    if root_cause_status == "no_clear_problem" or chain_status_id == "connected":
        return {
            "theme": "working_pattern",
            "hero_risk_id": None,
            "watch_focus": {},
        }

    renderer_guidance = (root_cause.get("renderer_guidance") or {})
    hero_risk_id = ""
    phase_targets = renderer_guidance.get("phase_targets") or {}
    if isinstance(phase_targets, dict):
        for phase_key in ("ffc", "release"):
            phase_target = phase_targets.get(phase_key) or {}
            hero_risk_id = str(phase_target.get("risk_id") or "").strip()
            if hero_risk_id:
                break
    if not hero_risk_id:
        anchor_risk_ids = renderer_guidance.get("anchor_risk_ids") or {}
        if isinstance(anchor_risk_ids, dict):
            for phase_key in ("ffc", "release"):
                hero_risk_id = str(anchor_risk_ids.get(phase_key) or "").strip()
                if hero_risk_id:
                    break

    return {
        "theme": "problem_pattern",
        "hero_risk_id": hero_risk_id or None,
        "watch_focus": {},
    }


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
    playback_mode: Optional[dict] = None,
    kinetic_chain: Optional[dict] = None,
    report_story: Optional[dict],
    root_cause: Optional[dict],
) -> dict:
    video_path = video.get("path")
    if not video_path or not os.path.exists(video_path):
        return {"available": False, "reason": "missing_video_path"}

    total_frames = int(video.get("total_frames") or len(pose_frames) or 0)
    start_frame, end_frame = _walkthrough_render_window(
        events=events,
        total_frames=total_frames,
        playback_mode=playback_mode,
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
            kinetic_chain=kinetic_chain,
            report_story=report_story,
            root_cause=root_cause,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            pause_seconds=WALKTHROUGH_PAUSE_SECONDS,
            slow_motion_factor=WALKTHROUGH_SLOW_MOTION_FACTOR,
            end_summary_seconds=WALKTHROUGH_END_SUMMARY_SECONDS,
            playback_mode=playback_mode,
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

    resolved_path = str(render_result.get("path") or output_path)
    if not os.path.exists(resolved_path):
        logger.warning(
            "[analyze:walkthrough_missing_after_render] run_id=%s expected_path=%s",
            run_id,
            resolved_path,
        )
        return {"available": False, "reason": "render_artifact_missing"}

    artifact_name = os.path.basename(resolved_path)
    upload_result = upload_render_artifact(
        resolved_path,
        artifact_name=artifact_name,
    )
    logger.info(
        "[analyze:walkthrough_storage] run_id=%s storage_backend=%s uploaded=%s bucket=%s object=%s reason=%s",
        run_id,
        upload_result.get("storage_backend") or "local",
        bool(upload_result.get("uploaded")),
        upload_result.get("bucket") or "-",
        upload_result.get("object_name") or "-",
        upload_result.get("reason") or "-",
    )

    public_render_result = {
        key: value
        for key, value in render_result.items()
        if key != "path"
    }

    return {
        **public_render_result,
        "renderer_version": WALKTHROUGH_RENDERER_VERSION,
        "artifact_type": "walkthrough_mp4",
        "storage_backend": upload_result.get("storage_backend") or "local",
        "storage_uploaded": bool(upload_result.get("uploaded")),
        "storage_bucket": upload_result.get("bucket"),
        "storage_object": upload_result.get("object_name"),
        "relative_url": f"/renders/{artifact_name}",
        "url": _public_asset_url(f"/renders/{artifact_name}"),
    }


# ------------------------------------------------------------
# Serve Visual Evidence
# ------------------------------------------------------------
VISUALS_DIR = "/tmp/actionlab_frames"
if os.path.isdir(VISUALS_DIR):
    app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")
    logger.info(f"Mounted visual evidence directory: {VISUALS_DIR}")
else:
    logger.warning(f"Visual evidence directory not found: {VISUALS_DIR}")


@app.get("/renders/{filename}")
def get_walkthrough_render(filename: str):
    safe_name = normalize_render_filename(filename)
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=404, detail="Render not found")

    remote_bytes = download_render_artifact(safe_name)
    if remote_bytes is not None:
        logger.info(
            "[renders:get] filename=%s source=gcs bytes=%s",
            safe_name,
            len(remote_bytes),
        )
        return Response(content=remote_bytes, media_type="video/mp4")

    local_path = os.path.join(RENDERS_DIR, safe_name)
    if os.path.exists(local_path):
        logger.info(
            "[renders:get] filename=%s source=local path=%s",
            safe_name,
            local_path,
        )
        return FileResponse(local_path, media_type="video/mp4")

    logger.warning("[renders:get] filename=%s source=missing", safe_name)
    raise HTTPException(status_code=404, detail="Render not found")


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
    actor: str = Form(None),  # legacy compatibility field; auth identity is authoritative
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
        f"content_type={getattr(file, 'content_type', None) or '-'} "
        f"filename={getattr(file, 'filename', None) or '-'} "
        f"user_agent={_compact_header(_request_header(request, 'user-agent') or '-')}"
    )

    content_type = (getattr(file, "content_type", None) or "").lower()
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
    prior_results: List[Dict[str, Any]] = []
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

        player_profile = get_player_profile(player_id)
        hand = str((player_profile or {}).get("handedness") or "").strip().upper() or None
        effective_age_group = str((player_profile or {}).get("age_group") or "").strip().upper() or None
        effective_season = (player_profile or {}).get("season")

        if hand not in {"R", "L"} or not effective_age_group or effective_season is None:
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
            remember_player_profile(
                player_id=player_id,
                handedness=hand,
                age_group=effective_age_group,
                season=effective_season,
            )

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

        prior_results = _load_recent_expert_history(
            db=db,
            player_id=player_id,
            limit=_get_deterministic_expert_engine().history_window_runs,
        )

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
        playback_mode = _playback_mode_v1(video=video)

        resolved_hand, events, hand_resolution = _resolve_analysis_hand(
            pose_frames=pose_frames,
            fps=fps_val,
            delivery_window=video.get("coarse_delivery_window"),
            preferred_hand=hand,
        )
        hand = resolved_hand
        logger.info(
            "[analyze:hand_resolution] request_id=%s run_id=%s preferred_hand=%s resolved_hand=%s margin=%s reason=%s",
            request_id,
            run_id,
            hand_resolution.get("preferred_hand"),
            hand_resolution.get("resolved_hand"),
            hand_resolution.get("score_margin"),
            hand_resolution.get("resolution_reason"),
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
            hand=hand,
        )

        # ------------------------------------------------------------
        # Estimated Release Speed (Research)
        # ------------------------------------------------------------
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
            playback_mode=playback_mode,
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

        # ------------------------------------------------------------
        # Clinician Layer
        # ------------------------------------------------------------
        clinician = _get_clinician_engine().build(
            elbow=elbow,
            risks=risks,
            interpretation=interpretation,
            action=action,
        )
        deterministic_expert = _get_deterministic_expert_engine().build(
            events=events,
            action=action,
            risks=risks,
            basics=basics,
            interpretation=interpretation,
            estimated_release_speed=estimated_release_speed,
            prior_results=prior_results,
            account_role=getattr(current_account, "role", None),
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
                "profile_hand": hand_resolution.get("preferred_hand"),
                "age_group": effective_age_group,
                "season": effective_season,
                "hand_resolution_v1": hand_resolution,
            },
            "video": {
                "fps": video.get("fps"),
                "total_frames": video.get("total_frames"),
                "playback_mode_v1": playback_mode,
            },
            "events": events,
            "elbow": elbow,
            "estimated_release_speed": estimated_release_speed,
            "action": action,
            "risks": risks,
            "basics": basics,
            "interpretation": interpretation,
            "clinician": clinician,
            "deterministic_expert_v1": deterministic_expert,
            "capture_quality_v1": deterministic_expert.get("capture_quality_v1"),
            "mechanics_evidence_v1": deterministic_expert.get("mechanics_evidence_v1"),
            "kinetic_chain_v1": deterministic_expert.get("kinetic_chain_v1"),
            "render_reasoning_v1": deterministic_expert.get("render_reasoning_v1"),
            "mechanism_explanation_v1": deterministic_expert.get("mechanism_explanation_v1"),
            "prescription_plan_v1": deterministic_expert.get("prescription_plan_v1"),
            "history_plan_v1": deterministic_expert.get("history_plan_v1"),
            "coach_diagnosis_v1": deterministic_expert.get("coach_diagnosis_v1"),
            "presentation_payload_v1": deterministic_expert.get("presentation_payload_v1"),
            "frontend_surface_v1": deterministic_expert.get("frontend_surface_v1"),
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
            playback_mode=playback_mode,
            kinetic_chain=deterministic_expert.get("kinetic_chain_v1"),
            report_story=_deterministic_render_story_context(deterministic_expert),
            root_cause=((deterministic_expert.get("coach_diagnosis_v1") or {}).get("root_cause")),
        )

        # ------------------------------------------------------------
        # Persist
        # ------------------------------------------------------------
        persistence_status = _persist_analysis_result(
            request_id=request_id,
            run_id=run_id,
            result=result,
            video=video,
            bowler_type=bowler_type,
            actor_obj=actor_obj,
            effective_age_group=effective_age_group,
            effective_season=effective_season,
        )
        if isinstance(persistence_status, dict) and persistence_status.get("persisted"):
            background_tasks.add_task(
                _persist_learning_case_best_effort,
                request_id=request_id,
                run_id=run_id,
                result=result,
                account_id=str(current_account.account_id),
            )
            background_tasks.add_task(
                _sync_prescription_followups_best_effort,
                request_id=request_id,
                run_id=run_id,
            )
            background_tasks.add_task(
                persist_analysis_completed_notification_best_effort,
                account_id=str(current_account.account_id),
                result=result,
            )

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
app.include_router(notification_router)
