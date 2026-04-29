"""
Video + Pose loader for ActionLab V14.

Responsibilities:
- Save uploaded video to temp path (stream-safe)
- Decode video frames with OpenCV
- Run MediaPipe Pose (single pass, no frame drop)
- Return:
    - video metadata
    - pose_frames (per-frame landmarks)
    - empty events dict (to be filled later)
"""

import cv2
import os
import tempfile
import shutil
import time
from typing import List, Dict, Any, Tuple

import mediapipe as mp
from app.common.logger import get_logger
from app.workers.windowing.delivery_window import detect_delivery_window


logger = get_logger(__name__)

_mp_pose = mp.solutions.pose

MIN_WINDOW_CONFIDENCE = 0.30
MIN_WINDOW_SECONDS = 1.6
LATE_FALLBACK_MIN_SECONDS = 2.6
LATE_FALLBACK_CLIP_RATIO = 0.38
SLOW_MOTION_FALLBACK_MIN_SECONDS = 4.0
SLOW_MOTION_FALLBACK_CLIP_RATIO = 0.30
SLOW_MOTION_POST_SECONDS = 0.6
TEMP_UPLOAD_ROOT_PREFIX = "actionlab_upload_"
TEMP_UPLOAD_PREFIX = f"{TEMP_UPLOAD_ROOT_PREFIX}{os.getpid()}_"
STALE_TEMP_UPLOAD_MAX_AGE_SECONDS = 24 * 60 * 60


def _is_likely_slow_motion_video(video: Dict[str, Any]) -> bool:
    fps = float(video.get("fps") or 0.0)
    total_frames = int(video.get("total_frames") or 0)
    if fps <= 0.0 or total_frames <= 0:
        return False
    duration_seconds = total_frames / fps
    if duration_seconds >= 12.0 and total_frames >= 240:
        return True
    if fps <= 40.0 and duration_seconds >= 8.0 and total_frames >= 240:
        return True
    return False


def _conservative_late_window(video: Dict[str, Any], delivery_window: Dict[str, Any]) -> Dict[str, Any]:
    total_frames = int(video.get("total_frames") or 0)
    fps = float(video.get("fps") or 0.0)
    if total_frames <= 0 or fps <= 0.0:
        return {
            "start": 0,
            "end": max(0, total_frames - 1),
            "source": "full_clip_fallback",
            "reason": str(delivery_window.get("reason") or "window_unavailable"),
        }

    hint = delivery_window.get("release_hint")
    try:
        hint_frame = int(hint) if hint is not None else None
    except Exception:
        hint_frame = None

    if hint_frame is not None:
        if _is_likely_slow_motion_video(video):
            pre_frames = max(
                int(round(SLOW_MOTION_FALLBACK_MIN_SECONDS * fps)),
                int(round(total_frames * SLOW_MOTION_FALLBACK_CLIP_RATIO)),
            )
            post_frames = max(12, int(round(SLOW_MOTION_POST_SECONDS * fps)))
            start = max(0, hint_frame - pre_frames)
            end = min(total_frames - 1, hint_frame + post_frames)
        else:
            start = int(delivery_window.get("analysis_start") or 0)
            end = int(delivery_window.get("analysis_end") or (total_frames - 1))
            start = max(0, min(total_frames - 1, start))
            end = max(start, min(total_frames - 1, end))
    else:
        width = min(
            total_frames,
            max(
                24,
                int(round(max(LATE_FALLBACK_MIN_SECONDS * fps, total_frames * LATE_FALLBACK_CLIP_RATIO))),
            ),
        )
        start = max(0, total_frames - width)
        end = total_frames - 1

    return {
        "start": start,
        "end": end,
        "source": "late_window_fallback",
        "reason": str(delivery_window.get("reason") or "window_unavailable"),
        "confidence": round(float(delivery_window.get("confidence") or 0.0), 3),
    }


def _create_pose_tracker():
    return _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,     # IMPORTANT: no smoothing at loader level
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _save_upload_to_temp_path(upload_file) -> str:
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=TEMP_UPLOAD_PREFIX,
        suffix=".mp4",
    )
    with tmp as f:
        shutil.copyfileobj(upload_file.file, f)
    return tmp.name


def cleanup_stale_temp_uploads(*, max_age_seconds: int = STALE_TEMP_UPLOAD_MAX_AGE_SECONDS) -> Dict[str, int]:
    scanned = 0
    removed = 0
    now = int(time.time())
    temp_dir = tempfile.gettempdir()
    try:
        names = os.listdir(temp_dir)
    except Exception as exc:
        logger.warning("[loader] stale_temp_cleanup_failed dir=%s error=%s", temp_dir, exc)
        return {"scanned": 0, "removed": 0}

    for name in names:
        if not name.startswith(TEMP_UPLOAD_ROOT_PREFIX):
            continue
        path = os.path.join(temp_dir, name)
        scanned += 1
        try:
            stat = os.stat(path)
            age_seconds = max(0, now - int(stat.st_mtime))
            if age_seconds >= max_age_seconds:
                os.remove(path)
                removed += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("[loader] stale_temp_cleanup_failed path=%s error=%s", path, exc)

    return {"scanned": scanned, "removed": removed}


def _read_video_metadata(video_path: str) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open uploaded video")

    video = {
        "path": video_path,
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    return cap, video


def _placeholder_pose_frames(total_frames: int) -> List[Dict[str, Any]]:
    return [{"frame": idx, "landmarks": None} for idx in range(max(0, int(total_frames or 0)))]


def _resolve_pose_window(video: Dict[str, Any], delivery_window: Dict[str, Any]) -> Dict[str, Any]:
    total_frames = int(video.get("total_frames") or 0)
    fps = float(video.get("fps") or 0.0)
    full_window = {
        "start": 0,
        "end": max(0, total_frames - 1),
        "source": "full_clip_fallback",
    }
    if total_frames <= 0 or fps <= 0.0:
        return full_window

    if not delivery_window.get("available"):
        return _conservative_late_window(video, delivery_window)

    confidence = float(delivery_window.get("confidence") or 0.0)
    start = int(delivery_window.get("analysis_start") or 0)
    end = int(delivery_window.get("analysis_end") or (total_frames - 1))
    start = max(0, min(total_frames - 1, start))
    end = max(start, min(total_frames - 1, end))
    width = end - start + 1
    min_frames = max(12, int(round(MIN_WINDOW_SECONDS * fps)))
    if confidence < MIN_WINDOW_CONFIDENCE or width < min_frames:
        delivery_window = dict(delivery_window)
        delivery_window["reason"] = (
            "low_window_confidence"
            if confidence < MIN_WINDOW_CONFIDENCE
            else "window_too_short"
        )
        return _conservative_late_window(video, delivery_window)

    return {
        "start": start,
        "end": end,
        "source": "coarse_delivery_window",
        "confidence": round(confidence, 3),
    }


def load_video(upload_file):
    """
    Load video, extract pose frames.

    Returns:
        video: dict
        pose_frames: list[dict]
        events: dict (empty for now)
    """


    # -----------------------------
    # Save uploaded file (STREAM SAFE)
    # -----------------------------
    video_path = _save_upload_to_temp_path(upload_file)


    # -----------------------------
    # Open video
    # -----------------------------
    cap, video = _read_video_metadata(video_path)
    fps = float(video.get("fps") or 0.0)
    total_frames = int(video.get("total_frames") or 0)
    delivery_window = detect_delivery_window(video)
    pose_window = _resolve_pose_window(video, delivery_window)
    video["coarse_delivery_window"] = delivery_window
    video["pose_window"] = pose_window
    pose_frames = _placeholder_pose_frames(total_frames)
    pose_start = int(pose_window.get("start") or 0)
    pose_end = int(pose_window.get("end") or max(0, total_frames - 1))

    logger.info(
        "[loader] pose_window source=%s start=%s end=%s total_frames=%s confidence=%s reason=%s",
        pose_window.get("source"),
        pose_start,
        pose_end,
        total_frames,
        pose_window.get("confidence", "-"),
        pose_window.get("reason", "-"),
    )

    # -----------------------------
    # Frame-by-frame pose extraction
    # -----------------------------
    with _create_pose_tracker() as pose_tracker:
        if pose_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pose_start)

        frame_idx = pose_start
        while frame_idx <= pose_end:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)

            if results.pose_landmarks:
                landmarks = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                    for lm in results.pose_landmarks.landmark
                ]
            else:
                landmarks = None  # Explicitly mark missing pose

            pose_frames[frame_idx] = {
                "frame": frame_idx,
                "landmarks": landmarks,
            }

            if frame_idx % 100 == 0:
                logger.info(f"Processed pose frame {frame_idx}/{total_frames}")

            frame_idx += 1

    cap.release()


    # -----------------------------
    # Return canonical payload
    # -----------------------------
    events = {}  # UAH / Release to be computed later

    return video, pose_frames, events
