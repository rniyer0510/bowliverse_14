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
import tempfile
import shutil
from typing import List, Dict, Any, Tuple

import mediapipe as mp
from app.common.logger import get_logger
from app.workers.windowing.delivery_window import detect_delivery_window


logger = get_logger(__name__)

_mp_pose = mp.solutions.pose

MIN_WINDOW_CONFIDENCE = 0.30
MIN_WINDOW_SECONDS = 1.6


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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with tmp as f:
        shutil.copyfileobj(upload_file.file, f)
    return tmp.name


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
        full_window["reason"] = str(delivery_window.get("reason") or "window_unavailable")
        return full_window

    confidence = float(delivery_window.get("confidence") or 0.0)
    start = int(delivery_window.get("analysis_start") or 0)
    end = int(delivery_window.get("analysis_end") or (total_frames - 1))
    start = max(0, min(total_frames - 1, start))
    end = max(start, min(total_frames - 1, end))
    width = end - start + 1
    min_frames = max(12, int(round(MIN_WINDOW_SECONDS * fps)))
    if confidence < MIN_WINDOW_CONFIDENCE or width < min_frames:
        full_window["reason"] = (
            "low_window_confidence"
            if confidence < MIN_WINDOW_CONFIDENCE
            else "window_too_short"
        )
        return full_window

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
