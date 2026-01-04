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
from typing import List, Dict, Any

import mediapipe as mp
from app.common.logger import get_logger


logger = get_logger(__name__)

# -----------------------------
# MediaPipe Pose Init (once)
# -----------------------------
_mp_pose = mp.solutions.pose
_pose = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=False,     # IMPORTANT: no smoothing at loader level
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def load_video(upload_file):
    """
    Load video, extract pose frames.

    Returns:
        video: dict
        pose_frames: list[dict]
        events: dict (empty for now)
    """

    logger.info("Starting video upload stream")

    # -----------------------------
    # Save uploaded file (STREAM SAFE)
    # -----------------------------
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with tmp as f:
        shutil.copyfileobj(upload_file.file, f)

    logger.info(f"Video saved to temp path: {tmp.name}")

    # -----------------------------
    # Open video
    # -----------------------------
    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        raise RuntimeError("Failed to open uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video opened | fps={fps:.2f} total_frames={total_frames}"
    )

    pose_frames: List[Dict[str, Any]] = []
    frame_idx = 0

    # -----------------------------
    # Frame-by-frame pose extraction
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _pose.process(rgb)

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

        pose_frames.append(
            {
                "frame": frame_idx,
                "landmarks": landmarks,
            }
        )

        if frame_idx % 100 == 0:
            logger.info(f"Processed pose frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()

    logger.info("Pose extraction completed")

    # -----------------------------
    # Return canonical payload
    # -----------------------------
    video = {
        "path": tmp.name,
        "fps": fps,
        "total_frames": total_frames,
    }

    events = {}  # UAH / Release to be computed later

    return video, pose_frames, events

