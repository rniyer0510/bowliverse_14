"""
Pose worker (V14, Pass 4A).

- Decoder: OpenCV
- Pose: MediaPipe
- Output: per-frame pose landmarks in full-frame normalized coordinates
- No frame drop
- ROI-based tracking with fallback reacquisition
- Outside-ROI masking before pose inference to suppress clutter
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

MIN_VISIBILITY = 0.45
ROI_PADDING_RATIO = 0.18
ROI_MIN_SIZE = 96
ROI_FAILURE_REACQUIRE_FRAMES = 4
OUTSIDE_ROI_DIM_FACTOR = 0.18


def _valid_landmark(lm: dict, min_visibility: float = MIN_VISIBILITY) -> bool:
    if not isinstance(lm, dict):
        return False
    x = lm.get("x")
    y = lm.get("y")
    v = lm.get("visibility")
    return (
        isinstance(x, (int, float))
        and isinstance(y, (int, float))
        and isinstance(v, (int, float))
        and v >= min_visibility
    )


def _pose_bbox_from_landmarks(
    landmarks: Optional[List[dict]],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(landmarks, list):
        return None

    core_indices = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    ]

    pts = []
    for idx in core_indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if not _valid_landmark(lm):
            continue
        px = int(round(float(lm["x"]) * frame_w))
        py = int(round(float(lm["y"]) * frame_h))
        pts.append((px, py))

    if len(pts) < 4:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    bw = max(ROI_MIN_SIZE, x1 - x0)
    bh = max(ROI_MIN_SIZE, y1 - y0)
    pad_x = int(round(bw * ROI_PADDING_RATIO))
    pad_y = int(round(bh * ROI_PADDING_RATIO))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(frame_w, x1 + pad_x)
    y1 = min(frame_h, y1 + pad_y)

    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _full_frame_human_ok(landmarks: Optional[List[dict]]) -> bool:
    if not isinstance(landmarks, list) or len(landmarks) <= RIGHT_ANKLE:
        return False

    torso_ok = all(
        _valid_landmark(landmarks[idx])
        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    )
    if not torso_ok:
        return False

    left_leg_ok = _valid_landmark(landmarks[LEFT_KNEE]) and _valid_landmark(landmarks[LEFT_ANKLE])
    right_leg_ok = _valid_landmark(landmarks[RIGHT_KNEE]) and _valid_landmark(landmarks[RIGHT_ANKLE])
    return left_leg_ok or right_leg_ok


def _mask_outside_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    masked = (frame.astype(np.float32) * OUTSIDE_ROI_DIM_FACTOR).astype(np.uint8)
    masked[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return masked


def _run_pose_on_frame(pose, frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return pose.process(rgb)


def _result_to_fullframe_landmarks(
    result,
    *,
    roi: Optional[Tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> Optional[List[dict]]:
    if not result or not result.pose_landmarks:
        return None

    output: List[dict] = []
    if roi is None:
        for lm in result.pose_landmarks.landmark:
            output.append(
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": getattr(lm, "visibility", 0.0),
                }
            )
        return output

    x0, y0, x1, y1 = roi
    roi_w = max(1, x1 - x0)
    roi_h = max(1, y1 - y0)

    for lm in result.pose_landmarks.landmark:
        full_x = (x0 + (lm.x * roi_w)) / float(frame_w)
        full_y = (y0 + (lm.y * roi_h)) / float(frame_h)
        output.append(
            {
                "x": full_x,
                "y": full_y,
                "z": lm.z,
                "visibility": getattr(lm, "visibility", 0.0),
            }
        )
    return output


def run_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)

    pose_frames = []
    frame_idx = 0
    tracked_roi: Optional[Tuple[int, int, int, int]] = None
    roi_fail_streak = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        used_roi = None
        landmarks = None

        if tracked_roi is not None and roi_fail_streak < ROI_FAILURE_REACQUIRE_FRAMES:
            used_roi = tracked_roi
            masked_frame = _mask_outside_roi(frame, used_roi)
            roi_frame = masked_frame[used_roi[1]:used_roi[3], used_roi[0]:used_roi[2]]

            if roi_frame.size > 0:
                result = _run_pose_on_frame(pose, roi_frame)
                landmarks = _result_to_fullframe_landmarks(
                    result,
                    roi=used_roi,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )

        if landmarks is None or not _full_frame_human_ok(landmarks):
            result = _run_pose_on_frame(pose, frame)
            landmarks = _result_to_fullframe_landmarks(
                result,
                roi=None,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            used_roi = None

        if _full_frame_human_ok(landmarks):
            tracked_roi = _pose_bbox_from_landmarks(landmarks, frame_w, frame_h)
            roi_fail_streak = 0
        else:
            roi_fail_streak += 1
            if roi_fail_streak >= ROI_FAILURE_REACQUIRE_FRAMES:
                tracked_roi = None

        pose_frames.append(
            {
                "frame": frame_idx,
                "landmarks": landmarks,
                "roi": tracked_roi,
                "roi_mode": used_roi is not None,
            }
        )

        frame_idx += 1

    cap.release()
    pose.close()

    return pose_frames
