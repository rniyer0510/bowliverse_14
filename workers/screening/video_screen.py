from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.workers.events.delivery_guard import detect_delivery_candidates

POSE_MIN_VIS = 0.20
MAX_SCREEN_DURATION_SEC = 15.0
PROMINENT_HEIGHT_RATIO = 0.85
PROMINENT_AREA_RATIO = 0.70
DUPLICATE_IOU = 0.35
MIN_PERSON_WEIGHT = 0.25
MIN_PERSON_HEIGHT_RATIO = 0.20
MAX_SAMPLES = 8

_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def _duration_seconds(video: Dict[str, Any]) -> float:
    fps = float(video.get("fps") or 0.0)
    total_frames = int(video.get("total_frames") or 0)
    if fps <= 0.0 or total_frames <= 0:
        return 0.0
    return total_frames / fps


def _sample_indices(total_frames: int, max_samples: int = MAX_SAMPLES) -> List[int]:
    total_frames = int(total_frames or 0)
    if total_frames <= 0:
        return []
    sample_count = min(max_samples, total_frames)
    return sorted(
        {
            int(round(idx))
            for idx in np.linspace(0, total_frames - 1, num=sample_count)
        }
    )


def _pose_box(
    pose_frame: Optional[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
) -> Optional[Dict[str, float]]:
    landmarks = (pose_frame or {}).get("landmarks")
    if not isinstance(landmarks, list):
        return None

    xs: List[float] = []
    ys: List[float] = []
    for point in landmarks:
        if not isinstance(point, dict):
            continue
        if float(point.get("visibility", 0.0)) < POSE_MIN_VIS:
            continue
        xs.append(float(point.get("x", 0.0)) * frame_w)
        ys.append(float(point.get("y", 0.0)) * frame_h)

    if len(xs) < 6 or len(ys) < 6:
        return None

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(frame_w), max(xs))
    y2 = min(float(frame_h), max(ys))

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad_x = 0.10 * w
    pad_y = 0.12 * h

    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(frame_w), x2 + pad_x)
    y2 = min(float(frame_h), y2 + pad_y)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    return {
        "x": x1,
        "y": y1,
        "w": w,
        "h": h,
        "cx": x1 + (w / 2.0),
        "cy": y1 + (h / 2.0),
        "area": w * h,
    }


def _box_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]

    inter_x1 = max(a["x"], b["x"])
    inter_y1 = max(a["y"], b["y"])
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union = float(a["area"] + b["area"] - inter_area)
    if union <= 1e-6:
        return 0.0
    return inter_area / union


def _nms_boxes(
    boxes: List[Dict[str, float]],
    iou_threshold: float = DUPLICATE_IOU,
) -> List[Dict[str, float]]:
    ordered = sorted(
        boxes,
        key=lambda box: (
            float(box.get("weight", 0.0)),
            float(box.get("area", 0.0)),
        ),
        reverse=True,
    )
    kept: List[Dict[str, float]] = []
    for candidate in ordered:
        if all(_box_iou(candidate, existing) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _detect_people_boxes(frame: np.ndarray) -> List[Dict[str, float]]:
    rects, weights = _HOG.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    frame_h, _ = frame.shape[:2]
    min_person_h = max(80.0, frame_h * MIN_PERSON_HEIGHT_RATIO)

    boxes: List[Dict[str, float]] = []
    for (x, y, w, h), weight in zip(rects, weights):
        score = float(weight)
        if score < MIN_PERSON_WEIGHT or float(h) < min_person_h:
            continue
        boxes.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "cx": float(x) + (float(w) / 2.0),
                "cy": float(y) + (float(h) / 2.0),
                "area": float(w) * float(h),
                "weight": score,
            }
        )
    return _nms_boxes(boxes)


def _choose_primary_box(
    boxes: List[Dict[str, float]],
    pose_box: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not boxes:
        return None
    if not pose_box:
        return max(
            boxes,
            key=lambda box: (box["h"], box["area"], box.get("weight", 0.0)),
        )

    def _score(box: Dict[str, float]) -> float:
        iou = _box_iou(box, pose_box)
        center_delta = abs(box["cx"] - pose_box["cx"]) / max(1.0, pose_box["w"])
        height_ratio = min(box["h"], pose_box["h"]) / max(
            box["h"],
            pose_box["h"],
            1.0,
        )
        return (iou * 4.0) + height_ratio - center_delta

    return max(boxes, key=_score)


def _is_competing_primary(
    candidate: Dict[str, float],
    primary: Dict[str, float],
) -> bool:
    if _box_iou(candidate, primary) > DUPLICATE_IOU:
        return False

    height_ratio = candidate["h"] / max(primary["h"], 1.0)
    area_ratio = candidate["area"] / max(primary["area"], 1.0)
    return (
        height_ratio >= PROMINENT_HEIGHT_RATIO
        or area_ratio >= PROMINENT_AREA_RATIO
    )


def assess_primary_subject(
    video: Dict[str, Any],
    pose_frames: List[Dict[str, Any]],
) -> Dict[str, Any]:
    video_path = str(video.get("path") or "")
    total_frames = int(video.get("total_frames") or len(pose_frames) or 0)
    sample_indices = _sample_indices(total_frames)
    if not video_path or not sample_indices:
        return {
            "passed": True,
            "status": "warn",
            "method": "insufficient_samples",
            "sample_count": 0,
            "detector_frames": 0,
            "frames_with_competing_primary": [],
            "frames_with_minor_people": [],
            "max_prominent_people": 1,
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "passed": True,
            "status": "warn",
            "method": "video_unavailable",
            "sample_count": len(sample_indices),
            "detector_frames": 0,
            "frames_with_competing_primary": [],
            "frames_with_minor_people": [],
            "max_prominent_people": 1,
        }

    competitor_frames: List[int] = []
    minor_people_frames: List[int] = []
    detector_frames = 0
    max_prominent_people = 1

    try:
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            detector_boxes = _detect_people_boxes(frame)
            if not detector_boxes:
                continue

            detector_frames += 1
            frame_h, frame_w = frame.shape[:2]
            pose_box = None
            if 0 <= frame_idx < len(pose_frames):
                pose_box = _pose_box(pose_frames[frame_idx], frame_w, frame_h)

            primary = _choose_primary_box(detector_boxes, pose_box)
            if primary is None:
                continue

            prominent_people = 1
            saw_minor = False
            for other in detector_boxes:
                if other is primary:
                    continue
                if _is_competing_primary(other, primary):
                    prominent_people += 1
                elif _box_iou(other, primary) <= DUPLICATE_IOU:
                    saw_minor = True

            if prominent_people > 1:
                competitor_frames.append(frame_idx)
            elif saw_minor:
                minor_people_frames.append(frame_idx)

            max_prominent_people = max(max_prominent_people, prominent_people)
    finally:
        cap.release()

    if detector_frames == 0:
        return {
            "passed": True,
            "status": "warn",
            "method": "detector_inconclusive",
            "sample_count": len(sample_indices),
            "detector_frames": 0,
            "frames_with_competing_primary": [],
            "frames_with_minor_people": [],
            "max_prominent_people": 1,
        }

    frame_threshold = (
        1
        if detector_frames < 4
        else max(3, int(math.ceil(detector_frames * 0.40)))
    )
    passed = len(competitor_frames) <= frame_threshold

    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "method": "hog_dominant_subject",
        "sample_count": len(sample_indices),
        "detector_frames": detector_frames,
        "frames_with_competing_primary": competitor_frames,
        "frames_with_minor_people": minor_people_frames,
        "max_prominent_people": max_prominent_people,
        "frame_threshold": frame_threshold,
    }


def run_preanalysis_screen(
    video: Dict[str, Any],
    pose_frames: List[Dict[str, Any]],
    hand: str,
) -> Dict[str, Any]:
    duration_sec = _duration_seconds(video)
    delivery_guard = detect_delivery_candidates(
        pose_frames=pose_frames,
        hand=hand,
        fps=float(video.get("fps") or 0.0),
    )
    primary_subject = assess_primary_subject(video=video, pose_frames=pose_frames)

    blocking_issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    duration_check = {
        "passed": duration_sec > 0.0 and duration_sec <= MAX_SCREEN_DURATION_SEC,
        "duration_sec": duration_sec,
        "max_duration_sec": MAX_SCREEN_DURATION_SEC,
    }
    if duration_sec <= 0.0:
        blocking_issues.append(
            {
                "code": "invalid_duration",
                "detail": "Could not determine video duration. Please upload a playable clip.",
            }
        )
    elif duration_sec > MAX_SCREEN_DURATION_SEC:
        blocking_issues.append(
            {
                "code": "video_too_long",
                "detail": "Please upload a single bowling clip under 15 seconds.",
            }
        )

    delivery_check = {
        "passed": int(delivery_guard.get("delivery_count") or 0) <= 1,
        "delivery_count": int(delivery_guard.get("delivery_count") or 0),
        "method": delivery_guard.get("method"),
        "candidate_frames": delivery_guard.get("candidate_frames") or [],
    }
    if not delivery_check["passed"]:
        blocking_issues.append(
            {
                "code": "multiple_deliveries",
                "detail": "Please upload a video with only one bowling delivery.",
            }
        )

    if not primary_subject.get("passed", True):
        warnings.append(
            {
                "code": "multiple_prominent_people",
                "detail": (
                    "Another prominent person was detected in several frames. "
                    "Analysis will proceed but results may be less reliable if "
                    "two active bowlers are present."
                ),
            }
        )
    elif primary_subject.get("status") == "warn":
        warnings.append(
            {
                "code": "primary_subject_inconclusive",
                "detail": (
                    "Primary-subject screening was inconclusive, so analysis "
                    "will still rely on pose stability checks."
                ),
            }
        )
    elif primary_subject.get("frames_with_minor_people"):
        warnings.append(
            {
                "code": "minor_bystanders_present",
                "detail": (
                    "Minor bystanders were detected, but one dominant bowler "
                    "remained clear across the sampled frames."
                ),
            }
        )
    if primary_subject.get("passed", True) and primary_subject.get("frames_with_competing_primary"):
        warnings.append(
            {
                "code": "primary_subject_competition_seen",
                "detail": (
                    "Another prominent person appeared in a few sampled frames, "
                    "but one bowler still remained dominant overall."
                ),
            }
        )

    return {
        "schema": "actionlab.screen.v1",
        "passed": len(blocking_issues) == 0,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "checks": {
            "duration": duration_check,
            "delivery": delivery_check,
            "primary_subject": primary_subject,
        },
    }
