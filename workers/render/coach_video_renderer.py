from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.common.logger import get_logger
from .render_load_watch import (
    _load_hotspot_regions,
    _load_watch_label,
    _preferred_ffc_cue_risk_id,
    _release_hotspot_risk_id,
    _summary_load_watch_text,
    _summary_symptom_text,
)

logger = get_logger(__name__)

DEFAULT_RENDER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "storage", "renders")
)
RENDER_DIR = os.getenv("ACTIONLAB_RENDER_DIR", DEFAULT_RENDER_DIR)
try:
    os.makedirs(RENDER_DIR, exist_ok=True)
except OSError:
    RENDER_DIR = "/tmp/actionlab_renders"
    os.makedirs(RENDER_DIR, exist_ok=True)

MIN_VISIBILITY = 0.35
SKELETON_COLOR = (240, 225, 70)
SKELETON_SHADOW = (32, 28, 18)
JOINT_OUTER = (255, 255, 255)
TEXT_COLOR = (246, 248, 252)
MUTED_TEXT = (204, 212, 222)
PANEL_BG = (24, 28, 34)
PANEL_EDGE = (62, 73, 88)
ACTIVE_FILL = (74, 194, 242)
ACTIVE_EDGE = (105, 222, 255)
INACTIVE_FILL = (44, 52, 62)
HOTSPOT_CORE = (48, 58, 235)
HOTSPOT_RING = (0, 132, 255)
HOTSPOT_SOFT = (90, 190, 255)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
)

TRACKED_JOINTS = tuple(sorted({idx for edge in SKELETON_EDGES for idx in edge}))

PHASES: Tuple[Dict[str, str], ...] = (
    {
        "key": "approach",
        "title": "Approach",
        "subtitle": "Building into the crease",
    },
    {
        "key": "bfc",
        "title": "Back Foot Contact",
        "subtitle": "Back foot lands",
    },
    {
        "key": "ffc",
        "title": "Front Foot Contact",
        "subtitle": "Front foot lands and supports the action",
    },
    {
        "key": "release",
        "title": "Release",
        "subtitle": "Ball comes out",
    },
)

RISK_TITLE_BY_ID: Dict[str, str] = {
    "knee_brace_failure": "Front-Leg Support",
    "lateral_trunk_lean": "Trunk Lean",
    "hip_shoulder_mismatch": "Shoulder Timing",
    "foot_line_deviation": "Foot Line",
    "front_foot_braking_shock": "Landing Load",
    "trunk_rotation_snap": "Trunk Rotation",
}

FEATURE_TO_RENDER_LABEL: Dict[str, str] = {
    "front_leg_support": "Front-Leg Support",
    "trunk_lean": "Trunk Lean",
    "upper_body_opening": "Shoulder Timing",
    "action_flow": "Action Flow",
    "front_foot_line": "Foot Line",
    "trunk_rotation_load": "Trunk Rotation",
}

FFC_DEPENDENT_RISKS = {
    "knee_brace_failure",
    "foot_line_deviation",
    "front_foot_braking_shock",
}

MIN_FFC_STORY_CONFIDENCE = 0.35
MIN_EVENT_CHAIN_QUALITY = 0.20


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _risk_lookup(risks: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for risk in risks or []:
        if not isinstance(risk, dict):
            continue
        risk_id = str(risk.get("risk_id") or "").strip()
        if risk_id:
            lookup[risk_id] = risk
    return lookup


def _risk_weight(risk: Optional[Dict[str, Any]]) -> float:
    if not isinstance(risk, dict):
        return 0.0
    signal = float(risk.get("signal_strength") or 0.0)
    confidence = float(risk.get("confidence") or 0.0)
    return max(0.0, signal) * max(0.0, confidence)


def _event_confidence(events: Optional[Dict[str, Any]], key: str) -> float:
    event = (events or {}).get(key) or {}
    confidence = _safe_float(event.get("confidence"))
    if confidence is None:
        return 1.0 if _safe_int(event.get("frame")) is not None else 0.0
    return max(0.0, confidence)


def _event_chain_quality(events: Optional[Dict[str, Any]]) -> float:
    quality = _safe_float((((events or {}).get("event_chain") or {}).get("quality")))
    if quality is None:
        return 1.0
    return max(0.0, quality)


def _supports_ffc_story(events: Optional[Dict[str, Any]]) -> bool:
    return (
        _event_confidence(events, "ffc") >= MIN_FFC_STORY_CONFIDENCE
        and _event_chain_quality(events) >= MIN_EVENT_CHAIN_QUALITY
    )


def _speed_display_text(speed: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(speed, dict):
        return None
    if not speed.get("available"):
        return None
    display = str(speed.get("display") or "").strip()
    if not display:
        return None
    return display


def _risk_supported_for_phase(
    risk_id: Optional[str],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
) -> bool:
    if not risk_id:
        return False
    if phase_key == "ffc":
        return risk_id in FFC_DEPENDENT_RISKS and _supports_ffc_story(events)
    if phase_key == "release":
        return risk_id in {
            "lateral_trunk_lean",
            "hip_shoulder_mismatch",
            "trunk_rotation_snap",
            "front_foot_braking_shock",
        }
    return False


def _story_feature_labels(report_story: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report_story, dict):
        return []
    labels: List[str] = []
    watch_focus = report_story.get("watch_focus") or {}
    watch_label = str(watch_focus.get("label") or "").strip()
    if watch_label:
        labels.append(watch_label)
    for item in report_story.get("key_metrics") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        label = str(item.get("label") or "").strip()
        resolved = FEATURE_TO_RENDER_LABEL.get(key) or label
        if resolved and resolved not in labels:
            labels.append(resolved)
    return labels


def _positive_recap_lines(report_story: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report_story, dict):
        return []
    theme = str(report_story.get("theme") or "").strip()
    if theme not in {"working_pattern", "good_base"}:
        return []

    lines: List[str] = []
    watch_focus = report_story.get("watch_focus") or {}
    watch_label = str(watch_focus.get("label") or "").strip()
    if watch_label:
        lines.append(f"Keep watching {watch_label}")

    positive_by_key = {
        "upper_body_alignment": "Upper body stays aligned",
        "lower_body_alignment": "Lower body stays aligned",
        "whole_body_alignment": "Action shape stays connected",
        "momentum_forward": "Carries forward well",
        "front_leg_support": "Front leg support looks steady",
        "trunk_lean": "Body stays fairly tall",
        "upper_body_opening": "Top half stays in order",
        "action_flow": "Action carries through well",
        "front_foot_line": "Feet stay balanced and in line",
    }

    for item in report_story.get("key_metrics") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        label = positive_by_key.get(key)
        if not label or label in lines:
            continue
        lines.append(label)
        if len(lines) >= 4:
            break

    if not lines:
        headline = str(report_story.get("headline") or "").strip()
        if "strong working pattern" in headline.lower():
            lines = ["Strong working pattern", "Keep repeating this shape"]
        else:
            lines = ["Action has a good base", "Keep repeating this shape"]
    return lines[:4]


def _story_risk_for_phase(
    report_story: Optional[Dict[str, Any]],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not isinstance(report_story, dict):
        return None
    hero_risk_id = str(report_story.get("hero_risk_id") or "").strip()
    if _risk_supported_for_phase(hero_risk_id, phase_key=phase_key, events=events):
        return hero_risk_id

    watch_focus = report_story.get("watch_focus") or {}
    watch_key = str(watch_focus.get("key") or "").strip()
    mapped = {
        "front_leg_support": "knee_brace_failure",
        "front_foot_line": "foot_line_deviation",
        "trunk_lean": "lateral_trunk_lean",
        "upper_body_opening": "hip_shoulder_mismatch",
        "action_flow": "front_foot_braking_shock",
        "trunk_rotation_load": "trunk_rotation_snap",
    }.get(watch_key)
    if _risk_supported_for_phase(mapped, phase_key=phase_key, events=events):
        return mapped
    return None


def _format_action_label(action: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(action, dict):
        return None
    raw = str(action.get("action") or "").strip()
    if not raw or raw.upper() == "UNKNOWN":
        return None
    return raw.replace("_", " ").title()


def _front_leg_joints(hand: Optional[str]) -> Tuple[int, int, int]:
    is_left_handed = str(hand or "R").upper().startswith("L")
    if is_left_handed:
        return RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
    return LEFT_HIP, LEFT_KNEE, LEFT_ANKLE


def _foot_indices(hand: Optional[str]) -> Tuple[int, int, int]:
    is_left_handed = str(hand or "R").upper().startswith("L")
    if is_left_handed:
        return RIGHT_FOOT_INDEX, RIGHT_HEEL, LEFT_FOOT_INDEX
    return LEFT_FOOT_INDEX, LEFT_HEEL, RIGHT_FOOT_INDEX


def _make_output_path(output_path: Optional[str]) -> str:
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path
    name = f"skeleton_{uuid.uuid4().hex[:12]}.mp4"
    return os.path.join(RENDER_DIR, name)


def _intermediate_render_path(final_path: str) -> str:
    base_dir = os.path.dirname(final_path) or RENDER_DIR
    name = f"render_tmp_{uuid.uuid4().hex[:12]}.mp4"
    return os.path.join(base_dir, name)


def _finalize_render_video(intermediate_path: str, final_path: str) -> Tuple[str, str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return intermediate_path, "mp4v"

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        intermediate_path,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        final_path,
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except Exception as exc:
        logger.warning("[coach_video_renderer] ffmpeg finalize failed: %s", exc)
        return intermediate_path, "mp4v"

    if completed.returncode != 0 or not os.path.exists(final_path):
        logger.warning(
            "[coach_video_renderer] ffmpeg finalize returned %s; keeping intermediate encode",
            completed.returncode,
        )
        return intermediate_path, "mp4v"

    try:
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)
    except OSError:
        pass
    return final_path, "h264"


def _point_from_landmarks(
    landmarks: Optional[List[Dict[str, Any]]],
    idx: int,
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    if not isinstance(landmarks, list) or idx >= len(landmarks):
        return None
    point = landmarks[idx]
    if not isinstance(point, dict):
        return None
    vis = _safe_float(point.get("visibility"))
    x = _safe_float(point.get("x"))
    y = _safe_float(point.get("y"))
    if vis is None or x is None or y is None or vis < MIN_VISIBILITY:
        return None
    return (x * width, y * height)


def _smooth_series(values: Iterable[Optional[float]], sigma: float) -> Optional[np.ndarray]:
    value_list = list(values)
    if not value_list:
        return None
    valid = np.array([value is not None for value in value_list], dtype=bool)
    if valid.sum() < 3:
        return None
    idx = np.arange(len(value_list), dtype=float)
    arr = np.zeros(len(value_list), dtype=float)
    arr[valid] = [float(value) for value in value_list if value is not None]
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return gaussian_filter1d(arr, sigma=max(1.0, sigma))


def _build_smoothed_tracks(
    pose_frames: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: float,
) -> Dict[int, Dict[str, Any]]:
    sigma = max(1.0, float(fps or 30.0) * 0.03)
    tracks: Dict[int, Dict[str, Any]] = {}
    for joint_idx in TRACKED_JOINTS:
        raw_points = [
            _point_from_landmarks((frame or {}).get("landmarks"), joint_idx, width=width, height=height)
            for frame in pose_frames
        ]
        xs = _smooth_series([point[0] if point else None for point in raw_points], sigma=sigma)
        ys = _smooth_series([point[1] if point else None for point in raw_points], sigma=sigma)
        tracks[joint_idx] = {
            "raw": raw_points,
            "xs": xs,
            "ys": ys,
        }
    return tracks


def _track_point(
    tracks: Dict[int, Dict[str, Any]],
    joint_idx: int,
    frame_idx: int,
) -> Optional[Tuple[int, int]]:
    track = tracks.get(joint_idx) or {}
    xs = track.get("xs")
    ys = track.get("ys")
    raw = track.get("raw") or []
    if xs is not None and ys is not None and frame_idx < len(xs) and frame_idx < len(ys):
        return (int(round(float(xs[frame_idx]))), int(round(float(ys[frame_idx]))))
    if frame_idx < len(raw) and raw[frame_idx] is not None:
        point = raw[frame_idx]
        return (int(round(point[0])), int(round(point[1])))
    return None


def _frame_point(
    pose_frames: List[Dict[str, Any]],
    *,
    frame_idx: int,
    joint_idx: int,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    if frame_idx < 0 or frame_idx >= len(pose_frames):
        return None
    point = _point_from_landmarks((pose_frames[frame_idx] or {}).get("landmarks"), joint_idx, width=width, height=height)
    if point is None:
        return None
    return (int(round(point[0])), int(round(point[1])))


def _draw_joint(frame: np.ndarray, point: Tuple[int, int], scale: int) -> None:
    outer = max(3, scale // 180)
    inner = max(2, outer - 1)
    cv2.circle(frame, point, outer + 2, SKELETON_SHADOW, -1, cv2.LINE_AA)
    cv2.circle(frame, point, outer, JOINT_OUTER, -1, cv2.LINE_AA)
    cv2.circle(frame, point, inner, SKELETON_COLOR, -1, cv2.LINE_AA)


def _draw_skeleton(frame: np.ndarray, tracks: Dict[int, Dict[str, Any]], frame_idx: int) -> None:
    scale = min(frame.shape[0], frame.shape[1])
    shadow_thickness = max(5, scale // 110)
    line_thickness = max(3, scale // 180)
    for start_idx, end_idx in SKELETON_EDGES:
        start = _track_point(tracks, start_idx, frame_idx)
        end = _track_point(tracks, end_idx, frame_idx)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, SKELETON_SHADOW, shadow_thickness, cv2.LINE_AA)
        cv2.line(frame, start, end, SKELETON_COLOR, line_thickness, cv2.LINE_AA)
    for joint_idx in TRACKED_JOINTS:
        point = _track_point(tracks, joint_idx, frame_idx)
        if point is None:
            continue
        _draw_joint(frame, point, scale)


def _overlay_panel(
    frame: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int],
    alpha: float = 0.78,
) -> None:
    x0 = max(0, min(frame.shape[1] - 1, x0))
    x1 = max(0, min(frame.shape[1], x1))
    y0 = max(0, min(frame.shape[0] - 1, y0))
    y1 = max(0, min(frame.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), edge_color, 2, cv2.LINE_AA)


def _phase_cut_points(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    last = max(start + 3, stop - 1)
    span = max(4, stop - start)
    bfc = _safe_int(((events or {}).get("bfc") or {}).get("frame"))
    ffc = _safe_int(((events or {}).get("ffc") or {}).get("frame"))
    uah = _safe_int(((events or {}).get("uah") or {}).get("frame"))
    release = _safe_int(((events or {}).get("release") or {}).get("frame"))
    phase_release = uah if uah is not None and uah >= (ffc if ffc is not None else start) else release

    cp1 = bfc if bfc is not None else start + int(round(span * 0.32))
    cp2 = ffc if ffc is not None else start + int(round(span * 0.58))
    cp3 = phase_release if phase_release is not None else start + int(round(span * 0.82))

    cp1 = max(start + 1, min(last - 2, cp1))
    cp2 = max(cp1 + 1, min(last - 1, cp2))
    cp3 = max(cp2 + 1, min(last, cp3))
    return cp1, cp2, cp3


def _phase_index_for_frame(
    frame_idx: int,
    *,
    cp1: int,
    cp2: int,
    cp3: int,
) -> int:
    if frame_idx < cp1:
        return 0
    if frame_idx < cp2:
        return 1
    if frame_idx < cp3:
        return 2
    return 3


def _draw_phase_rail(
    frame: np.ndarray,
    *,
    phase_idx: int,
    progress: float,
) -> None:
    width = frame.shape[1]
    height = frame.shape[0]
    rail_x0 = int(round(width * 0.05))
    rail_x1 = int(round(width * 0.95))
    rail_y0 = int(round(height * 0.90))
    rail_h = int(round(height * 0.05))
    gap = int(round(width * 0.012))
    segment_w = max(30, int((rail_x1 - rail_x0 - gap * (len(PHASES) - 1)) / len(PHASES)))

    for idx, phase in enumerate(PHASES):
        seg_x0 = rail_x0 + idx * (segment_w + gap)
        seg_x1 = seg_x0 + segment_w
        active = idx == phase_idx
        fill = ACTIVE_FILL if active else INACTIVE_FILL
        edge = ACTIVE_EDGE if active else PANEL_EDGE
        _overlay_panel(
            frame,
            x0=seg_x0,
            y0=rail_y0,
            x1=seg_x1,
            y1=rail_y0 + rail_h,
            fill_color=fill,
            edge_color=edge,
            alpha=0.88 if active else 0.72,
        )
        label = phase["title"].replace(" Contact", "")
        text_scale = max(0.38, min(width, height) / 1800.0)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        text_x = seg_x0 + max(10, (segment_w - text_size[0]) // 2)
        text_y = rail_y0 + int(rail_h * 0.63)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            TEXT_COLOR if active else MUTED_TEXT,
            1,
            cv2.LINE_AA,
        )

    tracker_x = rail_x0 + int(round((rail_x1 - rail_x0) * max(0.0, min(1.0, progress))))
    tracker_y0 = rail_y0 - int(round(height * 0.012))
    tracker_y1 = rail_y0 + rail_h + int(round(height * 0.012))
    cv2.line(frame, (tracker_x, tracker_y0), (tracker_x, tracker_y1), ACTIVE_EDGE, 2, cv2.LINE_AA)


def _draw_phase_overlay(
    frame: np.ndarray,
    *,
    frame_idx: int,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> None:
    cp1, cp2, cp3 = _phase_cut_points(start=start, stop=stop, events=events)
    phase_idx = _phase_index_for_frame(frame_idx, cp1=cp1, cp2=cp2, cp3=cp3)
    progress = 0.0 if stop <= start else float(frame_idx - start) / float(max(1, stop - start - 1))
    _draw_phase_rail(frame, phase_idx=phase_idx, progress=progress)


def _draw_top_risk_panel(
    frame: np.ndarray,
    *,
    title: str,
    headline: str,
    body: str,
    accent: Tuple[int, int, int],
) -> None:
    width = frame.shape[1]
    height = frame.shape[0]
    card_w = int(round(width * 0.58))
    card_h = int(round(height * 0.14))
    x0 = int(round(width * 0.05))
    y0 = int(round(height * 0.05))
    x1 = min(width - 18, x0 + card_w)
    y1 = y0 + card_h
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=accent,
        alpha=0.84,
    )
    cv2.putText(
        frame,
        title,
        (x0 + 18, y0 + int(card_h * 0.28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.44, min(width, height) / 1400.0),
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        headline,
        (x0 + 18, y0 + int(card_h * 0.58)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.58, min(width, height) / 1100.0),
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        body,
        (x0 + 18, y0 + int(card_h * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, min(width, height) / 1550.0),
        MUTED_TEXT,
        1,
        cv2.LINE_AA,
    )


def _front_leg_support_caption(risk: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(risk, dict):
        return None
    signal = float(risk.get("signal_strength") or 0.0)
    if signal >= 0.65:
        return {
            "title": "Front leg is softer here.",
            "body": "The landing leg is not holding its shape as well as it should.",
        }
    if signal >= 0.35:
        return {
            "title": "Front leg needs a bit more support here.",
            "body": "The landing leg softens a little as the action comes through.",
        }
    return {
        "title": "Front leg stays fairly firm here.",
        "body": "The landing leg gives the action a steady base at contact.",
    }


def _draw_front_leg_support_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk: Optional[Dict[str, Any]],
) -> None:
    caption = _front_leg_support_caption(risk)
    if not caption:
        return

    hip_idx, knee_idx, ankle_idx = _front_leg_joints(hand)
    hip = _track_point(tracks, hip_idx, frame_idx)
    knee = _track_point(tracks, knee_idx, frame_idx)
    ankle = _track_point(tracks, ankle_idx, frame_idx)
    if knee is None:
        return

    scale = min(frame.shape[0], frame.shape[1])
    signal = float((risk or {}).get("signal_strength") or 0.0)
    accent = (90, 220, 255) if signal < 0.35 else ((0, 196, 255) if signal < 0.65 else (0, 126, 255))
    radius = max(16, scale // 14)
    thickness = max(3, scale // 160)

    cv2.circle(frame, knee, radius + 4, SKELETON_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.circle(frame, knee, radius, accent, thickness, cv2.LINE_AA)

    if hip is not None and ankle is not None:
        direction = (ankle[0] - hip[0], ankle[1] - hip[1])
        arrow_start = (int(round(knee[0] + direction[0] * 0.10)), int(round(knee[1] - radius * 1.25)))
        arrow_end = (int(round(knee[0] + direction[0] * 0.16)), int(round(knee[1] + radius * 1.20)))
    else:
        arrow_start = (knee[0], knee[1] - int(radius * 1.6))
        arrow_end = (knee[0], knee[1] + int(radius * 1.4))

    cv2.arrowedLine(
        frame,
        arrow_start,
        arrow_end,
        accent,
        thickness,
        cv2.LINE_AA,
        tipLength=0.16,
    )
    _draw_top_risk_panel(
        frame,
        title=RISK_TITLE_BY_ID["knee_brace_failure"],
        headline=caption["title"],
        body=caption["body"],
        accent=accent,
    )


def _trunk_lean_caption(risk: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(risk, dict):
        return None
    signal = float(risk.get("signal_strength") or 0.0)
    if signal >= 0.65:
        return {
            "title": "Body is falling away here.",
            "body": "The upper body is moving too far off line near release.",
        }
    if signal >= 0.35:
        return {
            "title": "Body is leaning a bit here.",
            "body": "The action is starting to move off line near release.",
        }
    return {
        "title": "Body stays fairly tall here.",
        "body": "The action is staying more upright through release.",
    }


def _draw_trunk_lean_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    risk: Optional[Dict[str, Any]],
) -> None:
    caption = _trunk_lean_caption(risk)
    if not caption:
        return
    left_shoulder = _track_point(tracks, LEFT_SHOULDER, frame_idx)
    right_shoulder = _track_point(tracks, RIGHT_SHOULDER, frame_idx)
    left_hip = _track_point(tracks, LEFT_HIP, frame_idx)
    right_hip = _track_point(tracks, RIGHT_HIP, frame_idx)
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return

    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
    hip_mid = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
    dx = shoulder_mid[0] - hip_mid[0]
    signal = float((risk or {}).get("signal_strength") or 0.0)
    accent = (90, 220, 255) if signal < 0.35 else ((0, 196, 255) if signal < 0.65 else (0, 126, 255))
    scale = min(frame.shape[0], frame.shape[1])
    thickness = max(3, scale // 160)
    offset_x = int(round(scale * 0.04))

    cv2.line(frame, hip_mid, shoulder_mid, accent, thickness, cv2.LINE_AA)
    arrow_dir = 1 if dx >= 0 else -1
    arrow_start = (shoulder_mid[0], shoulder_mid[1] - int(scale * 0.02))
    arrow_end = (shoulder_mid[0] + arrow_dir * offset_x, shoulder_mid[1] - int(scale * 0.08))
    cv2.arrowedLine(frame, arrow_start, arrow_end, accent, thickness, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(frame, shoulder_mid, max(12, scale // 18), accent, thickness, cv2.LINE_AA)

    _draw_top_risk_panel(
        frame,
        title=RISK_TITLE_BY_ID["lateral_trunk_lean"],
        headline=caption["title"],
        body=caption["body"],
        accent=accent,
    )


def _hip_shoulder_caption(risk: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(risk, dict):
        return None
    debug = risk.get("debug") or {}
    sequence_pattern = str(debug.get("sequence_pattern") or "").lower()
    signal = float(risk.get("signal_strength") or 0.0)
    if sequence_pattern == "shoulders_lead":
        if signal >= 0.45:
            return {
                "title": "Shoulders are starting too soon.",
                "body": "The shoulders are getting ahead of the hips near release.",
            }
        return {
            "title": "Shoulders are starting a bit early.",
            "body": "The top half is getting ahead slightly near release.",
        }
    if sequence_pattern == "hips_lead":
        if signal >= 0.45:
            return {
                "title": "Hips are getting ahead here.",
                "body": "The hips are separating from the shoulders near release.",
            }
        return {
            "title": "Hips are leading a little here.",
            "body": "The hips and shoulders are starting to split near release.",
        }
    if signal >= 0.45:
        return {
            "title": "Hips and shoulders are out of sync here.",
            "body": "The middle of the action is not staying together near release.",
        }
    return None


def _draw_hip_shoulder_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    risk: Optional[Dict[str, Any]],
) -> None:
    caption = _hip_shoulder_caption(risk)
    if not caption:
        return
    left_shoulder = _track_point(tracks, LEFT_SHOULDER, frame_idx)
    right_shoulder = _track_point(tracks, RIGHT_SHOULDER, frame_idx)
    left_hip = _track_point(tracks, LEFT_HIP, frame_idx)
    right_hip = _track_point(tracks, RIGHT_HIP, frame_idx)
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return

    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
    hip_mid = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
    signal = float((risk or {}).get("signal_strength") or 0.0)
    accent = (90, 220, 255) if signal < 0.45 else (0, 196, 255)
    scale = min(frame.shape[0], frame.shape[1])
    thickness = max(3, scale // 170)

    cv2.line(frame, left_shoulder, right_shoulder, accent, thickness, cv2.LINE_AA)
    cv2.line(frame, left_hip, right_hip, accent, thickness, cv2.LINE_AA)
    cv2.line(frame, hip_mid, shoulder_mid, accent, max(2, thickness - 1), cv2.LINE_AA)

    debug = (risk or {}).get("debug") or {}
    sequence_pattern = str(debug.get("sequence_pattern") or "").lower()
    arrow_dx = int(round(scale * 0.05))
    if sequence_pattern == "shoulders_lead":
        cv2.arrowedLine(
            frame,
            (shoulder_mid[0] - arrow_dx, shoulder_mid[1] - arrow_dx),
            (shoulder_mid[0] + arrow_dx, shoulder_mid[1] - arrow_dx),
            accent,
            thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )
    elif sequence_pattern == "hips_lead":
        cv2.arrowedLine(
            frame,
            (hip_mid[0] - arrow_dx, hip_mid[1] + arrow_dx),
            (hip_mid[0] + arrow_dx, hip_mid[1] + arrow_dx),
            accent,
            thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )

    _draw_top_risk_panel(
        frame,
        title=RISK_TITLE_BY_ID["hip_shoulder_mismatch"],
        headline=caption["title"],
        body=caption["body"],
        accent=accent,
    )


def _draw_release_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    risk_by_id: Dict[str, Dict[str, Any]],
    report_story: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, Any]] = None,
) -> None:
    preferred_risk_id = _story_risk_for_phase(
        report_story,
        phase_key="release",
        events=events,
    )
    if preferred_risk_id == "hip_shoulder_mismatch":
        _draw_hip_shoulder_callout(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            risk=risk_by_id.get("hip_shoulder_mismatch"),
        )
        return
    if preferred_risk_id == "lateral_trunk_lean":
        _draw_trunk_lean_callout(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            risk=risk_by_id.get("lateral_trunk_lean"),
        )
        return
    if isinstance(report_story, dict) and str(report_story.get("theme") or "") in {
        "working_pattern",
        "good_base",
    }:
        return

    hip_shoulder = risk_by_id.get("hip_shoulder_mismatch")
    trunk_lean = risk_by_id.get("lateral_trunk_lean")
    if _risk_weight(hip_shoulder) >= max(0.20, _risk_weight(trunk_lean) + 0.03):
        _draw_hip_shoulder_callout(
            frame,
            tracks=tracks,
            frame_idx=frame_idx,
            risk=hip_shoulder,
        )
        return
    _draw_trunk_lean_callout(
        frame,
        tracks=tracks,
        frame_idx=frame_idx,
        risk=trunk_lean,
    )


def _draw_speed_chip(
    frame: np.ndarray,
    *,
    speed: Optional[Dict[str, Any]],
) -> None:
    display = _speed_display_text(speed)
    if not display:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    low_conf = str(speed.get("display_policy") or "") == "show_low_confidence"
    accent = (110, 210, 255) if low_conf else (70, 225, 140)
    title = "Ball Speed"
    value = display
    card_w = int(round(width * 0.28))
    card_h = int(round(height * 0.10))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05))
    y1 = y0 + card_h
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=accent,
        alpha=0.84,
    )
    cv2.putText(
        frame,
        title,
        (x0 + 16, y0 + int(card_h * 0.34)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.40, min(width, height) / 1500.0),
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        value,
        (x0 + 16, y0 + int(card_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.58, min(width, height) / 1100.0),
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def _draw_action_chip(
    frame: np.ndarray,
    *,
    action: Optional[Dict[str, Any]],
    below_speed: bool = False,
) -> None:
    label = _format_action_label(action)
    if not label:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    accent = (130, 214, 255)
    title = "Action Type"
    card_w = int(round(width * 0.28))
    card_h = int(round(height * 0.10))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    base_y = int(round(height * 0.05))
    if below_speed:
        base_y += card_h + int(round(height * 0.015))
    y0 = base_y
    y1 = y0 + card_h
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=accent,
        alpha=0.84,
    )
    cv2.putText(
        frame,
        title,
        (x0 + 16, y0 + int(card_h * 0.34)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.40, min(width, height) / 1500.0),
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x0 + 16, y0 + int(card_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.56, min(width, height) / 1120.0),
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def _draw_legality_chip(
    frame: np.ndarray,
    *,
    elbow: Optional[Dict[str, Any]],
    stack_index: int = 0,
) -> None:
    verdict = str((elbow or {}).get("verdict") or "").strip().upper()
    if not verdict:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    title = "Legality"
    if verdict == "LEGAL":
        value = "Legal"
        accent = (70, 225, 140)
    elif verdict == "ILLEGAL":
        value = "Illegal"
        accent = (72, 92, 235)
    else:
        value = verdict.title()
        accent = (120, 210, 255)
    card_w = int(round(width * 0.28))
    card_h = int(round(height * 0.10))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05)) + stack_index * (
        card_h + int(round(height * 0.015))
    )
    y1 = y0 + card_h
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=accent,
        alpha=0.84,
    )
    cv2.putText(
        frame,
        title,
        (x0 + 16, y0 + int(card_h * 0.34)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.40, min(width, height) / 1500.0),
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        value,
        (x0 + 16, y0 + int(card_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.56, min(width, height) / 1120.0),
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def _summary_issue_lines(
    risk_by_id: Dict[str, Dict[str, Any]],
    *,
    events: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
) -> List[str]:
    positive_lines = _positive_recap_lines(report_story)
    if positive_lines:
        return positive_lines

    story_labels = _story_feature_labels(report_story)
    if story_labels:
        filtered = [
            label
            for label in story_labels
            if label != "Front-Leg Support" or _supports_ffc_story(events)
        ]
        if filtered:
            return filtered[:4]

    ranked: List[Tuple[float, str]] = []
    allow_ffc_stories = _supports_ffc_story(events)
    for risk_id, risk in (risk_by_id or {}).items():
        if risk_id in FFC_DEPENDENT_RISKS and not allow_ffc_stories:
            continue
        weight = _risk_weight(risk)
        signal = float((risk or {}).get("signal_strength") or 0.0)
        if signal < 0.20:
            continue
        title = RISK_TITLE_BY_ID.get(risk_id)
        if not title:
            continue
        ranked.append((weight, title))
    ranked.sort(reverse=True)
    deduped: List[str] = []
    for _, title in ranked:
        if title not in deduped:
            deduped.append(title)
        if len(deduped) >= 4:
            break
    return deduped


def _draw_end_summary(
    frame: np.ndarray,
    *,
    risk_by_id: Dict[str, Dict[str, Any]],
    events: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]],
    speed: Optional[Dict[str, Any]],
    elbow: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)

    width = frame.shape[1]
    height = frame.shape[0]
    top_y = int(round(height * 0.05))
    top_h = int(round(height * 0.14))
    top_w = int(round(width * 0.42))
    left_x = int(round(width * 0.05))
    right_x = int(round(width * 0.53))
    symptom_text = _summary_symptom_text(risk_by_id, events=events, report_story=report_story)
    load_watch_text = _summary_load_watch_text(risk_by_id, events=events, report_story=report_story)
    for x0, title, body, accent in (
        (left_x, "Symptom", symptom_text, (92, 220, 255)),
        (right_x, "Load Watch", load_watch_text, (0, 132, 255)),
    ):
        x1 = min(width - 18, x0 + top_w)
        y1 = top_y + top_h
        _overlay_panel(frame, x0=x0, y0=top_y, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
        cv2.putText(frame, title, (x0 + 16, top_y + int(top_h * 0.24)), cv2.FONT_HERSHEY_SIMPLEX, max(0.40, min(width, height) / 1500.0), accent, 1, cv2.LINE_AA)
        for idx, line in enumerate(str(body or "").splitlines()[:2]):
            cv2.putText(frame, line, (x0 + 16, top_y + int(top_h * (0.54 + idx * 0.22))), cv2.FONT_HERSHEY_SIMPLEX, max(0.54, min(width, height) / 1150.0), TEXT_COLOR, 2 if idx == 0 else 1, cv2.LINE_AA)

    bottom_y = int(round(height * 0.73))
    stat_h = int(round(height * 0.13))
    gap = int(round(width * 0.03))
    stat_w = int(round((width - (left_x * 2) - (gap * 2)) / 3.0))
    stats = [
        ("Speed", _speed_display_text(speed) or "-", (70, 225, 140)),
        ("Action Type", _format_action_label(action) or "-", (130, 214, 255)),
    ]
    verdict = str((elbow or {}).get("verdict") or "").strip().upper()
    legality_value = "Legal" if verdict == "LEGAL" else (verdict.title() if verdict else "-")
    legality_accent = (70, 225, 140) if verdict == "LEGAL" else ((72, 92, 235) if verdict == "ILLEGAL" else (120, 210, 255))
    stats.append(("Legality", legality_value, legality_accent))
    for idx, (title, value, accent) in enumerate(stats):
        x0 = left_x + idx * (stat_w + gap)
        x1 = x0 + stat_w
        y1 = bottom_y + stat_h
        _overlay_panel(frame, x0=x0, y0=bottom_y, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
        cv2.putText(frame, title, (x0 + 14, bottom_y + int(stat_h * 0.26)), cv2.FONT_HERSHEY_SIMPLEX, max(0.34, min(width, height) / 1700.0), accent, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x0 + 14, bottom_y + int(stat_h * 0.64)), cv2.FONT_HERSHEY_SIMPLEX, max(0.48, min(width, height) / 1250.0), TEXT_COLOR, 2, cv2.LINE_AA)


def _draw_foot_line_overlay(
    frame: np.ndarray,
    *,
    pose_frames: List[Dict[str, Any]],
    frame_idx: int,
    events: Optional[Dict[str, Any]],
    hand: Optional[str],
    risk: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(risk, dict):
        return
    width = frame.shape[1]
    height = frame.shape[0]
    front_toe_idx, front_heel_idx, back_toe_idx = _foot_indices(hand)
    bfc_frame = _safe_int(((events or {}).get("bfc") or {}).get("frame"))
    back_toe = _frame_point(pose_frames, frame_idx=bfc_frame if bfc_frame is not None else frame_idx, joint_idx=back_toe_idx, width=width, height=height)
    front_toe = _frame_point(pose_frames, frame_idx=frame_idx, joint_idx=front_toe_idx, width=width, height=height)
    front_heel = _frame_point(pose_frames, frame_idx=frame_idx, joint_idx=front_heel_idx, width=width, height=height)
    if not (back_toe and front_toe and front_heel):
        return

    signal = float(risk.get("signal_strength") or 0.0)
    accent = (120, 210, 255) if signal < 0.35 else ((0, 196, 255) if signal < 0.65 else (0, 126, 255))
    muted = (190, 202, 214)
    thickness = max(2, min(width, height) // 190)
    dx = front_toe[0] - back_toe[0]
    dy = front_toe[1] - back_toe[1]
    line_end = (back_toe[0] + int(round(dx * 1.10)), back_toe[1] + int(round(dy * 1.10)))

    cv2.line(frame, back_toe, line_end, muted, thickness, cv2.LINE_AA)
    cv2.line(frame, front_heel, front_toe, accent, thickness + 1, cv2.LINE_AA)
    cv2.circle(frame, front_toe, max(8, min(width, height) // 40), accent, thickness + 1, cv2.LINE_AA)


def _draw_hotspot_marker(
    frame: np.ndarray,
    *,
    center: Tuple[int, int],
    scale: int,
    weight: float,
    pulse_phase: float,
) -> None:
    overlay = frame.copy()
    pulse = 0.5 - 0.5 * np.cos(float(pulse_phase) * np.pi * 2.0)
    pulse_weight = max(0.0, min(1.0, weight))
    base = max(7, scale // 50)
    inner_ring = int(round(base * (1.7 + pulse_weight * 0.35)))
    mid_ring = int(round(base * (2.45 + pulse_weight * 0.45 + pulse * 0.35)))
    outer_ring = int(round(base * (3.15 + pulse_weight * 0.55 + pulse * 0.65)))
    halo_alpha = 0.18 + pulse * 0.12 + pulse_weight * 0.05
    cv2.circle(overlay, center, outer_ring, HOTSPOT_SOFT, -1, cv2.LINE_AA)
    cv2.circle(overlay, center, mid_ring, HOTSPOT_RING, max(2, scale // 280), cv2.LINE_AA)
    cv2.circle(overlay, center, inner_ring, HOTSPOT_RING, max(2, scale // 280), cv2.LINE_AA)
    cv2.circle(overlay, center, base, HOTSPOT_CORE, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, min(0.42, halo_alpha), frame, 1.0 - min(0.42, halo_alpha), 0.0, frame)


def _load_watch_support_text(load_watch_text: str) -> str:
    lower = str(load_watch_text or "").lower()
    if "lower back" in lower or "side trunk" in lower:
        return "This is where extra body load may build if the pattern repeats."
    return "This is where extra body load may build if the pattern repeats."


def _draw_load_watch_card(
    frame: np.ndarray,
    *,
    load_watch_text: str,
) -> None:
    width = frame.shape[1]
    height = frame.shape[0]
    card_w = int(round(width * 0.48))
    card_h = int(round(height * 0.14))
    top_y = int(round(height * 0.05))
    left_x = int(round(width * 0.05))
    x1 = min(width - 18, left_x + card_w)
    y1 = top_y + card_h
    accent = (0, 132, 255)
    _overlay_panel(frame, x0=left_x, y0=top_y, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    cv2.putText(frame, "Load watch", (left_x + 16, top_y + int(card_h * 0.24)), cv2.FONT_HERSHEY_SIMPLEX, max(0.40, min(width, height) / 1500.0), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, load_watch_text, (left_x + 16, top_y + int(card_h * 0.52)), cv2.FONT_HERSHEY_SIMPLEX, max(0.58, min(width, height) / 1120.0), TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, _load_watch_support_text(load_watch_text), (left_x + 16, top_y + int(card_h * 0.80)), cv2.FONT_HERSHEY_SIMPLEX, max(0.30, min(width, height) / 1900.0), MUTED_TEXT, 1, cv2.LINE_AA)


def _draw_load_watch_phase(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk_id: Optional[str],
    risk_by_id: Dict[str, Dict[str, Any]],
    load_watch_text: str,
    pulse_phase: float,
) -> None:
    if not risk_id:
        return
    regions = _load_hotspot_regions(
        tracks=tracks,
        frame_idx=frame_idx,
        hand=hand,
        risk_id=risk_id,
        risk_by_id=risk_by_id,
    )
    if not regions:
        return
    scale = min(frame.shape[0], frame.shape[1])
    _draw_load_watch_card(frame, load_watch_text=load_watch_text)
    for region in regions:
        center = region.get("center")
        if not isinstance(center, tuple) or len(center) != 2:
            continue
        _draw_hotspot_marker(
            frame,
            center=(int(center[0]), int(center[1])),
            scale=scale,
            weight=float(region.get("weight") or 0.8),
            pulse_phase=pulse_phase,
        )


def _pause_anchor_frames(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> Dict[int, str]:
    anchors: Dict[int, str] = {}
    for key in ("bfc", "ffc", "release"):
        frame_value = _safe_int(((events or {}).get(key) or {}).get("frame"))
        if frame_value is None:
            continue
        if start <= frame_value < stop:
            anchors[int(frame_value)] = key
    return dict(sorted(anchors.items()))


def render_skeleton_video(
    *,
    video_path: str,
    pose_frames: List[Dict[str, Any]],
    events: Optional[Dict[str, Any]] = None,
    hand: Optional[str] = None,
    action: Optional[Dict[str, Any]] = None,
    elbow: Optional[Dict[str, Any]] = None,
    risks: Optional[List[Dict[str, Any]]] = None,
    estimated_release_speed: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    pause_seconds: float = 3.0,
    slow_motion_factor: float = 3.0,
    end_summary_seconds: float = 2.5,
) -> Dict[str, Any]:
    if not video_path or not os.path.exists(video_path):
        return {"available": False, "reason": "missing_video_path"}
    if not pose_frames:
        return {"available": False, "reason": "missing_pose_frames"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"available": False, "reason": "video_open_failed"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(pose_frames))

    if width <= 0 or height <= 0:
        cap.release()
        return {"available": False, "reason": "missing_video_geometry"}

    start = max(0, int(start_frame))
    stop = min(total_frames, len(pose_frames), int(end_frame) if end_frame is not None else min(total_frames, len(pose_frames)))
    if stop <= start:
        cap.release()
        return {"available": False, "reason": "empty_render_window"}

    out_path = _make_output_path(output_path)
    intermediate_path = _intermediate_render_path(out_path)
    writer = cv2.VideoWriter(
        intermediate_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return {"available": False, "reason": "writer_open_failed"}

    try:
        tracks = _build_smoothed_tracks(pose_frames, width=width, height=height, fps=fps)
        pause_frames = max(0, int(round(float(pause_seconds or 0.0) * fps)))
        pause_anchors = _pause_anchor_frames(start=start, stop=stop, events=events)
        ffc_frame = _safe_int(((events or {}).get("ffc") or {}).get("frame"))
        release_frame = _safe_int(((events or {}).get("release") or {}).get("frame"))
        slow_motion_extra_frames = max(
            0, int(round(float(slow_motion_factor or 1.0))) - 1
        )
        slow_motion_start = None
        slow_motion_end = None
        if (
            ffc_frame is not None
            and release_frame is not None
            and start <= ffc_frame <= release_frame < stop
        ):
            slow_motion_start = ffc_frame
            slow_motion_end = release_frame
        risk_by_id = _risk_lookup(risks)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        frames_rendered = 0
        final_summary_frame: Optional[np.ndarray] = None
        while frame_idx < stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _draw_skeleton(frame, tracks, frame_idx)
            _draw_phase_overlay(frame, frame_idx=frame_idx, start=start, stop=stop, events=events)
            final_summary_frame = frame.copy()
            writer.write(frame)
            frames_rendered += 1
            if (
                slow_motion_start is not None
                and slow_motion_end is not None
                and slow_motion_start <= frame_idx <= slow_motion_end
            ):
                for _ in range(slow_motion_extra_frames):
                    writer.write(frame)
                    frames_rendered += 1
            pause_key = pause_anchors.get(frame_idx)
            if pause_frames > 0 and pause_key:
                paused_frame = frame.copy()
                hotspot_payload: Optional[Dict[str, str]] = None
                if pause_key == "ffc":
                    if _supports_ffc_story(events):
                        preferred_ffc_risk = _preferred_ffc_cue_risk_id(
                            risk_by_id,
                            report_story=report_story,
                            events=events,
                        )
                        should_draw_front_leg = preferred_ffc_risk == "knee_brace_failure"
                        should_draw_foot_line = preferred_ffc_risk == "foot_line_deviation"
                        if not preferred_ffc_risk:
                            preferred_ffc_risk = _story_risk_for_phase(
                                report_story,
                                phase_key="ffc",
                                events=events,
                            )
                        if not preferred_ffc_risk:
                            story_theme = str((report_story or {}).get("theme") or "")
                            if not report_story or story_theme not in {"working_pattern", "good_base"}:
                                front_leg_weight = _risk_weight(risk_by_id.get("knee_brace_failure"))
                                foot_line_weight = _risk_weight(risk_by_id.get("foot_line_deviation"))
                                if front_leg_weight >= foot_line_weight and front_leg_weight > 0.0:
                                    preferred_ffc_risk = "knee_brace_failure"
                                elif foot_line_weight > 0.0:
                                    preferred_ffc_risk = "foot_line_deviation"
                        if preferred_ffc_risk:
                            should_draw_front_leg = preferred_ffc_risk == "knee_brace_failure"
                            should_draw_foot_line = preferred_ffc_risk == "foot_line_deviation"
                        if should_draw_front_leg:
                            _draw_front_leg_support_callout(
                                paused_frame,
                                tracks=tracks,
                                frame_idx=frame_idx,
                                hand=hand,
                                risk=risk_by_id.get("knee_brace_failure"),
                            )
                        if should_draw_foot_line:
                            _draw_foot_line_overlay(
                                paused_frame,
                                pose_frames=pose_frames,
                                frame_idx=frame_idx,
                                events=events,
                                hand=hand,
                                risk=risk_by_id.get("foot_line_deviation"),
                            )
                        if preferred_ffc_risk:
                            hotspot_payload = {
                                "risk_id": preferred_ffc_risk,
                                "symptom_text": _summary_symptom_text(risk_by_id, events=events, report_story=report_story),
                                "load_watch_text": _load_watch_label(preferred_ffc_risk) or _summary_load_watch_text(risk_by_id, events=events, report_story=report_story),
                            }
                elif pause_key == "release":
                    release_hotspot_risk = _release_hotspot_risk_id(
                        risk_by_id,
                        events=events,
                        report_story=report_story,
                    )
                    _draw_release_callout(
                        paused_frame,
                        tracks=tracks,
                        frame_idx=frame_idx,
                        risk_by_id=risk_by_id,
                        report_story=report_story,
                        events=events,
                    )
                    _draw_speed_chip(
                        paused_frame,
                        speed=estimated_release_speed,
                    )
                    speed_visible = _speed_display_text(estimated_release_speed) is not None
                    _draw_action_chip(
                        paused_frame,
                        action=action,
                        below_speed=speed_visible,
                    )
                    _draw_legality_chip(
                        paused_frame,
                        elbow=elbow,
                        stack_index=2 if speed_visible else 1,
                    )
                    if release_hotspot_risk:
                        hotspot_payload = {
                            "risk_id": release_hotspot_risk,
                            "symptom_text": _summary_symptom_text(risk_by_id, events=events, report_story=report_story),
                            "load_watch_text": _load_watch_label(release_hotspot_risk) or _summary_load_watch_text(risk_by_id, events=events, report_story=report_story),
                        }
                cue_hold = pause_frames if hotspot_payload is None else max(1, int(round(pause_frames * 0.45)))
                hotspot_hold = 0 if hotspot_payload is None else max(1, pause_frames - cue_hold)
                for _ in range(cue_hold):
                    writer.write(paused_frame)
                    frames_rendered += 1
                for hotspot_idx in range(hotspot_hold):
                    hotspot_frame = frame.copy()
                    pulse_phase = 0.0 if hotspot_hold <= 1 else float(hotspot_idx) / float(hotspot_hold - 1)
                    _draw_load_watch_phase(
                        hotspot_frame,
                        tracks=tracks,
                        frame_idx=frame_idx,
                        hand=hand,
                        risk_id=str((hotspot_payload or {}).get("risk_id") or ""),
                        risk_by_id=risk_by_id,
                        load_watch_text=str((hotspot_payload or {}).get("load_watch_text") or ""),
                        pulse_phase=pulse_phase,
                    )
                    writer.write(hotspot_frame)
                    frames_rendered += 1
            frame_idx += 1

        summary_hold_frames = max(0, int(round(float(end_summary_seconds or 0.0) * fps)))
        if final_summary_frame is not None and summary_hold_frames > 0:
            summary_frame = final_summary_frame.copy()
            _draw_end_summary(
                summary_frame,
                risk_by_id=risk_by_id,
                events=events,
                action=action,
                speed=estimated_release_speed,
                elbow=elbow,
                report_story=report_story,
            )
            for _ in range(summary_hold_frames):
                writer.write(summary_frame)
                frames_rendered += 1
        writer.release()
        cap.release()
    except Exception as exc:
        writer.release()
        cap.release()
        try:
            if os.path.exists(intermediate_path):
                os.remove(intermediate_path)
        except OSError:
            pass
        logger.exception("[coach_video_renderer] Skeleton render failed: %s", exc)
        return {"available": False, "reason": "render_failed", "detail": str(exc)}

    final_path, encoding = _finalize_render_video(intermediate_path, out_path)

    logger.info(
        "[coach_video_renderer] Rendered skeleton video path=%s frames=%s fps=%.2f",
        final_path,
        frames_rendered,
        fps,
    )
    return {
        "available": True,
        "path": final_path,
        "fps": round(fps, 3),
        "frames_rendered": frames_rendered,
        "width": width,
        "height": height,
        "start_frame": start,
        "end_frame": max(start, stop - 1),
        "style": "skeleton_phase_v1",
        "pause_seconds": round(float(pause_seconds or 0.0), 2),
        "slow_motion_factor": round(float(slow_motion_factor or 1.0), 2),
        "end_summary_seconds": round(float(end_summary_seconds or 0.0), 2),
        "encoding": encoding,
    }
