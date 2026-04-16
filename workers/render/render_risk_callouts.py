from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .render_constants import (
    ACTIVE_EDGE, JOINT_OUTER, LEFT_HIP, LEFT_SHOULDER, PANEL_BG, PANEL_EDGE,
    RIGHT_HIP, RIGHT_SHOULDER, RISK_TITLE_BY_ID, SKELETON_SHADOW, TEXT_COLOR, MUTED_TEXT
)
from .render_helpers import (
    _foot_indices, _front_leg_joints, _format_action_label, _risk_weight,
    _speed_display_text, _story_risk_for_phase, _supports_ffc_story,
)
from .render_phase import _overlay_panel
from .render_tracks import _frame_point, _track_point

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


