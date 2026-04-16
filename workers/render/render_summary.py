from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .render_constants import ACTIVE_EDGE, MUTED_TEXT, PANEL_BG, PANEL_EDGE, TEXT_COLOR
from .render_helpers import _positive_recap_lines, _story_feature_labels, _supports_ffc_story, _risk_weight, _speed_display_text
from .render_constants import FFC_DEPENDENT_RISKS, RISK_TITLE_BY_ID
from .render_phase import _overlay_panel
from .render_risk_callouts import _draw_action_chip, _draw_legality_chip, _draw_speed_chip

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
    card_w = int(round(width * 0.60))
    card_h = int(round(height * 0.24))
    x0 = int(round(width * 0.05))
    y0 = int(round(height * 0.08))
    x1 = x0 + card_w
    y1 = y0 + card_h
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=PANEL_EDGE,
        alpha=0.88,
    )

    story_theme = str((report_story or {}).get("theme") or "")
    heading = "Action recap" if story_theme in {"working_pattern", "good_base"} else "What to watch"
    cv2.putText(
        frame,
        heading,
        (x0 + 18, y0 + int(card_h * 0.20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.46, min(width, height) / 1450.0),
        MUTED_TEXT,
        1,
        cv2.LINE_AA,
    )

    issue_lines = _summary_issue_lines(
        risk_by_id,
        events=events,
        report_story=report_story,
    )
    if not issue_lines:
        if story_theme in {"working_pattern", "good_base"}:
            issue_lines = ["Strong working pattern", "Keep repeating this shape"]
        else:
            issue_lines = ["No major issue flagged"]
    start_y = y0 + int(card_h * 0.42)
    row_gap = int(card_h * 0.18)
    for idx, line in enumerate(issue_lines[:4]):
        row_y = start_y + idx * row_gap
        cv2.putText(
            frame,
            line,
            (x0 + 30, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.44, min(width, height) / 1500.0),
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (x0 + 18, row_y - 4), max(3, min(width, height) // 220), ACTIVE_EDGE, -1, cv2.LINE_AA)

    speed_visible = _speed_display_text(speed) is not None
    _draw_speed_chip(frame, speed=speed)
    _draw_action_chip(
        frame,
        action=action,
        below_speed=speed_visible,
    )
    _draw_legality_chip(
        frame,
        elbow=elbow,
        stack_index=2 if speed_visible else 1,
    )


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


