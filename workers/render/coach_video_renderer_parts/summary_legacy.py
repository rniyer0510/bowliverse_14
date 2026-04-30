from __future__ import annotations
from .shared import *
from .analytics import _speed_display_text
from .story_logic import _format_action_label
from .drawing_base import _apply_bottom_scrim
from .misc_helpers import _summary_telemetry_layout, _story_headline_and_support
from .story_overlay import _draw_story_overlay_card
from .themed_telemetry import _draw_themed_summary_metric_card
from .pil_context import _frame_draw_context, _commit_frame_draw_context

def _draw_end_summary_legacy(
    frame: np.ndarray,
    *,
    risk_by_id: Dict[str, Dict[str, Any]],
    events: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]],
    speed: Optional[Dict[str, Any]],
    elbow: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
    root_cause: Optional[Dict[str, Any]] = None,
) -> None:
    # Keep summary typography on the same themed font family as the BFC/FFC/Release cards.
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=4, sigmaY=4)
    cv2.addWeighted(blurred, 0.16, frame, 0.84, 0.0, frame)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame)

    width = frame.shape[1]
    height = frame.shape[0]
    top_y = int(round(height * 0.045))
    top_h = int(round(height * 0.18))
    top_gap = int(round(height * 0.018))
    top_w = int(round(width * 0.92))
    left_x = int(round(width * 0.04))
    _apply_bottom_scrim(
        frame,
        start_y=int(round(height * 0.60)),
        end_y=height,
        max_alpha=0.52,
    )
    symptom_text = _summary_symptom_text(
        risk_by_id,
        events=events,
        report_story=report_story,
        root_cause=root_cause,
    )
    load_watch_text = _summary_load_watch_text(
        risk_by_id,
        events=events,
        report_story=report_story,
        root_cause=root_cause,
    )
    symptom_title = _summary_symptom_title(
        report_story=report_story,
        root_cause=root_cause,
    )
    load_watch_title = _summary_load_watch_title(
        report_story=report_story,
        root_cause=root_cause,
    )
    x0 = left_x
    y0 = top_y
    x1 = min(width - 18, x0 + top_w)
    y1 = y0 + top_h
    headline, support = _story_headline_and_support(str(symptom_text or ""))
    secondary_body = str(load_watch_text or "").strip()
    if secondary_body:
        secondary_body = f"{str(load_watch_title or '').upper()}: {secondary_body}"
    if support:
        merged_body = f"{support} {secondary_body}".strip()
    else:
        merged_body = secondary_body
    final_y1 = _draw_story_overlay_card(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        title=symptom_title,
        headline=headline,
        body=merged_body,
        accent=(92, 220, 255),
        title_scale_boost=1.18,
        headline_scale_boost=1.10,
        body_scale_boost=1.10,
        headline_max_lines=2,
        body_max_lines=3,
        unify_body_with_headline=True,
    )
    current_top_y = int(final_y1) + top_gap

    if root_cause_status != "not_interpretable":
        bottom_y = max(int(round(height * 0.695)), current_top_y + int(round(height * 0.03)))
        telemetry = _summary_telemetry_layout(width, height)
        stat_h = int(telemetry["stat_h"])
        gap = int(telemetry["gap"])
        stat_w = int(telemetry["stat_w"])
        stats = [
            ("Speed", _speed_display_text(speed) or "-", (70, 225, 140)),
            ("Action Type", _format_action_label(action) or "-", (130, 214, 255)),
        ]
        verdict = str((elbow or {}).get("verdict") or "").strip().upper()
        if verdict == "LEGAL":
            legality_value = "Legal"
            legality_accent = (70, 225, 140)
        elif verdict == "ILLEGAL":
            legality_value = "Illegal"
            legality_accent = (72, 92, 235)
        elif verdict == "BORDERLINE":
            legality_value = "Borderline"
            legality_accent = (0, 196, 255)
        elif verdict == "SUSPECT":
            legality_value = "Suspect"
            legality_accent = (0, 196, 255)
        else:
            legality_value = "-"
            legality_accent = (120, 210, 255)
        stats.append(("Legality", legality_value, legality_accent))
        image, overlay, draw = _frame_draw_context(frame)
        for idx, (title, value, accent) in enumerate(stats):
            x0 = left_x + idx * (stat_w + gap)
            x1 = x0 + stat_w
            y1 = bottom_y + stat_h
            _draw_themed_summary_metric_card(
                draw,
                x0=x0,
                y0=bottom_y,
                x1=x1,
                y1=y1,
                title=title,
                value=value,
                accent=accent,
                width=width,
                height=height,
            )
        _commit_frame_draw_context(frame, image, overlay)
def _draw_end_summary(
    frame: np.ndarray,
    *,
    risk_by_id: Dict[str, Dict[str, Any]],
    events: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]],
    speed: Optional[Dict[str, Any]],
    elbow: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
    root_cause: Optional[Dict[str, Any]] = None,
) -> None:
    _draw_end_summary_legacy(
        frame,
        risk_by_id=risk_by_id,
        events=events,
        action=action,
        speed=speed,
        elbow=elbow,
        report_story=report_story,
        root_cause=root_cause,
    )
