from __future__ import annotations

from .shared import *
from .analytics import _speed_display_text
from .story_logic import _format_action_label
from .drawing_base import _overlay_panel


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
    card_w = int(round(width * 0.28))
    card_h = int(round(height * 0.10))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05))
    y1 = y0 + card_h
    _overlay_panel(frame, x0=x0, y0=y0, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    cv2.putText(frame, title, (x0 + 16, y0 + int(card_h * 0.34)), cv2.FONT_HERSHEY_SIMPLEX, max(0.40, min(width, height) / 1500.0), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, display, (x0 + 16, y0 + int(card_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, max(0.58, min(width, height) / 1100.0), TEXT_COLOR, 2, cv2.LINE_AA)


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
    y0 = int(round(height * 0.05)) + (card_h + int(round(height * 0.015)) if below_speed else 0)
    y1 = y0 + card_h
    _overlay_panel(frame, x0=x0, y0=y0, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    cv2.putText(frame, title, (x0 + 16, y0 + int(card_h * 0.34)), cv2.FONT_HERSHEY_SIMPLEX, max(0.40, min(width, height) / 1500.0), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x0 + 16, y0 + int(card_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, max(0.56, min(width, height) / 1120.0), TEXT_COLOR, 2, cv2.LINE_AA)


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
        value, accent = "Legal", (70, 225, 140)
    elif verdict == "ILLEGAL":
        value, accent = "Illegal", (72, 92, 235)
    else:
        value, accent = verdict.title(), (120, 210, 255)
    card_w = int(round(width * 0.28))
    card_h = int(round(height * 0.10))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05)) + stack_index * (card_h + int(round(height * 0.015)))
    y1 = y0 + card_h
    _overlay_panel(frame, x0=x0, y0=y0, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    cv2.putText(frame, title, (x0 + 16, y0 + int(card_h * 0.34)), cv2.FONT_HERSHEY_SIMPLEX, max(0.40, min(width, height) / 1500.0), accent, 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x0 + 16, y0 + int(card_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, max(0.64, min(width, height) / 980.0), TEXT_COLOR, 2, cv2.LINE_AA)
