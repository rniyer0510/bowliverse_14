from __future__ import annotations

from .shared import *
from .analytics import _speed_display_text
from .story_logic import _format_action_label
from .drawing_base import _overlay_panel
from .font_utils import _load_theme_font
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context


def _draw_chip_text(
    frame: np.ndarray,
    *,
    x0: int,
    y0: int,
    card_w: int,
    card_h: int,
    title: str,
    value: str,
    accent: Tuple[int, int, int],
) -> None:
    if Image is None or ImageDraw is None:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    title_font = _load_theme_font(LABEL_FONT_FILE, max(17, int(round(min(width, height) * 0.036))))
    value_font = _load_theme_font(BODY_FONT_FILE, max(24, int(round(min(width, height) * 0.050))))
    if title_font is None or value_font is None:
        return
    image, overlay, draw = _frame_draw_context(frame)
    draw.text(
        (x0 + 16, y0 + int(card_h * 0.14)),
        title,
        font=title_font,
        fill=_bgr_to_rgb(accent, 255),
    )
    draw.text(
        (x0 + 16, y0 + int(card_h * 0.44)),
        value,
        font=value_font,
        fill=_bgr_to_rgb(TEXT_COLOR, 255),
    )
    _commit_frame_draw_context(frame, image, overlay)


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
    card_w = int(round(width * 0.32))
    card_h = int(round(height * 0.12))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05))
    y1 = y0 + card_h
    _overlay_panel(frame, x0=x0, y0=y0, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    _draw_chip_text(frame, x0=x0, y0=y0, card_w=card_w, card_h=card_h, title=title, value=display, accent=accent)


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
    card_w = int(round(width * 0.32))
    card_h = int(round(height * 0.12))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05)) + (card_h + int(round(height * 0.015)) if below_speed else 0)
    y1 = y0 + card_h
    _overlay_panel(frame, x0=x0, y0=y0, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    _draw_chip_text(frame, x0=x0, y0=y0, card_w=card_w, card_h=card_h, title=title, value=label, accent=accent)


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
    card_w = int(round(width * 0.32))
    card_h = int(round(height * 0.12))
    x1 = int(round(width * 0.95))
    x0 = x1 - card_w
    y0 = int(round(height * 0.05)) + stack_index * (card_h + int(round(height * 0.015)))
    y1 = y0 + card_h
    _overlay_panel(frame, x0=x0, y0=y0, x1=x1, y1=y1, fill_color=PANEL_BG, edge_color=accent, alpha=0.84)
    _draw_chip_text(frame, x0=x0, y0=y0, card_w=card_w, card_h=card_h, title=title, value=value, accent=accent)
