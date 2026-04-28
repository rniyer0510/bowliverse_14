from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context
from .themed_story import _draw_themed_story_card
from .themed_card_shell import _draw_themed_card_shell

def _bubble_copy(
    *,
    title: str,
    headline: str,
    body: str,
) -> str:
    for candidate in (headline, title, body):
        resolved = " ".join(str(candidate or "").split())
        if resolved:
            return resolved
    return ""
def _reading_hold_frames(
    *,
    text: str,
    fps: float,
    minimum_seconds: float = 1.55,
    max_seconds: float = 3.15,
) -> int:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return 0
    words = max(1, len(cleaned.split(" ")))
    reading_seconds = max(
        float(minimum_seconds),
        min(float(max_seconds), 0.92 + words * 0.31),
    )
    return max(1, int(round(max(1.0, float(fps)) * reading_seconds)))
def _draw_pointer_bubble(
    frame: np.ndarray,
    *,
    anchor: Tuple[int, int],
    text: str,
    accent: Tuple[int, int, int],
) -> None:
    message = " ".join(str(text or "").split())
    if not message or Image is None or ImageDraw is None:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    image, overlay, draw = _frame_draw_context(frame)
    card_x0 = int(round(width * 0.05))
    card_y0 = int(round(height * 0.05))
    card_x1 = int(round(width * 0.61))
    card_y1 = card_y0 + int(round(height * 0.21))
    card_w = max(1, card_x1 - card_x0)
    card_h = max(1, card_y1 - card_y0)
    rail_offset = _draw_themed_card_shell(
        draw,
        x0=card_x0,
        y0=card_y0,
        x1=card_x1,
        y1=card_y1,
        accent=accent,
        width=width,
        height=height,
    )
    inner_pad_x = max(16, int(round(card_w * 0.070))) + rail_offset
    inner_pad_y = max(14, int(round(card_h * 0.12)))
    content_width = max(40, card_x1 - card_x0 - inner_pad_x * 2)
    content_height = max(36, card_y1 - card_y0 - inner_pad_y * 2)
    headline_base = max(30, int(round(card_h * 0.24)))
    headline_min = max(22, int(round(card_h * 0.17)))
    headline_font = None
    headline_lines: List[str] = []
    line_gap = max(5, int(round(card_h * 0.045)))
    current_headline_base = headline_base
    for _ in range(10):
        headline_font, headline_lines = _fit_pil_wrapped_text(
            draw,
            message,
            font_file=BODY_FONT_FILE,
            base_size=current_headline_base,
            min_size=headline_min,
            max_width=content_width,
            max_lines=4,
        )
        total_h = 0
        if headline_font is not None and headline_lines:
            total_h = sum(_pil_text_size(draw, line, headline_font)[1] for line in headline_lines)
            total_h += line_gap * max(0, len(headline_lines) - 1)
        if total_h <= content_height:
            break
        current_headline_base = max(headline_min, current_headline_base - 2)
    current_y = card_y0 + inner_pad_y
    for line in headline_lines:
        if headline_font is None:
            break
        draw.text(
            (card_x0 + inner_pad_x, current_y),
            line,
            font=headline_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, headline_font)
        current_y += line_h + line_gap
    scale = min(width, height)
    line_x = card_x0 + int(round((card_x1 - card_x0) * 0.44))
    line_y0 = card_y1 + max(8, int(round(scale * 0.012)))
    line_y1 = line_y0 + max(22, int(round(scale * 0.044)))
    draw.line(
        (line_x, line_y0, line_x, line_y1),
        fill=(255, 255, 255, 168),
        width=max(2, int(round(scale * 0.004))),
    )
    draw.line(
        (line_x, line_y1, anchor[0], anchor[1]),
        fill=(255, 255, 255, 150),
        width=max(2, int(round(scale * 0.004))),
    )
    _commit_frame_draw_context(frame, image, overlay)
def _draw_top_risk_panel(
    frame: np.ndarray,
    *,
    title: str,
    headline: str,
    body: str,
    accent: Tuple[int, int, int],
    anchor: Optional[Tuple[int, int]] = None,
) -> None:
    if Image is None or ImageDraw is None:
        return
    if not any(str(part or "").strip() for part in (title, headline, body)):
        return
    width = frame.shape[1]
    height = frame.shape[0]
    image, overlay, draw = _frame_draw_context(frame)
    card_x0 = int(round(width * 0.05))
    card_y0 = int(round(height * 0.05))
    card_x1 = int(round(width * 0.58))
    card_y1 = card_y0 + int(round(height * 0.20))
    _draw_themed_story_card(
        draw,
        x0=card_x0,
        y0=card_y0,
        x1=card_x1,
        y1=card_y1,
        title=str(title or ""),
        headline=str(headline or ""),
        body=str(body or ""),
        accent=accent,
        width=width,
        height=height,
        title_scale_boost=1.0,
        headline_scale_boost=1.08,
        body_scale_boost=1.0,
    )
    if anchor is not None:
        scale = min(width, height)
        line_x = card_x0 + int(round((card_x1 - card_x0) * 0.44))
        line_y0 = card_y1 + max(8, int(round(scale * 0.012)))
        line_y1 = line_y0 + max(22, int(round(scale * 0.044)))
        draw.line(
            (line_x, line_y0, line_x, line_y1),
            fill=(255, 255, 255, 168),
            width=max(2, int(round(scale * 0.004))),
        )
        draw.line(
            (line_x, line_y1, anchor[0], anchor[1]),
            fill=(255, 255, 255, 150),
            width=max(2, int(round(scale * 0.004))),
        )
    _commit_frame_draw_context(frame, image, overlay)
