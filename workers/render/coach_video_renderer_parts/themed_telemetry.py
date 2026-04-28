from __future__ import annotations
from .shared import *
from .font_utils import _load_theme_font, _fit_pil_wrapped_text, _pil_text_size
from .pil_context import _bgr_to_rgb

def _telemetry_value_layout(
    draw: Any,
    *,
    card_w: int,
    card_h: int,
    title: str,
    value: str,
    title_scale_boost: float,
    value_scale_boost: float,
) -> Tuple[Any, int, Any, List[str], List[int], int, int, int, int]:
    inner_pad_x = max(14, int(round(card_w * 0.11)))
    inner_pad_y = max(9, int(round(card_h * 0.16)))
    dot_r = max(4, int(round(card_h * 0.055)))
    title_size = max(11, int(round(card_h * 0.15 * title_scale_boost)))
    title_font = _load_theme_font(LABEL_FONT_FILE, title_size)
    value_font, value_lines = _fit_pil_wrapped_text(
        draw,
        str(value or ""),
        font_file=BODY_FONT_FILE,
        base_size=max(18, int(round(card_h * 0.28 * value_scale_boost))),
        min_size=max(12, int(round(card_h * 0.18 * value_scale_boost))),
        max_width=max(28, card_w - inner_pad_x * 2),
        max_lines=2,
    )
    title_h = _pil_text_size(draw, str(title or "").upper(), title_font)[1] if title_font is not None else max(10, int(round(card_h * 0.14)))
    value_heights = [_pil_text_size(draw, line, value_font)[1] for line in value_lines] if value_font is not None else []
    value_gap = max(2, int(round(card_h * 0.03)))
    needed_h = inner_pad_y * 2 + dot_r * 2 + title_h + max(7, int(round(card_h * 0.10)))
    if value_heights:
        needed_h += sum(value_heights) + value_gap * max(0, len(value_heights) - 1)
    return title_font, title_h, value_font, value_lines, value_heights, inner_pad_x, inner_pad_y, dot_r, needed_h

def _draw_themed_telemetry_pill(
    draw: Any,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    title: str,
    value: str,
    accent: Tuple[int, int, int],
    width: int,
    height: int,
    title_scale_boost: float = 1.0,
    value_scale_boost: float = 1.0,
) -> None:
    card_w = max(1, x1 - x0)
    card_h = max(1, y1 - y0)
    measure_draw = draw
    title_font, title_h, value_font, value_lines, value_heights, inner_pad_x, inner_pad_y, dot_r, needed_h = _telemetry_value_layout(
        measure_draw,
        card_w=card_w,
        card_h=card_h,
        title=title,
        value=value,
        title_scale_boost=title_scale_boost,
        value_scale_boost=value_scale_boost,
    )
    y1 = max(y1, y0 + needed_h)
    card_h = max(1, y1 - y0)
    radius = max(20, int(round(card_h * 0.46)))
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=_bgr_to_rgb((12, 14, 14), 232),
        outline=_bgr_to_rgb((34, 36, 35), 102),
        width=1,
    )
    dot_cx = x0 + inner_pad_x + dot_r
    dot_cy = y0 + inner_pad_y + dot_r + 1
    draw.ellipse(
        (dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r),
        fill=_bgr_to_rgb(accent, 255),
    )
    title_x = dot_cx + dot_r + max(7, int(round(card_w * 0.035)))
    title_y = y0 + inner_pad_y - 1
    if title_font is not None:
        draw.text(
            (title_x, title_y),
            str(title or "").upper(),
            font=title_font,
            fill=_bgr_to_rgb(THEME_TEXT_SECONDARY, 170),
        )
    value_y = title_y + title_h + max(7, int(round(card_h * 0.10)))
    for line in value_lines:
        if value_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, value_y),
            line,
            font=value_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        line_h = value_heights.pop(0) if value_heights else _pil_text_size(draw, line, value_font)[1]
        value_y += line_h + max(2, int(round(card_h * 0.03)))
def _draw_themed_stat_card(
    draw: Any,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    title: str,
    value: str,
    accent: Tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    card_w = max(1, x1 - x0)
    card_h = max(1, y1 - y0)
    radius = max(14, int(round(min(width, height) * 0.024)))
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=_bgr_to_rgb(THEME_SURFACE, 235),
        outline=_bgr_to_rgb(accent, 255),
        width=max(1, int(round(min(width, height) * 0.004))),
    )
    inner_pad_x = max(12, int(round(card_w * 0.09)))
    inner_pad_y = max(10, int(round(card_h * 0.14)))
    content_width = max(32, x1 - x0 - inner_pad_x * 2)
    title_font, title_lines = _fit_pil_wrapped_text(
        draw,
        str(title or ""),
        font_file=BODY_FONT_MEDIUM_FILE,
        base_size=max(15, int(round(card_h * 0.16))),
        min_size=max(12, int(round(card_h * 0.12))),
        max_width=content_width,
        max_lines=1,
    )
    value_font, value_lines = _fit_pil_wrapped_text(
        draw,
        str(value or ""),
        font_file=BODY_FONT_FILE,
        base_size=max(20, int(round(card_h * 0.22))),
        min_size=max(15, int(round(card_h * 0.16))),
        max_width=content_width,
        max_lines=2,
    )
    line_gap = max(2, int(round(card_h * 0.035)))
    current_y = y0 + inner_pad_y
    for line in title_lines:
        if title_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=title_font,
            fill=_bgr_to_rgb(THEME_TEXT_SECONDARY),
        )
        _, line_h = _pil_text_size(draw, line, title_font)
        current_y += line_h + line_gap
    current_y += max(4, int(round((y1 - y0) * 0.05)))
    for line in value_lines:
        if value_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=value_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, value_font)
        current_y += line_h + line_gap
