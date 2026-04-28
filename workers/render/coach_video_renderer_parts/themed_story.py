from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size, _pil_text_block_height
from .pil_context import _bgr_to_rgb
from .themed_card_shell import _draw_themed_card_shell

def _draw_themed_story_card(
    draw: Any,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    title: str,
    headline: str,
    body: str,
    accent: Tuple[int, int, int],
    width: int,
    height: int,
    title_scale_boost: float = 1.0,
    headline_scale_boost: float = 1.0,
    body_scale_boost: float = 1.0,
) -> None:
    card_w = max(1, x1 - x0)
    card_h = max(1, y1 - y0)
    rail_offset = _draw_themed_card_shell(
        draw,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        accent=accent,
        width=width,
        height=height,
    )
    inner_pad_x = max(14, int(round(card_w * 0.064))) + rail_offset
    inner_pad_y = max(11, int(round(card_h * 0.09)))
    content_width = max(40, x1 - x0 - inner_pad_x * 2)
    content_height = max(36, y1 - y0 - inner_pad_y * 2)
    title_base = max(18, int(round(card_h * 0.125 * title_scale_boost)))
    title_min = max(15, int(round(card_h * 0.098 * title_scale_boost)))
    headline_base = max(27, int(round(card_h * 0.195 * headline_scale_boost)))
    headline_min = max(20, int(round(card_h * 0.142 * headline_scale_boost)))
    body_base = max(17, int(round(card_h * 0.128 * body_scale_boost)))
    body_min = max(13, int(round(card_h * 0.094 * body_scale_boost)))
    title_font = None
    title_lines: List[str] = []
    headline_font = None
    headline_lines: List[str] = []
    body_font = None
    body_lines: List[str] = []
    line_gap = max(4, int(round(card_h * 0.040)))
    section_gap = max(12, int(round(card_h * 0.090)))
    current_title_base = title_base
    current_headline_base = headline_base
    current_body_base = body_base
    for _ in range(8):
        title_font, title_lines = _fit_pil_wrapped_text(
            draw,
            str(title or ""),
            font_file=LABEL_FONT_FILE,
            base_size=current_title_base,
            min_size=title_min,
            max_width=content_width,
            max_lines=2,
        )
        headline_font, headline_lines = _fit_pil_wrapped_text(
            draw,
            str(headline or ""),
            font_file=BODY_FONT_FILE,
            base_size=current_headline_base,
            min_size=headline_min,
            max_width=content_width,
            max_lines=2,
        )
        body_font, body_lines = _fit_pil_wrapped_text(
            draw,
            str(body or ""),
            font_file=BODY_FONT_MEDIUM_FILE,
            base_size=current_body_base,
            min_size=body_min,
            max_width=content_width,
            max_lines=1,
        )
        total_h = 0
        title_h = _pil_text_block_height(draw, title_lines, title_font, line_gap=line_gap)
        headline_h = _pil_text_block_height(draw, headline_lines, headline_font, line_gap=line_gap)
        body_h = _pil_text_block_height(draw, body_lines, body_font, line_gap=line_gap)
        if title_h:
            total_h += title_h
        if headline_h:
            if total_h:
                total_h += section_gap
            total_h += headline_h
        if body_h:
            if total_h:
                total_h += max(3, section_gap - 1)
            total_h += body_h
        if total_h <= content_height:
            break
        current_title_base = max(title_min, current_title_base - 1)
        current_headline_base = max(headline_min, current_headline_base - 2)
        current_body_base = max(body_min, current_body_base - 1)

    current_y = y0 + inner_pad_y
    for line in title_lines:
        if title_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            str(line or "").upper(),
            font=title_font,
            fill=_bgr_to_rgb(accent),
        )
        _, line_h = _pil_text_size(draw, str(line or "").upper(), title_font)
        current_y += line_h + line_gap
    if title_lines:
        current_y += section_gap
    for line in headline_lines:
        if headline_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=headline_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, headline_font)
        current_y += line_h + line_gap
    if headline_lines and body_lines:
        current_y += max(3, section_gap - 1)
    for line in body_lines:
        if body_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=body_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, body_font)
        current_y += line_h + line_gap
