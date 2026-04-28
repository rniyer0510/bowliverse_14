from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size, _pil_text_block_height, _phase_label_font_size
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
) -> int:
    card_w = max(1, x1 - x0)
    text_size = _phase_label_font_size(min(width, height))
    title_size = text_size
    title_font = _fit_pil_wrapped_text(
        draw,
        str(title or ""),
        font_file=LABEL_FONT_FILE,
        base_size=title_size,
        min_size=title_size,
        max_width=max(40, int(round(card_w * 0.75))),
        max_lines=2,
    )[0]
    rail_probe = max(3, int(round(card_w * 0.020))) + max(3, int(round(min(width, height) * 0.006)))
    inner_pad_x = max(14, int(round(card_w * 0.064))) + rail_probe
    content_width = max(40, x1 - x0 - inner_pad_x * 2)
    headline_font, headline_lines = _fit_pil_wrapped_text(
        draw,
        str(headline or ""),
        font_file=BODY_FONT_FILE,
        base_size=text_size,
        min_size=text_size,
        max_width=content_width,
        max_lines=4,
    )
    body_font, body_lines = _fit_pil_wrapped_text(
        draw,
        str(body or ""),
        font_file=BODY_FONT_MEDIUM_FILE,
        base_size=text_size,
        min_size=text_size,
        max_width=content_width,
        max_lines=3,
    )
    title_lines = _fit_pil_wrapped_text(
        draw,
        str(title or ""),
        font_file=LABEL_FONT_FILE,
        base_size=title_size,
        min_size=title_size,
        max_width=content_width,
        max_lines=2,
    )[1]
    line_gap = max(5, int(round(text_size * 0.30)))
    section_gap = max(10, int(round(text_size * 0.55)))
    title_h = _pil_text_block_height(draw, title_lines, title_font, line_gap=line_gap)
    headline_h = _pil_text_block_height(draw, headline_lines, headline_font, line_gap=line_gap)
    body_h = _pil_text_block_height(draw, body_lines, body_font, line_gap=line_gap)
    total_h = 0
    if title_h:
        total_h += title_h
    if headline_h:
        if total_h:
            total_h += section_gap
        total_h += headline_h
    if body_h:
        if total_h:
            total_h += max(4, section_gap - 2)
        total_h += body_h
    inner_pad_y = max(12, int(round(text_size * 0.70)))
    y1 = max(y1, y0 + inner_pad_y * 2 + total_h)
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
    return y1
