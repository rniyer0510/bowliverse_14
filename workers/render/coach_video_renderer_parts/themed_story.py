from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size, _pil_text_block_height, _phase_label_font_size
from .pil_context import _bgr_to_rgb
from .themed_card_shell import _draw_themed_card_shell

def _normalized_story_label(text: str) -> str:
    cleaned = "".join(ch.lower() for ch in str(text or "").strip() if ch.isalnum() or ch.isspace())
    return " ".join(cleaned.split())

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
    title_max_lines: int = 2,
    headline_max_lines: int = 4,
    body_max_lines: int = 3,
    vertical_align: str = "top",
) -> int:
    card_w = max(1, x1 - x0)
    scale = min(width, height)
    base_text_size = _phase_label_font_size(scale)
    normalized_title = _normalized_story_label(title)
    normalized_headline = _normalized_story_label(headline)
    if normalized_title and normalized_headline and normalized_title == normalized_headline:
        headline = ""
    title_size = max(14, int(round(base_text_size * 0.82 * max(0.7, float(title_scale_boost)))))
    headline_size = max(18, int(round(base_text_size * 1.08 * max(0.7, float(headline_scale_boost)))))
    body_size = max(14, int(round(base_text_size * 0.82 * max(0.7, float(body_scale_boost)))))
    rail_probe = max(3, int(round(card_w * 0.020))) + max(3, int(round(min(width, height) * 0.006)))
    left_pad = max(14, int(round(card_w * 0.064))) + rail_probe + 2
    right_pad = max(14, int(round(card_w * 0.058))) + 2
    top_pad = max(12, int(round(body_size * 0.72))) + 2
    bottom_pad = max(16, int(round(body_size * 0.90))) + 4
    inner_pad_x = left_pad
    content_width = max(40, x1 - x0 - inner_pad_x * 2)
    content_width = max(40, x1 - x0 - left_pad - right_pad)
    title_font, title_lines = _fit_pil_wrapped_text(
        draw,
        str(title or ""),
        font_file=LABEL_FONT_FILE,
        base_size=title_size,
        min_size=max(12, int(round(title_size * 0.90))),
        max_width=content_width,
        max_lines=max(1, int(title_max_lines)),
    )
    headline_font, headline_lines = _fit_pil_wrapped_text(
        draw,
        str(headline or ""),
        font_file=DISPLAY_FONT_FILE,
        base_size=headline_size,
        min_size=max(14, int(round(headline_size * 0.76))),
        max_width=content_width,
        max_lines=max(1, int(headline_max_lines)),
    )
    body_font, body_lines = _fit_pil_wrapped_text(
        draw,
        str(body or ""),
        font_file=LABEL_FONT_FILE,
        base_size=body_size,
        min_size=max(12, int(round(body_size * 0.78))),
        max_width=content_width,
        max_lines=max(1, int(body_max_lines)),
    )
    line_gap = max(4, int(round(body_size * 0.26)))
    section_gap = max(8, int(round(base_text_size * 0.42)))
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
    y1 = max(y1, y0 + top_pad + bottom_pad + total_h)
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
    inner_pad_x = max(left_pad, rail_offset + 2)

    available_h = max(1, y1 - y0 - top_pad - bottom_pad)
    current_y = y0 + top_pad
    if str(vertical_align).strip().lower() == "center" and total_h < available_h:
        current_y = y0 + top_pad + int(round((available_h - total_h) / 2.0))
    for idx, line in enumerate(title_lines):
        if title_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            str(line or "").upper(),
            font=title_font,
            fill=_bgr_to_rgb(accent),
        )
        _, line_h = _pil_text_size(draw, str(line or "").upper(), title_font)
        current_y += line_h
        if idx < len(title_lines) - 1:
            current_y += line_gap
    if title_lines:
        current_y += section_gap
    for idx, line in enumerate(headline_lines):
        if headline_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=headline_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, headline_font)
        current_y += line_h
        if idx < len(headline_lines) - 1:
            current_y += line_gap
    if headline_lines and body_lines:
        current_y += max(3, section_gap - 1)
    for idx, line in enumerate(body_lines):
        if body_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=body_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, body_font)
        current_y += line_h
        if idx < len(body_lines) - 1:
            current_y += line_gap
    return y1


def _draw_themed_insight_card(
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
    headline_max_lines: int = 4,
    body_max_lines: int = 3,
    unify_body_with_headline: bool = False,
) -> int:
    card_w = max(1, x1 - x0)
    scale = min(width, height)
    headline_text = str(headline or "").strip()
    body_text = str(body or "").strip()
    if headline_text:
        body_text = ""
    base_text_size = _phase_label_font_size(scale)
    title_size = max(15, int(round(base_text_size * 0.76 * max(0.7, float(title_scale_boost)))))
    headline_size = max(18, int(round(base_text_size * 0.90 * max(0.7, float(headline_scale_boost)))))
    body_size = max(14, int(round(base_text_size * 0.70 * max(0.7, float(body_scale_boost)))))
    body_font_file = LABEL_FONT_FILE
    if unify_body_with_headline:
        body_size = headline_size
        body_font_file = DISPLAY_FONT_FILE
    rail_probe = max(3, int(round(card_w * 0.020))) + max(3, int(round(min(width, height) * 0.006)))
    left_pad = max(16, int(round(card_w * 0.070))) + rail_probe + 3
    right_pad = max(14, int(round(card_w * 0.060))) + 3
    top_pad = max(14, int(round(body_size * 0.95)))
    bottom_pad = max(16, int(round(body_size * 1.00)))
    content_width = max(40, x1 - x0 - left_pad - right_pad)

    title_font, title_lines = _fit_pil_wrapped_text(
        draw,
        str(title or ""),
        font_file=LABEL_FONT_FILE,
        base_size=title_size,
        min_size=max(12, int(round(title_size * 0.90))),
        max_width=content_width,
        max_lines=2,
    )
    headline_font, headline_lines = _fit_pil_wrapped_text(
        draw,
        headline_text,
        font_file=DISPLAY_FONT_FILE,
        base_size=headline_size,
        min_size=max(15, int(round(headline_size * 0.82))),
        max_width=content_width,
        max_lines=max(1, int(headline_max_lines)),
    )
    body_font, body_lines = _fit_pil_wrapped_text(
        draw,
        body_text,
        font_file=body_font_file,
        base_size=body_size,
        min_size=max(12, int(round(body_size * 0.86))),
        max_width=content_width,
        max_lines=max(1, int(body_max_lines)),
    )

    title_gap = max(6, int(round(body_size * 0.45)))
    line_gap = max(3, int(round(body_size * 0.24)))
    section_gap = max(8, int(round(base_text_size * 0.36)))
    title_h = _pil_text_block_height(draw, title_lines, title_font, line_gap=line_gap)
    headline_h = _pil_text_block_height(draw, headline_lines, headline_font, line_gap=line_gap)
    body_h = _pil_text_block_height(draw, body_lines, body_font, line_gap=line_gap)

    total_h = 0
    if title_h:
        total_h += title_h
    if headline_h:
        if total_h:
            total_h += title_gap
        total_h += headline_h
    if body_h:
        if total_h:
            total_h += section_gap
        total_h += body_h

    y1 = max(y1, y0 + top_pad + bottom_pad + total_h)
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
    inner_pad_x = max(left_pad, rail_offset + 4)
    current_y = y0 + top_pad

    for idx, line in enumerate(title_lines):
        if title_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            str(line or "").upper(),
            font=title_font,
            fill=_bgr_to_rgb(accent),
        )
        _, line_h = _pil_text_size(draw, str(line or "").upper(), title_font)
        current_y += line_h
        if idx < len(title_lines) - 1:
            current_y += line_gap
    if title_lines and headline_lines:
        current_y += title_gap

    for idx, line in enumerate(headline_lines):
        if headline_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=headline_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY),
        )
        _, line_h = _pil_text_size(draw, line, headline_font)
        current_y += line_h
        if idx < len(headline_lines) - 1:
            current_y += line_gap
    if headline_lines and body_lines:
        current_y += section_gap

    for idx, line in enumerate(body_lines):
        if body_font is None:
            break
        draw.text(
            (x0 + inner_pad_x, current_y),
            line,
            font=body_font,
            fill=_bgr_to_rgb(THEME_TEXT_PRIMARY if unify_body_with_headline else THEME_TEXT_SECONDARY),
        )
        _, line_h = _pil_text_size(draw, line, body_font)
        current_y += line_h
        if idx < len(body_lines) - 1:
            current_y += line_gap

    return y1
