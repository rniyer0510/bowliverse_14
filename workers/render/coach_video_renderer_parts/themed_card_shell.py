from __future__ import annotations
from .shared import *
from .font_utils import _load_theme_font, _fit_pil_wrapped_text, _pil_text_size, _pil_text_block_height
from .pil_context import _bgr_to_rgb

def _draw_themed_card_shell(
    draw: Any,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    accent: Tuple[int, int, int],
    width: int,
    height: int,
) -> int:
    radius = max(16, int(round(min(width, height) * 0.028)))
    shadow_shift = max(3, int(round(min(width, height) * 0.008)))
    draw.rounded_rectangle(
        (x0 + shadow_shift, y0 + shadow_shift, x1 + shadow_shift, y1 + shadow_shift),
        radius=radius,
        fill=(0, 0, 0, 86),
    )
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=_bgr_to_rgb((14, 16, 15), 244),
        outline=_bgr_to_rgb((42, 44, 43), 128),
        width=1,
    )
    rail_w = max(5, int(round((x1 - x0) * 0.020)))
    rail_inset = max(3, int(round(min(width, height) * 0.006)))
    draw.rounded_rectangle(
        (x0 + rail_inset, y0 + rail_inset, x0 + rail_inset + rail_w, y1 - rail_inset),
        radius=max(rail_w * 2, 10),
        fill=_bgr_to_rgb(accent, 255),
    )
    return rail_inset + rail_w
def _draw_themed_summary_card(
    draw: Any,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    title: str,
    body: str,
    accent: Tuple[int, int, int],
    width: int,
    height: int,
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
    inner_pad_x = max(16, int(round(card_w * 0.070))) + rail_offset
    inner_pad_y = max(13, int(round(card_h * 0.11)))
    title_font = _load_theme_font(
        LABEL_FONT_FILE,
        max(20, int(round(card_h * 0.145))),
    )
    body_font, lines = _fit_pil_wrapped_text(
        draw,
        str(body or ""),
        font_file=BODY_FONT_MEDIUM_FILE,
        base_size=max(17, int(round(card_h * 0.115))),
        min_size=max(14, int(round(card_h * 0.085))),
        max_width=max(40, x1 - x0 - inner_pad_x * 2),
        max_lines=3,
    )
    if title_font is not None:
        draw.text(
            (x0 + inner_pad_x, y0 + inner_pad_y),
            str(title or "").upper(),
            font=title_font,
            fill=_bgr_to_rgb(accent),
        )
        _, title_h = _pil_text_size(draw, str(title or "").upper(), title_font)
    else:
        title_h = max(12, int(round(card_h * 0.18)))
    line_gap = max(4, int(round(min(width, height) * 0.008)))
    current_y = y0 + inner_pad_y + title_h + max(16, int(round(card_h * 0.20)))
    for line in lines:
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
