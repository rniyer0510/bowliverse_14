from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context

def _draw_phase_anchor_panel(
    frame: np.ndarray,
    *,
    phase_key: str,
) -> None:
    config = {
        "bfc": {
            "title": "Back Foot Contact",
            "headline": "Back foot contact.",
            "body": "Watch how the body is organising before the front foot lands.",
            "accent": (92, 220, 255),
        },
        "ffc": {
            "title": "Front Foot Contact",
            "headline": "Front foot contact.",
            "body": "Watch how the body arrives over the landing base here.",
            "accent": (90, 220, 255),
        },
        "release": {
            "title": "Release",
            "headline": "Release.",
            "body": "Watch how the upper body is sequencing as the ball comes out.",
            "accent": (0, 132, 255),
        },
    }.get(str(phase_key).strip().lower())
    if not config:
        return
    if Image is None or ImageDraw is None:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    scale = min(width, height)
    image, overlay, draw = _frame_draw_context(frame)
    title_font, title_lines = _fit_pil_wrapped_text(
        draw,
        str(config["headline"]),
        font_file=DISPLAY_FONT_FILE,
        base_size=max(28, int(round(scale * 0.052))),
        min_size=max(24, int(round(scale * 0.044))),
        max_width=max(270, int(round(width * 0.60))),
        max_lines=1,
    )
    body_font, body_lines = _fit_pil_wrapped_text(
        draw,
        str(config["body"]),
        font_file=BODY_FONT_MEDIUM_FILE,
        base_size=max(22, int(round(scale * 0.042))),
        min_size=max(19, int(round(scale * 0.036))),
        max_width=max(290, int(round(width * 0.64))),
        max_lines=2,
    )
    if title_font is None or body_font is None or not title_lines or not body_lines:
        return
    line_gap = max(4, int(round(scale * 0.008)))
    body_gap = max(8, int(round(scale * 0.014)))
    title_w, title_h = _pil_text_size(draw, title_lines[0], title_font)
    body_heights = [_pil_text_size(draw, line, body_font)[1] for line in body_lines]
    body_widths = [_pil_text_size(draw, line, body_font)[0] for line in body_lines]
    body_w = max(body_widths) if body_widths else 0
    body_h = sum(body_heights) + line_gap * max(0, len(body_lines) - 1)
    pad_x = max(20, int(round(scale * 0.038)))
    pad_y = max(16, int(round(scale * 0.028)))
    accent_w = 3
    panel_w = max(title_w, body_w) + pad_x * 2 + accent_w + 10
    panel_h = title_h + body_gap + body_h + pad_y * 2
    x0 = max(10, int(round(width * 0.06)))
    y0 = max(10, int(round(height * 0.08)))
    x1 = min(width - 10, x0 + panel_w)
    y1 = min(height - 10, y0 + panel_h)
    radius = max(8, int(round(scale * 0.016)))
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=(18, 18, 18, 230),
        outline=(255, 255, 255, 38),
    )
    draw.rounded_rectangle(
        (x0 + 6, y0 + 6, x0 + 6 + accent_w, y1 - 6),
        radius=max(4, accent_w * 2),
        fill=_bgr_to_rgb(config["accent"], 255),
    )
    text_x = x0 + pad_x + accent_w + 8
    title_y = y0 + pad_y
    draw.text((text_x, title_y), title_lines[0], font=title_font, fill=(255, 255, 255, 255))
    current_y = title_y + title_h + body_gap
    for idx, line in enumerate(body_lines):
        draw.text((text_x, current_y), line, font=body_font, fill=(255, 255, 255, 230))
        current_y += body_heights[idx] + line_gap
    _commit_frame_draw_context(frame, image, overlay)
