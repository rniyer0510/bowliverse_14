from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context

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
    scale = min(width, height)
    image, overlay, draw = _frame_draw_context(frame)
    dim = Image.new("RGBA", image.size, (0, 0, 0, 26))
    overlay = Image.alpha_composite(dim, overlay)
    draw = ImageDraw.Draw(overlay)
    font, lines = _fit_pil_wrapped_text(
        draw,
        message,
        font_file=BODY_FONT_FILE,
        base_size=max(28, int(round(scale * 0.043))),
        min_size=max(22, int(round(scale * 0.033))),
        max_width=max(220, int(round(width * 0.56))),
        max_lines=2,
    )
    if font is None or not lines:
        return

    line_gap = max(5, int(round(scale * 0.010)))
    text_heights = [_pil_text_size(draw, line, font)[1] for line in lines]
    text_widths = [_pil_text_size(draw, line, font)[0] for line in lines]
    text_w = max(text_widths) if text_widths else 0
    text_h = sum(text_heights) + line_gap * max(0, len(lines) - 1)
    pad_x = max(26, int(round(scale * 0.030)))
    pad_y = max(18, int(round(scale * 0.022)))
    bubble_w = text_w + pad_x * 2
    bubble_h = text_h + pad_y * 2
    margin = max(18, int(round(scale * 0.022)))
    top_band_y = max(margin, int(round(height * 0.08)))
    second_band_y = max(margin, int(round(height * 0.20)))
    if anchor[0] < width * 0.48:
        primary_x0 = width * 0.34
        secondary_x0 = width * 0.10
    else:
        primary_x0 = width * 0.10
        secondary_x0 = width * 0.34

    candidates = [
        (primary_x0, top_band_y),
        (secondary_x0, top_band_y),
        (primary_x0, second_band_y),
        (secondary_x0, second_band_y),
    ]
    chosen_x0 = margin
    chosen_y0 = top_band_y
    best_score = float("-inf")
    for cand_x0, cand_y0 in candidates:
        x0 = max(margin, min(width - bubble_w - margin, int(cand_x0)))
        y0 = max(margin, min(height - bubble_h - margin, int(cand_y0)))
        x1 = x0 + bubble_w
        y1 = y0 + bubble_h
        bubble_center_x = (x0 + x1) / 2.0
        bubble_center_y = (y0 + y1) / 2.0
        distance = float(np.hypot(bubble_center_x - anchor[0], bubble_center_y - anchor[1]))
        top_bonus = 26.0 if y0 <= second_band_y + 4 else 8.0
        side_bonus = 12.0 if ((anchor[0] < width * 0.5 and x0 > width * 0.28) or (anchor[0] >= width * 0.5 and x0 < width * 0.28)) else 0.0
        edge_penalty = abs(bubble_center_x - width * 0.5) * 0.012
        score = top_bonus + side_bonus - distance * 0.016 - edge_penalty
        if score > best_score:
            best_score = score
            chosen_x0 = x0
            chosen_y0 = y0

    x0 = int(chosen_x0)
    y0 = int(chosen_y0)
    x1 = int(x0 + bubble_w)
    y1 = int(y0 + bubble_h)
    radius = max(22, int(round(scale * 0.028)))
    tail_base_y = y1 if anchor[1] >= y1 else y0
    tail_direction = 1 if tail_base_y == y1 else -1
    tail_base_x = int(max(x0 + radius, min(x1 - radius, anchor[0] * 0.82 + ((x0 + x1) / 2.0) * 0.18)))
    tail_half_w = max(14, int(round(scale * 0.020)))
    tail_tip = (
        int(anchor[0]),
        int(anchor[1]),
    )
    tail_left = (tail_base_x - tail_half_w, tail_base_y)
    tail_right = (tail_base_x + tail_half_w, tail_base_y)
    tail_mid = (
        tail_base_x,
        tail_base_y + tail_direction * max(10, int(round(scale * 0.020))),
    )

    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=(18, 18, 18, 230),
        outline=(255, 255, 255, 38),
        width=max(2, int(round(scale * 0.0035))),
    )
    draw.polygon(
        [tail_left, tail_mid, tail_tip, tail_mid, tail_right],
        fill=(18, 18, 18, 225),
        outline=(255, 255, 255, 38),
    )
    accent_w = max(5, int(round(bubble_w * 0.028)))
    draw.rounded_rectangle(
        (x0 + 6, y0 + 6, x0 + 6 + accent_w, y1 - 6),
        radius=max(6, accent_w * 2),
        fill=_bgr_to_rgb(accent, 255),
    )

    text_x = x0 + pad_x + accent_w + max(8, int(round(scale * 0.010)))
    current_y = y0 + pad_y
    for idx, line in enumerate(lines):
        draw.text(
            (text_x, current_y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
        )
        current_y += text_heights[idx] + line_gap
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
    if anchor is None:
        return
    bubble_text = _bubble_copy(title=title, headline=headline, body=body)
    if not bubble_text:
        return
    _draw_pointer_bubble(
        frame,
        anchor=anchor,
        text=bubble_text,
        accent=accent,
    )
