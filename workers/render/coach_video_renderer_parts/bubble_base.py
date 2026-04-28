from __future__ import annotations
from .shared import *
from .font_utils import _fit_pil_wrapped_text, _pil_text_size, _phase_label_font_size
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

def _compact_pointer_copy(text: str) -> str:
    cleaned = " ".join(str(text or "").replace(".", " ").replace(",", " ").split())
    if not cleaned:
        return ""
    words = cleaned.split(" ")
    if len(words) <= 4:
        return cleaned
    drop_words = {
        "a", "an", "and", "at", "for", "from", "here", "into", "of", "the",
        "then", "through", "to",
    }
    reduced = [word for word in words if word.lower() not in drop_words]
    if 1 <= len(reduced) <= 4:
        return " ".join(reduced)
    if len(reduced) > 4:
        return " ".join(reduced[:4])
    return " ".join(words[:4])
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

def _explanation_top_y(height: int) -> int:
    return max(92, int(round(height * 0.145)))

def _story_card_layout(
    *,
    width: int,
    height: int,
    anchor: Optional[Tuple[int, int]] = None,
    bowler_hand: Optional[str] = None,
    width_ratio: float = 0.28,
    margin_ratio: float = 0.05,
    top_height_ratio: float = 0.16,
) -> Dict[str, Any]:
    margin = int(round(width * margin_ratio))
    card_w = max(80, int(round(width * width_ratio)))
    top_y = max(84, int(round(height * 0.16)))
    card_h = int(round(height * top_height_ratio))
    side = "left"
    hand_hint = str(bowler_hand or "").strip().upper()
    if hand_hint == "L":
        side = "right"
    elif hand_hint == "R":
        side = "left"
    elif anchor is not None:
        left_space = max(0, anchor[0] - margin)
        right_space = max(0, (width - margin) - anchor[0])
        side = "right" if right_space >= left_space else "left"
    if side == "right":
        x1 = width - margin
        x0 = max(margin, x1 - card_w)
    else:
        x0 = margin
        x1 = min(width - margin, x0 + card_w)
    usable_w = max(1, x1 - x0)
    edge_inset = max(12, int(round(usable_w * 0.08)))
    line_x = x1 - edge_inset if side == "left" else x0 + edge_inset
    return {
        "x0": x0,
        "y0": top_y,
        "x1": x1,
        "y1": top_y + card_h,
        "line_x": line_x,
        "side": side,
    }

def _draw_pointer_bubble(
    frame: np.ndarray,
    *,
    anchor: Tuple[int, int],
    text: str,
    accent: Tuple[int, int, int],
    bowler_hand: Optional[str] = None,
) -> None:
    message = _compact_pointer_copy(text)
    if not message or Image is None or ImageDraw is None:
        return
    width = frame.shape[1]
    height = frame.shape[0]
    image, overlay, draw = _frame_draw_context(frame)
    layout = _story_card_layout(width=width, height=height, anchor=anchor, bowler_hand=bowler_hand)
    card_x0 = int(layout["x0"])
    card_y0 = int(layout["y0"])
    card_x1 = int(layout["x1"])
    card_y1 = int(layout["y1"])
    card_w = max(1, card_x1 - card_x0)
    scale = min(width, height)
    headline_base = _phase_label_font_size(scale)
    rail_probe = max(3, int(round(card_w * 0.020))) + max(3, int(round(scale * 0.006)))
    inner_pad_x = max(16, int(round(card_w * 0.070))) + rail_probe
    content_width = max(40, card_x1 - card_x0 - inner_pad_x * 2)
    headline_font, headline_lines = _fit_pil_wrapped_text(
        draw,
        message,
        font_file=BODY_FONT_FILE,
        base_size=headline_base,
        min_size=headline_base,
        max_width=content_width,
        max_lines=2,
    )
    line_gap = max(5, int(round(headline_base * 0.30)))
    inner_pad_y = max(12, int(round(headline_base * 0.65)))
    total_h = 0
    if headline_font is not None and headline_lines:
        total_h = sum(_pil_text_size(draw, line, headline_font)[1] for line in headline_lines)
        total_h += line_gap * max(0, len(headline_lines) - 1)
    card_y1 = max(card_y1, card_y0 + inner_pad_y * 2 + total_h)
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
    line_x = int(layout["line_x"])
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
    bowler_hand: Optional[str] = None,
) -> None:
    if Image is None or ImageDraw is None:
        return
    if not any(str(part or "").strip() for part in (title, headline, body)):
        return
    width = frame.shape[1]
    height = frame.shape[0]
    image, overlay, draw = _frame_draw_context(frame)
    layout = _story_card_layout(width=width, height=height, anchor=anchor, bowler_hand=bowler_hand)
    card_x0 = int(layout["x0"])
    card_y0 = int(layout["y0"])
    card_x1 = int(layout["x1"])
    card_y1 = int(layout["y1"])
    final_y1 = _draw_themed_story_card(
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
        headline_max_lines=3,
        body_max_lines=2,
        vertical_align="top",
    )
    if anchor is not None:
        scale = min(width, height)
        line_x = int(layout["line_x"])
        line_y0 = int(final_y1) + max(6, int(round(scale * 0.010)))
        line_y1 = line_y0 + max(30, int(round(scale * 0.070)))
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
