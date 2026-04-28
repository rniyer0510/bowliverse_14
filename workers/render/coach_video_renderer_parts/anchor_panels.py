from __future__ import annotations
from .shared import *
from .pil_context import _frame_draw_context, _commit_frame_draw_context
from .themed_story import _draw_themed_story_card

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
    image, overlay, draw = _frame_draw_context(frame)
    card_x0 = int(round(width * 0.05))
    card_y0 = max(92, int(round(height * 0.145)))
    card_x1 = int(round(width * 0.74))
    card_y1 = card_y0 + int(round(height * 0.16))
    _draw_themed_story_card(
        draw,
        x0=card_x0,
        y0=card_y0,
        x1=card_x1,
        y1=card_y1,
        title=str(config["title"]),
        headline=str(config["headline"]),
        body=str(config["body"]),
        accent=tuple(config["accent"]),
        width=width,
        height=height,
    )
    _commit_frame_draw_context(frame, image, overlay)
