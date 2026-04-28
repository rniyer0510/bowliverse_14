from __future__ import annotations
from .shared import *
from .themed_story import _draw_themed_story_card
from .pil_context import _frame_draw_context, _commit_frame_draw_context

def _draw_story_overlay_card(
    frame: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    title: str,
    headline: str,
    body: str,
    accent: Tuple[int, int, int],
    title_scale_boost: float = 1.0,
    headline_scale_boost: float = 1.0,
    body_scale_boost: float = 1.0,
) -> None:
    if Image is None or ImageDraw is None:
        return
    image, overlay, draw = _frame_draw_context(frame)
    _draw_themed_story_card(
        draw,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        title=title,
        headline=headline,
        body=body,
        accent=accent,
        width=frame.shape[1],
        height=frame.shape[0],
        title_scale_boost=title_scale_boost,
        headline_scale_boost=headline_scale_boost,
        body_scale_boost=body_scale_boost,
    )
    _commit_frame_draw_context(frame, image, overlay)
