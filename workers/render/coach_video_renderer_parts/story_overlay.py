from __future__ import annotations
from .shared import *
from .themed_story import _draw_themed_story_card

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
    return
