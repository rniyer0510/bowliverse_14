from __future__ import annotations
from .shared import *
from .bubble_base import _draw_top_risk_panel

def _draw_phase_anchor_panel(
    frame: np.ndarray,
    *,
    phase_key: str,
) -> None:
    config = {
        "bfc": {
            "title": "Back Foot Contact",
            "headline": "Back foot lands here.",
            "body": "Pause here to see how the body is entering the crease.",
            "accent": (92, 220, 255),
        },
        "ffc": {
            "title": "Front Foot Contact",
            "headline": "Front foot lands here.",
            "body": "Pause here to see how the landing sets up the release.",
            "accent": (90, 220, 255),
        },
        "release": {
            "title": "Release",
            "headline": "Ball comes out here.",
            "body": "Pause here to see what the chain is doing at release.",
            "accent": (0, 132, 255),
        },
    }.get(str(phase_key).strip().lower())
    if not config:
        return
    _draw_top_risk_panel(
        frame,
        title=str(config["title"]),
        headline=str(config["headline"]),
        body=str(config["body"]),
        accent=config["accent"],
    )
