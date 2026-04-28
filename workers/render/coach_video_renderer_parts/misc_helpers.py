from __future__ import annotations
from .shared import *

def _top_risk_panel_metrics(width: int, height: int) -> Dict[str, Any]:
    card_w = int(round(width * 0.66))
    card_h = int(round(height * 0.145))
    return {
        "card_w": card_w,
        "card_h": card_h,
        "title_scale": max(0.46, min(width, height) / 1320.0),
        "headline_base_scale": max(0.62, min(width, height) / 980.0),
        "headline_min_scale": max(0.48, min(width, height) / 1280.0),
        "headline_max_lines": 2,
        "body_base_scale": max(0.38, min(width, height) / 1480.0),
        "body_min_scale": max(0.31, min(width, height) / 1780.0),
        "body_max_lines": 1,
    }
def _summary_telemetry_layout(width: int, height: int) -> Dict[str, Any]:
    left_x = int(round(width * 0.04))
    gap = int(round(width * 0.024))
    stat_h = int(round(height * 0.085))
    stat_w = int(round((width - (left_x * 2) - (gap * 2)) / 3.0))
    return {
        "left_x": left_x,
        "gap": gap,
        "stat_h": stat_h,
        "stat_w": stat_w,
    }
def _story_headline_and_support(text: str) -> Tuple[str, str]:
    parts = [
        segment.strip()
        for segment in str(text or "").replace("\n", " ").split(".")
        if segment.strip()
    ]
    if not parts:
        return "", ""
    headline = parts[0]
    support = ". ".join(parts[1:]).strip()
    if support and not support.endswith("."):
        support = f"{support}."
    return headline, support
