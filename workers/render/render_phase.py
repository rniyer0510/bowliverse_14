from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .render_constants import ACTIVE_EDGE, ACTIVE_FILL, INACTIVE_FILL, MUTED_TEXT, PANEL_BG, PANEL_EDGE, PHASES, TEXT_COLOR
from .render_helpers import _safe_int

def _overlay_panel(
    frame: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int],
    alpha: float = 0.78,
) -> None:
    x0 = max(0, min(frame.shape[1] - 1, x0))
    x1 = max(0, min(frame.shape[1], x1))
    y0 = max(0, min(frame.shape[0] - 1, y0))
    y1 = max(0, min(frame.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), edge_color, 2, cv2.LINE_AA)


def _phase_cut_points(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    last = max(start + 3, stop - 1)
    span = max(4, stop - start)
    bfc = _safe_int(((events or {}).get("bfc") or {}).get("frame"))
    ffc = _safe_int(((events or {}).get("ffc") or {}).get("frame"))
    uah = _safe_int(((events or {}).get("uah") or {}).get("frame"))
    release = _safe_int(((events or {}).get("release") or {}).get("frame"))
    phase_release = uah if uah is not None and uah >= (ffc if ffc is not None else start) else release

    cp1 = bfc if bfc is not None else start + int(round(span * 0.32))
    cp2 = ffc if ffc is not None else start + int(round(span * 0.58))
    cp3 = phase_release if phase_release is not None else start + int(round(span * 0.82))

    cp1 = max(start + 1, min(last - 2, cp1))
    cp2 = max(cp1 + 1, min(last - 1, cp2))
    cp3 = max(cp2 + 1, min(last, cp3))
    return cp1, cp2, cp3


def _phase_index_for_frame(
    frame_idx: int,
    *,
    cp1: int,
    cp2: int,
    cp3: int,
) -> int:
    if frame_idx < cp1:
        return 0
    if frame_idx < cp2:
        return 1
    if frame_idx < cp3:
        return 2
    return 3


def _draw_phase_rail(
    frame: np.ndarray,
    *,
    phase_idx: int,
    progress: float,
) -> None:
    width = frame.shape[1]
    height = frame.shape[0]
    rail_x0 = int(round(width * 0.05))
    rail_x1 = int(round(width * 0.95))
    rail_y0 = int(round(height * 0.90))
    rail_h = int(round(height * 0.05))
    gap = int(round(width * 0.012))
    segment_w = max(30, int((rail_x1 - rail_x0 - gap * (len(PHASES) - 1)) / len(PHASES)))

    for idx, phase in enumerate(PHASES):
        seg_x0 = rail_x0 + idx * (segment_w + gap)
        seg_x1 = seg_x0 + segment_w
        active = idx == phase_idx
        fill = ACTIVE_FILL if active else INACTIVE_FILL
        edge = ACTIVE_EDGE if active else PANEL_EDGE
        _overlay_panel(
            frame,
            x0=seg_x0,
            y0=rail_y0,
            x1=seg_x1,
            y1=rail_y0 + rail_h,
            fill_color=fill,
            edge_color=edge,
            alpha=0.88 if active else 0.72,
        )
        label = phase["title"].replace(" Contact", "")
        text_scale = max(0.38, min(width, height) / 1800.0)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        text_x = seg_x0 + max(10, (segment_w - text_size[0]) // 2)
        text_y = rail_y0 + int(rail_h * 0.63)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            TEXT_COLOR if active else MUTED_TEXT,
            1,
            cv2.LINE_AA,
        )

    tracker_x = rail_x0 + int(round((rail_x1 - rail_x0) * max(0.0, min(1.0, progress))))
    tracker_y0 = rail_y0 - int(round(height * 0.012))
    tracker_y1 = rail_y0 + rail_h + int(round(height * 0.012))
    cv2.line(frame, (tracker_x, tracker_y0), (tracker_x, tracker_y1), ACTIVE_EDGE, 2, cv2.LINE_AA)


def _draw_phase_overlay(
    frame: np.ndarray,
    *,
    frame_idx: int,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> None:
    cp1, cp2, cp3 = _phase_cut_points(start=start, stop=stop, events=events)
    phase_idx = _phase_index_for_frame(frame_idx, cp1=cp1, cp2=cp2, cp3=cp3)
    progress = 0.0 if stop <= start else float(frame_idx - start) / float(max(1, stop - start - 1))
    _draw_phase_rail(frame, phase_idx=phase_idx, progress=progress)


def _draw_top_risk_panel(
    frame: np.ndarray,
    *,
    title: str,
    headline: str,
    body: str,
    accent: Tuple[int, int, int],
) -> None:
    width = frame.shape[1]
    height = frame.shape[0]
    card_w = int(round(width * 0.58))
    card_h = int(round(height * 0.14))
    x0 = int(round(width * 0.05))
    y0 = int(round(height * 0.05))
    x1 = min(width - 18, x0 + card_w)
    y1 = y0 + card_h
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=accent,
        alpha=0.84,
    )
    cv2.putText(
        frame,
        title,
        (x0 + 18, y0 + int(card_h * 0.28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.44, min(width, height) / 1400.0),
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        headline,
        (x0 + 18, y0 + int(card_h * 0.58)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.58, min(width, height) / 1100.0),
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        body,
        (x0 + 18, y0 + int(card_h * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, min(width, height) / 1550.0),
        MUTED_TEXT,
        1,
        cv2.LINE_AA,
    )


