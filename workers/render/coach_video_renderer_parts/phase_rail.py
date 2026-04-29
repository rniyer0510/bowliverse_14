from __future__ import annotations
from .shared import *
from .drawing_base import _apply_bottom_scrim, _overlay_panel
from .font_utils import _load_theme_font, _pil_text_size, _phase_label_font_size
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context
from .timeline_events import _phase_cut_points

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
    rail_y0 = int(round(height * 0.885))
    rail_h = int(round(height * 0.066))
    _apply_bottom_scrim(
        frame,
        start_y=int(round(height * 0.75)),
        end_y=height,
        max_alpha=0.55,
    )
    gap = int(round(width * 0.012))
    segment_w = max(30, int((rail_x1 - rail_x0 - gap * (len(PHASES) - 1)) / len(PHASES)))
    font_size = _phase_label_font_size(min(width, height))
    font = _load_theme_font(LABEL_FONT_FILE, font_size)
    measure_draw = None
    if font is not None and Image is not None and ImageDraw is not None:
        measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    text_instructions: List[Tuple[str, int, int, int, int, bool]] = []

    for idx, phase in enumerate(PHASES):
        seg_x0 = rail_x0 + idx * (segment_w + gap)
        seg_x1 = seg_x0 + segment_w
        active = idx == phase_idx
        done = idx < phase_idx
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
        label = str(phase.get("short_title") or phase.get("title") or "")
        if not label or measure_draw is None or font is None:
            continue
        text_w, text_h = _pil_text_size(measure_draw, label, font)
        text_x = seg_x0 + max(8, int(round((segment_w - text_w) / 2.0)))
        text_y = rail_y0 + max(4, int(round((rail_h - text_h) / 2.0)))
        text_instructions.append((label, text_x, text_y, text_w, text_h, active, done, seg_x0, seg_x1))

    tracker_x = rail_x0 + int(round((rail_x1 - rail_x0) * max(0.0, min(1.0, progress))))
    tracker_y0 = rail_y0 - int(round(height * 0.012))
    tracker_y1 = rail_y0 + rail_h + int(round(height * 0.012))
    cv2.line(frame, (tracker_x, tracker_y0), (tracker_x, tracker_y1), ACTIVE_EDGE, 2, cv2.LINE_AA)

    if not text_instructions or font is None or Image is None or ImageDraw is None:
        return
    image, overlay, draw = _frame_draw_context(frame)
    for label, text_x, text_y, text_w, text_h, active, done, seg_x0, seg_x1 in text_instructions:
        if active:
            pill_pad_x = 8
            pill_pad_y = 4
            draw.rounded_rectangle(
                (
                    text_x - pill_pad_x,
                    text_y - pill_pad_y,
                    text_x + text_w + pill_pad_x,
                    text_y + text_h + pill_pad_y,
                ),
                radius=4,
                fill=(255, 255, 255, 235),
            )
        elif done:
            underline_y = rail_y0 + rail_h - max(3, int(round(rail_h * 0.12)))
            underline_x0 = seg_x0 + max(10, int(round((seg_x1 - seg_x0) * 0.18)))
            underline_x1 = seg_x1 - max(10, int(round((seg_x1 - seg_x0) * 0.18)))
            draw.line(
                (underline_x0, underline_y, underline_x1, underline_y),
                fill=(255, 255, 255, 178),
                width=max(1, int(round(rail_h * 0.04))),
            )
        draw.text(
            (text_x, text_y),
            label,
            font=font,
            fill=(32, 36, 42, 255) if active else _bgr_to_rgb(MUTED_TEXT, 178 if done else 102),
        )
    _commit_frame_draw_context(frame, image, overlay)
def _draw_phase_overlay(
    frame: np.ndarray,
    *,
    frame_idx: int,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
    fps: float = 30.0,
) -> None:
    cp1, cp2, cp3 = _phase_cut_points(start=start, stop=stop, events=events, fps=fps)
    phase_idx = _phase_index_for_frame(frame_idx, cp1=cp1, cp2=cp2, cp3=cp3)
    progress = 0.0 if stop <= start else float(frame_idx - start) / float(max(1, stop - start - 1))
    _draw_phase_rail(frame, phase_idx=phase_idx, progress=progress)
